import sys

import pytest

from scripts import advance_native_2a as runner


def test_direct_formal_runs_full_selected_arms_without_pilots(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(runner.Queue, "train", lambda self, name, **kw: calls.append((name, kw)))
    monkeypatch.setattr(runner.Queue, "wait_run", lambda *a: pytest.fail("Unexpected acceptance wait"))
    monkeypatch.setattr(sys, "argv", ["queue", "--direct-formal", "--output", str(tmp_path),
        "--arm-names", "stratified-beta0", "stratified-beta010"])
    runner.main()
    assert [name for name, _ in calls] == ["stratified-beta0", "stratified-beta010"]
    for _, kw in calls:
        assert (kw["steps"], kw["prompts"], kw["group"]) == (140, 32, 8)
        assert "stop" not in kw and "resume" not in kw


def test_direct_formal_stops_scan_after_failed_arm(monkeypatch, tmp_path):
    calls = []
    def fail(self, name, **kw):
        calls.append(name)
        raise RuntimeError("numerical guard failed")
    monkeypatch.setattr(runner.Queue, "train", fail)
    monkeypatch.setattr(sys, "argv", ["queue", "--direct-formal", "--output", str(tmp_path)])
    with pytest.raises(RuntimeError, match="numerical guard failed"):
        runner.main()
    assert calls == ["fixed4-beta0"]


def test_fixed_formal_queue_accepts_two_gpus(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(runner.Queue, "train", lambda self, name, **kw: calls.append((name, self.gpus)))
    monkeypatch.setattr(sys, "argv", ["queue", "--direct-formal", "--output", str(tmp_path),
        "--gpus", "4", "5", "--arm-names", "fixed4-beta0"])
    runner.main()
    assert calls == [("fixed4-beta0", ["4", "5"])]


def test_serving_upgrade_never_changes_training_protocol():
    from scripts.run_native_2a import validate_resume_protocol
    previous=dict(arm='stratified',total_steps=140,precision='fp32',beta=0.,batched_decode=True)
    current=dict(previous,serving=dict(attention_implementation='paged-fused',max_running_requests=8,kv_token_capacity=16384))
    with pytest.raises(ValueError):
        validate_resume_protocol(previous,current)
    change=validate_resume_protocol(previous,current,True)
    assert change['previous']['attention_implementation']=='reference'
    with pytest.raises(ValueError):
        validate_resume_protocol(previous,dict(current,beta=.1),True)


def test_resumed_formal_queue_resumes_only_first_arm(monkeypatch,tmp_path):
    calls=[]
    monkeypatch.setattr(runner.Queue,'train',lambda self,name,**kw:calls.append((name,kw)))
    monkeypatch.setattr(sys,'argv',['queue','--direct-formal','--output',str(tmp_path),
        '--arm-names','stratified-beta0','stratified-beta002',
        '--resume-first-run',str(tmp_path/'old'),'--resume-boundary','4',
        '--attention-implementation','paged-fused'])
    runner.main()
    assert calls[0][1]['boundary']==4
    assert 'resume' not in calls[1][1]


def test_upgrade_waits_for_model_and_rollout_checkpoint(tmp_path):
    from scripts.upgrade_native_2a_queue import checkpoint_signature
    root=tmp_path/'checkpoints'
    model=root/'iter_0000004'
    model.mkdir(parents=True)
    (root/'rollout').mkdir()
    (root/'latest_checkpointed_iteration.txt').write_text('4')
    (model/'.metadata').write_bytes(b'metadata')
    assert checkpoint_signature(tmp_path,4) is None
    shard=model/'__0_0.distcp'
    shard.write_bytes(b'model optimizer rng')
    assert checkpoint_signature(tmp_path,4) is None
    (root/'rollout/global_dataset_state_dict_4.pt').write_bytes(b'dataset cursor')
    signature=checkpoint_signature(tmp_path,4)
    assert signature
    shard.write_bytes(b'changed')
    assert checkpoint_signature(tmp_path,4)!=signature
    (root/'latest_checkpointed_iteration.txt').write_text('3')
    assert checkpoint_signature(tmp_path,4) is None


def test_upgrade_checks_process_ownership(monkeypatch,tmp_path):
    from scripts import upgrade_native_2a_queue as upgrade
    class Process:
        def cmdline(self):
            return ['python','advance_native_2a.py','--output',str(tmp_path/'other')]
    monkeypatch.setattr(upgrade.psutil,'Process',lambda pid:Process())
    with pytest.raises(RuntimeError,match='different output'):
        upgrade.checked_process(123,tmp_path/'intended','advance_native_2a.py')


def test_bf16_fixed_replicas_preserve_formal_training_spec(monkeypatch,tmp_path):
    calls=[]
    monkeypatch.setattr(runner.Queue,'train',lambda self,name,**kw:calls.append((self.gpus,self.run_options,kw)))
    monkeypatch.setattr(sys,'argv',['queue','--direct-formal','--output',str(tmp_path),
        '--gpus','4','5','6','--arm-names','fixed4-beta0','--depth4-replicas','2',
        '--precision','bf16','--rollout-correction','tis','--attention-implementation','bf16-fast',
        '--cuda-graph','--max-running-requests','32','--kv-token-capacity','65536'])
    runner.main()
    gpus,options,training=calls[0]
    assert gpus==['4','5','6'] and options['depth4_replicas']==2
    assert options['precision']=='bf16' and options['rollout_correction']=='tis'
    assert (training['steps'],training['prompts'],training['group'])==(140,32,8)


def test_resume_cannot_mix_fp32_and_bf16_protocols():
    from scripts.run_native_2a import validate_resume_protocol
    old=dict(arm='stratified',total_steps=140,precision='fp32',beta=0.)
    with pytest.raises(ValueError,match='protocol'):
        validate_resume_protocol(old,dict(old,precision='bf16',rollout_correction=dict(mode='tis')),True)
