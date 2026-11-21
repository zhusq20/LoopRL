from types import SimpleNamespace as NS

import pytest
import torch

from looprl.native_2b import parse_trace, post_process_rewards
from slime_plugins.ouro_training import begin_audit, record_reuse, finish_reuse_audit
from test_ouro_training import options, fixture


def test_trace_alignment_and_rejection():
    assert parse_trace([[[0.,0.],[4.,-.2]],[1.,-2.]],2)==([4,1],[-.2,-2.])
    for raw in ([], [[4.,-.2]], [[0.,-.1],[4.,-.1]], [[4.,float('nan')],[1.,-.1]]):
        with pytest.raises(ValueError):parse_trace(raw,2)


def sample(index, exits, reward):
    return NS(index=index,group_index=0,weight_versions=['1'],response_length=len(exits),
        tokens=[10]*3+[20]*len(exits),metadata={},prompt='q',response='a',label='1',reward=reward,
        status=NS(name='COMPLETED'),rollout_log_probs=[-1.]*len(exits),
        get_reward_value=lambda args:reward,
        train_metadata=dict(ouro_step=1,ouro_beta=0.,ouro_exits=exits,ouro_gate_shift=3.,
            ouro_gate_log_probs=[-.2]*len(exits),ouro_token_log_probs=[-.8]*len(exits),
            executed_loop_tokens=12+sum(exits[1:])))


def test_price_counts_actual_decode_once_and_zero_price_ignores_depth(tmp_path):
    args=NS(ouro_beta=0.,ouro_total_steps=2,ouro_price_warmup=0.,ouro_gate_shift=3.,
            rollout_max_response_len=8,ouro_output=str(tmp_path))
    samples=[sample(0,[1,1,1],1.),sample(1,[4,4,4],1.)]
    assert post_process_rewards(args,samples)[1]==[0.,0.]
    args.ouro_beta=.2
    for s in samples:s.train_metadata['ouro_beta']=.2
    advantage=torch.tensor(post_process_rewards(args,samples)[1])
    # Prefill is dense even when the first readout exits early. Only the last
    # two exit actions change executed cost: 6 loops, not 9.
    torch.testing.assert_close(advantage,torch.tensor([.2*6/32/2,-.2*6/32/2]))
    samples[0].train_metadata['executed_loop_tokens']+=1
    with pytest.raises(ValueError,match='executed compute'):post_process_rewards(args,samples)


def test_joint_score_recovers_expected_reward_minus_cost_gradient():
    # Enumerate two decisions, each selecting exit and emitted token. The
    # second policy depends on both first actions, representing ragged history.
    torch.manual_seed(7)
    gate=torch.randn(2,2,2,dtype=torch.float64,requires_grad=True)
    token=torch.randn(2,2,2,2,dtype=torch.float64,requires_grad=True)
    exact=gate.new_zeros(());score=gate.new_zeros(())
    for e0 in range(2):
        for y0 in range(2):
            for e1 in range(2):
                for y1 in range(2):
                    logp=gate[0,0].log_softmax(-1)[e0]+token[0,0,e0].log_softmax(-1)[y0]
                    logp=logp+gate[1,y0].log_softmax(-1)[e1]+token[1,e0,e1].log_softmax(-1)[y1]
                    utility=float(y1==1)-.17*(4+e1+1)/8
                    exact=exact+logp.exp()*utility
                    score=score+logp.exp().detach()*utility*logp
    for actual,expected in zip(torch.autograd.grad(score,(gate,token),retain_graph=True),torch.autograd.grad(exact,(gate,token))):
        torch.testing.assert_close(actual,expected,atol=1e-12,rtol=1e-12)


def test_gate_audit_is_independent_of_joint_cancellation(tmp_path):
    data=fixture();q=torch.tensor([-.2,-.3,-.4,-.5,-.6])
    for m,part in zip(data['metadata'],q.split(data['response_lengths'])):m['ouro_gate_log_probs']=part.tolist()
    audits=dict(zip(data['partition'],data['log_probs']));gates=dict(zip(data['partition'],q.split(data['response_lengths'])))
    args=options(tmp_path,ouro_token_exit=True,custom_loss_function_path='looprl.native_2b.policy_loss')
    with pytest.raises(ValueError,match='separate gate'):begin_audit(args,data,audits)
    begin_audit(args,data,audits,gates)
    with pytest.raises(ValueError,match='gate grad/no-grad'):record_reuse(args,data,torch.cat(data['log_probs']),q+.01)
    begin_audit(args,data,audits,gates)
    record_reuse(args,data,torch.cat(data['log_probs']),q)
    report=finish_reuse_audit(args)
    assert report['policy_space']=='joint_exit_token' and report['gate_grad_no_grad_max_error']==0


def test_missing_resume_cannot_silently_start_base_training(tmp_path):
    import subprocess
    import sys
    from pathlib import Path
    script=Path(__file__).resolve().parents[2]/'scripts/run_native_2a.py'
    result=subprocess.run([sys.executable,str(script),'--output',str(tmp_path/'out'),
                           '--resume',str(tmp_path/'missing'),'--resume-step','1'],
                          capture_output=True,text=True)
    assert result.returncode!=0 and 'checkpoint directory does not exist' in result.stderr
    assert not (tmp_path/'out').exists()


def test_short_execution_boundary_preserves_full_optimizer_schedule():
    from slime.backends.megatron_utils.model import get_optimizer_param_scheduler
    def make(stop,restore=False):
        args=NS(num_rollout=stop,rollout_batch_size=4,n_samples_per_prompt=8,global_batch_size=32,
            ouro_total_steps=140,lr_decay_iters=140,lr_wsd_decay_iters=None,lr_warmup_fraction=None,
            lr_warmup_iters=14,lr_warmup_init=0.,lr=1e-6,min_lr=0.,lr_decay_style='constant',
            start_weight_decay=.1,end_weight_decay=.1,weight_decay_incr_style='constant',
            use_checkpoint_opt_param_scheduler=restore,override_opt_param_scheduler=False,
            lr_wsd_decay_style='linear')
        optimizer=torch.optim.AdamW([torch.nn.Parameter(torch.ones(1))],lr=1e-6)
        return get_optimizer_param_scheduler(args,optimizer)
    original=make(2);original.step(64)
    resumed=make(3);resumed.load_state_dict(original.state_dict())
    assert resumed.num_steps==64 and resumed.wd_incr_steps==140*32
    resumed.step(32);assert resumed.num_steps==96
    legacy=original.state_dict();legacy['wd_incr_steps']=64
    restored=make(3,restore=True);restored.load_state_dict(legacy)
    assert restored.num_steps==64 and restored.wd_incr_steps==64
    assert restored.start_wd==restored.end_wd==.1
