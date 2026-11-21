from types import SimpleNamespace as NS

import pytest
import torch

from slime_plugins.ouro_2a import policy_loss
from slime_plugins.ouro_training import (begin_audit, finish_reuse_audit,
    homogeneous_schedule, validate_reuse_config)


def options(tmp_path, **kw):
    values = dict(num_steps_per_rollout=1, use_rollout_logprobs=False, kl_coef=0,
        loss_type="custom_loss", custom_loss_function_path="looprl.native_2a.policy_loss",
        advantage_estimator="grpo", use_tis=True, ouro_rollout_correction="tis",
        tis_clip=2., tis_clip_low=0., calculate_per_token_loss=False,
        context_parallel_size=1, rollout_temperature=1., rollout_top_p=1.,
        eps_clip=.2, eps_clip_high=.2, rollout_max_response_len=2048,
        use_kl_loss=True, kl_loss_coef=.001, ouro_output=str(tmp_path))
    values.update(kw)
    return NS(**values)


def fixture():
    old = [torch.tensor([-.5, -1., -.7]), torch.tensor([-.2, -.8])]
    sampled = [old[0] - torch.tensor([.25, .8, 4.]).log(), old[1] - .02]
    data = dict(log_probs=old, rollout_log_probs=sampled,
        ref_log_probs=[x + .03 for x in old], advantages=[torch.tensor([2., -1., .5]), torch.tensor([0., -1.])],
        loss_masks=[torch.tensor([1., 1., 0.]), torch.ones(2)],
        partition=[7, 19], response_lengths=[3, 2], total_lengths=[5, 7],
        unconcat_tokens=[], metadata=[dict(ouro_depth=2, ouro_step=3), dict(ouro_depth=4, ouro_step=3)],
        num_microbatches=[2])
    return data


def test_reuse_preserves_loss_and_gradient_with_kl_tis_masks_and_zero_advantage(monkeypatch, tmp_path):
    import slime.backends.megatron_utils.loss as native
    monkeypatch.setattr(native, "get_log_probs_and_entropy", lambda logits, **kw:
        (None, {"log_probs": list(logits.split(kw["response_lengths"]))}))
    data = fixture()
    baseline = torch.cat(data["log_probs"]).requires_grad_()
    expected_loss, _ = policy_loss(options(tmp_path), data, baseline, None)
    expected_grad = torch.autograd.grad(expected_loss, baseline)[0]
    args = options(tmp_path, ouro_reuse_old_log_probs=True)
    begin_audit(args, data, dict(zip(data["partition"], data["log_probs"])))
    current = baseline.detach().clone().requires_grad_()
    batch = dict(data)
    del batch["log_probs"]
    actual_loss, _ = policy_loss(args, batch, current, None)
    actual_grad = torch.autograd.grad(actual_loss, current)[0]
    torch.testing.assert_close(actual_loss, expected_loss, rtol=0, atol=0)
    torch.testing.assert_close(actual_grad, expected_grad, rtol=0, atol=0)
    assert actual_grad[3] != 0  # Zero advantage still retains the KL gradient.
    report = finish_reuse_audit(args)
    assert report["samples"] == 2 and report["tokens"] == 5 and report["audit_samples"] == 2


def test_audit_rejects_mismatch_missing_samples_and_multi_update(monkeypatch, tmp_path):
    import slime.backends.megatron_utils.loss as native
    monkeypatch.setattr(native, "get_log_probs_and_entropy", lambda logits, **kw:
        (None, {"log_probs": list(logits.split(kw["response_lengths"]))}))
    args, data = options(tmp_path, ouro_reuse_old_log_probs=True), fixture()
    audits = dict(zip(data["partition"], data["log_probs"]))
    begin_audit(args, data, audits)
    with pytest.raises(ValueError, match="Incomplete"):
        finish_reuse_audit(args)
    with pytest.raises(ValueError, match="replay mismatch"):
        policy_loss(args, data, torch.cat(data["log_probs"]) + .01, None)
    with pytest.raises(ValueError, match="one deterministic"):
        validate_reuse_config(options(tmp_path, num_steps_per_rollout=2), data)
    data["rollout_log_probs"][0][0] = float("nan")
    with pytest.raises(ValueError, match="Invalid Ouro reference"):
        begin_audit(args, data, audits)


def test_grouped_schedule_preserves_every_sample_and_depth():
    metadata = [dict(ouro_depth=k) for k in [4, 2, 3, 2, 4, 4, 3]]
    schedule = homogeneous_schedule(metadata, [8, 3, 4, 2, 7, 9, 8], 2)
    assert sorted(i for batch in schedule for i in batch) == list(range(7))
    assert all(len(batch) <= 2 and len({metadata[i]["ouro_depth"] for i in batch}) == 1 for batch in schedule)


def test_packing_and_dp_preserve_global_loss_and_gradient_normalization(monkeypatch, tmp_path):
    import slime.backends.megatron_utils.loss as native
    monkeypatch.setattr(native, "get_log_probs_and_entropy", lambda logits, **kw:
        (None, {"log_probs": list(logits.split(kw["response_lengths"]))}))
    monkeypatch.setattr(native, "get_sum_of_sample_mean", lambda *a, **kw: None)
    monkeypatch.setattr(native, "load_function", lambda path: policy_loss)
    data = fixture()
    data["rollout_mask_sums"] = None
    args = options(tmp_path, recompute_loss_function=False, allgather_cp=False)
    expected = torch.cat(data["log_probs"]).clone().requires_grad_()
    expected_loss, _ = policy_loss(args, data, expected, None)
    expected_gradient = torch.autograd.grad(expected_loss/2, expected)[0]
    for world, schedule in [(1, [[0], [1]]), (1, [[0, 1]]), (2, [[0], [1]])]:
        monkeypatch.setattr(native.mpu, "get_data_parallel_world_size", lambda **kw: world)
        current = torch.cat(data["log_probs"]).clone().requires_grad_()
        pieces = current.split(data["response_lengths"])
        accumulated = 0
        microbatches_per_rank = len(schedule) if world == 1 else 1
        for indices in schedule:
            batch = {k: [v[i] for i in indices] if isinstance(v, list) and len(v) == 2 else v for k,v in data.items()}
            loss, _, _ = native.loss_function(args, batch, microbatches_per_rank, 2,
                                               torch.cat([pieces[i] for i in indices]))
            # Native schedule divides by microbatch count; DP averages ranks.
            accumulated = accumulated + loss/microbatches_per_rank/world
        torch.testing.assert_close(torch.autograd.grad(accumulated, current)[0], expected_gradient)


def test_training_upgrade_cannot_change_algorithm_or_precision():
    from scripts.run_native_2a import validate_resume_protocol
    old = dict(arm="stratified", precision="bf16", total_steps=140, beta=.1,
               activation_recompute=dict(granularity="full", method="uniform", layers=1))
    new = dict(old, training_execution=dict(reuse_old_log_probs=True, micro_batch_size=1,
                data_parallel_size=1, group_by_depth=False))
    assert validate_resume_protocol(old, new, allow_training_upgrade=True)
    for key, value in [("precision", "fp32"), ("beta", 0.), ("total_steps", 70)]:
        with pytest.raises(ValueError, match="protocol"):
            validate_resume_protocol(old, dict(new, **{key: value}), allow_training_upgrade=True)
