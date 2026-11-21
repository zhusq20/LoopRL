import asyncio
import concurrent.futures
from types import SimpleNamespace as NS

import pytest

from looprl.native_2a import price_at_step, reward, sample_seed, stratified_depth


def test_schedule_is_absolute_across_resume_and_keeps_price_units():
    uninterrupted = [price_at_step(.1, i, 140, .3) for i in range(140)]
    resumed = [price_at_step(.1, i, 140, .3) for i in range(10, 140)]
    assert uninterrupted[10:] == resumed
    assert uninterrupted[0] == 0 and uninterrupted[42] == .1
    assert price_at_step(.2, 10, 140, .3)/(4*4096) == price_at_step(.1, 10, 140, .3)/(4*2048)


def test_stratified_groups_are_balanced_and_completion_order_independent():
    for group in range(12):
        values = [stratified_depth(42, group, group*8+i, 8) for i in range(8)]
        assert sorted(values.count(k) for k in (2, 3, 4)) == [2, 3, 3]
        assert list(reversed(values)) == [stratified_depth(42, group, i, 8)
                                         for i in reversed(range(group*8, (group+1)*8))]
    assert sample_seed(42, 0, 1) != sample_seed(42, 0, 2)


def test_math_reward_rejects_unfinished_prompt_opened_thinking():
    s = NS(prompt="question<think>\n", response="candidate \\boxed{42}", label="42", train_metadata=None)
    assert asyncio.run(reward(NS(), s)) == 0
    assert s.train_metadata["math_score"]["reason"] == "unfinished_thinking"
    s.response += "</think>Final answer: \\boxed{42}"
    assert asyncio.run(reward(NS(), s)) == 1


def test_math_reward_uses_symbolic_equality_and_rejects_wrong_answer():
    s = NS(prompt="question", response="\\boxed{\\frac{1}{2}}", label="0.5", train_metadata=None)
    assert asyncio.run(reward(NS(), s)) == 1
    s.label = "0.6"
    assert asyncio.run(reward(NS(), s)) == 0


def test_math_reward_works_in_ray_style_worker_thread():
    s = NS(prompt="question", response="\\boxed{7}", label="7", train_metadata=None)
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        assert executor.submit(lambda: asyncio.run(reward(NS(), s))).result(timeout=30) == 1


def test_depth_action_is_counted_once_and_not_per_response_token(monkeypatch):
    import torch
    import slime_plugins.ouro_2a as base
    from looprl.native_2a import policy_loss
    monkeypatch.setattr(base, "policy_loss", lambda args, batch, logits, reducer:
                        (logits.sum()*0, {"loss": logits.sum().detach()*0}))
    for length in (1, 17):
        raw_q = torch.tensor([.1, .2, .7], requires_grad=True)
        logq = raw_q.log_softmax(-1)
        logits = torch.zeros(1, length, 8, requires_grad=True)
        logits.ouro_depth_log_probs = logq
        args = NS(ouro_learned_depth=True, eps_clip=.2, eps_clip_high=.2, rollout_max_response_len=100)
        batch = dict(metadata=[dict(ouro_depth=2, ouro_log_q=logq[0].item(), sequence_advantage=2.)])
        loss, _ = policy_loss(args, batch, logits, None)
        gradient = torch.autograd.grad(loss, raw_q)[0]
        expected = -2*(torch.tensor([1., 0., 0.])-raw_q.detach().softmax(-1))/100
        torch.testing.assert_close(gradient, expected)
