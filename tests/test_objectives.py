import itertools

import torch
from torch.nn import functional as F

from looprl.objectives import (batch_reward_scale, exit_log_probs, exit_marginal_loss,
    mixture_terms, priced_advantages, ramped_beta, sequence_depth_loss, stratified_depths)


def test_hazards_normalized_stable_and_closed_form_gradient():
    torch.manual_seed(7)
    a = torch.randn(5, 3, dtype=torch.float64, requires_grad=True)
    lp = torch.randn(5, 4, dtype=torch.float64).log_softmax(-1)
    lq = exit_log_probs(a)
    torch.testing.assert_close(lq.exp().sum(-1), torch.ones(5, dtype=torch.float64))
    mix = torch.logsumexp(lq + lp, -1)
    actual, = torch.autograd.grad(mix.sum(), a)
    rho = (lq + lp - mix[:, None]).exp()
    tail = rho.flip(-1).cumsum(-1).flip(-1)
    expected = rho[:, :-1] - a.sigmoid() * tail[:, :-1]
    torch.testing.assert_close(actual, expected)
    extremes = exit_log_probs(torch.tensor([[1000., -1000., 1000.]]))
    assert torch.isfinite(extremes).all()
    torch.testing.assert_close(extremes.exp().sum(), torch.tensor(1.))


def test_equal_readouts_have_no_reward_gate_gradient():
    a = torch.randn(3, dtype=torch.float64, requires_grad=True)
    lp = torch.full((4,), -.7, dtype=torch.float64)
    loss = torch.logsumexp(exit_log_probs(a) + lp, -1)
    grad, = torch.autograd.grad(loss, a)
    torch.testing.assert_close(grad, torch.zeros_like(grad), atol=1e-14, rtol=0)


def test_compute_price_is_not_cancelled_and_zero_variance_is_bounded():
    rewards = torch.tensor([1., 1., 0., 0.])
    groups = torch.tensor([0, 0, 1, 1])
    depths = torch.tensor([[2., 2.], [4., 4.], [2., 2.], [4., 4.]])
    mask = torch.ones_like(depths)
    a = priced_advantages(rewards, groups, depths, mask, .02, 2)
    b = priced_advantages(rewards, groups, depths, mask, .2, 2)
    torch.testing.assert_close(b.values, a.values * 10)
    assert batch_reward_scale(torch.ones(4)) == 1
    c = priced_advantages(torch.ones(4), groups, depths, mask, .2, 2)
    assert c.values.abs().max() <= .1


def test_other_position_cost_and_padding():
    rewards = torch.tensor([1., 0.])
    groups = torch.tensor([0, 0])
    depths = torch.tensor([[2., 4., 0.], [3., 0., 0.]])
    mask = depths > 0
    a = priced_advantages(rewards, groups, depths, mask, .1, 8, token_exits=True)
    for i, j in [(0, 0), (0, 1), (1, 0)]:
        other = depths[i].sum() - depths[i, j]
        expected = (rewards[i] - a.cost_coefficient * other - a.baseline[i]) / a.reward_scale
        torch.testing.assert_close(a.values[i, j], expected)
    assert (a.values[~mask] == 0).all()


def test_depth_score_occurs_once_per_sequence():
    lp = torch.zeros(2, 5, requires_grad=True)
    q = torch.zeros(2, requires_grad=True)
    mask = torch.tensor([[1, 1, 0, 0, 0], [1, 1, 1, 1, 1]])
    loss = sequence_depth_loss(lp, lp.detach(), mask, torch.ones(2), q, q.detach())
    gq, glp = torch.autograd.grad(loss, (q, lp))
    torch.testing.assert_close(gq, torch.full_like(q, -.5))
    torch.testing.assert_close(glp, -.5 * mask.float())


def test_kappa_one_equals_joint_score_at_old_policy():
    torch.manual_seed(4)
    a = torch.randn(2, 3, 3, dtype=torch.float64, requires_grad=True)
    logits = torch.randn(2, 3, 4, 3, dtype=torch.float64, requires_grad=True)
    lp = logits.log_softmax(-1)[..., 1]
    lq = exit_log_probs(a)
    exits = torch.tensor([[1, 3, 4], [2, 2, 3]])
    mask = torch.ones(2, 3, dtype=torch.float64)
    advantage = torch.tensor([[1., -2., 1.], [-1., .5, 2.]], dtype=torch.float64)
    mix, rho, _ = mixture_terms(lp, lq, exits)
    loss = exit_marginal_loss(lp, lq, exits, mask, advantage, mix.detach(), rho.detach(), 0., torch.tensor(1.))
    exact = -(advantage * (lp + lq).gather(-1, (exits - 1)[..., None]).squeeze(-1)).sum(-1).mean()
    actual = torch.autograd.grad(loss, (a, logits), retain_graph=True)
    expected = torch.autograd.grad(exact, (a, logits))
    for x, y in zip(actual, expected):
        torch.testing.assert_close(x, y)


def test_priced_gradient_matches_exact_two_step_process():
    """Exhaustive expectation, including exit-dependent future and analytic cost.

    Uses a constant independent baseline: finite-sample GRPO self-baselines are
    a separate estimator approximation, not part of the exact-score identity.
    """
    torch.manual_seed(2)
    a = torch.randn(3, dtype=torch.float64, requires_grad=True)
    b = torch.randn(4, 2, 3, dtype=torch.float64, requires_grad=True)
    l1 = torch.randn(4, 2, dtype=torch.float64, requires_grad=True)
    l2 = torch.randn(4, 2, 4, 2, dtype=torch.float64, requires_grad=True)
    params = (a, b, l1, l2)
    lq1, p1 = exit_log_probs(a), l1.log_softmax(-1)
    objective = surrogate = 0.
    c = .03
    for e1, y1, e2, y2 in itertools.product(range(4), range(2), range(4), range(2)):
        lq2 = exit_log_probs(b[e1, y1])
        p2 = l2[e1, y1].log_softmax(-1)
        lp = torch.stack((p1[:, y1], p2[:, y2]))[None]
        lq = torch.stack((lq1, lq2))[None]
        exits = torch.tensor([[e1 + 1, e2 + 1]])
        path_prob = (lq1[e1] + p1[e1, y1] + lq2[e2] + p2[e2, y2]).exp()
        reward = float(y1 == y2)
        costs = exits.to(torch.float64)
        objective = objective + path_prob * (reward - c * costs.sum())
        advantage = reward - c * (costs.sum() - costs)
        mix, rho, _ = mixture_terms(lp, lq, exits)
        surrogate = surrogate + path_prob.detach() * exit_marginal_loss(
            lp, lq, exits, torch.ones_like(exits), advantage,
            mix.detach(), rho.detach(), c, torch.tensor(1.))
    g1 = torch.autograd.grad(-objective, params, retain_graph=True)
    g2 = torch.autograd.grad(surrogate, params)
    for x, y in zip(g1, g2):
        torch.testing.assert_close(x, y, atol=1e-12, rtol=1e-10)


def test_strata_and_price_schedule():
    for seed in range(5):
        x = stratified_depths(8, (2, 3, 4), torch.Generator().manual_seed(seed))
        counts = [(x == k).sum().item() for k in (2, 3, 4)]
        assert max(counts) - min(counts) <= 1
    assert ramped_beta(.1, 0, 140) == 0
    assert ramped_beta(.1, 42, 140) == .1
    assert ramped_beta(.1, 139, 140) == .1
