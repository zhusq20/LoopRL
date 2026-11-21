"""Objectives from README §§6, 8. Depths are one-based everywhere.

PPO's separately clipped mixture/posterior surrogate has the exact joint
score gradient at the old policy, not at arbitrary off-policy parameters.
Group means follow the plan's GRPO estimator (finite-group self-baseline bias).
"""
from dataclasses import dataclass
import math

import torch
from torch import Tensor
from torch.nn import functional as F


def exit_log_probs(hazard_logits: Tensor) -> Tensor:
    """K-1 hazard logits -> normalized log q for K exits, final exit forced."""
    a = hazard_logits.float() if hazard_logits.dtype in (torch.bfloat16, torch.float16) else hazard_logits
    survival = F.logsigmoid(-a)
    prefix = F.pad(survival.cumsum(-1), (1, 0))
    return torch.cat((F.logsigmoid(a) + prefix[..., :-1], prefix[..., -1:]), -1)


def mixture_terms(loop_log_probs: Tensor, log_q: Tensor, exits: Tensor):
    """Selected-token likelihoods [..., K]; returns log mix, log rho_e, E[e]."""
    if loop_log_probs.shape != log_q.shape:
        raise ValueError("loop likelihoods and exit distribution must have identical shape")
    if torch.any((exits < 1) | (exits > log_q.shape[-1])):
        raise ValueError("exits must be one-based and within the loop range")
    joint = loop_log_probs + log_q
    mix = torch.logsumexp(joint, -1)
    posterior = joint.gather(-1, (exits.long() - 1).unsqueeze(-1)).squeeze(-1) - mix
    depths = torch.arange(1, log_q.shape[-1] + 1, device=log_q.device, dtype=log_q.dtype)
    return mix, posterior, (log_q.exp() * depths).sum(-1)


def batch_reward_scale(rewards: Tensor, epsilon: float = 1e-8) -> Tensor:
    """Std of binary outcome, NEVER std of priced return. Constant batch -> 1.

    Dividing a constant-reward batch by epsilon would explode the cost signal;
    unit scale keeps that batch's cost signal proportional to beta.
    """
    std = rewards.float().std(correction=0)
    return torch.where(std > epsilon, std, torch.ones_like(std)).detach()


def group_baseline(values: Tensor, groups: Tensor) -> Tensor:
    if values.ndim != 1 or groups.shape != values.shape:
        raise ValueError("values and group IDs must be vectors with equal shape")
    _, inverse = torch.unique(groups, return_inverse=True)
    sums = torch.zeros_like(values).scatter_add(0, inverse, values)
    counts = torch.zeros_like(values).scatter_add(0, inverse, torch.ones_like(values))
    return (sums / counts.clamp_min(1))[inverse].detach()


@dataclass
class Advantages:
    values: Tensor
    returns: Tensor
    baseline: Tensor
    reward_scale: Tensor
    cost_coefficient: float


def priced_advantages(rewards: Tensor, groups: Tensor, depths: Tensor, mask: Tensor,
                      beta: float, max_new_tokens: int, max_loops: int = 4,
                      token_exits: bool = False) -> Advantages:
    if beta < 0 or max_new_tokens < 1 or max_loops < 1:
        raise ValueError("invalid compute price or budget")
    if depths.shape != mask.shape or depths.shape[0] != rewards.numel():
        raise ValueError("depths/mask must be [batch, response positions]")
    valid_depths = depths[mask.bool()]
    if torch.any((valid_depths < 1) | (valid_depths > max_loops)):
        raise ValueError("invalid executed depth")
    c = beta / (max_loops * max_new_tokens)
    total_cost = (depths * mask).sum(-1) * c
    returns = rewards - total_cost
    baseline = group_baseline(returns, groups)
    scale = batch_reward_scale(rewards)
    if token_exits:
        # Own exit cost is handled analytically by c * E_q[e] in the loss.
        values = ((returns - baseline)[:, None] + c * depths) / scale
        values = values * mask
    else:
        values = (returns - baseline) / scale
    return Advantages(values.detach(), returns.detach(), baseline, scale, c)


def ramped_beta(target: float, step: int, steps: int, warmup_fraction: float = .3) -> float:
    if target < 0 or step < 0 or steps < 1 or not 0 <= warmup_fraction <= 1:
        raise ValueError("invalid price schedule")
    if warmup_fraction == 0:
        return target
    # Update 0 starts at zero. Ramp duration uses ceil, never silently rounds to 0.
    return target * min(1., step / max(1, math.ceil(steps * warmup_fraction)))


def ppo_surrogate(logp: Tensor, old_logp: Tensor, advantage: Tensor, clip: float = .2) -> Tensor:
    ratio = (logp - old_logp.detach()).exp()
    return torch.minimum(ratio * advantage.detach(), ratio.clamp(1 - clip, 1 + clip) * advantage.detach())


def sequence_depth_loss(logp: Tensor, old_logp: Tensor, mask: Tensor, advantages: Tensor,
                        log_depth: Tensor | None = None, old_log_depth: Tensor | None = None,
                        ref_logp: Tensor | None = None, kl_coef: float = 1e-3,
                        clip: float = .2, normalization: int = 1) -> Tensor:
    """2a: sum token scores and ONE depth score per trajectory, then batch mean.

    Separate token/depth PPO ratios share a return baseline. They match the
    joint-action score at the old policy. A fixed normalization preserves the
    relative weight of q and all LM actions across different response lengths.
    """
    loss = -(ppo_surrogate(logp, old_logp, advantages[:, None], clip) * mask).sum(-1)
    if log_depth is not None:
        if old_log_depth is None:
            raise ValueError("learned depth requires old depth likelihood")
        loss = loss - ppo_surrogate(log_depth, old_log_depth, advantages, clip)
    if ref_logp is not None and kl_coef:
        delta = ref_logp.detach() - logp
        kl = delta.exp() - delta - 1  # nonnegative sampled k3 estimate
        loss = loss + kl_coef * (kl * mask).sum(-1)
    return loss.mean() / normalization


def exit_marginal_loss(loop_logp: Tensor, log_q: Tensor, exits: Tensor, mask: Tensor,
                       advantages: Tensor, old_mix: Tensor, old_posterior: Tensor,
                       cost_coefficient: float, reward_scale: Tensor,
                       kappa: float = 1., clip: float = .2, normalization: int = 1) -> Tensor:
    mix, posterior, expected_depth = mixture_terms(loop_logp, log_q, exits)
    pg = ppo_surrogate(mix, old_mix, advantages, clip)
    pg = pg + kappa * ppo_surrogate(posterior, old_posterior, advantages, clip)
    cost = cost_coefficient * expected_depth / reward_scale.detach()
    return ((-pg + cost) * mask).sum(-1).mean() / normalization


def stratified_depths(count: int, choices: tuple[int, ...], generator=None) -> Tensor:
    """Randomized balanced strata, including random remainder when G % K != 0."""
    if count < 1 or not choices or len(set(choices)) != len(choices):
        raise ValueError("invalid rollout count or depth choices")
    choices_tensor = torch.tensor(choices)
    whole, remainder = divmod(count, len(choices))
    extra = choices_tensor[torch.randperm(len(choices), generator=generator)[:remainder]]
    samples = torch.cat([choices_tensor.repeat(whole), extra])
    return samples[torch.randperm(count, generator=generator)]
