"""Verified slime reward-hook contract, NOT an Ouro Megatron implementation.

Upstream inspected at THUDM/slime@79aa41a884ab0bf65222c94418f6357d3c31192b.
Usage: --custom-reward-post-process-path looprl.slime_hooks.priced_rewards
       --custom-rm-path looprl.slime_hooks.math_reward

The working single-rank HF Ouro backend is in slime_backend.py and is launched
by slime_train.py. Megatron conversion and a depth-aware SGLang sampler remain
separate future work; this reward hook alone is not a training pipeline.
"""
import torch
from torch.nn.utils.rnn import pad_sequence

from .objectives import priced_advantages
from .rewards import score_answer


async def math_reward(args, sample, **kwargs):
    if sample.label is None:
        raise ValueError("A gold answer is required")
    return score_answer("<think>\n" + sample.response, sample.label)["reward"]


def priced_rewards(args, samples):
    """Runs on the COMPLETE rollout batch before data-parallel sharding.

    Each sample must carry explicit group_index and metadata.looprl containing
    depths, effective beta and max_new_tokens. The runner owns price warmup.
    Returns raw outcomes and already centered/batch-scaled advantages, matching
    slime's reward postprocessing contract. No second std normalization allowed.
    """
    if not samples or isinstance(samples[0], list):
        raise ValueError("Expected a flat complete batch of single-trajectory samples")
    metadata = [sample.metadata["looprl"] for sample in samples]
    signatures = {(m["beta"], m["max_new_tokens"]) for m in metadata}
    if len(signatures) != 1:
        raise ValueError("Mixed compute pricing/budgets within a batch")
    if any(sample.group_index is None for sample in samples):
        raise ValueError("Explicit group_index is required; never infer groups after shuffling")
    raw = [float(sample.get_reward_value(args)) for sample in samples]
    depths_list = []
    for sample, m in zip(samples, metadata):
        if len(m["depths"]) != sample.response_length or len(set(m["depths"])) != 1:
            raise ValueError("This 2a hook requires one constant depth for every response token")
        depths_list.append(torch.tensor(m["depths"], dtype=torch.float32))
    depths = pad_sequence(depths_list, batch_first=True)
    beta, budget = signatures.pop()
    groups = torch.tensor([sample.group_index for sample in samples])
    a = priced_advantages(torch.tensor(raw), groups, depths, depths > 0, beta, budget)
    return raw, a.values.tolist()
