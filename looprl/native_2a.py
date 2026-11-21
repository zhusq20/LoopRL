"""Sequence-depth experiments using native slime actors and SGLang engines.

The run manifest fixes the protocol. Per-sample prices and seeds make replay
independent of asynchronous request completion and process restart order.
"""
import copy
import asyncio
import concurrent.futures
import hashlib
import json
import math
import multiprocessing
import random
import time
from pathlib import Path

_reward_pool = None


def reward_pool():
    global _reward_pool
    if _reward_pool is None:
        # math-verify uses signal.alarm; Ray's rollout event loop runs in a
        # worker thread. Separate processes retain the parser's timeout guard.
        _reward_pool = concurrent.futures.ProcessPoolExecutor(max_workers=2,
            mp_context=multiprocessing.get_context("spawn"))
    return _reward_pool


def price_at_step(target, step, total_steps, warmup):
    if target < 0 or total_steps < 1 or not 0 <= warmup <= 1 or step < 0:
        raise ValueError("Invalid compute-price schedule")
    return target * min(1., step / max(1, math.ceil(total_steps * warmup))) if warmup else target


def sample_seed(seed, group, index, purpose="tokens"):
    value = f"{seed}:{group}:{index}:{purpose}".encode()
    return int.from_bytes(hashlib.sha256(value).digest()[:4], "little") % (2**31 - 1)


def stratified_depth(seed, group, index, group_size):
    """Balanced per prompt, randomized extras, stable across restart/concurrency."""
    order = [2, 3, 4]
    random.Random(sample_seed(seed, group, 0, "depth")).shuffle(order)
    return order[(index % group_size) % len(order)]


def append_record(directory, filename, value):
    path = Path(directory) / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n")


async def generate(args, sample, sampling_params, evaluation=False):
    from slime.rollout.sglang_rollout import generate as base_generate

    step = args.ouro_rollout_id
    arm = args.ouro_arm
    depth_metadata = {}
    if arm == "fixed":
        depth = 4
    elif arm == "stratified":
        depth = stratified_depth(args.rollout_seed, sample.group_index, sample.index,
                                 args.n_samples_per_prompt)
    elif arm == "learned":
        from slime.rollout.sglang_rollout import GenerateState, _prepare_prompt_ids
        from slime.utils.http_utils import post
        state = GenerateState(args)
        prompt_ids = _prepare_prompt_ids(sample, state.tokenizer, state.processor)
        ip, port = args.sglang_model_routers["depth4"]
        output = await post(f"http://{ip}:{port}/generate", dict(input_ids=prompt_ids,
            sampling_params=dict(temperature=0., max_new_tokens=1), return_hidden_states="last"))
        logq = output["meta_info"]["hidden_states"]
        if len(logq) != 3 or not all(math.isfinite(x) for x in logq) or abs(sum(math.exp(x) for x in logq)-1) > 1e-5:
            raise ValueError("Depth engine did not return a normalized three-action policy")
        generator = random.Random(sample_seed(args.rollout_seed, sample.group_index, sample.index, "depth"))
        depth = generator.choices([2, 3, 4], weights=[math.exp(x) for x in logq])[0]
        depth_metadata = dict(ouro_log_q=logq[depth-2], ouro_all_log_q=logq,
            depth_prefill_loop_tokens=4*len(prompt_ids),
            depth_weight_version=output["meta_info"].get("weight_version"))
    else:
        raise ValueError(f"Unsupported native sequence-depth arm: {arm}")
    routers = args.sglang_model_routers
    name = f"depth{depth}"
    if name not in routers:
        raise ValueError(f"No dedicated SGLang engine for {name}; refusing fallback")
    routed = copy.copy(args)
    routed.sglang_router_ip, routed.sglang_router_port = routers[name]
    seed = sample_seed(args.rollout_seed, sample.group_index, sample.index)
    params = dict(sampling_params, sampling_seed=seed)
    sample.train_metadata = dict(sample.train_metadata or {}, **depth_metadata, ouro_depth=depth,
        ouro_step=step, ouro_seed=seed, ouro_beta=price_at_step(args.ouro_beta, step,
            args.ouro_total_steps, args.ouro_price_warmup))
    started = time.monotonic()
    result = await base_generate(routed, sample, params)
    if arm == "learned" and result.weight_versions != [depth_metadata["depth_weight_version"]]:
        raise ValueError("Depth action and token sampling used different weight versions")
    result.train_metadata["prefill_loop_tokens"] = depth*(len(result.tokens)-result.response_length) + depth_metadata.get("depth_prefill_loop_tokens", 0)
    result.train_metadata["generation_seconds"] = time.monotonic() - started
    return result


async def reward(args, sample, **kwargs):
    from looprl.rewards import score_answer
    # Ouro's template may have already opened <think> in the prompt.
    response = sample.response
    if sample.prompt.rfind("<think>") > sample.prompt.rfind("</think>"):
        response = "<think>" + response
    score = await asyncio.get_running_loop().run_in_executor(reward_pool(), score_answer, response, sample.label)
    sample.train_metadata = dict(sample.train_metadata or {}, math_score=score)
    return score["reward"]


def generate_rollout(args, rollout_id, data_source, evaluation=False):
    from slime.rollout.sglang_rollout import generate_rollout as base_rollout
    args.ouro_rollout_id = rollout_id
    return base_rollout(args, rollout_id, data_source, evaluation=evaluation)


def post_process_rewards(args, samples):
    from slime_plugins.ouro_2a import post_process_rewards as base_rewards
    steps = {s.train_metadata["ouro_step"] for s in samples}
    prices = {s.train_metadata["ouro_beta"] for s in samples}
    if len(steps) != 1 or len(prices) != 1:
        raise ValueError("Mixed rollout steps or compute prices in a native batch")
    step, beta = steps.pop(), prices.pop()
    versions = {v for s in samples for v in s.weight_versions}
    if len(versions) != 1 or any(len(s.weight_versions) != 1 for s in samples) or not all(str(v).isdigit() for v in versions):
        raise ValueError(f"All depth engines must sample one shared numeric weight version; got {versions}")
    expected = price_at_step(args.ouro_beta, step, args.ouro_total_steps, args.ouro_price_warmup)
    if beta != expected:
        raise ValueError("Rollout price differs from the saved experiment protocol")
    priced_args = copy.copy(args)
    priced_args.ouro_beta = beta
    raw, advantages = base_rewards(priced_args, samples)
    for s, a in zip(samples, advantages, strict=True):
        s.train_metadata["sequence_advantage"] = a
        append_record(args.ouro_output, "rollouts.jsonl", dict(step=step, index=s.index,
            group=s.group_index, metadata=s.metadata, train_metadata=s.train_metadata,
            prompt=s.prompt, response=s.response, label=s.label, reward=s.reward,
            response_length=s.response_length, tokens=s.tokens,
            rollout_log_probs=s.rollout_log_probs, weight_versions=s.weight_versions,
            status=s.status.name))
    n = len(samples)
    depths = [s.train_metadata["ouro_depth"] for s in samples]
    append_record(args.ouro_output, "rollout-metrics.jsonl", dict(step=step, beta=beta,
        samples=n, reward_mean=sum(raw)/n, mean_response_tokens=sum(s.response_length for s in samples)/n,
        mean_depth=sum(depths)/n, depth_counts={str(k): depths.count(k) for k in (2, 3, 4)},
        mean_loop_tokens=sum(k*s.response_length for k, s in zip(depths, samples))/n,
        mean_executed_loop_tokens=sum(s.train_metadata["prefill_loop_tokens"] +
            k*max(0, s.response_length-1) for k, s in zip(depths, samples))/n,
        truncation_rate=sum(s.status.name == "TRUNCATED" for s in samples)/n,
        nonzero_advantages=sum(abs(a) > 1e-10 for a in advantages),
        generation_seconds=sum(s.train_metadata["generation_seconds"] for s in samples)))
    return raw, advantages


def check_on_policy(args, rollout_id, rollout_data):
    import torch
    from slime_plugins.ouro_2a import check_on_policy as base_check
    base_check(args, rollout_id, rollout_data)
    ref = torch.cat(rollout_data["ref_log_probs"])
    if not torch.isfinite(ref).all():
        raise ValueError("Nonfinite frozen-reference probabilities")
    append_record(args.ouro_output, "consistency.jsonl", dict(step=rollout_id,
        sampler_recompute_max_error=float(rollout_data["ouro_logprob_max_error"]),
        reference_finite=True, tokens=ref.numel()))


def policy_loss(args, batch, logits, reducer):
    from slime_plugins.ouro_2a import policy_loss as base_loss
    checked = copy.copy(args)
    checked.rollout_data_postprocess_path = "slime_plugins.ouro_2a.check_on_policy"
    loss, metrics = base_loss(checked, batch, logits, reducer)
    if getattr(args, "ouro_learned_depth", False):
        import torch
        if len(batch["metadata"]) != 1:
            raise ValueError("Learned depth requires one trajectory per microbatch")
        meta = batch["metadata"][0]
        logq = logits.ouro_depth_log_probs[meta["ouro_depth"]-2]
        old = logq.new_tensor(meta["ouro_log_q"])
        error = (logq.detach()-old).abs().item()
        if not math.isfinite(error) or error > 1e-3:
            raise ValueError(f"Depth sampler/recompute mismatch {error} exceeds 0.001")
        advantage = logq.new_tensor(meta["sequence_advantage"])
        ratio = (logq-old).exp()
        depth_loss = -torch.minimum(ratio*advantage,
            ratio.clamp(1-args.eps_clip, 1+args.eps_clip_high)*advantage) / args.rollout_max_response_len
        loss = loss + depth_loss
        metrics.update(loss=loss.detach(), depth_loss=depth_loss.detach(), depth_logprob_error=logq.new_tensor(error))
    return loss, metrics


def model_forward_kwargs(args, batch):
    from slime_plugins.ouro_2a import model_forward_kwargs as base_kwargs
    kwargs = base_kwargs(args, batch)
    if getattr(args, "ouro_learned_depth", False):
        if len(batch["total_lengths"]) != 1:
            raise ValueError("Learned depth requires a single trajectory per microbatch")
        kwargs["ouro_prompt_length"] = batch["total_lengths"][0] - batch["response_lengths"][0]
    return kwargs

