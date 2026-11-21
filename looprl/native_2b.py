"""Token-wise hazard policy: actual exit/token joint score and executed cost.

The main arm uses the joint-action (kappa=1) policy gradient. A sampled full
trajectory cost enters the return once; there is no additional analytic cost
term. TIS operates on each joint exit/token action, not on token logits alone.
"""
import math
import time

from .native_2a import (append_record, check_on_policy, generate_rollout,
                       price_at_step, reward, sample_seed)


def parse_trace(raw, count):
    # SGLang FULL capture puts dense prompt rows in the first entry; subsequent
    # entries are one decode row. Only the last prompt row predicts token 0.
    if not raw:
        raise ValueError("Missing actual token-exit trace")
    rows = [raw[0][-1]] + raw[1:] if raw[0] and isinstance(raw[0][0], list) else raw
    if len(rows) != count or any(len(row) != 2 for row in rows):
        raise ValueError("Token/exit trace length mismatch")
    exits, gates = [], []
    for depth, logq in rows:
        if depth not in (1., 2., 3., 4.) or not math.isfinite(logq) or logq > 1e-6:
            raise ValueError("Invalid exit depth or gate probability")
        exits.append(int(depth));gates.append(float(logq))
    return exits, gates


async def generate(args, sample, sampling_params, evaluation=False):
    from slime.rollout.sglang_rollout import GenerateState, _prepare_prompt_ids
    from slime.utils.http_utils import post
    if sample.response_length:
        raise ValueError("Token-exit continuation requires its original KV trace; refusing fresh-prefill continuation")
    state = GenerateState(args)
    prompt = _prepare_prompt_ids(sample, state.tokenizer, state.processor)
    ip, port = args.sglang_model_routers["token_exit"]
    params = dict(sampling_params, sampling_seed=sample_seed(args.rollout_seed, sample.group_index, sample.index),
                  custom_params=dict(ouro_token_exit=True))
    started = time.monotonic()
    output = await post(f"http://{ip}:{port}/generate", dict(input_ids=prompt,
        sampling_params=params, return_logprob=True, return_hidden_states=True))
    info = output["meta_info"]
    values = info.get("output_token_logprobs", [])
    tokens, conditional = [v[1] for v in values], [v[0] for v in values]
    exits, gates = parse_trace(info.get("hidden_states"), len(tokens))
    if not tokens or not all(math.isfinite(x) for x in conditional):
        raise ValueError("Missing or nonfinite token probabilities")
    step = args.ouro_rollout_id
    sample.tokens = prompt
    sample.train_metadata = dict(sample.train_metadata or {}, ouro_mode="token_exit",
        # This key is the replay graph's maximum depth, not the sampled depth.
        ouro_depth=4, ouro_exits=exits, ouro_gate_log_probs=gates,
        ouro_token_log_probs=conditional, ouro_gate_shift=args.ouro_gate_shift,
        ouro_step=step, ouro_seed=params['sampling_seed'],
        ouro_beta=price_at_step(args.ouro_beta,step,args.ouro_total_steps,args.ouro_price_warmup),
        prefill_loop_tokens=4*len(prompt), executed_loop_tokens=4*len(prompt)+sum(exits[1:]),
        generation_seconds=time.monotonic()-started)
    sample.append_response_tokens(args, tokens=tokens,
        log_probs=[p+q for p,q in zip(conditional,gates,strict=True)],trainable=True,
        meta_info=info,text=output['text'])
    return sample


def post_process_rewards(args, samples):
    import torch
    from .objectives import batch_reward_scale, group_baseline
    if not samples:
        raise ValueError("Empty gated rollout")
    versions={v for s in samples for v in s.weight_versions}
    if len(versions)!=1 or any(len(s.weight_versions)!=1 for s in samples):
        raise ValueError("Every gated engine must use one shared policy version")
    steps={s.train_metadata['ouro_step'] for s in samples}
    prices={s.train_metadata['ouro_beta'] for s in samples}
    if len(steps)!=1 or len(prices)!=1:
        raise ValueError("Mixed gated update IDs/prices")
    step,beta=steps.pop(),prices.pop()
    if beta!=price_at_step(args.ouro_beta,step,args.ouro_total_steps,args.ouro_price_warmup):
        raise ValueError("Gated compute price differs from protocol")
    raw=[s.get_reward_value(args) for s in samples]
    costs=[]
    for s in samples:
        m=s.train_metadata;trace=m['ouro_exits'];n=s.response_length
        if len(trace)!=n or len(m['ouro_gate_log_probs'])!=n or len(m['ouro_token_log_probs'])!=n or any(e not in (1,2,3,4) for e in trace):
            raise ValueError("Incomplete sampled exit trajectory")
        expected=4*(len(s.tokens)-n)+sum(trace[1:])
        if m['executed_loop_tokens']!=expected or m['ouro_gate_shift']!=args.ouro_gate_shift:
            raise ValueError("Incorrect executed compute or gate shift")
        costs.append(expected/(4*args.rollout_max_response_len))
    reward_tensor=torch.tensor(raw,dtype=torch.float32)
    if not torch.isfinite(reward_tensor).all():raise ValueError("Nonfinite rewards")
    returns=reward_tensor-beta*torch.tensor(costs)
    groups=torch.tensor([s.group_index for s in samples])
    advantages=((returns-group_baseline(returns,groups))/batch_reward_scale(reward_tensor)).tolist()
    for s,value in zip(samples,advantages,strict=True):
        s.train_metadata['sequence_advantage']=value
        append_record(args.ouro_output,'rollouts.jsonl',dict(step=step,index=s.index,group=s.group_index,
            metadata=s.metadata,train_metadata=s.train_metadata,prompt=s.prompt,response=s.response,label=s.label,
            reward=s.reward,response_length=s.response_length,tokens=s.tokens,rollout_log_probs=s.rollout_log_probs,
            weight_versions=s.weight_versions,status=s.status.name))
    exits=[e for s in samples for e in s.train_metadata['ouro_exits']]
    append_record(args.ouro_output,'rollout-metrics.jsonl',dict(step=step,beta=beta,samples=len(samples),
        reward_mean=sum(raw)/len(raw),mean_depth=sum(exits)/len(exits),
        exit_counts={str(e):exits.count(e) for e in (1,2,3,4)},
        mean_response_tokens=len(exits)/len(samples),
        mean_executed_loop_tokens=sum(s.train_metadata['executed_loop_tokens'] for s in samples)/len(samples),
        truncation_rate=sum(s.status.name=='TRUNCATED' for s in samples)/len(samples),
        nonzero_advantages=sum(abs(v)>1e-10 for v in advantages)))
    return raw,advantages


def model_forward_kwargs(args,batch):
    metadata=batch['metadata']
    if any(m.get('ouro_mode')!='token_exit' or m['ouro_gate_shift']!=args.ouro_gate_shift for m in metadata):
        raise ValueError("Wrong token-exit replay protocol")
    return dict(ouro_exit_traces=[m['ouro_exits'] for m in metadata],
        ouro_total_lengths=batch['total_lengths'],ouro_response_lengths=batch['response_lengths'])


def policy_loss(args,batch,logits,reducer):
    from slime_plugins.ouro_2a import policy_loss as joint_loss
    if not args.ouro_token_exit or not args.ouro_reuse_old_log_probs:
        raise ValueError("Token-exit training requires audited single-update joint probability reuse")
    return joint_loss(args,batch,logits,reducer)
