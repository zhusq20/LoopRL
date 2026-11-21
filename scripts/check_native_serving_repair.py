"""Exercise real SGLang admission, long prompts, and token-exit trace integrity."""
import argparse
import json
import os
from pathlib import Path
import random
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from looprl.data import read_jsonl, write_json
from looprl.native_serving import configure_token_exit_serving, context_capacity, prompt_ids
from looprl.native_2b import parse_generation_trace


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--mode', choices=['fixed', 'gate', 'reproduce-retraction'], required=True)
    p.add_argument('--checkpoint', default='/tmp/looprl-models/Ouro-2.6B-Thinking')
    p.add_argument('--data', type=Path, default=ROOT/'data/math_train.jsonl')
    p.add_argument('--tokens', type=int, default=2048)
    p.add_argument('--requests', type=int, default=32)
    p.add_argument('--kv-capacity', type=int, default=65536)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if min(a.tokens, a.requests, a.kv_capacity) < 1:
        p.error('Token budget, request count and KV capacity must be positive')
    gated = a.mode != 'fixed'
    if a.mode == 'reproduce-retraction':
        # Observed ratio immediately before the original run's KV retraction.
        os.environ.update(SGLANG_INIT_NEW_TOKEN_RATIO='0.098',
                          SGLANG_MIN_NEW_TOKEN_RATIO_FACTOR='1')
    elif gated:
        configure_token_exit_serving(a.tokens)
    from sglang import Engine
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(a.checkpoint, trust_remote_code=True)
    rows = read_jsonl(a.data)
    random.Random(20261121).shuffle(rows)
    texts = [r['problem']+'\nPlease put your final answer in \\boxed{}.' for r in rows[256:]]
    model_limit = json.loads((Path(a.checkpoint)/'config.json').read_text())['max_position_embeddings']
    capacity = context_capacity(tokenizer, texts, a.tokens, model_limit, a.kv_capacity)
    encoded = [prompt_ids(tokenizer, text) for text in texts]
    longest = max(encoded, key=len)
    typical = next(ids for ids in encoded if 110 <= len(ids) <= 130)
    inputs = [longest] + [typical]*(a.requests-1)
    engine = Engine(model_path=a.checkpoint, trust_remote_code=True, dtype='bfloat16',
        tp_size=1, pp_size=1, context_length=capacity['context_length'],
        max_total_tokens=a.kv_capacity, max_running_requests=32, mem_fraction_static=.85,
        attention_backend='flashinfer', disable_cuda_graph=gated,
        cuda_graph_config=(None if gated else dict(decode=dict(backend='full',bs=[1,2,4,8,16,32]),
                                                  prefill=dict(backend='disabled'))),
        disable_overlap_schedule=True, disable_radix_cache=gated, chunked_prefill_size=-1,
        json_model_override_args=json.dumps(dict(total_ut_steps=4,ouro_token_exit=gated,
            ouro_gate_shift=3.,ouro_batched_decode=True,ouro_fused_ops=True)),
        return_hidden_states_mode='full' if gated else None, random_seed=42, log_level='info')
    try:
        started = time.monotonic()
        params = [dict(temperature=1.,top_p=1.,max_new_tokens=a.tokens,min_new_tokens=a.tokens,
                       sampling_seed=200+i,custom_params={'ouro_token_exit':True} if gated else {})
                  for i in range(len(inputs))]
        outputs = engine.generate(input_ids=inputs,sampling_params=params,
                                  return_logprob=True,return_hidden_states=gated)
        cases = []
        for ids, result in zip(inputs, outputs, strict=True):
            info = result['meta_info']
            values = info['output_token_logprobs']
            case = dict(prompt_tokens=len(ids),tokens=len(values),
                        completion_tokens=info['completion_tokens'],retractions=info['num_retractions'])
            if gated:
                try:
                    exits, gates = parse_generation_trace(info,len(values))
                    case.update(trace_valid=True,mean_depth=sum(exits)/len(exits))
                except ValueError as exc:
                    case.update(trace_valid=False,error=str(exc))
            cases.append(case)
        retractions = sum(c['retractions'] for c in cases)
        passed = (retractions > 0 if a.mode=='reproduce-retraction' else
                  all(c['tokens']==a.tokens and c.get('trace_valid',True) for c in cases)
                  and (not gated or retractions == 0))
        report = dict(passed=passed,mode=a.mode,capacity=capacity,requests=len(inputs),
                      kv_capacity=a.kv_capacity,seconds=time.monotonic()-started,
                      total_retractions=retractions,cases=cases,
                      scope='Serving correctness stress with forced output length; no accuracy or speedup claim')
        write_json(a.output,report)
        if a.mode=='reproduce-retraction':
            write_json(a.output.with_suffix('.responses.json'),outputs)
        print(json.dumps(report),flush=True)
        assert passed,report
    finally:
        engine.shutdown()


if __name__ == '__main__':
    main()
