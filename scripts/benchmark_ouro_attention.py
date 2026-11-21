"""Compare actual Ouro serving variants and save native-replay trajectories."""
import argparse
import json
from pathlib import Path
import time

from sglang import Engine


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',default='/tmp/looprl-models/Ouro-2.6B-Thinking')
    p.add_argument('--variant',choices=['reference','paged','paged-fused','bf16-fast'],required=True)
    p.add_argument('--rollouts',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--depth',type=int,default=4)
    p.add_argument('--dtype',choices=['float32','bfloat16'],default='float32')
    p.add_argument('--new-tokens',type=int,default=64)
    p.add_argument('--batch',type=int,default=8)
    p.add_argument('--concurrency',type=int,help='Engine limit; batch is the number of submitted requests')
    p.add_argument('--contexts',nargs='+',type=int,default=[512,2048])
    p.add_argument('--mem-fraction',type=float,default=.65)
    p.add_argument('--max-total-tokens',type=int,default=8192)
    p.add_argument('--math-prompts',action='store_true')
    p.add_argument('--cuda-graph',action='store_true')
    p.add_argument('--overlap',action='store_true')
    p.add_argument('--replay-group',type=int,help='Replay all stochastic requests of a saved prompt group')
    p.add_argument('--replay-count',type=int,help='Replay the first N saved requests with their original seeds')
    a=p.parse_args()
    concurrency=a.concurrency or a.batch
    if a.dtype!='float32' and a.variant not in ('reference','bf16-fast'):
        p.error('The custom paged kernels currently require FP32; use reference for the BF16 control')
    rows=[json.loads(s) for s in a.rollouts.read_text().splitlines()]
    args=dict(total_ut_steps=a.depth,ouro_batched_decode=a.variant!='bf16-fast',
              ouro_paged_attention=a.variant in ('paged','paged-fused'),ouro_fused_ops=a.variant in ('paged-fused','bf16-fast'))
    engine=Engine(model_path=a.checkpoint,trust_remote_code=True,dtype=a.dtype,
        tp_size=1,pp_size=1,context_length=4096,max_total_tokens=a.max_total_tokens,
        max_running_requests=concurrency,mem_fraction_static=a.mem_fraction,
        attention_backend='flashinfer' if a.variant=='bf16-fast' else 'triton' if a.cuda_graph else 'torch_native',
        disable_cuda_graph=not a.cuda_graph,
        cuda_graph_config={'decode':{'backend':'full','bs':sorted({1,2,4,8,16,concurrency}&set(range(1,concurrency+1)))},
                           'prefill':{'backend':'disabled'}} if a.cuda_graph else None,
        disable_overlap_schedule=not a.overlap,chunked_prefill_size=-1,
        json_model_override_args=json.dumps(args),random_seed=42,log_level='warning')
    report=dict(variant=a.variant,depth=a.depth,dtype=a.dtype,cuda_graph=a.cuda_graph,concurrency=concurrency,
                overlap=a.overlap,
                kv_token_capacity=a.max_total_tokens,forward_args=args,cases=[])
    saved=[]
    try:
        if a.replay_count is not None:
            selected=rows[:a.replay_count]
            requests=[[r['tokens'][:-r['response_length']] for r in selected]]
        elif a.replay_group is not None:
            selected=[r for r in rows if r['group']==a.replay_group]
            if len(selected)!=a.batch:
                raise ValueError('Replay group size differs from requested batch')
            requests=[[r['tokens'][:-r['response_length']] for r in selected]]
        elif a.math_prompts:
            unique={}
            for row in rows:
                unique.setdefault(row['group'],row)
            selected=list(unique.values())[:a.batch]
            requests=[[r['tokens'][:-r['response_length']] for r in selected]]
        else:
            source=next(r['tokens'] for r in rows if len(r['tokens'])>=max(a.contexts))
            requests=[[source[:length]]*a.batch for length in a.contexts]
        for prompts in requests:
            # Populate prompt cache and JIT compile before measuring generation.
            engine.generate(input_ids=prompts,sampling_params=dict(temperature=0,max_new_tokens=2,ignore_eos=True))
            started=time.monotonic()
            parameters=dict(temperature=0,max_new_tokens=a.new_tokens,ignore_eos=not a.math_prompts)
            if a.replay_group is not None or a.replay_count is not None:
                parameters=[dict(temperature=1,top_p=1,max_new_tokens=a.new_tokens,
                    sampling_seed=r['train_metadata']['ouro_seed']) for r in selected]
            outputs=engine.generate(input_ids=prompts,sampling_params=parameters,return_logprob=True,
                logprob_start_len=[len(ids)-1 for ids in prompts])
            elapsed=time.monotonic()-started
            row=dict(prompt_lengths=[len(ids) for ids in prompts],batch=len(prompts),
                seconds=elapsed,output_tokens=sum(len(o['output_ids']) for o in outputs))
            row['tokens_per_second']=row['output_tokens']/elapsed
            report['cases'].append(row)
            for ids,output in zip(prompts,outputs):
                response=output['output_ids']
                saved.append(dict(index=len(saved),step=0,group=len(saved),
                    train_metadata=dict(ouro_depth=a.depth),tokens=ids+response,
                    response_length=len(response),rollout_log_probs=[r[0] for r in output['meta_info']['output_token_logprobs']]))
            a.output.parent.mkdir(parents=True,exist_ok=True)
            a.output.write_text(json.dumps(report,indent=2)+'\n')
            a.output.with_suffix('.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in saved))
            print(json.dumps(row),flush=True)
    finally:
        engine.shutdown()


if __name__=='__main__':
    main()
