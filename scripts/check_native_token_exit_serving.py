"""Full-model SGLang early-exit sampling and native executed-trace replay."""
import argparse
import json
from pathlib import Path
import tempfile
import time

import torch
import torch.distributed as dist
from megatron.core import parallel_state
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from transformers import AutoTokenizer
from check_native_rollout_logprobs import load_model
from looprl.native_serving import configure_token_exit_serving
from looprl.native_2b import parse_generation_trace


def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',default='/tmp/looprl-models/Ouro-2.6B-Thinking')
    p.add_argument('--dtype',choices=['float32','bfloat16'],default='float32');p.add_argument('--tokens',type=int,default=32)
    p.add_argument('--gate-shift',type=float,default=3.);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    a.output.parent.mkdir(parents=True,exist_ok=True)
    configure_token_exit_serving(a.tokens)
    from sglang import Engine
    torch.set_num_threads(4);torch.cuda.set_device(0);torch.backends.cuda.matmul.allow_tf32=False
    tokenizer=AutoTokenizer.from_pretrained(a.checkpoint,trust_remote_code=True)
    prompts=[tokenizer.apply_chat_template([dict(role='user',content=q)],tokenize=True,add_generation_prompt=True,return_dict=False)
             for q in ['What is 2 + 3? Please put your answer in \\boxed{}.', 'Solve 3*x + 7 = 25. Explain and give a boxed answer.']]
    engine=Engine(model_path=a.checkpoint,trust_remote_code=True,dtype=a.dtype,tp_size=1,pp_size=1,
        context_length=4096,max_total_tokens=4096,max_running_requests=4,mem_fraction_static=.35,
        attention_backend='flashinfer' if a.dtype=='bfloat16' else 'torch_native',
        disable_cuda_graph=True,disable_overlap_schedule=True,disable_radix_cache=True,chunked_prefill_size=-1,
        json_model_override_args=json.dumps(dict(total_ut_steps=4,ouro_token_exit=True,ouro_gate_shift=a.gate_shift,
            ouro_batched_decode=True,ouro_fused_ops=a.dtype=='bfloat16')),
        return_hidden_states_mode='full',random_seed=42,log_level='warning')
    try:
        start=time.monotonic()
        outputs=engine.generate(input_ids=prompts,sampling_params=[dict(temperature=1.,top_p=1.,max_new_tokens=a.tokens,
            sampling_seed=71+i,ignore_eos=True,custom_params={'ouro_token_exit':True}) for i in range(len(prompts))],
            return_logprob=True,return_hidden_states=True)
        report=dict(passed=False,dtype=a.dtype,gate_shift=a.gate_shift,generation_seconds=time.monotonic()-start,cases=[])
        rows=[]
        for ids,result in zip(prompts,outputs,strict=True):
            tokens=[r[1] for r in result['meta_info']['output_token_logprobs']]
            conditional=[r[0] for r in result['meta_info']['output_token_logprobs']]
            exits,gates=parse_generation_trace(result['meta_info'],len(tokens))
            assert len(tokens)==a.tokens,len(tokens)
            assert all(e in (1,2,3,4) for e in exits)
            rows.append(dict(prompt=ids,response=tokens,exits=exits,gate_log_probs=gates,token_log_probs=conditional,text=result['text']))
        a.output.with_suffix('.samples.json').write_text(json.dumps(rows,indent=2)+'\n')
        with tempfile.TemporaryDirectory(prefix='ouro-exit-serving-') as temp:
            dist.init_process_group('nccl',init_method=f'file://{temp}/init',rank=0,world_size=1)
            parallel_state.initialize_model_parallel(tensor_model_parallel_size=1,pipeline_model_parallel_size=1)
            model_parallel_cuda_manual_seed(42)
            model=load_model(a.checkpoint,recompute='full',dtype=getattr(torch,a.dtype))
            model.token_exit=True;model.gate_shift=a.gate_shift;model.early_exit_gate.requires_grad_(True)
            try:
                for row in rows:
                    ids=torch.tensor([row['prompt']+row['response']],device='cuda');n=len(row['response']);offset=len(row['prompt'])-1
                    kwargs=dict(ouro_exit_traces=[row['exits']],ouro_total_lengths=[ids.shape[1]],ouro_response_lengths=[n])
                    with torch.no_grad():
                        logits=model(ids,**kwargs)
                        cond=logits[0,offset:offset+n].log_softmax(-1).gather(-1,ids[0,offset+1:offset+1+n,None]).squeeze(-1)
                        q=logits.ouro_exit_log_probs[offset:offset+n]
                        joint=(cond+q).detach()
                        sample_cond=torch.tensor(row['token_log_probs'],device='cuda');sample_q=torch.tensor(row['gate_log_probs'],device='cuda')
                        delta=joint-(sample_cond+sample_q);weights=delta.exp().clamp(max=2.)
                        case=dict(response_tokens=n,exit_counts={str(e):row['exits'].count(e) for e in (1,2,3,4)},
                            mean_depth=sum(row['exits'])/n,token_logprob_max_error=(cond-sample_cond).abs().max().item(),
                            gate_logprob_max_error=(q-sample_q).abs().max().item(),joint_logprob_max_error=delta.abs().max().item(),
                            tis_mean=weights.mean().item(),tis_ess=(weights.sum().square()/(n*weights.square().sum())).item())
                        assert torch.isfinite(delta).all() and torch.isfinite(weights).all()
                        if a.dtype=='float32':assert case['joint_logprob_max_error']<.001,case
                    model.train();model.zero_grad(set_to_none=True)
                    logits=model(ids,**kwargs)
                    cond=logits[0,offset:offset+n].log_softmax(-1).gather(-1,ids[0,offset+1:offset+1+n,None]).squeeze(-1)
                    score=cond+logits.ouro_exit_log_probs[offset:offset+n]
                    case['grad_no_grad_error']=(score.detach()-joint).abs().max().item();assert case['grad_no_grad_error']<.001
                    (-score.mean()).backward()
                    case['gate_gradient_norm']=model.early_exit_gate.weight.grad.float().norm().item()
                    case['lm_gradient_norm']=model.decoder.final_layernorm.weight.grad.float().norm().item()
                    assert case['gate_gradient_norm']>0 and case['lm_gradient_norm']>0
                    report['cases'].append(case);model.eval()
                report['passed']=True;a.output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)
            finally:
                parallel_state.destroy_model_parallel();dist.destroy_process_group()
    finally:engine.shutdown()

if __name__=='__main__':main()
