"""Measure real native forward/backward and retained activation memory."""
import argparse
import json
from pathlib import Path
import tempfile
import time

import torch
import torch.distributed as dist
from megatron.core import parallel_state
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from slime.backends.megatron_utils.data import PackedSeqParamsWithCPUOffsets
from check_native_rollout_logprobs import load_model


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',default='/tmp/looprl-models/Ouro-2.6B-Thinking')
    p.add_argument('--rollouts',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--recompute',choices=['full','selective','selective-mlp'],default='full')
    p.add_argument('--profile',action='store_true')
    p.add_argument('--memory-fraction',type=float,default=.6)
    p.add_argument('--dtype',choices=['float32','bfloat16'],default='float32')
    p.add_argument('--gemm',choices=['ieee','tf32x3'],default='ieee')
    a=p.parse_args()
    rows=[json.loads(s) for s in a.rollouts.read_text().splitlines()]
    row=max((r for r in rows if r['train_metadata']['ouro_depth']==4),key=lambda r:len(r['tokens']))
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.cuda.set_per_process_memory_fraction(a.memory_fraction)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='ouro-train-perf-') as temp:
        dist.init_process_group('nccl',init_method=f'file://{temp}/init',rank=0,world_size=1)
        parallel_state.initialize_model_parallel(tensor_model_parallel_size=1,pipeline_model_parallel_size=1)
        model_parallel_cuda_manual_seed(42)
        try:
            model=load_model(a.checkpoint,a.recompute,dtype=getattr(torch,a.dtype))
            tokens=torch.tensor([row['tokens']],device='cuda')
            length=tokens.shape[1];prompt=length-row['response_length']
            offsets=torch.tensor([0,length],device='cuda',dtype=torch.int32)
            packed=PackedSeqParamsWithCPUOffsets(cu_seqlens_q=offsets,cu_seqlens_kv=offsets,
                cu_seqlens_q_cpu=(0,length),max_seqlen_q=length,max_seqlen_kv=length,qkv_format='thd')
            def forward():
                logits=model(tokens,ouro_depth=4,packed_seq_params=packed)
                return logits[:,prompt-1:-1].float().log_softmax(-1).gather(-1,tokens[:,prompt:].unsqueeze(-1)).flatten()
            model.eval()
            with torch.no_grad():
                reference=forward()
            if a.gemm=='tf32x3':
                from experimental_ouro_linear import install
                install(model)
            model.train()
            report=dict(recompute=a.recompute,dtype=a.dtype,gemm=a.gemm,total_tokens=length,response_tokens=row['response_length'],runs=[])
            for _ in range(3):
                model.zero_grad(set_to_none=True)
                torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
                start=time.monotonic()
                values=forward()
                (-values.mean()).backward()
                torch.cuda.synchronize()
                result=dict(seconds=time.monotonic()-start,
                    max_error=float((reference-values.detach()).abs().max()),
                    gradient_norm=float(model.decoder.final_layernorm.weight.grad.norm()),
                    peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30)
                assert result['max_error']<=.001
                report['runs'].append(result)
                print(json.dumps(result),flush=True)
                a.output.write_text(json.dumps(report,indent=2)+'\n')
                del values
            torch.save(dict(log_probs=reference.cpu(),norm_gradient=model.decoder.final_layernorm.weight.grad.cpu()),
                       a.output.with_suffix('.pt'))
            if a.profile:
                model.zero_grad(set_to_none=True)
                with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as prof:
                    (-forward().mean()).backward()
                events=sorted(prof.key_averages(),key=lambda e:e.self_device_time_total,reverse=True)
                report['profile']=[dict(name=e.key,count=e.count,self_gpu_ms=e.self_device_time_total/1000,
                                       self_cpu_ms=e.self_cpu_time_total/1000) for e in events[:30]]
                a.output.write_text(json.dumps(report,indent=2)+'\n')
        finally:
            parallel_state.destroy_model_parallel();dist.destroy_process_group()


if __name__=='__main__':
    main()
