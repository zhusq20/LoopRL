"""Audit saved real trajectories against the native checkpoint, retaining errors."""
import argparse
import json
from pathlib import Path
import tempfile
import time

import torch
import torch.distributed as dist
from megatron.core import parallel_state
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.transformer_config import TransformerConfig
from slime.utils.model_config import load_hf_config
from slime.backends.megatron_utils.hf_to_megatron.common import SafetensorReader
from slime.backends.megatron_utils.hf_to_megatron.ouro import ouro_hf_tensor
from slime_plugins.models.ouro import OuroGPTModel, ouro_block_spec


def load_model(checkpoint, recompute="none", dtype=torch.float32):
    hf = load_hf_config(checkpoint, trust_remote_code=True)
    recompute_options = {}
    if recompute == "full":
        recompute_options = dict(recompute_granularity="full", recompute_method="uniform", recompute_num_layers=1)
    elif recompute == "selective":
        recompute_options = dict(recompute_granularity="selective", recompute_modules=["core_attn"])
    elif recompute == "selective-mlp":
        recompute_options = dict(recompute_granularity="selective", recompute_modules=["core_attn", "mlp"])
    config = TransformerConfig(num_layers=hf.num_hidden_layers, hidden_size=hf.hidden_size,
        ffn_hidden_size=hf.intermediate_size, num_attention_heads=hf.num_attention_heads,
        num_query_groups=hf.num_key_value_heads, kv_channels=hf.head_dim,
        hidden_dropout=0., attention_dropout=0., normalization="RMSNorm", layernorm_epsilon=hf.rms_norm_eps,
        gated_linear_unit=True, activation_func=torch.nn.functional.silu, add_bias_linear=False,
        attention_softmax_in_fp32=True, params_dtype=dtype, bf16=dtype==torch.bfloat16, use_cpu_initialization=True,
        persist_layer_norm=False, gradient_accumulation_fusion=False, bias_activation_fusion=False,
        masked_softmax_fusion=False, bias_dropout_fusion=False, perform_initialization=False, apply_rope_fusion=False,
        **recompute_options)
    model = OuroGPTModel(config=config, transformer_layer_spec=ouro_block_spec(config),
        vocab_size=hf.vocab_size, max_sequence_length=hf.max_position_embeddings,
        share_embeddings_and_output_weights=False, position_embedding_type="rope", rotary_base=hf.rope_theta).cuda().eval()
    reader = SafetensorReader(checkpoint)
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            parameter.copy_(ouro_hf_tensor(name, reader, hf))
    return model


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint",default="/tmp/looprl-models/Ouro-2.6B-Thinking")
    p.add_argument("--rollouts",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--limit",type=int)
    p.add_argument("--dtype",choices=["float32","bfloat16"],default="float32")
    args=p.parse_args()
    rows=[json.loads(s) for s in args.rollouts.read_text().splitlines()][:args.limit]
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    args.output.parent.mkdir(parents=True,exist_ok=True)
    report=dict(source=str(args.rollouts),checkpoint=args.checkpoint,dtype=args.dtype,cases=[])
    references=[]
    with tempfile.TemporaryDirectory(prefix="ouro-audit-") as tmp:
        dist.init_process_group("nccl",init_method=f"file://{tmp}/init",rank=0,world_size=1)
        parallel_state.initialize_model_parallel(tensor_model_parallel_size=1,pipeline_model_parallel_size=1)
        model_parallel_cuda_manual_seed(42)
        try:
            model=load_model(args.checkpoint,dtype=getattr(torch,args.dtype))
            with torch.inference_mode():
                for row in rows:
                    started=time.monotonic()
                    tokens=torch.tensor([row['tokens']],device='cuda')
                    length=tokens.shape[1];prompt=length-row['response_length']
                    bounds=torch.tensor([0,length],device='cuda',dtype=torch.int32)
                    packed=PackedSeqParams(cu_seqlens_q=bounds,cu_seqlens_kv=bounds,
                        max_seqlen_q=length,max_seqlen_kv=length,qkv_format='thd')
                    logits=model(tokens,ouro_depth=row['train_metadata']['ouro_depth'],packed_seq_params=packed)
                    values=logits[:,prompt-1:-1].float().log_softmax(-1).gather(-1,tokens[:,prompt:].unsqueeze(-1)).flatten().cpu()
                    sampled=torch.tensor(row['rollout_log_probs'])
                    difference=(values-sampled).abs()
                    worst=int(difference.argmax())
                    case=dict(index=row['index'],step=row['step'],total_tokens=length,
                        response_tokens=row['response_length'],max_error=float(difference.max()),
                        mean_error=float(difference.mean()),p99_error=float(difference.quantile(.99)),
                        ratio_outside_20_percent=float(((values-sampled).exp().sub(1).abs()>.2).float().mean()),
                        worst_response_position=worst,sampled_logp=float(sampled[worst]),native_logp=float(values[worst]),
                        seconds=time.monotonic()-started)
                    report['cases'].append(case)
                    references.append(dict(index=row['index'],step=row['step'],log_probs=values))
                    del logits
                    report['max_error']=max(r['max_error'] for r in report['cases'])
                    report['passed']=all(torch.isfinite(torch.tensor(r['max_error'])) and r['max_error']<=.001 for r in report['cases'])
                    args.output.write_text(json.dumps(report,indent=2)+'\n')
                    if len(references)%16==0 or case['max_error']>.001:
                        print(json.dumps(case),flush=True)
                torch.save(references,args.output.with_suffix('.pt'))
        finally:
            parallel_state.destroy_model_parallel();dist.destroy_process_group()
    print(json.dumps({'samples':len(references),'max_error':report['max_error'],'passed':report['passed']}),flush=True)


if __name__=='__main__':
    main()
