"""Verify vectorized native ragged replay against a serial executed-path oracle."""
import argparse
import json
from pathlib import Path
import tempfile

import torch
import torch.distributed as dist
from megatron.core import parallel_state
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.transformer_config import TransformerConfig
from slime_plugins.models.ouro import OuroGPTModel, ouro_block_spec
from slime_plugins.models.ouro_token_exit import recurrent_layer, exit_layout
from slime.backends.megatron_utils.data import PackedSeqParamsWithCPUOffsets


def make_model(recompute=None):
    config = TransformerConfig(num_layers=2, hidden_size=32, ffn_hidden_size=64,
        num_attention_heads=4, num_query_groups=2, kv_channels=8,
        hidden_dropout=0., attention_dropout=0., normalization="RMSNorm", layernorm_epsilon=1e-6,
        gated_linear_unit=True, activation_func=torch.nn.functional.silu, add_bias_linear=False,
        attention_softmax_in_fp32=True, params_dtype=torch.float32, use_cpu_initialization=True,
        persist_layer_norm=False, gradient_accumulation_fusion=False, bias_activation_fusion=False,
        masked_softmax_fusion=False, bias_dropout_fusion=False, apply_rope_fusion=False,
        recompute_granularity=recompute,
        **(dict(recompute_method="uniform", recompute_num_layers=1) if recompute else {}))
    model = OuroGPTModel(config=config, transformer_layer_spec=ouro_block_spec(config),
        vocab_size=64, max_sequence_length=128, share_embeddings_and_output_weights=False,
        position_embedding_type="rope", rotary_base=10000, token_exit=True, gate_shift=1.).cuda()
    return model


def serial(model, tokens, prompt, exits):
    """Per-query oracle: only compute that query through its real exit.

    Each layer has four persistent cache slots. Exited query K/V are copied to
    later loop slots without detaching, exactly as the serving engine will do.
    This is independent of the parallel replay's masking/scheduling.
    """
    from megatron.core.models.common.embeddings.rope_utils import apply_rotary_pos_emb
    from torch.nn import functional as F
    hidden, rope = model._preprocess(tokens, None)[:2]
    caches = [[(None, None) for _ in model.decoder.layers] for _ in range(4)]
    selected, gate_scores = [], []
    for position in range(tokens.shape[1]):
        h = hidden[position:position+1]
        response_index = position - prompt + 1
        readout = exits[response_index] if 0 <= response_index < len(exits) else 4
        depth = 4 if position < prompt else readout
        chosen, logq = None, h.new_zeros(())
        for loop in range(depth):
            for index, layer in enumerate(model.decoder.layers):
                a = layer.self_attention
                q, k, v = a.get_query_key_value_tensors(layer.input_layernorm(h))
                q = apply_rotary_pos_emb(q, rope[position:position+1], config=model.config)
                k = apply_rotary_pos_emb(k, rope[position:position+1], config=model.config)
                past_k, past_v = caches[loop][index]
                k_all = k if past_k is None else torch.cat([past_k, k])
                v_all = v if past_v is None else torch.cat([past_v, v])
                caches[loop][index] = k_all, v_all
                # length-1 SDPA query sees the entire prefix and itself.
                kk = k_all.permute(1,2,0,3).repeat_interleave(2,dim=1)
                vv = v_all.permute(1,2,0,3).repeat_interleave(2,dim=1)
                context = F.scaled_dot_product_attention(q.permute(1,2,0,3), kk, vv,
                    dropout_p=0., is_causal=False).permute(2,0,1,3).flatten(2)
                branch, _ = a.linear_proj(context)
                h = h + layer.post_self_attn_layernorm(branch)
                branch, _ = layer.mlp(layer.pre_mlp_layernorm(h))
                h = h + layer.post_mlp_layernorm(branch)
            h = model.decoder.final_layernorm(h)
            if loop + 1 == readout:
                chosen = h
            if loop < min(3, readout):
                hazard = model.early_exit_gate(h).squeeze().float()-model.gate_shift
                logq = logq + F.logsigmoid(hazard if loop+1 == readout else -hazard)
        for later in range(depth,4):
            for index in range(len(model.decoder.layers)):
                source = caches[depth-1][index]
                previous = caches[later][index]
                caches[later][index] = tuple(torch.cat([old, new[-1:]]) if old is not None
                    else new[-1:] for old,new in zip(previous,source,strict=True))
        selected.append(chosen)
        gate_scores.append(logq)
    states = torch.cat(selected)
    logits = model.output_layer(states)[0].transpose(0,1).float()
    return logits, torch.stack(gate_scores)


def main():
    p=argparse.ArgumentParser();p.add_argument("--output",type=Path,required=True);a=p.parse_args()
    torch.set_num_threads(4);torch.cuda.set_device(0);torch.backends.cuda.matmul.allow_tf32=False
    with tempfile.TemporaryDirectory(prefix="ouro-token-exit-") as temp:
        dist.init_process_group("nccl",init_method=f"file://{temp}/init",rank=0,world_size=1)
        parallel_state.initialize_model_parallel(tensor_model_parallel_size=1,pipeline_model_parallel_size=1)
        model_parallel_cuda_manual_seed(42)
        try:
            model=make_model();tokens=torch.tensor([[5,7,3,8,11,6,9,2]],device="cuda")
            traces=[[4,4,4,4,4],[1,4,2,4,1],[4,1,3,2,4]];report=dict(passed=False,cases=[])
            for trace in traces:
                model.zero_grad(set_to_none=True)
                logits=model(tokens,ouro_exit_traces=[trace],ouro_total_lengths=[8],ouro_response_lengths=[5])
                oracle,gates=serial(model,tokens,3,trace)
                error=(logits[:,2:7]-oracle[:,2:7]).abs().max().item()
                gate_error=(logits.ouro_exit_log_probs[2:7]-gates[2:7]).abs().max().item()
                torch.testing.assert_close(logits[:,2:7],oracle[:,2:7],atol=2e-6,rtol=2e-5)
                torch.testing.assert_close(logits.ouro_exit_log_probs[2:7],gates[2:7],atol=2e-6,rtol=2e-5)
                def objective(logits,gate):
                    target=tokens[:,3:8,None]
                    return -(logits[:,2:7].log_softmax(-1).gather(-1,target).sum()+gate[2:7].sum())
                actual=torch.autograd.grad(objective(logits,logits.ouro_exit_log_probs),tuple(model.parameters()),allow_unused=True)
                expected=torch.autograd.grad(objective(oracle,gates),tuple(model.parameters()),allow_unused=True)
                max_gradient_error=0.
                for left,right in zip(actual,expected,strict=True):
                    if left is None or right is None:
                        assert left is right
                    else:
                        torch.testing.assert_close(left,right,atol=2e-5,rtol=3e-4)
                        max_gradient_error=max(max_gradient_error,(left-right).abs().max().item())
                report['cases'].append(dict(exits=trace,logit_error=error,gate_error=gate_error,max_gradient_error=max_gradient_error))
            # The all-deep path equals the pre-existing dense four-loop model.
            model.token_exit=False
            dense=model(tokens,ouro_depth=4)
            model.token_exit=True
            replay=model(tokens,ouro_exit_traces=[traces[0]],ouro_total_lengths=[8],ouro_response_lengths=[5])
            torch.testing.assert_close(dense,replay,atol=2e-6,rtol=2e-5)
            # Checkpoint recomputation must not detach the KV links across loops.
            recomputed=make_model('full');recomputed.load_state_dict(model.state_dict())
            ref=model(tokens,ouro_exit_traces=[traces[1]],ouro_total_lengths=[8],ouro_response_lengths=[5])
            cp=recomputed(tokens,ouro_exit_traces=[traces[1]],ouro_total_lengths=[8],ouro_response_lengths=[5])
            g1=torch.autograd.grad(objective(ref,ref.ouro_exit_log_probs),tuple(model.parameters()),allow_unused=True)
            g2=torch.autograd.grad(objective(cp,cp.ouro_exit_log_probs),tuple(recomputed.parameters()),allow_unused=True)
            for x,y in zip(g1,g2,strict=True):
                if x is not None:torch.testing.assert_close(x,y,atol=2e-6,rtol=2e-5)
                else:assert y is None
            # Packed requests must be isolated and preserve exit/prompt indexing.
            bounds=torch.tensor([0,8,16],device='cuda',dtype=torch.int32)
            packed=PackedSeqParamsWithCPUOffsets(cu_seqlens_q=bounds,cu_seqlens_kv=bounds,
                cu_seqlens_q_cpu=(0,8,16),max_seqlen_q=8,max_seqlen_kv=8,qkv_format='thd')
            both=model(tokens.repeat(1,2),packed_seq_params=packed,ouro_exit_traces=traces[1:],ouro_total_lengths=[8,8],ouro_response_lengths=[5,5])
            for i,trace in enumerate(traces[1:]):
                one=model(tokens,ouro_exit_traces=[trace],ouro_total_lengths=[8],ouro_response_lengths=[5])
                torch.testing.assert_close(both[:,8*i:8*i+8],one,atol=2e-6,rtol=2e-5)
                torch.testing.assert_close(both.ouro_exit_log_probs[8*i:8*i+8],one.ouro_exit_log_probs,atol=2e-6,rtol=2e-5)
            report.update(passed=True,checkpointed_gradients=True,packed_isolation=True,dense_equivalence=True)
            a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))
        finally:
            parallel_state.destroy_model_parallel();dist.destroy_process_group()

if __name__=='__main__':main()
