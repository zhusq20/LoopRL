"""Replay the longest depth-four trajectory from the batch that exhausted memory."""
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


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="/tmp/looprl-models/Ouro-2.6B-Thinking")
    p.add_argument("--rollouts", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    rows = [json.loads(s) for s in a.rollouts.read_text().splitlines()]
    row = max((r for r in rows if r["train_metadata"]["ouro_depth"] == 4), key=lambda r: len(r["tokens"]))
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    hf = load_hf_config(a.checkpoint, trust_remote_code=True)
    with tempfile.TemporaryDirectory(prefix="ouro-recompute-") as temp:
        dist.init_process_group("nccl", init_method=f"file://{temp}/init", rank=0, world_size=1)
        parallel_state.initialize_model_parallel(tensor_model_parallel_size=1, pipeline_model_parallel_size=1)
        model_parallel_cuda_manual_seed(42)
        try:
            config = TransformerConfig(num_layers=hf.num_hidden_layers, hidden_size=hf.hidden_size,
                ffn_hidden_size=hf.intermediate_size, num_attention_heads=hf.num_attention_heads,
                num_query_groups=hf.num_key_value_heads, kv_channels=hf.head_dim,
                hidden_dropout=0., attention_dropout=0., normalization="RMSNorm", layernorm_epsilon=hf.rms_norm_eps,
                gated_linear_unit=True, activation_func=torch.nn.functional.silu, add_bias_linear=False,
                attention_softmax_in_fp32=True, params_dtype=torch.float32, use_cpu_initialization=True,
                persist_layer_norm=False, gradient_accumulation_fusion=False, bias_activation_fusion=False,
                masked_softmax_fusion=False, bias_dropout_fusion=False, perform_initialization=False,
                apply_rope_fusion=False, recompute_granularity="full", recompute_method="uniform", recompute_num_layers=1)
            model = OuroGPTModel(config=config, transformer_layer_spec=ouro_block_spec(config),
                vocab_size=hf.vocab_size, max_sequence_length=hf.max_position_embeddings,
                share_embeddings_and_output_weights=False, position_embedding_type="rope", rotary_base=hf.rope_theta).cuda()
            reader = SafetensorReader(a.checkpoint)
            with torch.no_grad():
                for name, parameter in model.named_parameters():
                    parameter.copy_(ouro_hf_tensor(name, reader, hf))
            tokens = torch.tensor([row["tokens"]], device="cuda")
            length = tokens.shape[1]
            prompt = length-row["response_length"]
            boundaries = torch.tensor([0, length], device="cuda", dtype=torch.int32)
            packed = PackedSeqParams(cu_seqlens_q=boundaries, cu_seqlens_kv=boundaries,
                max_seqlen_q=length, max_seqlen_kv=length, qkv_format="thd")
            def forward():
                logits = model(tokens, ouro_depth=4, packed_seq_params=packed)
                return logits[:, prompt-1:-1].log_softmax(-1).gather(-1, tokens[:, prompt:].unsqueeze(-1)).flatten()
            model.eval()
            with torch.no_grad():
                reference = forward()
            model.train()
            torch.cuda.reset_peak_memory_stats()
            started = time.monotonic()
            replay = forward()
            error = (reference-replay).abs().max().item()
            assert error < .001
            (-replay.mean()).backward()
            gradient = model.decoder.final_layernorm.weight.grad
            assert torch.isfinite(gradient).all() and gradient.norm() > 0
            report = dict(passed=True, source=str(a.rollouts), step=row["step"], index=row["index"],
                total_tokens=length, response_tokens=row["response_length"], depth=4,
                grad_no_grad_max_logprob_error=error, final_norm_gradient=gradient.norm().item(),
                peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30, seconds=time.monotonic()-started,
                limitation="Single native trajectory backward; excludes full trainer, reference backup and optimizer state")
            a.output.write_text(json.dumps(report, indent=2)+"\n")
            print(json.dumps(report))
        finally:
            parallel_state.destroy_model_parallel()
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
