"""Check SGLang's prompt depth distribution before/after actual weight sync."""
import argparse
import json
from pathlib import Path
import tempfile

import torch
from sglang import Engine
from transformers import AutoTokenizer


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="/tmp/looprl-models/Ouro-2.6B-Thinking")
    p.add_argument("--output", required=True)
    a = p.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(a.checkpoint, trust_remote_code=True)
    ids = tokenizer.apply_chat_template([dict(role="user", content="What is 2 + 3?")],
        tokenize=True, add_generation_prompt=True, return_dict=False)
    engine = Engine(model_path=a.checkpoint, trust_remote_code=True, dtype="float32",
        tp_size=1, pp_size=1, context_length=4096, max_total_tokens=4096,
        max_running_requests=2, mem_fraction_static=.3, attention_backend="torch_native",
        disable_cuda_graph=True, disable_overlap_schedule=True, chunked_prefill_size=-1,
        json_model_override_args=json.dumps({"total_ut_steps": 4, "ouro_learned_depth": True}),
        return_hidden_states_mode="last", random_seed=42, log_level="warning")
    try:
        def generate():
            return engine.generate(input_ids=ids, sampling_params=dict(temperature=0., max_new_tokens=1),
                                   return_hidden_states="last")
        before = generate()
        q = torch.tensor(before["meta_info"]["hidden_states"])
        torch.testing.assert_close(q.exp(), torch.tensor([.1, .1, .8]), atol=1e-6, rtol=0.)
        target = torch.tensor([.2, .3, .5])
        sync = engine.update_weights_from_tensor([("model.sequence_depth_head.bias", target.log())])
        after = generate()
        updated = torch.tensor(after["meta_info"]["hidden_states"])
        torch.testing.assert_close(updated.exp(), target, atol=1e-6, rtol=0.)
        # A nonzero head is required to test prompt features, rather than just
        # matching biases. Compare the real serving path with native Megatron.
        from megatron.core import parallel_state
        from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
        from megatron.core.transformer.transformer_config import TransformerConfig
        from slime.utils.model_config import load_hf_config
        from slime.backends.megatron_utils.hf_to_megatron.common import SafetensorReader
        from slime.backends.megatron_utils.hf_to_megatron.ouro import ouro_hf_tensor
        from slime_plugins.models.ouro import OuroGPTModel, ouro_block_spec
        import torch.distributed as dist
        torch.set_num_threads(4)
        hf = load_hf_config(a.checkpoint, trust_remote_code=True)
        with tempfile.TemporaryDirectory(prefix="ouro-head-check-") as temp:
            dist.init_process_group("nccl", init_method=f"file://{temp}/init", rank=0, world_size=1)
            parallel_state.initialize_model_parallel(tensor_model_parallel_size=1, pipeline_model_parallel_size=1)
            model_parallel_cuda_manual_seed(42)
            config = TransformerConfig(num_layers=hf.num_hidden_layers, hidden_size=hf.hidden_size,
                ffn_hidden_size=hf.intermediate_size, num_attention_heads=hf.num_attention_heads,
                num_query_groups=hf.num_key_value_heads, kv_channels=hf.head_dim,
                hidden_dropout=0., attention_dropout=0., normalization="RMSNorm", layernorm_epsilon=hf.rms_norm_eps,
                gated_linear_unit=True, activation_func=torch.nn.functional.silu, add_bias_linear=False,
                attention_softmax_in_fp32=True, params_dtype=torch.float32, use_cpu_initialization=True,
                persist_layer_norm=False, gradient_accumulation_fusion=False, bias_activation_fusion=False,
                masked_softmax_fusion=False, bias_dropout_fusion=False, perform_initialization=False, apply_rope_fusion=False)
            native = OuroGPTModel(config=config, transformer_layer_spec=ouro_block_spec(config),
                vocab_size=hf.vocab_size, max_sequence_length=hf.max_position_embeddings,
                share_embeddings_and_output_weights=False, position_embedding_type="rope",
                rotary_base=hf.rope_theta, learned_depth=True).cuda().eval()
            reader = SafetensorReader(a.checkpoint)
            with torch.no_grad():
                for name, parameter in native.named_parameters():
                    parameter.copy_(ouro_hf_tensor(name, reader, hf))
                generator = torch.Generator().manual_seed(7)
                weights = torch.randn(3, hf.hidden_size, generator=generator)*.01
                native.sequence_depth_head.weight.copy_(weights)
                native.sequence_depth_head.bias.copy_(target.log())
            engine.update_weights_from_tensor([("model.sequence_depth_head.weight", weights)])
            real = torch.tensor(generate()["meta_info"]["hidden_states"], device="cuda")
            inputs = torch.tensor([ids], device="cuda")
            feature = native(inputs, ouro_depth=4, ouro_return_prompt_feature=True)
            expected = native.sequence_depth_head(feature).log_softmax(-1)
            error = (real-expected).abs().max().item()
            assert error < 1e-3, error
            (-expected[0]).backward()
            head_gradient = native.sequence_depth_head.weight.grad.norm().item()
            lm_gradient = native.decoder.final_layernorm.weight.grad.norm().item()
            assert head_gradient > 0 and lm_gradient > 0
            parallel_state.destroy_model_parallel()
            dist.destroy_process_group()
        report = dict(passed=True, before=q.tolist(), after=updated.tolist(), sync=str(sync),
                      prompt_tokens=len(ids), nonzero_head_logprob_max_error=error,
                      head_gradient_norm=head_gradient, lm_gradient_norm=lm_gradient,
                      scope="real SGLang head initialization/sync and full 2.6B Megatron prompt-feature/gradient check")
        path = Path(a.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2)+"\n")
        print(json.dumps(report), flush=True)
    finally:
        engine.shutdown()


if __name__ == "__main__":
    main()
