"""Check the actual pretrained Ouro checkpoint against native Megatron.

Requires the pinned Megatron on PYTHONPATH and one GPU. This is a numerical
acceptance check, not a training run or a speed benchmark.
"""

import argparse
import json
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace

import torch
import torch.distributed as dist
from torch.nn import functional as F
from transformers import AutoConfig, AutoModelForCausalLM
from megatron.core import parallel_state
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.transformer_config import TransformerConfig

from slime.backends.megatron_utils.hf_to_megatron.ouro import ouro_hf_tensor
from slime.backends.megatron_utils.megatron_to_hf.ouro import convert_ouro_to_hf
from slime_plugins.models.ouro import OuroGPTModel, ouro_block_spec


class Reader(dict):
    def get_tensor(self, name):
        return self[name]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--lengths", type=int, nargs="+", default=[128, 2048])
    parser.add_argument("--depths", type=int, nargs="+", default=[2, 3, 4])
    parser.add_argument("--gradient-length", type=int, default=16)
    parser.add_argument("--dtype", choices=["float32", "bfloat16"], default="float32")
    parser.add_argument("--sglang-output", type=Path, nargs="*", default=[])
    args = parser.parse_args()
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.cuda.set_device(0)
    torch.manual_seed(42)
    with tempfile.TemporaryDirectory(prefix="ouro-native-check-") as temp:
        dist.init_process_group("nccl", init_method=f"file://{temp}/init", rank=0, world_size=1)
        parallel_state.initialize_model_parallel(tensor_model_parallel_size=1, pipeline_model_parallel_size=1)
        model_parallel_cuda_manual_seed(42)
        try:
            run(args)
        finally:
            parallel_state.destroy_model_parallel()
            dist.destroy_process_group()


def run(args):
    dtype = getattr(torch, args.dtype)
    hf_config = AutoConfig.from_pretrained(args.checkpoint, trust_remote_code=True)
    hf = AutoModelForCausalLM.from_pretrained(args.checkpoint, trust_remote_code=True,
        torch_dtype=dtype, attn_implementation="sdpa").cuda().eval()
    config = TransformerConfig(num_layers=hf_config.num_hidden_layers,
        hidden_size=hf_config.hidden_size, ffn_hidden_size=hf_config.intermediate_size,
        num_attention_heads=hf_config.num_attention_heads,
        num_query_groups=hf_config.num_key_value_heads, kv_channels=hf_config.head_dim,
        hidden_dropout=0., attention_dropout=0., normalization="RMSNorm",
        layernorm_epsilon=hf_config.rms_norm_eps, gated_linear_unit=True, activation_func=F.silu,
        add_bias_linear=False, attention_softmax_in_fp32=True, params_dtype=dtype,
        use_cpu_initialization=True, persist_layer_norm=False, gradient_accumulation_fusion=False,
        bias_activation_fusion=False, masked_softmax_fusion=False, bias_dropout_fusion=False,
        perform_initialization=False, apply_rope_fusion=False)
    native = OuroGPTModel(config=config, transformer_layer_spec=ouro_block_spec(config),
        vocab_size=hf_config.vocab_size, max_sequence_length=hf_config.max_position_embeddings,
        share_embeddings_and_output_weights=hf_config.tie_word_embeddings,
        position_embedding_type="rope", rotary_base=hf_config.rope_theta).cuda().eval()
    state = Reader(hf.state_dict())
    conversion_args = SimpleNamespace(hidden_size=config.hidden_size,
        num_attention_heads=config.num_attention_heads, num_query_groups=config.num_query_groups,
        kv_channels=config.kv_channels)
    covered = set()
    with torch.no_grad():
        for name, parameter in native.named_parameters():
            parameter.copy_(ouro_hf_tensor(name, state, hf_config))
            covered.update(key for key, _ in convert_ouro_to_hf(conversion_args, name, parameter))
    assert covered == set(state), "Mapping must cover every actual HF tensor"
    report = dict(checkpoint=args.checkpoint, dtype=args.dtype, torch=torch.__version__,
        physical_layers=len(native.decoder.layers), hf_tensors=len(state),
        parameters=sum(p.numel() for p in native.parameters()), cases=[], gradients=[])
    del state
    for length in args.lengths:
        tokens = torch.randint(0, hf_config.vocab_size, (1, length), device="cuda")
        labels = tokens.roll(-1, 1)
        for depth in args.depths:
            start = time.monotonic()
            hf.model.total_ut_steps = depth
            calls = []
            handle = native.decoder.layers[0].register_forward_hook(lambda *a: calls.append(1))
            with torch.no_grad():
                expected = hf(tokens, use_cache=False, exit_at_step=depth - 1).logits.float()
                actual = native(tokens, ouro_depth=depth).float()
                error = (expected - actual).abs().max().item()
                p = F.log_softmax(expected, dim=-1).gather(-1, labels.unsqueeze(-1))
                q = F.log_softmax(actual, dim=-1).gather(-1, labels.unsqueeze(-1))
                logprob_error = (p - q).abs().max().item()
                agreement = (expected.argmax(-1) == actual.argmax(-1)).float().mean().item()
            handle.remove()
            assert len(calls) == depth
            case = dict(length=length, depth=depth, decoder_calls=len(calls),
                max_logit_error=error, max_logprob_error=logprob_error,
                argmax_agreement=agreement, seconds=time.monotonic() - start)
            report["cases"].append(case)
            print(json.dumps(case), flush=True)
            torch.testing.assert_close(actual, expected, atol=1e-3, rtol=1e-4)
            assert logprob_error < 1e-3
            del actual, expected, p, q
    if args.gradient_length:
        tokens = torch.randint(0, hf_config.vocab_size, (1, args.gradient_length), device="cuda")
        for depth in args.depths:
            hf.zero_grad(set_to_none=True)
            native.zero_grad(set_to_none=True)
            hf.model.total_ut_steps = depth
            for model in [hf, native]:
                kwargs = dict(use_cache=False, exit_at_step=depth - 1) if model is hf else dict(ouro_depth=depth)
                output = model(tokens, **kwargs)
                logits = output.logits if model is hf else output
                F.cross_entropy(logits.reshape(-1, hf_config.vocab_size), tokens.roll(-1, 1).flatten()).backward()
                del logits, output
            gradients = Reader({name: p.grad for name, p in hf.named_parameters()})
            max_error = 0.
            squared_error = squared_ref = 0.
            for name, parameter in native.named_parameters():
                if name.startswith("early_exit_gate."):
                    assert parameter.grad is None
                    continue
                target = ouro_hf_tensor(name, gradients, hf_config)
                difference = parameter.grad - target
                max_error = max(max_error, difference.abs().max().item())
                squared_error += difference.double().square().sum().item()
                squared_ref += target.double().square().sum().item()
                torch.testing.assert_close(parameter.grad, target, atol=1e-3, rtol=1e-3)
            case = dict(depth=depth, length=args.gradient_length, max_gradient_error=max_error,
                        relative_l2_error=(squared_error / squared_ref) ** 0.5)
            report["gradients"].append(case)
            print(json.dumps(case), flush=True)
            assert case["relative_l2_error"] < 1e-3
    report["serving_cases"] = []
    for path in args.sglang_output:
        serving = json.loads(path.read_text())
        depth = serving["depth"]
        hf.model.total_ut_steps = depth
        pairs = list(zip(serving["input_ids"], serving["batch_outputs"], strict=True))
        pairs.append((serving["input_ids"][0], serving["repeated_output"]))
        for index, (prompt, generated) in enumerate(pairs):
            response = generated["output_ids"]
            sampled = generated["meta_info"]["output_token_logprobs"]
            assert response == [row[1] for row in sampled]
            tokens = torch.tensor([prompt + response], device="cuda")
            with torch.no_grad():
                expected = hf(tokens, use_cache=False, exit_at_step=depth - 1).logits.float()
                actual = native(tokens, ouro_depth=depth).float()
                targets = torch.tensor(response, device="cuda").view(1, -1, 1)
                start = len(prompt) - 1
                native_logp = actual[:, start:-1].log_softmax(-1).gather(-1, targets).flatten()
                hf_logp = expected[:, start:-1].log_softmax(-1).gather(-1, targets).flatten()
                sampled_logp = torch.tensor([row[0] for row in sampled], device="cuda")
                native_error = (native_logp - sampled_logp).abs().max().item()
                hf_error = (hf_logp - sampled_logp).abs().max().item()
            case = dict(source=str(path), depth=depth, index=index, prompt_length=len(prompt),
                response_length=len(response), max_native_logprob_error=native_error,
                max_hf_logprob_error=hf_error, cached_tokens=generated["meta_info"].get("cached_tokens", 0))
            report["serving_cases"].append(case)
            print(json.dumps(case), flush=True)
            assert max(native_error, hf_error) < 1e-3
        assert serving["repeated_output"]["output_ids"] == serving["batch_outputs"][0]["output_ids"]
    report["passed"] = True
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
