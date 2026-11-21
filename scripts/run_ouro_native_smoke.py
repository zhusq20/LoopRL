"""Run unmodified slime.train with its native Megatron and SGLang actors.

This uses an explicitly synthetic reward to verify parameter updates and
checkpoint/sync plumbing. It does not measure math quality or 2a benefit.
Select two free GPUs with CUDA_VISIBLE_DEVICES. PYTHONPATH must contain the
pinned slime, Megatron and patched SGLang source roots.
"""

import argparse
import json
import os
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--depth", type=int, choices=[2, 3, 4], default=2)
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--resume", help="Native slime checkpoint root; --steps remains the total rollout count")
    parser.add_argument("--resume-step", type=int, help="Restore a specific saved rollout boundary")
    parser.add_argument("--prompt-data", help="Reuse the original diagnostic data file when resuming")
    options = parser.parse_args()
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
    if len(visible) != 2 or len(set(visible)) != 2:
        raise ValueError("Select exactly two distinct free GPUs with CUDA_VISIBLE_DEVICES")
    output = Path(options.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    if options.resume and not options.prompt_data:
        raise ValueError("Resume requires the original --prompt-data path")
    data = Path(options.prompt_data).resolve() if options.prompt_data else output / "diagnostic-prompts.jsonl"
    if options.prompt_data:
        if not data.is_file():
            raise ValueError(f"Missing original diagnostic data: {data}")
    else:
        data.write_text(json.dumps(dict(prompt="Name one animal and describe it in a short sentence.", label="diagnostic")) + "\n")
    config = json.loads((Path(options.checkpoint) / "config.json").read_text())
    argv = ["train.py", "--actor-num-nodes", "1", "--actor-num-gpus-per-node", "1",
        "--rollout-num-gpus", "1", "--num-gpus-per-node", "2",
        "--hf-checkpoint", options.checkpoint, "--load", options.resume or options.checkpoint,
        "--save", str(output / "checkpoints"), "--save-interval", "1",
        "--spec", "slime_plugins.models.ouro", "get_ouro_model_provider",
        "--transformer-impl", "local", "--train-fp32", "--ouro-depth", str(options.depth),
        "--disable-checkpoint-metadata-cache",
        "--num-layers", str(config["num_hidden_layers"]),
        "--hidden-size", str(config["hidden_size"]),
        "--ffn-hidden-size", str(config["intermediate_size"]),
        "--num-attention-heads", str(config["num_attention_heads"]),
        "--group-query-attention", "--num-query-groups", str(config["num_key_value_heads"]),
        "--kv-channels", str(config["head_dim"]), "--vocab-size", str(config["vocab_size"]),
        "--max-position-embeddings", str(config["max_position_embeddings"]),
        "--swiglu", "--disable-bias-linear", "--normalization", "RMSNorm",
        "--norm-epsilon", str(config["rms_norm_eps"]), "--position-embedding-type", "rope",
        "--rotary-base", str(int(config["rope_theta"])), "--untie-embeddings-and-output-weights",
        "--no-gradient-accumulation-fusion", "--no-bias-swiglu-fusion", "--no-bias-dropout-fusion",
        "--no-masked-softmax-fusion", "--no-rope-fusion", "--no-persist-layer-norm",
        "--attention-dropout", "0", "--hidden-dropout", "0",
        "--tensor-model-parallel-size", "1", "--pipeline-model-parallel-size", "1",
        "--context-parallel-size", "1", "--micro-batch-size", "1", "--global-batch-size", "2",
        "--prompt-data", str(data), "--input-key", "prompt", "--label-key", "label",
        "--apply-chat-template", "--rollout-batch-size", "1", "--n-samples-per-prompt", "2",
        "--num-rollout", str(options.steps), "--num-steps-per-rollout", "1",
        "--rollout-max-response-len", str(options.max_new_tokens),
        "--rollout-temperature", "1", "--rollout-top-p", "1",
        "--custom-rm-path", "slime_plugins.ouro_smoke.reward",
        "--custom-reward-post-process-path", "slime_plugins.ouro_2a.post_process_rewards",
        "--custom-model-forward-kwargs-path", "slime_plugins.ouro_2a.model_forward_kwargs",
        "--rollout-data-postprocess-path", "slime_plugins.ouro_2a.check_on_policy",
        "--loss-type", "custom_loss", "--custom-loss-function-path", "slime_plugins.ouro_2a.policy_loss",
        "--advantage-estimator", "grpo", "--eps-clip", "0.2", "--eps-clip-high", "0.2",
        "--kl-coef", "0", "--kl-loss-coef", "0", "--entropy-coef", "0",
        "--optimizer", "adam", "--lr", "1e-6", "--lr-decay-style", "constant",
        "--weight-decay", "0", "--adam-beta1", "0.9", "--adam-beta2", "0.95",
        "--rollout-num-gpus-per-engine", "1", "--sglang-dtype", "float32",
        "--sglang-json-model-override-args", json.dumps({"total_ut_steps": options.depth}),
        "--sglang-attention-backend", "torch_native", "--sglang-disable-cuda-graph",
        "--sglang-disable-overlap-schedule", "--sglang-chunked-prefill-size", "-1",
        "--sglang-context-length", "4096", "--sglang-max-total-tokens", "8192",
        "--sglang-mem-fraction-static", "0.5", "--sglang-max-running-requests", "4",
        "--dump-details", str(output / "details")]
    if options.resume_step is not None:
        if not options.resume:
            raise ValueError("--resume-step requires --resume")
        argv += ["--ckpt-step", str(options.resume_step)]
    (output / "launch.json").write_text(json.dumps(dict(argv=argv, diagnostic_reward=True), indent=2) + "\n")
    sys.argv = argv
    import ray
    from slime.utils.arguments import parse_args
    import train
    args, restore_plan = parse_args(return_restore_plan=True)
    ray.init(address="local", num_gpus=2, num_cpus=16, include_dashboard=False,
             _temp_dir=str(output / "ray"))
    try:
        train.main(args, restore_plan)
    finally:
        ray.shutdown()


if __name__ == "__main__":
    main()
