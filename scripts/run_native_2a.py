"""Launch a reproducible real-MATH 2a run through unmodified slime.train."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from looprl.data import read_jsonl, sha256, write_json


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="/tmp/looprl-models/Ouro-2.6B-Thinking")
    p.add_argument("--output", required=True)
    p.add_argument("--arm", choices=["fixed", "stratified", "learned"], default="stratified")
    p.add_argument("--steps", type=int, default=10)
    p.add_argument("--prompts", type=int, default=2)
    p.add_argument("--group-size", type=int, default=6)
    p.add_argument("--max-new-tokens", type=int, default=2048)
    p.add_argument("--beta", type=float, default=.1)
    p.add_argument("--price-warmup", type=float, default=.3)
    p.add_argument("--gate-learning-rate", type=float, default=1e-4)
    p.add_argument("--batched-decode", action="store_true")
    p.add_argument("--seed", type=int, default=20261121)
    p.add_argument("--save-interval", type=int, default=5)
    p.add_argument("--resume")
    p.add_argument("--resume-step", type=int)
    p.add_argument("--stop-after", type=int)
    p.add_argument("--dry-run", action="store_true")
    options = p.parse_args()
    if min(options.steps, options.prompts, options.group_size, options.max_new_tokens) < 1:
        p.error("Steps, batch size and token budget must be positive")
    if options.group_size < 3 and options.arm == "stratified":
        p.error("Stratified runs need at least three samples per prompt")
    if options.beta < 0 or not 0 <= options.price_warmup <= 1:
        p.error("Invalid compute-price schedule")
    if options.stop_after is not None and not 1 <= options.stop_after <= options.steps:
        p.error("stop-after must be within the original total steps")
    depths = [4] if options.arm == "fixed" else [2, 3, 4]
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
    if len(visible) != len(depths)+1 or len(set(visible)) != len(visible):
        p.error(f"Select {len(depths)+1} distinct free GPUs with CUDA_VISIBLE_DEVICES")
    if not options.dry_run:
        memory = subprocess.check_output(["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"], text=True)
        used = dict(line.split(", ") for line in memory.strip().splitlines())
        if any(int(used[g]) > 1024 for g in visible):
            raise RuntimeError("Selected GPUs are occupied")
    output = Path(options.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    checkpoint = str(Path(options.checkpoint).resolve())
    # A fixed held-out validation split is excluded from every training arm.
    rows = read_jsonl(ROOT / "data/math_train.jsonl")
    random.Random(options.seed).shuffle(rows)
    validation, training = rows[:256], rows[256:]
    data = output / "math-train.jsonl"
    with data.open("w") as f:
        for row in training:
            f.write(json.dumps(dict(prompt=row["problem"]+"\nPlease put your final answer in \\boxed{}.",
                label=row["answer"], metadata={"id": row["id"], "split": "train"}), ensure_ascii=False)+"\n")
    write_json(output / "validation-ids.json", [r["id"] for r in validation])
    with (output/"validation.jsonl").open("w") as f:
        for row in validation:
            f.write(json.dumps(row, ensure_ascii=False)+"\n")
    protocol = dict(backend="slime_native_megatron_sglang", diagnostic_reward=False,
        arm=options.arm, total_steps=options.steps, prompts=options.prompts,
        group_size=options.group_size, max_new_tokens=options.max_new_tokens,
        beta=options.beta, price_warmup=options.price_warmup, seed=options.seed,
        precision="fp32", optimizer="AdamW", adam_betas=[.9, .99], weight_decay=.1,
        learning_rate=1e-6, gate_learning_rate=options.gate_learning_rate,
        lr_warmup_fraction=.1, clip_grad=.1, kl_coef=.001,
        temperature=1., top_p=1., reward="looprl.rewards.score_answer; math-verify==0.8.0",
        batched_decode=options.batched_decode,
        activation_recompute=dict(granularity="full", method="uniform", layers=1),
        shared_depth_weight_sync=True,
        train_sha256=sha256(data), validation_sha256=sha256(output / "validation-ids.json"),
        checkpoint=checkpoint, checkpoint_revision="f1edd81e7ac41355db670500ceaf204e0f73af68",
        checkpoint_config_sha256=sha256(Path(checkpoint)/"config.json"))
    if options.resume:
        previous = json.loads((Path(options.resume).resolve().parent/"protocol.json").read_text())
        previous.setdefault("batched_decode", False)
        if previous != protocol:
            raise ValueError("Resume protocol differs from original run")
    write_json(output/"protocol.json", protocol)
    config = json.loads((Path(checkpoint)/"config.json").read_text())
    overrides = dict(ouro_arm=options.arm, ouro_total_steps=options.steps,
        ouro_price_warmup=options.price_warmup, ouro_output=str(output),
        ouro_shared_depth_engines=True,
        ouro_learned_depth=options.arm == "learned", ouro_gate_learning_rate=options.gate_learning_rate)
    write_json(output/"custom.json", overrides)
    deployment = {"sglang": [dict(name=f"depth{k}", update_weights=True,
        server_groups=[dict(worker_type="regular", num_gpus=1,
            overrides={"json_model_override_args": json.dumps({"total_ut_steps": k,
                "ouro_learned_depth": options.arm == "learned", "ouro_batched_decode": options.batched_decode}),
                **({"return_hidden_states_mode": "last"} if options.arm == "learned" else {})})]) for k in depths]}
    write_json(output/"engines.json", deployment)
    argv = ["train.py", "--actor-num-nodes", "1", "--actor-num-gpus-per-node", "1",
        "--rollout-num-gpus", str(len(depths)), "--num-gpus-per-node", str(len(visible)),
        "--hf-checkpoint", checkpoint, "--load", options.resume or checkpoint,
        "--ref-load", checkpoint, "--save", str(output/"checkpoints"),
        "--save-interval", str(options.save_interval),
        "--spec", "slime_plugins.models.ouro", "get_ouro_model_provider",
        "--transformer-impl", "local", "--train-fp32", "--ouro-depth", "4",
        "--ouro-beta", str(options.beta), "--disable-checkpoint-metadata-cache",
        "--num-layers", str(config["num_hidden_layers"]), "--hidden-size", str(config["hidden_size"]),
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
        "--context-parallel-size", "1", "--micro-batch-size", "1",
        "--recompute-granularity", "full", "--recompute-method", "uniform", "--recompute-num-layers", "1",
        "--global-batch-size", str(options.prompts*options.group_size),
        "--prompt-data", str(data), "--input-key", "prompt", "--label-key", "label",
        "--apply-chat-template", "--rollout-batch-size", str(options.prompts),
        "--n-samples-per-prompt", str(options.group_size), "--rollout-shuffle",
        "--num-rollout", str(options.stop_after or options.steps), "--num-steps-per-rollout", "1",
        "--rollout-max-response-len", str(options.max_new_tokens),
        "--rollout-temperature", "1", "--rollout-top-p", "1", "--rollout-seed", str(options.seed),
        "--seed", str(options.seed), "--rollout-function-path", "looprl.native_2a.generate_rollout",
        "--custom-generate-function-path", "looprl.native_2a.generate",
        "--custom-rm-path", "looprl.native_2a.reward",
        "--custom-reward-post-process-path", "looprl.native_2a.post_process_rewards",
        "--custom-model-forward-kwargs-path", "looprl.native_2a.model_forward_kwargs",
        "--rollout-data-postprocess-path", "looprl.native_2a.check_on_policy",
        "--loss-type", "custom_loss", "--custom-loss-function-path", "looprl.native_2a.policy_loss",
        "--advantage-estimator", "grpo", "--eps-clip", ".2", "--eps-clip-high", ".2",
        "--kl-coef", "0", "--use-kl-loss", "--kl-loss-coef", ".001", "--entropy-coef", "0",
        "--optimizer", "adam", "--lr", "1e-6", "--lr-decay-style", "constant",
        "--lr-decay-iters", str(options.steps), "--lr-warmup-iters", str(max(1, math.ceil(options.steps*.1))),
        "--clip-grad", ".1", "--weight-decay", ".1", "--adam-beta1", ".9", "--adam-beta2", ".99",
        "--rollout-num-gpus-per-engine", "1", "--sglang-dtype", "float32",
        "--sglang-config", str(output/"engines.json"),
        "--sglang-attention-backend", "torch_native", "--sglang-disable-cuda-graph",
        "--sglang-disable-overlap-schedule", "--sglang-chunked-prefill-size", "-1",
        "--sglang-context-length", str(max(4096, options.max_new_tokens+2048)),
        "--sglang-max-total-tokens", str(max(16384, 4*(options.max_new_tokens+2048))),
        "--sglang-mem-fraction-static", ".55", "--sglang-max-running-requests", "8",
        "--custom-config-path", str(output/"custom.json"), "--dump-details", str(output/"details")]
    if options.resume_step is not None:
        if not options.resume:
            p.error("resume-step requires resume")
        argv += ["--ckpt-step", str(options.resume_step)]
    write_json(output/"launch.json", dict(argv=argv, gpus=visible))
    source = output/"source"
    source.mkdir()
    for name in ["looprl/native_2a.py", "scripts/run_native_2a.py", "third_party/slime/slime_plugins/ouro_2a.py",
                 "third_party/slime/slime_plugins/models/ouro.py", "third_party/slime/slime_plugins/sglang_models/ouro.py"]:
        dest = source/name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT/name, dest)
    # Actors import exclusively from this run's snapshot, even while further
    # experiments are developed in the shared working tree.
    shutil.copytree(ROOT/"looprl", source/"looprl", dirs_exist_ok=True,
                    ignore=shutil.ignore_patterns("__pycache__"))
    for name in ["slime", "slime_plugins"]:
        shutil.copytree(ROOT/"third_party/slime"/name, source/"third_party/slime"/name,
            dirs_exist_ok=True, ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copy2(ROOT/"third_party/slime/train.py", source/"third_party/slime/train.py")
    snapshot_paths = [str(source), str(source/"third_party/slime")]
    sys.path[:0] = snapshot_paths
    os.environ["PYTHONPATH"] = os.pathsep.join(snapshot_paths+[os.environ.get("PYTHONPATH", "")])
    # Python's already-imported package __path__ and Ray workers' cwd otherwise
    # take precedence over the snapshot's PYTHONPATH.
    for name in list(sys.modules):
        if name == "looprl" or name.startswith("looprl."):
            del sys.modules[name]
    os.chdir(source)
    # Spawned reward workers re-import __main__; point them at this snapshot
    # too, so its ROOT insertion cannot restore the live working directory.
    sys.modules["__main__"].__file__ = str(source/"scripts/run_native_2a.py")
    write_json(output/"source-manifest.json", {str(f.relative_to(source)): sha256(f)
        for f in source.rglob("*.py")})
    if options.dry_run:
        print(output)
        return
    sys.argv = argv
    import ray
    import train
    from slime.utils.arguments import parse_args
    args, restore_plan = parse_args(return_restore_plan=True)
    write_json(output/"status.json", dict(status="running", started_unix=time.time(), pid=os.getpid()))
    ray.init(address="local", num_gpus=len(visible), num_cpus=24, include_dashboard=False,
             _temp_dir=str(Path("/tmp")/("ray-2a-"+hashlib.sha256(str(output).encode()).hexdigest()[:12])))
    try:
        train.main(args, restore_plan)
    except BaseException as exc:
        write_json(output/"status.json", dict(status="failed", error=repr(exc), finished_unix=time.time()))
        raise
    else:
        write_json(output/"status.json", dict(status="completed", completed_steps=options.stop_after or options.steps,
            finished_unix=time.time()))
    finally:
        ray.shutdown()


if __name__ == "__main__":
    main()
