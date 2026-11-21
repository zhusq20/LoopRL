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


def validate_resume_protocol(previous, current, allow_serving_upgrade=False, allow_training_upgrade=False):
    previous, current = dict(previous), dict(current)
    for value in (previous,current):
        value.setdefault("batched_decode", False)
        value.setdefault("rollout_correction",dict(mode="none",clip_low=0.,clip_high=2.,level="token"))
        value["serving"] = dict(value.get("serving", dict(attention_implementation="reference",
            max_running_requests=8, kv_token_capacity=16384)))
        value["serving"].setdefault("cuda_graph",False)
        value["serving"].setdefault("memory_fraction",.55)
        value["serving"].setdefault("depth4_replicas",1)
        value.setdefault("training_execution", dict(reuse_old_log_probs=False,
            micro_batch_size=1, data_parallel_size=1, group_by_depth=False))
    if previous == current:
        return None
    before, after = dict(previous), dict(current)
    old_serving, new_serving = before.pop("serving"), after.pop("serving")
    if allow_training_upgrade:
        old_training, new_training = before.pop("training_execution"), after.pop("training_execution")
        old_recompute, new_recompute = before.pop("activation_recompute", None), after.pop("activation_recompute", None)
        old_kernels, new_kernels = dict(old_serving), dict(new_serving)
        old_kernels.pop("depth4_replicas", None)
        new_kernels.pop("depth4_replicas", None)
        if before != after or old_kernels != new_kernels:
            raise ValueError("Resume protocol differs from original run outside validated execution settings")
        return dict(previous_training=old_training, current_training=new_training,
                    previous_recompute=old_recompute, current_recompute=new_recompute,
                    previous_serving=old_serving, current_serving=new_serving)
    if not allow_serving_upgrade or before != after:
        raise ValueError("Resume protocol differs from original run")
    return dict(previous=old_serving, current=new_serving)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="/tmp/looprl-models/Ouro-2.6B-Thinking")
    p.add_argument("--output", required=True)
    p.add_argument("--arm", choices=["fixed", "stratified", "learned", "token_exit"], default="stratified")
    p.add_argument("--steps", type=int, default=10)
    p.add_argument("--prompts", type=int, default=2)
    p.add_argument("--group-size", type=int, default=6)
    p.add_argument("--max-new-tokens", type=int, default=2048)
    p.add_argument("--beta", type=float, default=.1)
    p.add_argument("--price-warmup", type=float, default=.3)
    p.add_argument("--gate-learning-rate", type=float, default=1e-4)
    p.add_argument("--gate-shift", type=float, default=3.)
    p.add_argument("--serving-replicas", type=int, default=1)
    p.add_argument("--batched-decode", action="store_true")
    p.add_argument("--attention-implementation", choices=["reference", "paged", "paged-fused", "bf16-fast"],
                   default="reference", help="Serving kernels recorded alongside precision and rollout correction")
    p.add_argument("--precision",choices=["fp32","bf16"],default="fp32")
    p.add_argument("--rollout-correction",choices=["none","tis"],default="none")
    p.add_argument("--tis-clip",type=float,default=2.)
    p.add_argument("--cuda-graph",action="store_true")
    p.add_argument("--serving-memory-fraction",type=float,default=.55)
    p.add_argument("--depth4-replicas",type=int,default=1)
    p.add_argument("--trainer-gpus", type=int, default=1)
    p.add_argument("--micro-batch-size", type=int, default=1)
    p.add_argument("--reuse-old-log-probs", action="store_true")
    p.add_argument("--recompute", choices=["full", "selective", "selective-mlp", "none"], default="full")
    p.add_argument("--training-upgrade-report", type=Path)
    p.add_argument("--max-running-requests", type=int, default=8)
    p.add_argument("--kv-token-capacity", type=int, default=16384)
    p.add_argument("--seed", type=int, default=20261121)
    p.add_argument("--save-interval", type=int, default=5)
    p.add_argument("--engine-base-port", type=int,
                   help="Start of this run's SGLang port range; defaults to a range per GPU allocation")
    p.add_argument("--resume")
    p.add_argument("--resume-step", type=int)
    p.add_argument("--serving-upgrade-report", type=Path,
                   help="Explicitly allow a validated serving-only change at a saved resume boundary")
    p.add_argument("--stop-after", type=int)
    p.add_argument("--dry-run", action="store_true")
    options = p.parse_args()
    if options.resume:
        # Actors run from an isolated source directory. Resolve before chdir;
        # slime otherwise treats a missing path as a new HF finetune.
        options.resume = str(Path(options.resume).resolve())
        if not Path(options.resume).is_dir():
            p.error('Resume checkpoint directory does not exist')
        if options.resume_step is not None:
            boundary=Path(options.resume)/f'iter_{options.resume_step:07d}'
            cursor=Path(options.resume)/'rollout'/f'global_dataset_state_dict_{options.resume_step}.pt'
            if not (boundary/'.metadata').is_file() or not cursor.is_file():
                p.error('Resume requires a complete model and rollout checkpoint at the requested boundary')
    # A native run must not depend on an interactive shell exporting this.
    # Without it SGLang silently falls back to the generic Transformers model.
    package = os.environ.get("SGLANG_EXTERNAL_MODEL_PACKAGE", "slime_plugins.sglang_models")
    if package != "slime_plugins.sglang_models":
        p.error("Native Ouro requires SGLANG_EXTERNAL_MODEL_PACKAGE=slime_plugins.sglang_models")
    os.environ["SGLANG_EXTERNAL_MODEL_PACKAGE"] = package
    if min(options.steps, options.prompts, options.group_size, options.max_new_tokens) < 1:
        p.error("Steps, batch size and token budget must be positive")
    if options.group_size < 3 and options.arm == "stratified":
        p.error("Stratified runs need at least three samples per prompt")
    if options.beta < 0 or not 0 <= options.price_warmup <= 1:
        p.error("Invalid compute-price schedule")
    if options.max_running_requests < 1 or options.kv_token_capacity < options.max_new_tokens + 2048:
        p.error("Invalid decode concurrency or KV capacity")
    if options.precision=="bf16" and options.rollout_correction!="tis":
        p.error("BF16 training requires explicit TIS rollout correction")
    if options.precision=="bf16" and options.arm=="learned":
        p.error("BF16 learned-depth needs separate validation of depth-action probability correction")
    if options.attention_implementation=="bf16-fast" and options.precision!="bf16":
        p.error("bf16-fast requires BF16 training and serving")
    if options.precision=="bf16" and options.attention_implementation in ("paged","paged-fused"):
        p.error("Custom paged kernels require FP32")
    if not 0 < options.serving_memory_fraction < 1 or not math.isfinite(options.tis_clip) or options.tis_clip < 1:
        p.error("Invalid serving memory fraction or TIS threshold")
    if options.stop_after is not None and not 1 <= options.stop_after <= options.steps:
        p.error("stop-after must be within the original total steps")
    if options.serving_upgrade_report and not options.resume:
        p.error("serving-upgrade-report requires resume")
    if options.training_upgrade_report and not options.resume:
        p.error("training-upgrade-report requires resume")
    if options.trainer_gpus < 1 or options.micro_batch_size < 1 or (options.prompts*options.group_size) % options.trainer_gpus:
        p.error("Positive training sizes and a global batch divisible by trainer-gpus are required")
    if options.reuse_old_log_probs and (options.rollout_correction != "tis" or options.arm == "learned"):
        p.error("Probability reuse currently requires fixed/stratified TIS training")
    if options.trainer_gpus > 1 and not options.reuse_old_log_probs:
        p.error("Multi-GPU Ouro requires the aggregated reuse audit")
    token_exit = options.arm == "token_exit"
    if token_exit and (not options.reuse_old_log_probs or options.cuda_graph or options.recompute != "full"):
        p.error("Token-exit training requires audited probability reuse, full functional recomputation and eager serving")
    if token_exit and (options.serving_replicas < 1 or not math.isfinite(options.gate_shift)):
        p.error("Invalid token-exit gate shift or serving replicas")
    depths = [4] if options.arm in ("fixed", "token_exit") else [2, 3, 4]
    if options.depth4_replicas < 1:
        p.error("depth4-replicas must be positive")
    rollout_gpus=options.serving_replicas if token_exit else len(depths)+options.depth4_replicas-1
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
    if len(visible) != rollout_gpus+options.trainer_gpus or len(set(visible)) != len(visible):
        p.error(f"Select {rollout_gpus+options.trainer_gpus} distinct free GPUs with CUDA_VISIBLE_DEVICES")
    engine_base_port = options.engine_base_port or (25000 + 1000*int(visible[0]))
    if not 1024 <= engine_base_port <= 64000:
        p.error("engine-base-port must leave room for the engine port range")
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
        precision=options.precision, optimizer="AdamW", adam_betas=[.9, .99], weight_decay=.1,
        learning_rate=1e-6, gate_learning_rate=options.gate_learning_rate,
        lr_warmup_fraction=.1, clip_grad=.1, kl_coef=.001,
        temperature=1., top_p=1., reward="looprl.rewards.score_answer; math-verify==0.8.0",
        batched_decode=options.batched_decode,
        activation_recompute=dict(granularity="full", method="uniform", layers=1),
        shared_depth_weight_sync=True,
        train_sha256=sha256(data), validation_sha256=sha256(output / "validation-ids.json"),
        checkpoint=checkpoint, checkpoint_revision="f1edd81e7ac41355db670500ceaf204e0f73af68",
        checkpoint_config_sha256=sha256(Path(checkpoint)/"config.json"))
    # Keep implementation choices in the reproducibility protocol. A resume
    # cannot silently switch kernels or admission settings.
    protocol["serving"] = dict(attention_implementation=options.attention_implementation,
        max_running_requests=options.max_running_requests,
        kv_token_capacity=options.kv_token_capacity,cuda_graph=options.cuda_graph,
        memory_fraction=options.serving_memory_fraction,depth4_replicas=options.depth4_replicas)
    protocol["rollout_correction"] = dict(mode=options.rollout_correction,
        clip_low=0.,clip_high=options.tis_clip,level="token")
    protocol["training_execution"] = dict(reuse_old_log_probs=options.reuse_old_log_probs,
        micro_batch_size=options.micro_batch_size, data_parallel_size=options.trainer_gpus,
        group_by_depth=options.micro_batch_size > 1)
    if token_exit:
        protocol["token_exit"] = dict(scheme="2b", exits=[1,2,3,4], gate_shift=options.gate_shift,
            final_exit_forced=True, prompt_kv="dense_four_loops", past_kv="last_exited",
            objective="sampled_joint_exit_token_ppo", kappa=1.,
            compute_price="executed_prefill_plus_decode_loop_tokens", cost_normalizer=4*options.max_new_tokens,
            reference_kl="joint_exit_token_on_recorded_history", correction="joint_action_tokenwise_TIS")
        protocol["serving"]["serving_replicas"] = options.serving_replicas
        protocol["serving"]["radix_cache"] = False
    if options.recompute != "full":
        protocol["activation_recompute"] = (dict(granularity=None) if options.recompute == "none"
            else dict(granularity="selective", modules=["core_attn"] +
                      (["mlp"] if options.recompute == "selective-mlp" else [])))
    if options.resume:
        previous = json.loads((Path(options.resume).resolve().parent/"protocol.json").read_text())
        change = validate_resume_protocol(previous, protocol, bool(options.serving_upgrade_report),
                                          bool(options.training_upgrade_report))
        if change and options.training_upgrade_report:
            validation = json.loads(options.training_upgrade_report.read_text())
            execution = {key: protocol[key] for key in ("training_execution", "activation_recompute", "serving")}
            if (validation.get("passed") is not True
                    or execution not in validation.get("validated_configurations", [])
                    or validation.get("checkpoint_revision") != protocol["checkpoint_revision"]):
                raise ValueError("Training upgrade lacks a passing matching validation report")
            hashes = validation.get("source_sha256", {})
            required = {"looprl/native_2a.py", "third_party/slime/slime_plugins/ouro_training.py",
                        "third_party/slime/slime_plugins/ouro_2a.py",
                        "third_party/slime/slime/backends/megatron_utils/actor.py",
                        "third_party/slime/slime/backends/megatron_utils/model.py"}
            if not required <= hashes.keys() or any(sha256(ROOT/name) != digest for name, digest in hashes.items()):
                raise ValueError("Training source differs from the validated implementation")
            write_json(output/"training-upgrade.json", dict(**change,
                validation_report=str(options.training_upgrade_report.resolve()),
                validation_sha256=sha256(options.training_upgrade_report),
                source_run=str(Path(options.resume).resolve().parent), boundary=options.resume_step))
        elif change:
            validation = json.loads(options.serving_upgrade_report.read_text())
            if (validation.get("passed") is not True
                    or validation.get("attention_implementation") != options.attention_implementation
                    or validation.get("checkpoint_revision") != protocol["checkpoint_revision"]):
                raise ValueError("Serving upgrade lacks a passing matching validation report")
            required = {"third_party/slime/slime_plugins/ouro_attention.py",
                        "third_party/slime/slime_plugins/ouro_kernels.py",
                        "third_party/slime/slime_plugins/sglang_models/ouro.py"}
            if set(validation.get("source_sha256", {})) != required or any(
                    sha256(ROOT/name) != digest for name, digest in validation["source_sha256"].items()):
                raise ValueError("Serving source differs from the validated implementation")
            write_json(output/"serving-upgrade.json", dict(**change,
                validation_report=str(options.serving_upgrade_report.resolve()),
                validation_sha256=sha256(options.serving_upgrade_report),
                source_run=str(Path(options.resume).resolve().parent), boundary=options.resume_step))
    write_json(output/"protocol.json", protocol)
    config = json.loads((Path(checkpoint)/"config.json").read_text())
    overrides = dict(ouro_arm=options.arm, ouro_total_steps=options.steps,
        ouro_price_warmup=options.price_warmup, ouro_output=str(output),
        ouro_rollout_correction=options.rollout_correction,
        ouro_reuse_old_log_probs=options.reuse_old_log_probs,
        ouro_group_microbatches=options.micro_batch_size > 1,
        ouro_token_exit=token_exit, ouro_gate_shift=options.gate_shift,
        ouro_shared_depth_engines=True,
        ouro_engine_base_port=engine_base_port,
        ouro_learned_depth=options.arm == "learned", ouro_gate_learning_rate=options.gate_learning_rate)
    write_json(output/"custom.json", overrides)
    deployment = {"sglang": [dict(name="token_exit" if token_exit else f"depth{k}", update_weights=True,
        server_groups=[dict(worker_type="regular", num_gpus=options.serving_replicas if token_exit else options.depth4_replicas if k==4 else 1,
            overrides={"json_model_override_args": json.dumps({"total_ut_steps": k,
                "ouro_learned_depth": options.arm == "learned", "ouro_batched_decode": options.batched_decode,
                "ouro_token_exit": token_exit, "ouro_gate_shift": options.gate_shift,
                "ouro_paged_attention": options.attention_implementation in ("paged","paged-fused"),
                "ouro_fused_ops": options.attention_implementation in ("paged-fused","bf16-fast")}),
                **({"return_hidden_states_mode": "full"} if token_exit else
                   {"return_hidden_states_mode": "last"} if options.arm == "learned" else {})})]) for k in depths]}
    write_json(output/"engines.json", deployment)
    argv = ["train.py", "--actor-num-nodes", "1", "--actor-num-gpus-per-node", str(options.trainer_gpus),
        "--rollout-num-gpus", str(rollout_gpus), "--num-gpus-per-node", str(len(visible)),
        "--hf-checkpoint", checkpoint, "--load", options.resume or checkpoint,
        "--ref-load", checkpoint, "--save", str(output/"checkpoints"),
        "--save-interval", str(options.save_interval),
        "--spec", "slime_plugins.models.ouro", "get_ouro_model_provider",
        "--transformer-impl", "local", "--train-fp32" if options.precision=="fp32" else "--bf16", "--ouro-depth", "4",
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
        "--context-parallel-size", "1", "--micro-batch-size", str(options.micro_batch_size),
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
        "--rollout-num-gpus-per-engine", "1", "--sglang-dtype", "float32" if options.precision=="fp32" else "bfloat16",
        "--sglang-config", str(output/"engines.json"),
        "--sglang-attention-backend", "flashinfer" if options.attention_implementation=="bf16-fast" else "triton" if options.cuda_graph else "torch_native",
        "--sglang-disable-overlap-schedule", "--sglang-chunked-prefill-size", "-1",
        "--sglang-context-length", str(max(4096, options.max_new_tokens+2048)),
        "--sglang-max-total-tokens", str(options.kv_token_capacity),
        "--sglang-mem-fraction-static", str(options.serving_memory_fraction), "--sglang-max-running-requests", str(options.max_running_requests),
        "--custom-config-path", str(output/"custom.json"), "--dump-details", str(output/"details")]
    if options.recompute == "full":
        argv += ["--recompute-granularity", "full", "--recompute-method", "uniform", "--recompute-num-layers", "1"]
    elif options.recompute != "none":
        argv += ["--recompute-granularity", "selective", "--recompute-modules", "core_attn"]
        if options.recompute == "selective-mlp":
            argv += ["mlp"]
    if token_exit:
        argv = [arg.replace("looprl.native_2a.", "looprl.native_2b.") for arg in argv]
        argv += ["--sglang-disable-radix-cache"]
    if options.precision=="bf16":
        argv += ["--accumulate-allreduce-grads-in-fp32"]
    if options.rollout_correction=="tis":
        argv += ["--use-tis","--tis-clip",str(options.tis_clip),"--tis-clip-low","0"]
    if options.cuda_graph:
        sizes=sorted({1,2,4,8,16,32,options.max_running_requests}&set(range(1,options.max_running_requests+1)))
        argv += ["--sglang-cuda-graph-config",json.dumps(dict(decode=dict(backend="full",bs=sizes),prefill=dict(backend="disabled")))]
    else:
        argv += ["--sglang-disable-cuda-graph"]
    if options.resume_step is not None:
        if not options.resume:
            p.error("resume-step requires resume")
        argv += ["--ckpt-step", str(options.resume_step)]
    if options.resume:
        # --stop-after is an execution boundary, not a new optimizer schedule.
        # Protocol equality was checked above; restore its saved scheduler.
        argv += ['--use-checkpoint-opt-param-scheduler', '--exit-on-missing-checkpoint']
    write_json(output/"launch.json", dict(argv=argv, gpus=visible, engine_base_port=engine_base_port,
        external_model_package=package))
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
    if options.resume and (Path(args.load).resolve()!=Path(options.resume) or args.finetune or args.no_load_optim):
        raise ValueError('Requested checkpoint resume was replaced by base finetuning; refusing to train')
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
