"""Run Ouro through the pinned slime training loop with a custom backend."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import uuid

from .data import write_json

SLIME_REVISION = "79aa41a884ab0bf65222c94418f6357d3c31192b"


def parser():
    p = argparse.ArgumentParser()
    p.add_argument("--slime-path", default="third_party/slime")
    p.add_argument("--model", default="ByteDance/Ouro-2.6B-Thinking")
    p.add_argument("--data", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--steps", type=int, default=140)
    p.add_argument("--prompts-per-step", type=int, default=32)
    p.add_argument("--group-size", type=int, default=8)
    p.add_argument("--max-new-tokens", type=int, default=2048)
    p.add_argument("--arm", choices=["fixed", "stratified", "learned"], default="stratified")
    p.add_argument("--depths", type=int, nargs="+", default=[2, 3, 4])
    p.add_argument("--precision", choices=["fp32", "bf16"], default="fp32")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--limit", type=int, default=7498)
    p.add_argument("--offset", type=int, default=0)
    p.add_argument("--beta", type=float, default=.1)
    p.add_argument("--price-warmup", type=float, default=.3)
    p.add_argument("--learning-rate", type=float, default=1e-6)
    p.add_argument("--gate-learning-rate", type=float, default=1e-4)
    p.add_argument("--kl-coef", type=float, default=1e-3)
    p.add_argument("--logprob-tolerance", type=float, default=.001)
    p.add_argument("--resume")
    p.add_argument("--save-interval", type=int)
    p.add_argument("--stop-after", type=int, help="Stop at this absolute update count, retaining the original --steps schedule")
    return p


def native_args(args):
    """Explicit options consumed by native slime components; no fake modules."""
    defaults = dict(
        actor_num_nodes=1, actor_num_gpus_per_node=1, megatron_config_path=None,
        num_rollout=min(args.steps, args.stop_after) if args.stop_after is not None else args.steps,
        start_rollout_id=None, release_train=False,
        offload_rollout=False, offload_train=False, check_weight_update_equal=False,
        eval_interval=None, use_critic=False, num_critic_only_steps=0,
        use_kl_loss=args.kl_coef != 0, use_opd=False, use_routing_replay=False,
        train_backend="ouro", rollout_external=True, rollout_data_transport="object-store",
        update_weight_mode="full", update_weight_transport="tensor", update_weight_start_version=0,
        debug_train_only=False, debug_rollout_only=False, use_wandb=False, use_tensorboard=False,
        no_save_optim=False, save=str(Path(args.output) / "checkpoints"),
        custom_reward_post_process_path="looprl.slime_hooks.priced_rewards",
        custom_convert_samples_to_train_data_path=None, reward_key=None,
        advantage_estimator="grpo", rewards_normalization=True, grpo_std_normalization=False,
        n_samples_per_prompt=args.group_size, rollout_batch_size=args.prompts_per_step,
        rollout_top_p=1., use_score_centering=False, use_rollout_routing_replay=False,
        train_env_vars={"OMP_NUM_THREADS": "4", "TOKENIZERS_PARALLELISM": "false"},
    )
    for name, value in defaults.items():
        setattr(args, name, value)
    if args.save_interval is None:
        args.save_interval = args.steps
    return args


def run(args):
    from .experiments import new_run
    path = Path(args.output).resolve()
    args.output = str(path)
    args.data = str(Path(args.data).resolve())
    slime = Path(args.slime_path).resolve()
    revision = subprocess.check_output(["git", "-C", str(slime), "rev-parse", "HEAD"], text=True).strip()
    if revision != SLIME_REVISION:
        raise RuntimeError(f"Expected pinned slime revision {SLIME_REVISION}, got {revision}")
    if args.steps < 1 or args.group_size < 2 or args.prompts_per_step < 1 or args.max_new_tokens < 1:
        raise ValueError("Invalid experiment dimensions")
    if args.stop_after is not None and not 1 <= args.stop_after <= args.steps:
        raise ValueError("--stop-after must be between 1 and --steps")
    if args.beta < 0 or not 0 <= args.price_warmup <= 1 or args.logprob_tolerance <= 0:
        raise ValueError("Invalid price schedule or numerical tolerance")
    if args.learning_rate <= 0 or args.gate_learning_rate <= 0 or args.kl_coef < 0:
        raise ValueError("Invalid optimization settings")
    if args.limit < 1 or args.offset < 0 or (args.save_interval is not None and args.save_interval < 1):
        raise ValueError("Invalid data selection or checkpoint interval")
    if not args.depths or len(set(args.depths)) != len(args.depths) or any(d not in range(1, 5) for d in args.depths):
        raise ValueError("Invalid depth candidates")
    if args.arm == "learned" and len(args.depths) < 2:
        raise ValueError("Learned depth requires at least two candidates")
    # Explicitly select only idle GPUs; the caller supplies the physical pair.
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
    if len(visible) != 2 or any(not x.isdigit() for x in visible) or visible[0] == visible[1]:
        raise ValueError("Set CUDA_VISIBLE_DEVICES to exactly two distinct physical GPU indices")
    inventory = subprocess.check_output(["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"], text=True)
    memory = {line.split(",")[0].strip(): int(line.split(",")[1]) for line in inventory.splitlines()}
    if any(memory.get(gpu, 10**9) > 1024 for gpu in visible):
        raise RuntimeError("Requested GPUs are not idle; no existing process will be stopped")
    sys.path.insert(0, str(slime))
    repo = str(Path(__file__).resolve().parents[1])
    os.environ["PYTHONPATH"] = os.pathsep.join([repo, str(slime), os.environ.get("PYTHONPATH", "")])
    os.environ["RAY_USAGE_STATS_ENABLED"] = "0"
    args = native_args(args)
    args.sync_dir = f"/tmp/looprl-sync-{uuid.uuid4().hex[:10]}"
    path = new_run(args)
    manifest = json.loads((path / "manifest.json").read_text())
    patch = subprocess.check_output(["git", "-C", str(slime), "diff", "--", "train.py"], text=True)
    manifest.update(backend="slime_custom_ouro", slime_training=True, slime_revision=revision,
                    train_loop=str(slime / "train.py"), sampler_backend="hf", trainer_ranks=1,
                    physical_gpus=visible, slime_patch=patch)
    write_json(path / "manifest.json", manifest)
    import ray
    from ray.util.placement_group import placement_group, remove_placement_group
    from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
    from .slime_backend import OuroRolloutManager, OuroTrainRayActor
    spec = importlib.util.spec_from_file_location("looprl_pinned_slime_train", slime / "train.py")
    native = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(native)
    if "actor_cls" not in __import__("inspect").signature(native.train).parameters:
        raise RuntimeError("Apply integrations/slime/custom-actor.patch before running")
    manager, pg, actors = None, None, []
    ray.init(address="local", num_cpus=8, num_gpus=2, include_dashboard=False,
             namespace=f"looprl-{uuid.uuid4().hex[:10]}", object_store_memory=512 * 1024**2,
             _temp_dir=f"/tmp/lr-ray-{uuid.uuid4().hex[:8]}", _node_ip_address="127.0.0.1")
    try:
        pg = placement_group([{"CPU": 4, "GPU": 1}, {"CPU": 4, "GPU": 1}], strategy="PACK")
        ray.get(pg.ready(), timeout=120)
        manager = ray.remote(num_gpus=1, num_cpus=2)(OuroRolloutManager).options(
            scheduling_strategy=PlacementGroupSchedulingStrategy(placement_group=pg, placement_group_bundle_index=1)).remote(args)
        pgs = {"actor": (pg, [0], [0]), "rollout": (pg, [1], [1])}
        # This is slime's actual loop: native model creation, train dispatch,
        # checkpoint boundaries, pause/sync/resume and rollout lifecycle.
        native.train(args, pgs, manager, None, None, actor_cls=OuroTrainRayActor)
        write_json(path / "status.json", {"status": "completed" if args.num_rollout == args.steps else "stopped_at_checkpoint", "completed_steps": args.num_rollout,
                   "backend": "slime_custom_ouro", "finished_unix": time.time()})
    except BaseException as error:
        write_json(path / "status.json", {"status": "failed", "error": str(error), "backend": "slime_custom_ouro"})
        raise
    finally:
        if manager is not None:
            try:
                actors = ray.get(manager.trainer_handles.remote(), timeout=15)
            except Exception:
                pass
            for actor in actors:
                ray.kill(actor, no_restart=True)
            ray.kill(manager, no_restart=True)
        if pg is not None:
            remove_placement_group(pg)
        ray.shutdown()


def main():
    args = parser().parse_args()
    manifest_existed = (Path(args.output) / "manifest.json").exists()
    try:
        run(args)
    except BaseException as error:
        if not manifest_existed and (Path(args.output) / "manifest.json").exists():
            write_json(Path(args.output) / "status.json", {"status": "failed", "error": str(error), "backend": "slime_custom_ouro"})
        raise


if __name__ == "__main__":
    main()
