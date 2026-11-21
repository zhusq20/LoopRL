"""Ouro backend for slime's native train loop, RayTrainGroup and BatchBuilder.

The first backend supports one trainer GPU and one HF sampler GPU. It retains
Ouro's actual recurrent weights; it is not a conversion to a dense 192-layer
model, nor an SGLang/Megatron implementation. FP32 is the validated consistency
mode; all log-probability checks remain enabled for every accepted batch.
"""
from dataclasses import asdict
import gc
import hashlib
import json
import os
from pathlib import Path
import shutil
import time

import ray
import torch

from .data import sha256, write_json
from .experiments import append_record, seed_everything, selected_rows
from .objectives import batch_reward_scale, ramped_beta, sequence_depth_loss, stratified_depths
from .ouro import DepthPrior, OuroAdapter, Rollout


def configure_device(args):
    """Respect the GPU allocated by Ray, including slime's NOSET env mode."""
    ids = ray.get_gpu_ids()
    if len(ids) != 1:
        raise RuntimeError(f"Expected exactly one allocated GPU, got {ids}")
    allocated = str(int(float(ids[0])))
    # Some Ray worker environments arrive pinned to one physical CPU core.
    # Use only CPUs available to the driver, with four CPUs per physical GPU.
    allowed = args.driver_cpu_affinity
    if allowed:
        start = int(allocated) * 4 % len(allowed)
        cpus = {allowed[(start + i) % len(allowed)] for i in range(min(4, len(allowed)))}
        for task in Path("/proc/self/task").iterdir():
            try:
                os.sched_setaffinity(int(task.name), cpus)
            except ProcessLookupError:
                pass
    device_index = 0
    if not torch.cuda.is_initialized():
        os.environ["CUDA_VISIBLE_DEVICES"] = allocated
    else:
        visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
        if allocated not in visible:
            raise RuntimeError("Initialized CUDA context does not expose the assigned GPU")
        device_index = visible.index(allocated)
    torch.set_num_threads(4)
    torch.cuda.set_device(device_index)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    return f"cuda:{device_index}"


def validate_resume(state, args):
    for key in ("model", "precision", "arm", "depths", "max_new_tokens", "seed", "steps", "beta", "price_warmup",
                "prompts_per_step", "group_size", "limit", "offset", "learning_rate", "gate_learning_rate", "kl_coef"):
        if state["args"][key] != getattr(args, key):
            raise ValueError(f"Resume changes the experiment setting {key}")
    if state.get("data_sha256") != sha256(args.data):
        raise ValueError("Resume requires the same training data contents")
    if state["completed_steps"] >= args.num_rollout:
        raise ValueError("Resume has no remaining updates")
    if state["weight_version"] != state["completed_steps"]:
        raise ValueError("Checkpoint update count and weight version disagree")


def load_adapter(args):
    if args.precision not in ("fp32", "bf16"):
        raise ValueError("Unknown precision")
    adapter = OuroAdapter.load(args.model, configure_device(args),
                               dtype=torch.float32 if args.precision == "fp32" else torch.bfloat16)
    configure_device(args)
    return adapter


def file_digest(path):
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for block in iter(lambda: source.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stage_checkpoint(source, destination):
    """Sequentially stage a shared-filesystem checkpoint before mmap loading."""
    source, destination = Path(source), Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    size = source.stat().st_size
    if shutil.disk_usage(destination.parent).free < size:
        raise RuntimeError("Insufficient local disk for checkpoint staging")
    started = time.monotonic()
    digest = hashlib.sha256()
    with source.open("rb") as reader, destination.open("xb") as writer:
        for block in iter(lambda: reader.read(16 * 1024 * 1024), b""):
            digest.update(block)
            writer.write(block)
    if destination.stat().st_size != size:
        raise RuntimeError("Incomplete checkpoint copy")
    return {"source": str(source), "local_path": str(destination), "bytes": size,
            "sha256": digest.hexdigest(), "seconds": time.monotonic() - started}


def parameter_probe(model, prior=None):
    """Deterministic sync fingerprint over named probes, not a whole-model hash."""
    digest = hashlib.sha256()
    modules = [("model", model)] + ([("prior", prior)] if prior is not None else [])
    for prefix, module in modules:
        for name, parameter in module.named_parameters():
            x = parameter.detach().reshape(-1)
            stride = max(1, x.numel() // 16)
            digest.update((prefix + "." + name).encode())
            digest.update(x[::stride][:16].float().cpu().numpy().tobytes())
    return digest.hexdigest()


class OuroTrainRayActor:
    """Actor contract consumed directly by slime.ray.actor_group.RayTrainGroup."""

    def __init__(self, world_size, rank, master_addr, master_port):
        if world_size != 1 or rank != 0:
            raise NotImplementedError("This validated backend currently supports one trainer rank")
        self.master_addr, self.master_port = "127.0.0.1", 0
        self.manager = None
        self.version = 0

    def get_master_addr_and_port(self):
        return self.master_addr, self.master_port

    def init(self, args, role, with_ref=False, with_opd_teacher=False):
        if role != "actor" or with_opd_teacher:
            raise ValueError("Only the scheme-2a actor role is implemented")
        self.args = args
        seed_everything(args.seed)
        self.adapter = load_adapter(args)
        self.reference = OuroAdapter.load(args.model, self.adapter.device,
            dtype=next(self.adapter.model.parameters()).dtype) if with_ref else None
        if self.reference is not None:
            self.reference.model.requires_grad_(False)
        self.prior = DepthPrior(self.adapter.model.config.hidden_size, tuple(args.depths)).to(self.adapter.device) if args.arm == "learned" else None
        parameter_groups = [{"params": list(self.adapter.model.parameters()), "lr": args.learning_rate}]
        if self.prior is not None:
            parameter_groups.append({"params": list(self.prior.parameters()), "lr": args.gate_learning_rate})
        self.optimizer = torch.optim.AdamW(parameter_groups, betas=(.9, .99), weight_decay=.1, foreach=False)
        self.adapter.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        self.start_rollout_id = 0
        if args.resume:
            local_checkpoint = Path(args.sync_dir) / "resume.pt"
            staged = stage_checkpoint(args.resume, local_checkpoint)
            write_json(Path(args.output) / "checkpoint-stage.json", staged)
            state = torch.load(local_checkpoint, map_location="cpu", weights_only=False, mmap=True)
            validate_resume(state, args)
            self.adapter.model.load_state_dict(state["model"])
            if self.prior is not None:
                self.prior.load_state_dict(state["depth_prior"])
            self.optimizer.load_state_dict(state["optimizer"])
            self.start_rollout_id = state["completed_steps"]
            self.version = state["weight_version"]
            torch.set_rng_state(state["torch_rng_state"])
            torch.cuda.set_rng_state_all(state["cuda_rng_state"])
            del state
            local_checkpoint.unlink()
        configure_device(args)
        write_json(Path(args.output) / "trainer.json", {
            "pid": os.getpid(), "ray_gpu_ids": ray.get_gpu_ids(), "precision": args.precision,
            "cpu_affinity": sorted(os.sched_getaffinity(0)),
            "parameter_count": sum(p.numel() for p in self.adapter.model.parameters()),
            "physical_decoder_layers": len(self.adapter.model.model.layers), "max_loops": 4,
            "gradient_checkpointing": True, "optimizer": "AdamW", "optimizer_parameter_dtype": str(next(self.adapter.model.parameters()).dtype)})
        return self.start_rollout_id

    def set_rollout_manager(self, manager):
        self.manager = manager

    def train(self, rollout_id, rollout_data, external_data=None):
        if external_data is not None:
            raise ValueError("Critic data is not supported")
        batch = ray.get(rollout_data) if isinstance(rollout_data, ray.ObjectRef) else rollout_data
        if batch["looprl"]["weight_version"] != self.version:
            raise RuntimeError("Stale rollout weights: refusing an off-policy batch")
        args = self.args
        count = len(batch["tokens"])
        if count != args.prompts_per_step * args.group_size:
            raise ValueError("Incomplete rollout batch; global reward scaling would change")
        trajectories = []
        for tokens, length, meta in zip(batch["tokens"], batch["response_lengths"], batch["metadata"]):
            ids = tokens.tolist() if torch.is_tensor(tokens) else tokens
            trajectories.append(Rollout(ids[:-length], ids[-length:], [], [meta["depth"]] * length,
                                         "", not meta["truncated"], 0., meta["prefill_loop_tokens"]))
        self.adapter.model.eval()
        old, refs, errors = [], [], []
        started = time.monotonic()
        for i, rollout in enumerate(trajectories):
            with torch.no_grad():
                lp = self.adapter.response_log_probs(rollout)
                sampled = torch.as_tensor(batch["rollout_log_probs"][i], device=self.adapter.device)
                error = float((lp - sampled).abs().max())
                if error > args.logprob_tolerance:
                    raise RuntimeError(f"Sampler/trainer mismatch {error:.8g} > {args.logprob_tolerance}")
                old.append(lp.detach())
                refs.append(self.reference.response_log_probs(rollout).detach() if self.reference else None)
                errors.append(error)
                if self.prior is not None:
                    q = self.prior(self.adapter.prompt_feature(rollout.prompt_ids))[0, args.depths.index(rollout.depths[0])]
                    if abs(float(q) - batch["metadata"][i]["old_log_depth"]) > args.logprob_tolerance:
                        raise RuntimeError("Depth prior likelihood mismatch")
        self.optimizer.zero_grad(set_to_none=True)
        self.adapter.model.train()
        torch.cuda.reset_peak_memory_stats()
        loss_total, grad_errors = 0., []
        for i, rollout in enumerate(trajectories):
            lp = self.adapter.response_log_probs(rollout)
            error = float((lp.detach() - old[i]).abs().max())
            if error > args.logprob_tolerance:
                raise RuntimeError(f"Gradient-enabled forward mismatch {error:.8g}")
            grad_errors.append(error)
            q = self.prior(self.adapter.prompt_feature(rollout.prompt_ids))[0, args.depths.index(rollout.depths[0])][None] if self.prior is not None else None
            loss = sequence_depth_loss(lp[None], old[i][None], torch.as_tensor(batch["loss_masks"][i], device=lp.device)[None],
                torch.tensor([batch["rewards"][i]], device=lp.device), q,
                torch.tensor([batch["metadata"][i]["old_log_depth"]], device=lp.device) if q is not None else None,
                refs[i][None] if refs[i] is not None else None, args.kl_coef, normalization=args.max_new_tokens) / count
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite loss")
            loss.backward()
            loss_total += float(loss.detach())
            del lp, loss, q
        parameters = list(self.adapter.model.parameters()) + (list(self.prior.parameters()) if self.prior is not None else [])
        norm = torch.nn.utils.clip_grad_norm_(parameters, .1)
        if not torch.isfinite(norm):
            raise RuntimeError("Non-finite gradient norm")
        # A fixed parameter probe distinguishes an optimizer call from an actual
        # change. FP32 master parameters avoid BF16 sub-ULP update loss.
        probe = self.adapter.model.model.layers[0].mlp.down_proj.weight
        before = probe.detach().clone()
        prior_before = self.prior.head.weight.detach().clone() if self.prior else None
        warmup = max(1, int(args.steps * .1 + .999999))
        scale = min(1., (rollout_id + 1) / warmup)
        for i, group in enumerate(self.optimizer.param_groups):
            group["lr"] = (args.learning_rate if i == 0 else args.gate_learning_rate) * scale
        self.optimizer.step()
        delta = float((probe.detach() - before).norm())
        prior_delta = float((self.prior.head.weight.detach() - prior_before).norm()) if self.prior else None
        del before, prior_before
        self.version += 1
        record = {"step": rollout_id, "weight_version": self.version,
            "beta": batch["looprl"]["beta"], "precision": args.precision,
            "reward_scale": float(batch_reward_scale(torch.tensor(batch["raw_reward"]))),
            "cost_coefficient": batch["looprl"]["beta"] / (4 * args.max_new_tokens),
            "advantages": batch["rewards"],
            "backend": "slime_custom_ouro", "reward_mean": sum(batch["raw_reward"]) / count,
            "truncation_rate": sum(batch["truncated"]) / count,
            "response_lengths": batch["response_lengths"], "depths": [r.depths[0] for r in trajectories],
            "loss": loss_total, "grad_norm_before_clip": float(norm),
            "sampler_trainer_logp_max_error": max(errors), "grad_forward_logp_max_error": max(grad_errors),
            "lm_probe_update_l2": delta, "depth_prior_update_l2": prior_delta,
            "peak_memory_gib": torch.cuda.max_memory_allocated() / 1024**3,
            "training_seconds": time.monotonic() - started}
        append_record(Path(args.output) / "metrics.jsonl", record)
        print(json.dumps(record), flush=True)
        self.optimizer.zero_grad(set_to_none=True)
        return record

    def update_weights(self):
        if self.manager is None:
            raise RuntimeError("No rollout manager attached")
        self.adapter.model.eval()
        root = Path(self.args.sync_dir)
        root.mkdir(parents=True, exist_ok=True)
        path = root / f"weights-{self.version:06d}.pt"
        temporary = path.with_suffix(".tmp")
        payload = {"model": {k: v.detach().cpu() for k, v in self.adapter.model.state_dict().items()},
                   "depth_prior": {k: v.detach().cpu() for k, v in self.prior.state_dict().items()} if self.prior else None,
                   "version": self.version, "precision": self.args.precision}
        torch.save(payload, temporary)
        temporary.replace(path)
        del payload
        expected = parameter_probe(self.adapter.model, self.prior)
        digest = file_digest(path)
        received = ray.get(self.manager.install_weights.remote(str(path), digest, expected))
        if received["version"] != self.version or received["probe"] != expected:
            raise RuntimeError("Weight synchronization failed")
        append_record(Path(self.args.output) / "weight-sync.jsonl", {**received, "file_sha256": digest})

    def clear_memory(self):
        self.optimizer.zero_grad(set_to_none=True)
        gc.collect()
        torch.cuda.empty_cache()

    def save_model(self, rollout_id, force_sync=False):
        path = Path(self.args.save) / f"step-{rollout_id + 1:06d}.pt"
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".tmp")
        torch.save({"model": self.adapter.model.state_dict(),
                    "depth_prior": self.prior.state_dict() if self.prior else None,
                    "optimizer": self.optimizer.state_dict(), "completed_steps": rollout_id + 1,
                    "weight_version": self.version, "args": vars(self.args),
                    "data_sha256": sha256(self.args.data),
                    "torch_rng_state": torch.get_rng_state(), "cuda_rng_state": torch.cuda.get_rng_state_all()}, temporary)
        temporary.replace(path)
        write_json(Path(self.args.output) / "checkpoint.json", {"path": str(path), "completed_steps": rollout_id + 1})


class OuroRolloutManager:
    """Custom manager/sampler implementing the native slime train-loop contract."""

    def __init__(self, args):
        from slime.data.batch_builder import BatchBuilder
        self.args = args
        self.adapter = load_adapter(args)
        self.prior = DepthPrior(self.adapter.model.config.hidden_size, tuple(args.depths)).to(self.adapter.device) if args.arm == "learned" else None
        self.rows = selected_rows(args, require_train=True)
        self.builder = BatchBuilder(args)
        self.version = -1
        self.actors = []
        self.ready = False
        self.paused = False
        self.next_rollout = 0

    def register_training_actors(self, role, actors, args):
        if role != "actor":
            raise ValueError("Unsupported role")
        self.actors = actors
        return {"update_weight_start_version": 0}

    def trainer_handles(self):
        return self.actors

    def install_weights(self, path, digest, expected_probe):
        if file_digest(path) != digest:
            raise RuntimeError("Weight artifact checksum mismatch")
        state = torch.load(path, map_location="cpu", mmap=True, weights_only=True)
        if state["precision"] != self.args.precision:
            raise ValueError("Sampler and trainer must use the same precision")
        self.adapter.model.load_state_dict(state["model"])
        if self.prior is not None:
            self.prior.load_state_dict(state["depth_prior"])
        self.version = state["version"]
        actual = parameter_probe(self.adapter.model, self.prior)
        if actual != expected_probe:
            raise RuntimeError("Installed weights do not match trainer parameter probes")
        del state
        gc.collect()
        return {"version": self.version, "probe": actual, "sampler_pid": os.getpid(), "ray_gpu_ids": ray.get_gpu_ids()}

    def load(self, rollout_id):
        self.next_rollout = rollout_id + 1

    def training_ready(self):
        if self.version < 0:
            raise RuntimeError("Weights must be synchronized before sampling")
        self.ready = True

    def generate(self, rollout_id):
        from slime.utils.types import Sample
        from .rewards import score_answer
        if not self.ready or self.paused or rollout_id != self.next_rollout:
            raise RuntimeError("Invalid rollout lifecycle state")
        args = self.args
        beta = ramped_beta(args.beta, rollout_id, args.steps, args.price_warmup)
        samples = []
        generator = torch.Generator().manual_seed(args.seed + rollout_id * 100003)
        for group in range(args.prompts_per_step):
            row = self.rows[(rollout_id * args.prompts_per_step + group) % len(self.rows)]
            prompt = self.adapter.prompt_ids(row["problem"])
            with torch.no_grad():
                logq = self.prior(self.adapter.prompt_feature(prompt))[0] if self.prior else None
            if args.arm == "fixed":
                depths = [4] * args.group_size
            elif args.arm == "stratified":
                depths = stratified_depths(args.group_size, tuple(args.depths), generator).tolist()
            else:
                depths = [args.depths[k] for k in torch.multinomial(logq.exp().cpu(), args.group_size, replacement=True, generator=generator).tolist()]
            for i, depth in enumerate(depths):
                seed = args.seed + 100003 * rollout_id + 997 * group + i
                rollout = self.adapter.generate(prompt, depth, args.max_new_tokens, seed=seed)
                # The prompt feature is computed once per group and shared by
                # its samples. Account for its actual extra four-loop pass.
                if self.prior is not None and i == 0:
                    rollout.prefill_loop_tokens += 4 * len(prompt)
                score = score_answer("<think>\n" + rollout.text, row["answer"])
                meta = {"depth": depth, "group_index": group, "weight_version": self.version,
                        "old_log_depth": float(logq[args.depths.index(depth)]) if logq is not None else 0.,
                        "truncated": not rollout.terminated, "prefill_loop_tokens": rollout.prefill_loop_tokens,
                        "id": row["id"], "seed": seed}
                sample = Sample(index=len(samples), rollout_id=len(samples), group_index=group,
                    prompt=row["problem"], tokens=prompt + rollout.response_ids,
                    response=rollout.text, response_length=len(rollout.response_ids), label=row["answer"],
                    reward=score["reward"], rollout_log_probs=rollout.log_probs,
                    status=Sample.Status.COMPLETED if rollout.terminated else Sample.Status.TRUNCATED,
                    weight_versions=[str(self.version)], train_metadata=meta,
                    metadata={"looprl": {"depths": rollout.depths, "beta": beta, "max_new_tokens": args.max_new_tokens}})
                samples.append(sample)
                append_record(Path(args.output) / "rollouts.jsonl", {**asdict(rollout), **score, **meta, "step": rollout_id})
                print(f"slime rollout={rollout_id} group={group} sample={i} depth={depth} tokens={sample.response_length} reward={sample.reward}", flush=True)
        batch = self.builder.convert(samples)
        batch["looprl"] = {"weight_version": self.version, "beta": beta, "rollout_id": rollout_id}
        return ray.put(batch)

    def training_completed(self, rollout_id):
        self.next_rollout = rollout_id + 1
        write_json(Path(self.args.output) / "status.json", {"status": "running", "completed_steps": self.next_rollout,
                   "total_steps": self.args.steps, "backend": "slime_custom_ouro"})

    def pause_rollout_admission(self):
        before = self.paused
        self.paused = True
        return before

    def resume_rollout_admission(self, was_paused):
        self.paused = was_paused

    def save(self, rollout_id):
        write_json(Path(self.args.output) / "rollout-state.json", {
            "completed_steps": rollout_id + 1, "next_rollout": self.next_rollout,
            "data_sha256": sha256(self.args.data), "rollout_weight_version": self.version})
