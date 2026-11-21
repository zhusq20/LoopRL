"""Bounded single-GPU reference experiments; all samples and settings are saved."""
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import random
import subprocess
import time
import shutil

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence

from .data import read_jsonl, sha256, write_json
from .objectives import priced_advantages, ramped_beta, sequence_depth_loss, stratified_depths
from .ouro import DepthPrior, MODEL_ID, MODEL_REVISION, OuroAdapter
from .rewards import score_answer


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def new_run(args):
    path = Path(args.output)
    path.mkdir(parents=True, exist_ok=True)
    if (path / "manifest.json").exists():
        raise FileExistsError(f"Refusing to overwrite an existing run: {path}")
    import transformers
    gpu = subprocess.run(["nvidia-smi", "--query-gpu=index,name,uuid,memory.used", "--format=csv,noheader"],
                         capture_output=True, text=True, check=False).stdout
    git = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True, check=False).stdout
    # Include source hash, not just git HEAD: experiments are often run before committing.
    source = {str(p): sha256(p) for p in sorted(Path("looprl").glob("*.py"))}
    manifest = {"args": vars(args), "model_id": MODEL_ID, "model_revision": MODEL_REVISION,
                "git_head": git, "git_dirty": dirty, "source_sha256": source,
                "torch": torch.__version__, "transformers": transformers.__version__,
                "gpu_inventory": gpu, "started_unix": time.time(),
                "backend": "hf_reference", "slime_training": False,
                "data_sha256": sha256(args.data) if getattr(args, "data", None) else None}
    write_json(path / "manifest.json", manifest)
    write_json(path / "status.json", {"status": "running", "started_unix": manifest["started_unix"]})
    for source_path in Path("looprl").glob("*.py"):
        destination = path / "source" / "looprl" / source_path.name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, destination)
    return path


def append_record(path, record):
    with Path(path).open("a") as f:
        f.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")


def selected_rows(args, require_train=False):
    rows = read_jsonl(args.data)
    if require_train and any(row.get("split") != "train" for row in rows):
        raise ValueError("Training requires explicitly marked train records; evaluation data is forbidden")
    if not require_train and any(row.get("split") == "train" for row in rows):
        raise ValueError("Evaluation requires a held-out dataset")
    order = list(range(len(rows)))
    random.Random(args.seed).shuffle(order)
    selected = [rows[i] for i in order[args.offset:args.offset + args.limit]]
    if not selected:
        raise ValueError("No samples selected")
    return selected


def summarize(records):
    result = {"samples": len(records), "by_depth": {}}
    for depth in sorted({str(r["depth"]) for r in records}):
        xs = [r for r in records if str(r["depth"]) == depth]
        repeats = sorted({r["repeat"] for r in xs})
        accuracies = [np.mean([r["reward"] for r in xs if r["repeat"] == repeat]) for repeat in repeats]
        result["by_depth"][depth] = {
            "samples": len(xs), "accuracy": float(np.mean([r["reward"] for r in xs])),
            "repeat_accuracy_std": float(np.std(accuracies, ddof=1)) if len(accuracies) > 1 else None,
            "mean_response_tokens": float(np.mean([len(r["response_ids"]) for r in xs])),
            "mean_loop_tokens": float(np.mean([sum(r["depths"]) for r in xs])),
            "mean_prefill_loop_tokens": float(np.mean([r["prefill_loop_tokens"] for r in xs])),
            "mean_executed_total_loop_tokens": float(np.mean([r["prefill_loop_tokens"] + sum(r["depths"]) - r["depths"][0] for r in xs])),
            "mean_depth": sum(sum(r["depths"]) for r in xs) / sum(len(r["depths"]) for r in xs),
            "truncation_rate": float(np.mean([not r["terminated"] for r in xs])),
        }
    grouped = {}
    for row in records:
        grouped.setdefault((row["id"], row["repeat"]), {})[str(row["depth"])] = row["reward"]
    paired = [g for g in grouped.values() if all(str(k) in g for k in (2, 3, 4))]
    if paired:
        gains = np.array([max(g["2"], g["3"], g["4"]) - g["4"] for g in paired])
        result["M4"] = {"paired_samples": len(paired), "oracle_accuracy": float(np.mean([max(g["2"], g["3"], g["4"]) for g in paired])),
                         "oracle_minus_4": float(gains.mean()),
                         "shallow_correct_deep_wrong": float(np.mean([max(g["2"], g["3"]) > g["4"] for g in paired])),
                         "note": "Per-sample stochastic oracle; do not interpret as a learned policy."}
    diagnostics = [r for r in records if r.get("diagnostics")]
    if diagnostics:
        gaps = np.array([g for r in diagnostics for g in r["diagnostics"]["gaps"]]).clip(0)
        result["M1"] = {"gap_mean": float(gaps.mean()), "positions": len(gaps)}
        for fraction in (.01, .05):
            result["M1"][f"top_{round(fraction * 100)}pct_share"] = float(np.sort(gaps)[-max(1, math.ceil(len(gaps) * fraction)):].sum() / gaps.sum()) if gaps.sum() else 0.
        result["M2"] = {str(int(reward)): float(np.mean([g for r in diagnostics if r["reward"] == reward for g in r["diagnostics"]["gaps"]]))
                         if any(r["reward"] == reward for r in diagnostics) else None for reward in (0., 1.)}
        counts = np.array([r["diagnostics"]["agreement_counts"] for r in diagnostics]).sum(0)
        result["M3"] = {f"loop{k}_vs_4": float(counts[k - 1] / len(gaps)) for k in (1, 2, 3)}
    return result


def evaluate(args):
    path = new_run(args)
    rows = selected_rows(args)
    seed_everything(args.seed)
    adapter = OuroAdapter.load(args.model, args.device)
    prior = None
    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        adapter.model.load_state_dict(checkpoint["model"])
        if args.mode == "learned":
            if checkpoint["depth_prior"] is None:
                raise ValueError("Checkpoint does not contain a learned depth prior")
            prior = DepthPrior(adapter.model.config.hidden_size, tuple(checkpoint["args"]["depths"])).to(adapter.device)
            prior.load_state_dict(checkpoint["depth_prior"])
    elif args.mode == "learned":
        raise ValueError("Learned-depth evaluation requires --checkpoint")
    records = []
    for i, row in enumerate(rows):
        prompt = adapter.prompt_ids(row["problem"])
        for repeat in range(args.repeats):
            seed = args.seed + 100003 * i + repeat
            for depth in args.depths:
                if args.mode == "learned":
                    with torch.no_grad():
                        q = prior(adapter.prompt_feature(prompt))[0]
                        gen = torch.Generator(device=adapter.device).manual_seed(seed)
                        sampled_depth = prior.choices[int(torch.multinomial(q.exp(), 1, generator=gen))]
                    rollout = adapter.generate(prompt, sampled_depth, args.max_new_tokens, args.temperature, args.top_p, seed + 1)
                    rollout.prefill_loop_tokens += 4 * len(prompt)
                    label = "learned"
                elif args.mode == "fixed":
                    rollout = adapter.generate(prompt, depth, args.max_new_tokens, args.temperature, args.top_p, seed)
                    label = depth
                else:
                    rollout = adapter.generate_ragged(prompt, args.max_new_tokens, args.gate_shift,
                                                      args.temperature, seed, args.mode, args.threshold, top_p=args.top_p)
                    label = args.mode
                # Include <think> prefill marker so unfinished reasoning cannot earn a reward.
                score = score_answer("<think>\n" + rollout.text, row["answer"])
                record = {**asdict(rollout), **score, "id": row["id"], "gold": row["answer"],
                          "depth": label, "repeat": repeat, "seed": seed}
                if args.diagnostics and depth == 4 and args.mode == "fixed":
                    record["diagnostics"] = adapter.diagnostics(rollout)
                records.append(record)
                append_record(path / "samples.jsonl", record)
                summary = summarize(records)
                write_json(path / "summary.json", summary)
                print(json.dumps({"completed": len(records), "id": row["id"], "depth": label,
                                  "reward": score["reward"], "tokens": len(rollout.response_ids),
                                  "terminated": rollout.terminated, "seconds": rollout.seconds}), flush=True)
                if args.mode != "fixed":
                    break
    write_json(path / "status.json", {"status": "completed", "samples": len(records), "finished_unix": time.time()})
    return summarize(records)


def train(args):
    if args.group_size < 2:
        raise ValueError("GRPO requires at least two rollouts per prompt")
    if args.top_p != 1:
        raise ValueError("Reference training requires top_p=1 for sampler/update support consistency")
    if args.replay_mode == "cached" and args.max_new_tokens > 64:
        raise ValueError("Cached differentiable replay is a short-run reference (at most 64 tokens)")
    path = new_run(args)
    rows = selected_rows(args, require_train=True)
    seed_everything(args.seed)
    adapter = OuroAdapter.load(args.model, args.device)
    reference = OuroAdapter.load(args.model, args.device) if args.kl_coef else None
    if reference:
        reference.model.requires_grad_(False)
    prior = DepthPrior(adapter.model.config.hidden_size, tuple(args.depths)).to(adapter.device) if args.arm == "learned" else None
    params = [{"params": list(adapter.model.parameters()), "lr": args.learning_rate}]
    if prior:
        params.append({"params": list(prior.parameters()), "lr": args.gate_learning_rate})
    if args.optimizer == "adamw8bit":
        from bitsandbytes.cextension import lib
        if lib is None or not getattr(lib, "compiled_with_cuda", False):
            raise RuntimeError("bitsandbytes CUDA library unavailable; use --optimizer adamw for reference validation")
        from bitsandbytes.optim import AdamW8bit
        optimizer = AdamW8bit(params, betas=(.9, .99), weight_decay=.1)
    else:
        optimizer = torch.optim.AdamW(params, betas=(.9, .99), weight_decay=.1)
    if args.replay_mode == "full":
        adapter.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    generator = torch.Generator().manual_seed(args.seed)
    for step in range(args.steps):
        started = time.monotonic()
        adapter.model.eval()
        trajectories, rewards, groups, depth_logps = [], [], [], []
        beta = ramped_beta(args.beta, step, args.steps, args.price_warmup)
        for group in range(args.prompts_per_step):
            row = rows[(step * args.prompts_per_step + group) % len(rows)]
            prompt = adapter.prompt_ids(row["problem"])
            with torch.no_grad():
                q = prior(adapter.prompt_feature(prompt))[0] if prior else None
            if args.arm == "fixed":
                depths = [4] * args.group_size
            elif args.arm == "stratified":
                depths = stratified_depths(args.group_size, tuple(args.depths), generator).tolist()
            else:
                depths = [args.depths[k] for k in torch.multinomial(q.exp().cpu(), args.group_size, replacement=True, generator=generator).tolist()]
            for rollout_index, depth in enumerate(depths):
                seed = args.seed + step * 100003 + group * 997 + rollout_index
                rollout = adapter.generate(prompt, depth, args.max_new_tokens, args.temperature, args.top_p, seed)
                score = score_answer("<think>\n" + rollout.text, row["answer"])
                trajectories.append(rollout)
                rewards.append(score["reward"])
                groups.append(group)
                depth_logps.append(float(q[args.depths.index(depth)]) if q is not None else 0.)
                append_record(path / "rollouts.jsonl", {**asdict(rollout), **score, "step": step,
                    "group": group, "id": row["id"], "seed": seed, "old_log_depth": depth_logps[-1]})
                print(f"step={step} group={group} sample={rollout_index} depth={depth} reward={score['reward']} tokens={len(rollout.response_ids)}", flush=True)
        device = adapter.device
        depths = pad_sequence([torch.tensor(r.depths, device=device) for r in trajectories], batch_first=True)
        mask = depths > 0
        advantage = priced_advantages(torch.tensor(rewards, device=device), torch.tensor(groups, device=device),
                                      depths, mask, beta, args.max_new_tokens)
        # Recompute old policy with the exact same update forward; independently
        # check it against sampler log probabilities before accepting any update.
        old, refs, errors = [], [], []
        with torch.no_grad():
            for rollout in trajectories:
                lp = adapter.response_log_probs(rollout, args.temperature, replay_mode=args.replay_mode)
                error = float((lp - torch.tensor(rollout.log_probs, device=device)).abs().max())
                errors.append(error)
                if error > args.logprob_tolerance:
                    raise RuntimeError(f"rollout/update mismatch {error:.6f} > {args.logprob_tolerance}; refusing update")
                old.append(lp.detach())
                refs.append(reference.response_log_probs(rollout, args.temperature, replay_mode=args.replay_mode).detach() if reference else None)
        optimizer.zero_grad(set_to_none=True)
        adapter.model.train()
        total_loss = 0.
        update_errors = []
        for i, rollout in enumerate(trajectories):
            lp = adapter.response_log_probs(rollout, args.temperature, replay_mode=args.replay_mode)
            gradient_forward_error = float((lp.detach() - old[i]).abs().max())
            update_errors.append(gradient_forward_error)
            if gradient_forward_error > args.logprob_tolerance:
                raise RuntimeError(f"Gradient-enabled update mismatch: {gradient_forward_error:.6f}")
            q = prior(adapter.prompt_feature(rollout.prompt_ids))[0, args.depths.index(rollout.depths[0])][None] if prior else None
            loss = sequence_depth_loss(lp[None], old[i][None], torch.ones_like(lp)[None], advantage.values[i:i+1],
                q, torch.tensor([depth_logps[i]], device=device) if prior else None,
                refs[i][None] if refs[i] is not None else None, args.kl_coef,
                normalization=args.max_new_tokens) / len(trajectories)
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite training loss")
            loss.backward()
            total_loss += float(loss.detach())
        norm = torch.nn.utils.clip_grad_norm_([p for pg in params for p in pg["params"]] if prior else adapter.model.parameters(), .1)
        if not torch.isfinite(norm):
            raise RuntimeError("Non-finite gradient norm")
        warmup = max(1, math.ceil(args.steps * .1))
        lr_scale = min(1., (step + 1) / warmup)
        for i, pg in enumerate(optimizer.param_groups):
            pg["lr"] = (args.learning_rate if i == 0 else args.gate_learning_rate) * lr_scale
        optimizer.step()
        prior_grad_norm = float(torch.stack([p.grad.float().square().sum() for p in prior.parameters() if p.grad is not None]).sum().sqrt()) if prior else None
        record = {"step": step, "beta": beta, "cost_coefficient": advantage.cost_coefficient,
                  "reward_mean": float(np.mean(rewards)), "reward_std": float(np.std(rewards)),
                  "reward_scale": float(advantage.reward_scale), "loss": total_loss,
                  "grad_norm_before_clip": float(norm), "rollout_update_logp_max_error": max(errors),
                  "grad_enabled_logp_max_error": max(update_errors),
                  "depth_prior_grad_norm_after_clip": prior_grad_norm,
                  "mean_response_tokens": float(np.mean([len(r.response_ids) for r in trajectories])),
                  "mean_loop_tokens": float(np.mean([r.loop_tokens for r in trajectories])),
                  "mean_depth": float(depths.sum() / mask.sum()), "seconds": time.monotonic() - started,
                  "truncation_rate": float(np.mean([not r.terminated for r in trajectories]))}
        append_record(path / "metrics.jsonl", record)
        print(json.dumps(record), flush=True)
        write_json(path / "status.json", {"status": "running", "completed_steps": step + 1, "total_steps": args.steps})
    checkpoint = {"model": adapter.model.state_dict(), "depth_prior": prior.state_dict() if prior else None,
                  "optimizer": optimizer.state_dict(), "args": vars(args), "completed_steps": args.steps,
                  "torch_rng_state": torch.get_rng_state(), "cuda_rng_state": torch.cuda.get_rng_state_all()}
    torch.save(checkpoint, path / "checkpoint.pt")
    write_json(path / "status.json", {"status": "completed", "completed_steps": args.steps, "finished_unix": time.time()})
    return record
