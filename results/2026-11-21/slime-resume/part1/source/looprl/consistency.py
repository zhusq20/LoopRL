"""Reproducible long-sequence cached/full/gradient-enabled comparison."""
import argparse
import gc
import json
from pathlib import Path
import time

import torch

from .data import read_jsonl, sha256, write_json
from .ouro import MODEL_REVISION, OuroAdapter, Rollout


def error_statistics(actual, expected):
    error = actual.detach() - expected.detach()
    return {"max_abs_error": float(error.abs().max()), "mean_abs_error": float(error.abs().mean()),
            "rms_error": float(error.square().mean().sqrt()),
            "ratio_max_abs_deviation": float((error.exp() - 1).abs().max())}


def check(args):
    destination = Path(args.output)
    if destination.exists():
        raise FileExistsError(destination)
    destination.mkdir(parents=True)
    row = read_jsonl(args.rollouts)[args.record]
    if max(args.lengths) > len(row["response_ids"]):
        raise ValueError("Recorded response is shorter than the requested check")
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    dtype = {"bf16": torch.bfloat16, "fp32": torch.float32}[args.precision]
    adapter = OuroAdapter.load(args.model, args.device, dtype=dtype)
    report = {"args": vars(args), "source_sha256": sha256(args.rollouts), "model_revision": MODEL_REVISION,
              "torch": torch.__version__, "records": [], "passed": False,
              "scope": "Fixed recorded text; numerical/gradient validation, not on-policy RL or accuracy."}
    for depth in args.depths:
        for length in args.lengths:
            rollout = Rollout(row["prompt_ids"], row["response_ids"][:length], [], [depth] * length,
                              "", False, 0., depth * len(row["prompt_ids"]))
            started = time.monotonic()
            adapter.model.eval()
            with torch.no_grad():
                cached = adapter.response_log_probs(rollout, replay_mode="cached")
                full = adapter.response_log_probs(rollout)
            entry = {"depth": depth, "tokens": length, "prompt_tokens": len(rollout.prompt_ids),
                     **error_statistics(full, cached), "forward_seconds": time.monotonic() - started}
            if entry["max_abs_error"] > args.tolerance:
                report["records"].append(entry)
                write_json(destination / "results.json", report)
                raise RuntimeError(f"Sampler/recompute mismatch: {entry}")
            if args.backward:
                adapter.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
                adapter.model.train()
                torch.cuda.reset_peak_memory_stats()
                started = time.monotonic()
                lp = adapter.response_log_probs(rollout)
                error = float((lp.detach() - full).abs().max())
                loss = -lp.mean()
                loss.backward()
                norm = float(torch.nn.utils.clip_grad_norm_(adapter.model.parameters(), .1))
                if error > args.tolerance or not torch.isfinite(torch.tensor(norm)) or norm <= 0:
                    raise RuntimeError("Gradient-enabled validation failed")
                entry.update(grad_forward_max_abs_error=error, grad_norm_before_clip=norm,
                             backward_seconds=time.monotonic() - started,
                             peak_memory_gib=torch.cuda.max_memory_allocated() / 1024**3)
                del lp, loss
                adapter.model.zero_grad(set_to_none=True)
            report["records"].append(entry)
            write_json(destination / "results.json", report)
            print(json.dumps(entry), flush=True)
            del cached, full
            gc.collect()
            torch.cuda.empty_cache()
    report["passed"] = True
    write_json(destination / "results.json", report)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="ByteDance/Ouro-2.6B-Thinking")
    p.add_argument("--rollouts", required=True)
    p.add_argument("--record", type=int, default=0)
    p.add_argument("--output", required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--precision", choices=["fp32", "bf16"], default="fp32")
    p.add_argument("--depths", type=int, nargs="+", default=[2, 3, 4])
    p.add_argument("--lengths", type=int, nargs="+", default=[512, 2048])
    p.add_argument("--tolerance", type=float, default=.001)
    p.add_argument("--backward", action="store_true")
    args = p.parse_args()
    if min(args.lengths) < 1 or any(k not in range(1, 5) for k in args.depths):
        raise ValueError("Invalid lengths or depths")
    check(args)


if __name__ == "__main__":
    main()
