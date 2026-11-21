"""Compare uninterrupted and resumed runs, including every checkpoint tensor."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import math
from pathlib import Path

import torch


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--control", required=True)
    p.add_argument("--resumed", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--model-atol", type=float, default=0.)
    p.add_argument("--optimizer-atol", type=float, default=0.)
    p.add_argument("--logprob-atol", type=float, default=0.)
    p.add_argument("--local-cache", help="Stage shared-filesystem checkpoints here before mmap comparison")
    args = p.parse_args()
    torch.set_num_threads(4)
    roots = [Path(args.control), Path(args.resumed)]
    checkpoints = [json.loads((root / "checkpoint.json").read_text())["path"] for root in roots]
    local_paths, staging = checkpoints, []
    if args.local_cache:
        from looprl.slime_backend import stage_checkpoint
        local_paths = [str(Path(args.local_cache) / f"checkpoint-{i}.pt") for i in range(2)]
        with ThreadPoolExecutor(max_workers=2) as pool:
            staging = list(pool.map(lambda pair: stage_checkpoint(*pair), zip(checkpoints, local_paths)))
    states = [torch.load(path, map_location="cpu", weights_only=False, mmap=True) for path in local_paths]
    report = {"checkpoints": checkpoints, "tensors_compared": 0, "elements_compared": 0,
              "tolerances": {"model": args.model_atol, "optimizer": args.optimizer_atol, "logprob": args.logprob_atol},
              "staging": staging, "bitwise_difference_count": 0, "max_abs_error_by_section": {}, "mismatches": [], "passed": False}

    def compare(a, b, path):
        section = path.split(".")[0]
        if torch.is_tensor(a) and torch.is_tensor(b):
            report["tensors_compared"] += 1
            report["elements_compared"] += a.numel()
            equal = a.dtype == b.dtype and a.shape == b.shape and torch.equal(a, b)
            if not equal:
                report["bitwise_difference_count"] += 1
                if a.shape == b.shape and a.dtype == b.dtype and a.is_floating_point():
                    error = float((a - b).abs().max())
                    report["max_abs_error_by_section"][section] = max(error, report["max_abs_error_by_section"].get(section, 0.))
                    tolerance = args.optimizer_atol if section == "optimizer" else args.model_atol
                    equal = math.isfinite(error) and error <= tolerance
        elif isinstance(a, dict) and isinstance(b, dict):
            equal = a.keys() == b.keys()
            for key in a.keys() & b.keys():
                compare(a[key], b[key], f"{path}.{key}")
        elif isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
            equal = len(a) == len(b)
            for i, (x, y) in enumerate(zip(a, b)):
                compare(x, y, f"{path}[{i}]")
        else:
            equal = type(a) is type(b) and a == b
            if not equal and isinstance(a, float) and isinstance(b, float) and section.startswith("sample"):
                report["bitwise_difference_count"] += 1
                error = abs(a - b)
                report["max_abs_error_by_section"]["sample_float"] = max(error, report["max_abs_error_by_section"].get("sample_float", 0.))
                equal = math.isfinite(error) and error <= args.logprob_atol
        if not equal:
            report["mismatches"].append(path)

    for key in ("model", "depth_prior", "optimizer", "torch_rng_state", "cuda_rng_state",
                "completed_steps", "weight_version", "data_sha256"):
        compare(states[0][key], states[1][key], key)
    samples = [[json.loads(line) for line in (root / "rollouts.jsonl").read_text().splitlines()] for root in roots]
    lookup = {(r["step"], r["id"], r["seed"]): r for r in samples[0]}
    report["resumed_samples_compared"] = len(samples[1])
    if not samples[1]:
        report["mismatches"].append("empty resumed rollout log")
    for row in samples[1]:
        key = row["step"], row["id"], row["seed"]
        reference = lookup.get(key)
        if reference is None:
            report["mismatches"].append(f"missing control sample {key}")
            continue
        for field in ("prompt_ids", "response_ids", "depths", "log_probs", "old_log_depth", "reward", "weight_version"):
            compare(reference[field], row[field], f"sample{key}.{field}")
    report["passed"] = not report["mismatches"]
    report["bitwise_equal"] = report["passed"] and report["bitwise_difference_count"] == 0
    Path(args.output).write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
