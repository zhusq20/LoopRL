"""Launch a bounded, paired stage-0 pilot, one fixed depth per unused GPU.

No process killing, service ports, or cluster-wide state changes. PID/argv/logs
are recorded so only these jobs can be monitored or stopped later.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--data", default="data/math500.jsonl")
    p.add_argument("--gpus", nargs="+", type=int, default=[2, 3, 4, 5])
    p.add_argument("--depths", nargs="+", type=int, default=[1, 2, 3, 4])
    p.add_argument("--limit", type=int, default=8)
    p.add_argument("--repeats", type=int, default=2)
    p.add_argument("--max-new-tokens", type=int, default=4096)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--top-p", type=float, default=.7)
    args = p.parse_args()
    if len(args.gpus) != len(args.depths) or len(set(args.gpus)) != len(args.gpus):
        raise ValueError("Assign one distinct GPU per depth")
    inventory = subprocess.check_output(["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"], text=True)
    used = {int(line.split(",")[0]): int(line.split(",")[1]) for line in inventory.splitlines()}
    for gpu in args.gpus:
        if used.get(gpu, 10**9) > 1024:
            raise RuntimeError(f"GPU {gpu} is occupied ({used.get(gpu)} MiB); refusing launch")
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    workers = []
    for gpu, depth in zip(args.gpus, args.depths):
        destination = output / f"depth{depth}"
        command = [sys.executable, "-u", "-m", "looprl", "evaluate", "--model", args.model,
                   "--data", args.data, "--output", str(destination), "--depths", str(depth),
                   "--limit", str(args.limit), "--repeats", str(args.repeats),
                   "--max-new-tokens", str(args.max_new_tokens), "--seed", str(args.seed),
                   "--top-p", str(args.top_p)]
        if depth == 4:
            command.append("--diagnostics")
        environment = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS="4", TOKENIZERS_PARALLELISM="false")
        log_path = output / f"depth{depth}.log"
        with log_path.open("w") as log:
            child = subprocess.Popen(command, env=environment, stdout=log, stderr=subprocess.STDOUT,
                                     stdin=subprocess.DEVNULL, start_new_session=True)
        workers.append({"pid": child.pid, "gpu": gpu, "depth": depth, "argv": command,
                        "log": str(log_path), "output": str(destination)})
    manifest = {"started_unix": time.time(), "pilot_only": True, "workers": workers,
                "planned_samples": len(args.depths) * args.limit * args.repeats,
                "interpretation": "4096-token pilot is not the long-budget paper baseline; report truncation."}
    (output / "launch.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
