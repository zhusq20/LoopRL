import argparse
import json
from pathlib import Path


def parser():
    root = argparse.ArgumentParser(description="Scheme 2: reproducible reference validation")
    sub = root.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare-data")
    prep.add_argument("--output", default="data")
    summary = sub.add_parser("summarize")
    summary.add_argument("inputs", nargs="+")
    summary.add_argument("--output", required=True)
    for command in ("preflight", "evaluate", "train", "history-probe"):
        p = sub.add_parser(command)
        p.add_argument("--model", default="ByteDance/Ouro-2.6B-Thinking")
        p.add_argument("--output", required=True)
        p.add_argument("--device", default="cuda:0")
        p.add_argument("--seed", type=int, default=42)
        p.add_argument("--logprob-tolerance", type=float, default=.15,
                       help="BF16 absolute logp bound; report actual error, never suppress mismatch")
        if command == "preflight":
            p.add_argument("--tokens", type=int, default=8)
            continue
        p.add_argument("--data", required=True)
        p.add_argument("--limit", type=int, default=16 if command == "evaluate" else 7500)
        p.add_argument("--offset", type=int, default=0)
        p.add_argument("--depths", type=int, nargs="+", default=[2, 3, 4])
        p.add_argument("--max-new-tokens", type=int, default=2048)
        p.add_argument("--temperature", type=float, default=1.)
        p.add_argument("--top-p", type=float, default=1.)
        if command == "history-probe":
            p.set_defaults(limit=2, max_new_tokens=32)
            continue
        if command == "evaluate":
            p.add_argument("--repeats", type=int, default=1)
            p.add_argument("--diagnostics", action="store_true")
            p.add_argument("--checkpoint")
            p.add_argument("--mode", choices=["fixed", "learned", "sample", "q_exit", "confidence"], default="fixed")
            p.add_argument("--gate-shift", type=float, default=0.)
            p.add_argument("--threshold", type=float, default=.5)
        else:
            p.add_argument("--arm", choices=["fixed", "stratified", "learned"], default="stratified")
            p.add_argument("--steps", type=int, default=140)
            p.add_argument("--prompts-per-step", type=int, default=32)
            p.add_argument("--group-size", type=int, default=8)
            p.add_argument("--beta", type=float, default=0.)
            p.add_argument("--price-warmup", type=float, default=.3)
            p.add_argument("--learning-rate", type=float, default=1e-6)
            p.add_argument("--gate-learning-rate", type=float, default=1e-4)
            p.add_argument("--kl-coef", type=float, default=1e-3)
            p.add_argument("--optimizer", choices=["adamw8bit", "adamw"], default="adamw8bit")
            p.add_argument("--replay-mode", choices=["full", "cached"], default="full")
    return root


def execute(args):
    if getattr(args, "depths", None) and any(k not in (1, 2, 3, 4) for k in args.depths):
        raise ValueError("depths must be in 1..4")
    if args.command == "prepare-data":
        from .data import prepare
        result = prepare(args.output)
    elif args.command == "preflight":
        from .validation import preflight
        result = preflight(args)
    elif args.command == "evaluate":
        from .experiments import evaluate
        result = evaluate(args)
    elif args.command == "train":
        from .experiments import train
        result = train(args)
    elif args.command == "history-probe":
        from .history import history_probe
        result = history_probe(args)
    else:
        from .data import read_jsonl, write_json
        from .experiments import summarize
        records = [r for p in args.inputs for r in read_jsonl(p)]
        result = summarize(records)
        write_json(args.output, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


def main():
    args = parser().parse_args()
    output = Path(args.output)
    existed = (output / "manifest.json").exists()
    try:
        execute(args)
    except Exception as error:
        if not existed and (output / "manifest.json").exists():
            from .data import write_json
            write_json(output / "status.json", {"status": "failed", "error_type": type(error).__name__, "error": str(error)})
        raise


if __name__ == "__main__":
    main()
