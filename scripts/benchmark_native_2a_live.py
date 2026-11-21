"""Short diagnostic requests against idle Ouro engines, without changing weights.

Run only while the selected rollout engines are idle. Results include request
latency and server timing; these cached-prefix probes are not training throughput.
"""
import argparse
import hashlib
import json
import time
import urllib.request
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--host", required=True)
    p.add_argument("--engines", nargs="+", required=True, help="depth:port")
    p.add_argument("--input", required=True, help="Saved rollout JSONL")
    p.add_argument("--output", required=True)
    p.add_argument("--contexts", nargs="+", type=int, default=[512, 2048])
    p.add_argument("--batches", nargs="+", type=int, default=[1, 8])
    p.add_argument("--new-tokens", type=int, default=32)
    args = p.parse_args()
    with Path(args.input).open() as stream:
        tokens = next(row["tokens"] for line in stream
                      if len((row := json.loads(line))["tokens"]) >= max(args.contexts))
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "started_unix": time.time(),
        "note": "Live-engine synthetic cached-prefix probe; no weights or server settings changed."
                " Short requests do not represent complete RL iteration throughput.",
        "input_tokens_sha256": hashlib.sha256(json.dumps(tokens).encode()).hexdigest(),
        "cases": [],
    }
    for engine in args.engines:
        depth, port = map(int, engine.split(":"))
        base = f"http://{args.host}:{port}"

        def request(path, body=None):
            req = urllib.request.Request(base + path,
                data=None if body is None else json.dumps(body).encode(),
                headers={"Content-Type": "application/json"})
            with opener.open(req, timeout=90) as response:
                return json.load(response)

        info = request("/get_server_info")
        report.setdefault("engines", {})[engine] = {
            k: info.get(k) for k in ("dtype", "enable_tf32_matmul", "max_running_requests",
                "attention_backend", "disable_cuda_graph", "max_total_tokens")}
        if max(args.batches) > info["max_running_requests"]:
            raise ValueError("Requested batch exceeds engine admission limit")
        for context in args.contexts:
            ids = tokens[:context]
            request("/generate", {"input_ids": ids, "sampling_params": {
                "temperature": 0, "sampling_seed": 914, "max_new_tokens": 1, "ignore_eos": True}})
            for batch in args.batches:
                started = time.monotonic()
                results = request("/generate", {"input_ids": [ids] * batch,
                    "sampling_params": {"temperature": 0, "sampling_seed": 914,
                        "max_new_tokens": args.new_tokens, "ignore_eos": True},
                    "return_logprob": True, "logprob_start_len": context - 1})
                elapsed = time.monotonic() - started
                metadata = [{k: v for k, v in result["meta_info"].items()
                             if "logprob" not in k and k != "hidden_states"} for result in results]
                count = sum(m["completion_tokens"] for m in metadata)
                row = dict(depth=depth, port=port, context_tokens=context, batch=batch,
                    elapsed_seconds=elapsed, output_tokens=count, tokens_per_second=count/elapsed,
                    metadata=metadata)
                report["cases"].append(row)
                output.write_text(json.dumps(report, indent=2) + "\n")
                print(json.dumps({k: v for k, v in row.items() if k != "metadata"}), flush=True)
    report["finished_unix"] = time.time()
    output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
