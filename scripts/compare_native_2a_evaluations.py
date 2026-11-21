"""Paired evaluation deltas with question-level bootstrap uncertainty."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from looprl.data import read_jsonl, write_json

FIELDS = {"accuracy": lambda r: r["score"]["reward"],
          "response_tokens": lambda r: r["response_tokens"],
          "executed_loop_tokens": lambda r: r["executed_loop_tokens"],
          "truncation_rate": lambda r: float(r["truncated"])}


def load_evaluation(path):
    protocol = json.loads((path/"protocol.json").read_text())
    if json.loads((path/"status.json").read_text())["status"] != "completed":
        raise ValueError(f"Incomplete evaluation: {path}")
    rows = read_jsonl(path/"samples.jsonl")
    keyed = {(r["id"], r["repeat"]): r for r in rows}
    required = {(i, repeat) for i in protocol["ids"] for repeat in range(protocol["repeats"])}
    if keyed.keys() != required or len(keyed) != len(rows):
        raise ValueError(f"Missing or duplicate paired samples: {path}")
    return protocol, keyed


def compare(reference, candidate, *, replicates=10000, seed=20261121):
    (left_protocol, left), (right_protocol, right) = map(load_evaluation, (reference, candidate))
    for field in ("data_sha256", "ids", "repeats", "max_new_tokens", "seed", "dtype", "temperature", "top_p", "backend"):
        if left_protocol[field] != right_protocol[field]:
            raise ValueError(f"Unpaired evaluation setting: {field}")
    if left.keys() != right.keys():
        raise ValueError("Evaluation sample keys differ")
    ids = left_protocol["ids"]
    if not ids or replicates < 1:
        raise ValueError("Nonempty evaluations and positive bootstrap count required")
    rng = np.random.default_rng(seed)
    # Repeated samples from one question stay together. Treating them as
    # independent questions would understate uncertainty.
    indices = rng.integers(len(ids), size=(replicates, len(ids)))
    metrics = {}
    for name, extract in FIELDS.items():
        arrays = [np.array([np.mean([extract(rows[i, r]) for r in range(left_protocol["repeats"])])
                           for i in ids], dtype=np.float64) for rows in (left, right)]
        if not all(np.isfinite(a).all() for a in arrays):
            raise ValueError(f"Nonfinite evaluation metric: {name}")
        delta = arrays[1]-arrays[0]
        interval = np.quantile(delta[indices].mean(axis=1), [.025, .975]).tolist()
        metrics[name] = dict(reference_mean=float(arrays[0].mean()), candidate_mean=float(arrays[1].mean()),
            paired_delta=float(delta.mean()), question_bootstrap_95_interval=interval)
    return dict(reference=str(reference), candidate=str(candidate), questions=len(ids),
        repeats=left_protocol["repeats"], metrics=metrics,
        bootstrap=dict(unit="question", replicates=replicates, seed=seed),
        reference_policy={k:left_protocol[k] for k in ("checkpoint", "mode", "depth")},
        candidate_policy={k:right_protocol[k] for k in ("checkpoint", "mode", "depth")},
        limitation="Paired descriptive intervals; no multiple-comparison correction or automatic H4/H5 acceptance. Report truncation and compute with accuracy.")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--reference", type=Path, required=True)
    p.add_argument("--candidate", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    report = compare(a.reference, a.candidate)
    write_json(a.output, report)
    print(json.dumps(report))


if __name__ == "__main__":
    main()
