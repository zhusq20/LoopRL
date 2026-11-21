"""Select a compute price using held-out training examples only."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from looprl.data import read_jsonl, sha256, write_json


def select(evaluations, validation, plan):
    ids = {r["id"] for r in read_jsonl(validation)}
    data_hash = sha256(validation)
    results = []
    expected_keys = None
    expected_settings = None
    for arm in plan["arms"]:
        if arm["arm"] not in ("stratified", "token_exit") or not arm["beta"]:
            continue
        gated=arm['arm']=='token_exit'
        samples = []
        for depth in ((4,) if gated else (2, 3, 4)):
            path = evaluations/arm["name"]/('token_exit' if gated else f"depth{depth}")
            protocol = json.loads((path/"protocol.json").read_text())
            status = json.loads((path/"status.json").read_text())
            if status["status"] != "completed" or protocol["data_sha256"] != data_hash:
                raise ValueError(f"Incomplete evaluation or wrong validation data: {path}")
            export = json.loads((Path(protocol["checkpoint"])/"export.json").read_text())
            trained = export["protocol"]
            if trained["arm"] != arm['arm'] or trained["beta"] != arm["beta"] or trained["total_steps"] != plan["formal_steps"]:
                raise ValueError(f"Wrong trained checkpoint for price arm: {path}")
            if gated and protocol.get('gate_shift')!=trained['token_exit']['gate_shift']:
                raise ValueError('Evaluation changed the trained token gate shift')
            if Path(export["source"]).name != f"iter_{plan['formal_steps']-1:07d}":
                raise ValueError(f"Incomplete training checkpoint: {path}")
            settings = {k: protocol[k] for k in ("repeats", "max_new_tokens", "seed", "dtype", "temperature", "top_p", "backend")}
            if settings != plan["validation"]["evaluation"]:
                raise ValueError(f"Evaluation differs from the predeclared validation protocol: {path}")
            if expected_settings is None:
                expected_settings = settings
            if settings != expected_settings or protocol["mode"] != ('token_exit' if gated else "fixed") or protocol["depth"] != depth:
                raise ValueError(f"Unpaired evaluation protocol: {path}")
            rows = read_jsonl(path/"samples.jsonl")
            keys = {(r["id"], r["repeat"]) for r in rows}
            required = {(identifier, repeat) for identifier in ids for repeat in range(protocol["repeats"])}
            if keys != required or len(keys) != len(rows) or (not gated and any(r["depth"] != depth for r in rows)):
                raise ValueError(f"Missing, duplicated or unexpected validation samples: {path}")
            if gated and any(len(r.get('exits',[]))!=r['response_tokens'] or
                    any(e not in (1,2,3,4) for e in r['exits']) or
                    r['executed_loop_tokens']!=4*r['prompt_tokens']+sum(r['exits'][1:]) for r in rows):
                raise ValueError('Missing or invalid actual gated validation trajectory')
            if expected_keys is None:
                expected_keys = keys
            if keys != expected_keys:
                raise ValueError("Validation comparisons are not paired")
            samples.extend(rows)
        price = plan["validation"]["selection_reference_cost_per_loop_token"]
        cost_key='executed_loop_tokens' if gated else 'loop_tokens'
        results.append(dict(name=arm["name"], beta=arm["beta"], samples=len(samples),
            accuracy=sum(r["score"]["reward"] for r in samples)/len(samples),
            mean_response_loop_tokens=sum(r["loop_tokens"] for r in samples)/len(samples),
            mean_selection_loop_tokens=sum(r[cost_key] for r in samples)/len(samples),
            selection_cost=cost_key,
            utility=sum(r["score"]["reward"]-price*r[cost_key] for r in samples)/len(samples)))
    if not results:
        raise ValueError("No positive-price depth-policy arms in the experiment plan")
    best = min(results, key=lambda r: (-r["utility"], r["beta"]))
    import math
    nearest = min((r for r in results if r != best),
        key=lambda r: (abs(math.log(r["beta"]/best["beta"])), r["beta"]))
    return dict(selected_beta=best["beta"], same_price_fixed4_betas=[best["beta"], nearest["beta"]],
        validation_sha256=data_hash, criteria=plan["validation"], results=results,
        limitation="Hyperparameter selection on validation; not held-out test performance")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--evaluations", type=Path, required=True)
    p.add_argument("--validation", type=Path, required=True)
    p.add_argument("--plan", type=Path, default=ROOT/"configs/native2a-plan.json")
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    plan = json.loads(a.plan.read_text())
    import random
    training = read_jsonl(ROOT/"data/math_train.jsonl")
    random.Random(plan["seed"]).shuffle(training)
    if {r["id"]:r for r in read_jsonl(a.validation)} != {r["id"]:r for r in training[:256]}:
        raise ValueError("Selection input is not the predeclared held-out training split")
    report = select(a.evaluations, a.validation, plan)
    write_json(a.output, report)
    print(json.dumps(report))


if __name__ == "__main__":
    main()
