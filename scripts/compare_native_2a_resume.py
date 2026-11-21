"""Compare a resumed run against continuation from the SAME saved boundary."""
import argparse
import json
from pathlib import Path
import sys

import torch
from torch.distributed.checkpoint import FileSystemReader, load
from torch.distributed.checkpoint.metadata import TensorStorageMetadata
from megatron.core.dist_checkpointing import load_common_state_dict

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from looprl.data import read_jsonl, write_json


def read_tensors(path):
    reader = FileSystemReader(path)
    metadata = reader.read_metadata()
    state = {name: torch.empty(item.size, dtype=item.properties.dtype)
             for name, item in metadata.state_dict_metadata.items() if isinstance(item, TensorStorageMetadata)}
    load(state, storage_reader=reader)
    return state


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--control", type=Path, required=True)
    p.add_argument("--resumed", type=Path, required=True)
    p.add_argument("--step", type=int, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    torch.set_num_threads(8)
    launch = json.loads((a.resumed/"launch.json").read_text())["argv"]
    assert "--ckpt-step" in launch and "--load" in launch
    boundary = int(launch[launch.index("--ckpt-step")+1])
    assert Path(launch[launch.index("--load")+1]).resolve() == (a.control/"checkpoints").resolve()
    assert boundary < a.step
    paths = [root/"checkpoints"/f"iter_{a.step:07d}" for root in (a.control, a.resumed)]
    left, right = [read_tensors(path) for path in paths]
    failures = []
    if left.keys() != right.keys():
        failures.append("checkpoint_tensor_keys_differ")
    maximum = {"model": 0., "optimizer": 0.}
    bitwise = True
    count = elements = 0
    for key in left.keys() & right.keys():
        x, y = left[key], right[key]
        if x.shape != y.shape or x.dtype != y.dtype:
            failures.append(key+":schema")
            continue
        section = "optimizer" if key.startswith("optimizer.") else "model"
        tolerance = 1e-10 if section == "optimizer" else 1e-7
        error = 0.
        for xs, ys in zip(x.flatten().split(1_000_000), y.flatten().split(1_000_000), strict=True):
            if not torch.isfinite(xs).all() or not torch.isfinite(ys).all():
                failures.append(key+":nonfinite")
            error = max(error, (xs-ys).abs().max().item())
        maximum[section] = max(maximum[section], error)
        bitwise &= torch.equal(x, y)
        if error > tolerance:
            failures.append(key)
        count += 1
        elements += x.numel()
    common = [load_common_state_dict(path) for path in paths]
    if common[0]["iteration"] != common[1]["iteration"] or common[0]["opt_param_scheduler"] != common[1]["opt_param_scheduler"]:
        failures.append("iteration_or_scheduler_differs")
    for state in common:
        if {g["step"] for g in state["optimizer"]["optimizer"]["param_groups"]} != {a.step+1}:
            failures.append("optimizer_step_wrong")
    selected = [{(r["step"], r["index"]): r for r in read_jsonl(root/"rollouts.jsonl")
                 if boundary < r["step"] <= a.step} for root in (a.control, a.resumed)]
    logp_error = 0.
    if selected[0].keys() != selected[1].keys():
        failures.append("sample_keys_differ")
    for key in selected[0].keys() & selected[1].keys():
        x, y = selected[0][key], selected[1][key]
        if any(x[k] != y[k] for k in ("tokens", "reward", "label", "group")):
            failures.append(f"sample:{key}")
            continue
        for field in ("ouro_depth", "ouro_seed", "ouro_beta"):
            if x["train_metadata"][field] != y["train_metadata"][field]:
                failures.append(f"sample:{key}:{field}")
        if x['train_metadata'].get('ouro_mode')=='token_exit':
            for field in ('ouro_mode','ouro_exits','ouro_gate_shift','executed_loop_tokens'):
                if x['train_metadata'].get(field)!=y['train_metadata'].get(field):
                    failures.append(f'sample:{key}:{field}')
        error = max(abs(u-v) for u,v in zip(x["rollout_log_probs"], y["rollout_log_probs"], strict=True))
        logp_error = max(logp_error, error)
        bitwise &= x["rollout_log_probs"] == y["rollout_log_probs"]
    if logp_error > 1e-5:
        failures.append("sampling_logprob_tolerance_exceeded")
    # The restored source cursor must advance identically, with no skipped questions.
    cursors = [torch.load(root/"checkpoints/rollout"/f"global_dataset_state_dict_{a.step}.pt",
                         weights_only=False, map_location="cpu") for root in (a.control, a.resumed)]
    for field in ("sample_offset", "epoch_id", "sample_group_index", "sample_index"):
        if cursors[0][field] != cursors[1][field]:
            failures.append("data_cursor:"+field)
    report = dict(passed=not failures, bitwise_equal=bitwise and not failures,
        boundary=boundary, compared_step=a.step, checkpoints=list(map(str, paths)),
        tolerances=dict(model=1e-7, optimizer=1e-10, sampling_logp=1e-5),
        maximum_absolute_error=maximum, max_sampling_logprob_error=logp_error,
        tensor_count=count, elements=elements, samples=len(selected[1]), failures=failures)
    write_json(a.output, report)
    print(json.dumps({k:v for k,v in report.items() if k != "failures"}))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
