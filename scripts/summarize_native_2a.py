"""Fail-closed acceptance report for real-reward native 2a training."""
import argparse
import ast
import json
import math
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from looprl.data import read_jsonl, write_json


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    protocol = json.loads((a.run/"protocol.json").read_text())
    gated = protocol['arm'] == 'token_exit'
    status = json.loads((a.run/"status.json").read_text())
    log = re.sub(r"\x1b\[[0-9;]*m", "", a.run.with_suffix(".log").read_text())
    steps = [dict(step=int(step), metrics=ast.literal_eval(value)) for step, value in
             re.findall(r"model\.py:\d+ - step (\d+): (\{[^\n]*\})", log)]
    rollouts = read_jsonl(a.run/"rollouts.jsonl") if (a.run/"rollouts.jsonl").exists() else []
    metrics = read_jsonl(a.run/"rollout-metrics.jsonl") if (a.run/"rollout-metrics.jsonl").exists() else []
    checks = read_jsonl(a.run/"consistency.jsonl") if (a.run/"consistency.jsonl").exists() else []
    failures = []
    if status["status"] != "completed":
        failures.append("run_not_completed")
    launch = json.loads((a.run/"launch.json").read_text())["argv"]
    start = int(launch[launch.index("--ckpt-step")+1])+1 if "--ckpt-step" in launch else 0
    finish = int(launch[launch.index("--num-rollout")+1])
    if [s["step"] for s in steps] != list(range(start, finish)):
        failures.append("missing_or_duplicate_training_steps")
    correction=protocol.get("rollout_correction",{}).get("mode","none")
    bad_probability_check = any(not math.isfinite(c["sampler_recompute_max_error"]) for c in checks)
    if correction=="tis":
        cap=protocol["rollout_correction"]["clip_high"]
        bad_probability_check |= any(c.get("correction")!="tis" or
            not 0<c.get("tis_weight_mean",0)<=cap or not 0<c.get("tis_ess_fraction",0)<=1.000001 or
            not 0<=c.get("tis_clip_fraction",-1)<=1 for c in checks)
    else:
        bad_probability_check |= any(c["sampler_recompute_max_error"] > .001 for c in checks)
    if len(checks) != len(steps) or bad_probability_check:
        failures.append("missing_or_failed_probability_checks")
    execution = protocol.get("training_execution", {})
    if execution.get("reuse_old_log_probs"):
        expected_samples = protocol["prompts"]*protocol["group_size"]
        for check in checks:
            error = check.get("grad_no_grad_max_error", float("inf"))
            if (check.get("old_policy_source") != "detached_training_forward"
                    or not check.get("reference_finite")
                    or not math.isfinite(error) or error > .001
                    or check.get("samples") != expected_samples
                    or check.get("data_parallel_size") != execution["data_parallel_size"]
                    or check.get("audit_samples", 0) < (1 if protocol["arm"] in ("fixed", "token_exit") else 3)):
                failures.append(f"invalid_reuse_audit_at_{check['step']}")
            if gated and (check.get('policy_space') != 'joint_exit_token' or
                          not math.isfinite(check.get('gate_grad_no_grad_max_error',float('inf'))) or
                          check.get('gate_grad_no_grad_max_error',float('inf')) > .001):
                failures.append(f"invalid_gate_audit_at_{check['step']}")
    if any(not math.isfinite(v) for s in steps for v in s["metrics"].values()):
        failures.append("nonfinite_training_metrics")
    if not any(s["metrics"]["train/grad_norm"] > 0 for s in steps):
        failures.append("no_nonzero_gradient")
    if len(rollouts) != (finish-start)*protocol["prompts"]*protocol["group_size"]:
        failures.append("incorrect_sample_count")
    depths = sorted({r["train_metadata"]["ouro_depth"] for r in rollouts})
    if protocol["arm"] == "stratified" and depths != [2, 3, 4]:
        failures.append("missing_depth_coverage")
    if protocol["arm"] == "fixed" and depths != [4]:
        failures.append("fixed_arm_not_depth4")
    if gated:
        for r in rollouts:
            m=r['train_metadata'];trace=m.get('ouro_exits',[]);n=r['response_length']
            if (len(trace)!=n or any(e not in (1,2,3,4) for e in trace) or
                    m.get('executed_loop_tokens') != 4*(len(r['tokens'])-n)+sum(trace[1:])):
                failures.append('invalid_executed_exit_trace');break
        if len({e for r in rollouts for e in r['train_metadata'].get('ouro_exits',[])}) < 2:
            failures.append('no_actual_exit_variation')
    if any(r["reward"] not in (0., 1.) or r["reward"] != r["train_metadata"]["math_score"]["reward"] for r in rollouts):
        failures.append("invalid_math_reward")
    if start == 0 and not any(r["reward"] == 1 for r in rollouts):
        failures.append("no_successful_real_math_answer")
    for step in range(start, finish):
        versions = {v for r in rollouts if r["step"] == step for v in r["weight_versions"]}
        if len(versions) != 1:
            failures.append(f"inconsistent_engine_weight_versions_at_{step}")
    validation_ids = set(json.loads((a.run/"validation-ids.json").read_text()))
    if any(r["metadata"]["id"] in validation_ids for r in rollouts):
        failures.append("validation_data_in_training")
    if not failures:
        import torch
        from safetensors import safe_open
        from torch.distributed.checkpoint import FileSystemReader, load
        from megatron.core.dist_checkpointing import load_common_state_dict
        checkpoint = a.run/"checkpoints"/f"iter_{finish-1:07d}"
        reader = FileSystemReader(checkpoint)
        md = reader.read_metadata()
        names = ["decoder.final_layernorm.weight", "optimizer.state.exp_avg.decoder.final_layernorm.weight",
                 "optimizer.state.exp_avg_sq.decoder.final_layernorm.weight", "early_exit_gate.weight", "early_exit_gate.bias"]
        master_key=None
        if protocol["precision"]=="bf16":
            candidates=[f"optimizer.state.{kind}.decoder.final_layernorm.weight" for kind in ("param","fp32_param")]
            found=[key for key in candidates if key in md.state_dict_metadata]
            if len(found)!=1:
                raise ValueError(f"Expected one FP32 master parameter, found {found}")
            master_key=found[0]
            names.append(master_key)
        if protocol["arm"] == "learned":
            names += ["sequence_depth_head.weight", "sequence_depth_head.bias",
                      "optimizer.state.exp_avg.sequence_depth_head.weight"]
        gate_master=None
        if gated:
            names += [f'optimizer.state.{kind}.early_exit_gate.weight' for kind in ('exp_avg','exp_avg_sq')]
            if protocol['precision']=='bf16':
                found=[f'optimizer.state.{kind}.early_exit_gate.weight' for kind in ('param','fp32_param')
                       if f'optimizer.state.{kind}.early_exit_gate.weight' in md.state_dict_metadata]
                if len(found)!=1:raise ValueError('Missing unique FP32 gate master parameter')
                gate_master=found[0];names.append(gate_master)
        tensors = {k: torch.empty(md.state_dict_metadata[k].size,
            dtype=md.state_dict_metadata[k].properties.dtype) for k in names}
        load(tensors, storage_reader=reader)
        with safe_open(str(Path(protocol["checkpoint"])/"model.safetensors"), framework="pt") as base:
            delta = (tensors[names[0]]-base.get_tensor("model.norm.weight").float()).abs().max().item()
            master_delta=(tensors[master_key]-base.get_tensor("model.norm.weight").float()).abs().max().item() if protocol["precision"]=="bf16" else delta
            if master_delta == 0:
                failures.append("parameters_did_not_update")
            if gated:
                gate_delta=(tensors[gate_master or 'early_exit_gate.weight'].float()-base.get_tensor('model.early_exit_gate.weight').float()).abs().max().item()
                if not gate_delta:failures.append('token_exit_gate_did_not_update')
                if gate_master and tensors[gate_master].dtype!=torch.float32:failures.append('gate_master_not_fp32')
            else:
                for suffix in ("weight", "bias"):
                    if not torch.equal(tensors[f"early_exit_gate.{suffix}"], base.get_tensor(f"model.early_exit_gate.{suffix}").float()):
                        failures.append("frozen_token_gate_changed")
        moments = {k: tensors[k].abs().max().item() for k in names if k.startswith(("optimizer.state.exp_avg.","optimizer.state.exp_avg_sq."))}
        if protocol["precision"]=="bf16" and any(tensors[k].dtype!=torch.float32 for k in [master_key,*moments]):
            failures.append("bf16_master_or_optimizer_state_is_not_fp32")
        if protocol["precision"]=="bf16" and tensors[names[0]].dtype!=torch.bfloat16:
            failures.append("bf16_model_parameters_have_wrong_dtype")
        if not all(moments.values()):
            failures.append("optimizer_moments_are_zero")
        common = load_common_state_dict(checkpoint)
        adam_steps = {g["step"] for g in common["optimizer"]["optimizer"]["param_groups"]}
        if adam_steps != {finish} or common["opt_param_scheduler"]["num_steps"] != finish*protocol["prompts"]*protocol["group_size"]:
            failures.append("optimizer_or_scheduler_progress_mismatch")
        checkpoint_report = dict(path=str(checkpoint), final_norm_max_delta=delta,
            master_final_norm_max_delta=master_delta,
            model_dtype=str(tensors[names[0]].dtype),
            master_dtype=str(tensors[master_key].dtype) if master_key else str(tensors[names[0]].dtype),
            optimizer_moments=moments, adam_steps=sorted(adam_steps),
            scheduler_samples=common["opt_param_scheduler"]["num_steps"])
        if protocol["arm"] == "learned":
            changed = tensors["sequence_depth_head.weight"].abs().max().item()
            checkpoint_report["depth_head_weight_max"] = changed
            if not changed:
                failures.append("learned_depth_head_did_not_update")
        if gated:
            checkpoint_report['gate_master_max_delta']=gate_delta
            checkpoint_report['gate_model_dtype']=str(tensors['early_exit_gate.weight'].dtype)
    else:
        checkpoint_report = None
    report = dict(passed=not failures, failures=failures, protocol=protocol, run=str(a.run),
        steps=steps, rollout_metrics=metrics, consistency=checks, checkpoint=checkpoint_report,
        real_math_correct=sum(r["reward"] for r in rollouts), samples=len(rollouts),
        limitation="Engineering and training acceptance; no held-out improvement claim; strict resume checked separately")
    write_json(a.output, report)
    print(json.dumps(dict(passed=not failures, failures=failures, steps=len(steps), samples=len(rollouts))))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
