"""Check native smoke evidence, including actual checkpoint parameter changes."""

import argparse
import ast
import json
import math
from pathlib import Path
import re

from safetensors import safe_open
import torch
from torch.distributed.checkpoint import FileSystemReader, load
from megatron.core.dist_checkpointing import load_common_state_dict


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    text = re.sub(r"\x1b\[[0-9;]*m", "", args.log.read_text())
    steps = []
    for number, value in re.findall(r"model\.py:\d+ - step (\d+): (\{[^\n]*\})", text):
        metrics = ast.literal_eval(value)
        assert all(math.isfinite(v) for v in metrics.values())
        assert metrics["train/grad_norm"] > 0, "A zero-gradient smoke does not demonstrate an update"
        steps.append(dict(step=int(number), metrics=metrics))
    assert steps, "No completed native training steps"
    samples = []
    for path in sorted((args.run / "details/train_data").glob("*.pt")):
        batch = torch.load(path, map_location="cpu", weights_only=False)
        for item in batch["samples"]:
            error = (item["log_probs"] - item["rollout_log_probs"]).abs().max().item()
            assert error <= 1e-3
            samples.append(dict(rollout_id=batch["rollout_id"], sample_index=item["sample_index"],
                depth=item["metadata"]["ouro_depth"], response_length=item["response_lengths"],
                max_logprob_error=error))
    latest = int((args.run / "checkpoints/latest_checkpointed_iteration.txt").read_text())
    assert latest == steps[-1]["step"]
    checkpoint = args.run / "checkpoints" / f"iter_{latest:07d}"
    reader = FileSystemReader(checkpoint)
    metadata = reader.read_metadata()
    keys = ["decoder.final_layernorm.weight", "early_exit_gate.weight", "early_exit_gate.bias",
            "optimizer.state.exp_avg.decoder.final_layernorm.weight",
            "optimizer.state.exp_avg_sq.decoder.final_layernorm.weight"]
    tensors = {key: torch.empty(metadata.state_dict_metadata[key].size,
                                dtype=metadata.state_dict_metadata[key].properties.dtype) for key in keys}
    load(tensors, storage_reader=reader)
    reference = {}
    for shard in args.base_checkpoint.glob("*.safetensors"):
        with safe_open(shard, framework="pt", device="cpu") as weights:
            for key in ["model.norm.weight", "model.early_exit_gate.weight", "model.early_exit_gate.bias"]:
                if key in weights.keys():
                    reference[key] = weights.get_tensor(key).float()
    norm = tensors["decoder.final_layernorm.weight"]
    difference = (norm - reference["model.norm.weight"]).abs()
    assert difference.max() > 0, "Checkpoint parameters must differ from initial weights"
    for suffix in ["weight", "bias"]:
        torch.testing.assert_close(tensors[f"early_exit_gate.{suffix}"],
            reference[f"model.early_exit_gate.{suffix}"], atol=0, rtol=0)
    moments = {}
    for name in ["exp_avg", "exp_avg_sq"]:
        value = tensors[f"optimizer.state.{name}.decoder.final_layernorm.weight"]
        moments[name] = value.abs().max().item()
        assert moments[name] > 0
    report = dict(passed=True, diagnostic_reward=True, run=str(args.run), steps=steps,
        samples=samples, checkpoint=str(checkpoint), checkpoint_iteration=latest,
        final_norm_changed_elements=int(difference.count_nonzero()),
        final_norm_max_parameter_delta=difference.max().item(),
        optimizer_moment_maxima=moments, frozen_exit_gate_unchanged=True,
        limitation="Synthetic reward, fixed depth, TP=PP=CP=1; not evidence of algorithm benefit")
    common = load_common_state_dict(checkpoint)
    optimizer_steps = {g["step"] for g in common["optimizer"]["optimizer"]["param_groups"]}
    assert optimizer_steps == {latest + 1}, "Adam step must advance across checkpoint restore"
    report["optimizer_step"] = latest + 1
    report["scheduler_num_steps"] = common["opt_param_scheduler"]["num_steps"]
    launch = json.loads((args.run / "launch.json").read_text())["argv"]
    if "--ckpt-step" in launch:
        restored_step = int(launch[launch.index("--ckpt-step") + 1])
        restored = Path(launch[launch.index("--load") + 1]) / f"iter_{restored_step:07d}"
        assert steps[0]["step"] == restored_step + 1
        prior = load_common_state_dict(restored)
        assert prior["iteration"] == restored_step
        assert {g["step"] for g in prior["optimizer"]["optimizer"]["param_groups"]} == {restored_step + 1}
        assert report["scheduler_num_steps"] > prior["opt_param_scheduler"]["num_steps"]
        previous = {key: torch.empty_like(value) for key, value in tensors.items()}
        load(previous, storage_reader=FileSystemReader(restored))
        update = (norm - previous["decoder.final_layernorm.weight"]).abs()
        assert update.max() > 0, "Restored parameters must change during the resumed update"
        report["resume"] = dict(checkpoint=str(restored), iteration=restored_step,
            optimizer_step_before=restored_step + 1,
            final_norm_max_delta_after_resume=update.max().item(),
            limitation="Verifies restoration and continued updates, not bitwise equivalence to a continuous run")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
