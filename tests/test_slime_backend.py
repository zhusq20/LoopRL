import importlib.util
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch

from looprl.consistency import error_statistics


def test_log_probability_error_reports_probability_ratio():
    metrics = error_statistics(torch.tensor([-.9, -2.2]), torch.tensor([-1., -2.]))
    assert metrics["max_abs_error"] == pytest.approx(.2)
    assert metrics["ratio_max_abs_deviation"] == pytest.approx(1 - torch.exp(torch.tensor(-.2)).item())


def test_nonfinite_probabilities_never_pass_consistency():
    with pytest.raises(ValueError, match="Non-finite"):
        error_statistics(torch.tensor([float("nan")]), torch.tensor([-1.]))


@pytest.fixture
def slime_path():
    path = Path(os.environ.get("LOOPRL_SLIME_PATH", "third_party/slime"))
    if not (path / "slime").exists():
        pytest.skip("Set LOOPRL_SLIME_PATH to the pinned, patched checkout for integration tests")
    sys.path.insert(0, str(path.resolve()))
    yield path
    sys.path.pop(0)


def test_actual_slime_batch_builder_preserves_depth_and_global_pricing(slime_path):
    from slime.data.batch_builder import BatchBuilder
    from slime.utils.types import Sample
    from looprl.slime_train import parser, native_args
    args = native_args(parser().parse_args(["--data", "unused", "--output", "unused"]))
    samples = []
    for i, (group, depth, reward) in enumerate([(3, 2, 1.), (8, 2, 0.), (3, 4, 1.), (8, 4, 0.)]):
        samples.append(Sample(index=i, group_index=group, tokens=[1, 2, 7, 8], response_length=2,
            reward=reward, rollout_log_probs=[-.1, -.2],
            metadata={"looprl": {"depths": [depth, depth], "beta": .1, "max_new_tokens": 2}},
            train_metadata={"depth": depth}))
    data = BatchBuilder(args).convert(samples)
    assert data["rewards"] == pytest.approx([.05, .05, -.05, -.05], abs=1e-6)
    assert data["raw_reward"] == [1., 0., 1., 0.]
    assert [m["depth"] for m in data["metadata"]] == [2, 2, 4, 4]
    assert data["rollout_log_probs"] == [[-.1, -.2]] * 4


def test_actual_native_train_loop_accepts_custom_actor(slime_path):
    spec = importlib.util.spec_from_file_location("test_native_slime_train", slime_path / "train.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    import inspect
    assert "actor_cls" in inspect.signature(module.train).parameters


def test_stale_sampler_weights_rejected_before_training():
    pytest.importorskip("ray")
    from looprl.slime_backend import OuroTrainRayActor
    actor = OuroTrainRayActor(1, 0, None, None)
    actor.version = 3
    with pytest.raises(RuntimeError, match="Stale"):
        actor.train(0, {"looprl": {"weight_version": 2}})


def test_parameter_probe_detects_update():
    pytest.importorskip("ray")
    from looprl.slime_backend import parameter_probe
    model = torch.nn.Linear(4, 3)
    before = parameter_probe(model)
    with torch.no_grad():
        model.weight[0, 0] += .1
    assert parameter_probe(model) != before


def test_resume_rejects_changed_schedule_and_dataset(tmp_path):
    pytest.importorskip("ray")
    from looprl.slime_backend import validate_resume
    from looprl.data import sha256
    data = tmp_path / "data.jsonl"
    data.write_text("{}\n")
    settings = dict(model="ouro", precision="fp32", arm="stratified", depths=[2, 4],
                    max_new_tokens=2048, seed=42, steps=140, beta=.1, price_warmup=.3,
                    prompts_per_step=32, group_size=8, limit=7498, offset=0,
                    learning_rate=1e-6, gate_learning_rate=1e-4, kl_coef=.001)
    state = {"args": settings, "completed_steps": 1, "weight_version": 1, "data_sha256": sha256(data)}
    args = SimpleNamespace(**settings, data=str(data), num_rollout=140)
    validate_resume(state, args)
    args.beta = .2
    with pytest.raises(ValueError, match="beta"):
        validate_resume(state, args)
    args.beta = .1
    args.offset = 1
    with pytest.raises(ValueError, match="offset"):
        validate_resume(state, args)
    args.offset = 0
    data.write_text('{"changed":true}\n')
    with pytest.raises(ValueError, match="data"):
        validate_resume(state, args)
