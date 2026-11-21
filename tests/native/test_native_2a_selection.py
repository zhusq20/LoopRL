import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("select_native_2a_beta", ROOT/"scripts/select_native_2a_beta.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


@pytest.fixture
def evaluation(tmp_path):
    validation = tmp_path/"validation.jsonl"
    validation.write_text(json.dumps(dict(id="validation/example"))+"\n")
    plan = dict(formal_steps=140, arms=[dict(name=f"beta{beta}", arm="stratified", beta=beta) for beta in (.02, .05, .1)],
        validation=dict(selection_reference_cost_per_loop_token=.01,
            evaluation=dict(repeats=1, max_new_tokens=2048, seed=42, dtype="float32",
                temperature=1., top_p=1., backend="sglang_native_ouro")))
    for arm in plan["arms"]:
        exported = tmp_path/(arm["name"]+"-model")
        exported.mkdir()
        (exported/"export.json").write_text(json.dumps(dict(source="checkpoints/iter_0000139",
            protocol=dict(arm="stratified", beta=arm["beta"], total_steps=140))))
        for depth in (2, 3, 4):
            dest = tmp_path/arm["name"]/f"depth{depth}"
            dest.mkdir(parents=True)
            protocol = dict(data_sha256=module.sha256(validation), checkpoint=str(exported), repeats=1,
                max_new_tokens=2048, seed=42, dtype="float32", temperature=1., top_p=1.,
                backend="sglang_native_ouro", mode="fixed", depth=depth)
            (dest/"protocol.json").write_text(json.dumps(protocol))
            (dest/"status.json").write_text(json.dumps(dict(status="completed")))
            row = dict(id="validation/example", repeat=0, depth=depth, score=dict(reward=1.),
                loop_tokens=200 if arm["beta"] == .02 else 100)
            (dest/"samples.jsonl").write_text(json.dumps(row)+"\n")
    return tmp_path, validation, plan


def test_predeclared_cost_and_lower_price_tiebreak(evaluation):
    root, validation, plan = evaluation
    report = module.select(root, validation, plan)
    assert report["selected_beta"] == .05
    assert report["same_price_fixed4_betas"] == [.05, .1]
    assert [r["utility"] for r in report["results"]] == [-1., 0., 0.]


@pytest.mark.parametrize("fault", ["wrong_data", "missing_question", "duplicate", "wrong_checkpoint"])
def test_rejects_unpaired_or_mislabeled_evaluation(evaluation, fault):
    root, validation, plan = evaluation
    target = root/"beta0.05/depth3"
    if fault == "wrong_data":
        path = target/"protocol.json"
        protocol = json.loads(path.read_text())
        protocol["data_sha256"] = "test-set-hash"
        path.write_text(json.dumps(protocol))
    elif fault == "wrong_checkpoint":
        path = root/"beta0.05-model/export.json"
        export = json.loads(path.read_text())
        export["protocol"]["beta"] = .2
        path.write_text(json.dumps(export))
    else:
        path = target/"samples.jsonl"
        path.write_text("" if fault == "missing_question" else path.read_text()*2)
    with pytest.raises(ValueError):
        module.select(root, validation, plan)
