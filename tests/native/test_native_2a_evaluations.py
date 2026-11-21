import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("native_2a_evaluations", ROOT/"scripts/compare_native_2a_evaluations.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def fixture(root, name, responses, repeats=1):
    path = root/name
    path.mkdir()
    ids = [str(i) for i in range(len(responses))]
    protocol = dict(ids=ids, repeats=repeats, data_sha256="same", max_new_tokens=2048,
        seed=42, dtype="float32", temperature=1., top_p=1., backend="sglang_native_ouro",
        checkpoint=name, mode="fixed", depth=4)
    (path/"protocol.json").write_text(json.dumps(protocol))
    (path/"status.json").write_text(json.dumps(dict(status="completed")))
    with (path/"samples.jsonl").open("w") as f:
        for i, reward in zip(ids, responses, strict=True):
            for repeat in range(repeats):
                f.write(json.dumps(dict(id=i, repeat=repeat, score=dict(reward=reward),
                    response_tokens=10, executed_loop_tokens=100, truncated=False))+"\n")
    return path


def test_repeated_answers_do_not_artificially_shrink_question_uncertainty(tmp_path):
    a = fixture(tmp_path, "a", [0., 0., 0., 0.])
    b = fixture(tmp_path, "b", [0., 1., 0., 1.])
    c = fixture(tmp_path, "c", [0., 0., 0., 0.], repeats=16)
    d = fixture(tmp_path, "d", [0., 1., 0., 1.], repeats=16)
    one = module.compare(a, b)["metrics"]["accuracy"]
    sixteen = module.compare(c, d)["metrics"]["accuracy"]
    assert one == sixteen
    assert one["paired_delta"] == .5
    assert one["question_bootstrap_95_interval"] == [0., 1.]


def test_different_generation_budgets_are_rejected(tmp_path):
    a = fixture(tmp_path, "a", [0., 1.])
    b = fixture(tmp_path, "b", [0., 1.])
    path = b/"protocol.json"
    protocol = json.loads(path.read_text())
    protocol["max_new_tokens"] = 8192
    path.write_text(json.dumps(protocol))
    with pytest.raises(ValueError, match="max_new_tokens"):
        module.compare(a, b)
