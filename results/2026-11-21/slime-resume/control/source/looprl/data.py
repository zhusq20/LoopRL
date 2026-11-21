"""Pinned MATH train / MATH-500 test inputs with local SHA256 manifests."""
import hashlib
import json
from pathlib import Path

from .rewards import last_boxed

MATH500_REV = "6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be"
MATH_TRAIN_REV = "21a5633873b6a120296cce3e2df9d5550074f4a3"
SUBJECTS = ("algebra", "counting_and_probability", "geometry", "intermediate_algebra",
            "number_theory", "prealgebra", "precalculus")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def prepare(destination):
    import requests
    import pandas as pd
    from io import BytesIO
    dest = Path(destination)
    dest.mkdir(parents=True, exist_ok=True)
    url = f"https://huggingface.co/datasets/HuggingFaceH4/MATH-500/resolve/{MATH500_REV}/test.jsonl"
    response = requests.get(url, timeout=120)
    response.raise_for_status()
    rows = [json.loads(line) for line in response.text.splitlines() if line.strip()]
    evaluation = [{"id": str(r.get("unique_id", i)), "problem": r["problem"], "answer": r["answer"],
                   "source": "MATH-500", "split": "test", "subject": r.get("subject", ""),
                   "level": r.get("level")} for i, r in enumerate(rows)]
    train, sources, skipped = [], [{"url": url, "sha256": hashlib.sha256(response.content).hexdigest()}], []
    for subject in SUBJECTS:
        url = f"https://huggingface.co/datasets/EleutherAI/hendrycks_math/resolve/{MATH_TRAIN_REV}/{subject}/train-00000-of-00001.parquet"
        response = requests.get(url, timeout=120)
        response.raise_for_status()
        sources.append({"url": url, "sha256": hashlib.sha256(response.content).hexdigest()})
        for i, row in enumerate(pd.read_parquet(BytesIO(response.content)).to_dict("records")):
            answer = last_boxed(row["solution"])
            if answer is None:
                skipped.append({"id": f"train/{subject}/{i}", "reason": "no_boxed_gold_answer"})
                continue
            train.append({"id": f"train/{subject}/{i}", "problem": row["problem"], "answer": answer,
                          "source": "MATH", "split": "train", "subject": subject, "level": row.get("level")})
    eval_problems = {r["problem"].strip() for r in evaluation}
    if any(r["problem"].strip() in eval_problems for r in train):
        raise ValueError("Exact problem overlap between training and evaluation inputs")
    for name, items in [("math_train.jsonl", train), ("math500.jsonl", evaluation)]:
        with (dest / name).open("w") as f:
            for item in items:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
    manifest = {"sources": sources, "train_count": len(train), "test_count": len(evaluation), "skipped": skipped,
                "files": {name: sha256(dest / name) for name in ("math_train.jsonl", "math500.jsonl")}}
    write_json(dest / "manifest.json", manifest)
    return manifest
