"""Observe a launcher-owned experiment and persist partial/final summaries."""
import argparse
import json
from pathlib import Path
import time

from .data import read_jsonl, write_json
from .experiments import summarize


def process_matches(worker):
    try:
        actual = Path(f"/proc/{worker['pid']}/cmdline").read_bytes().split(b"\0")
    except OSError:
        return False
    return [a.decode() for a in actual if a] == worker["argv"]


def snapshot(launch):
    workers, records = [], []
    for worker in launch["workers"]:
        path = Path(worker["output"])
        sample_path = path / "samples.jsonl"
        rows = []
        if sample_path.exists():
            # A concurrently appended final line can be incomplete. The next
            # snapshot will read it after the writer has finished the record.
            for line in sample_path.read_text().splitlines():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    break
        state = json.loads((path / "status.json").read_text()) if (path / "status.json").exists() else {}
        alive = process_matches(worker)
        status = state.get("status")
        if status != "completed":
            status = "running" if alive else "failed_or_stopped"
        workers.append({"depth": worker["depth"], "pid": worker["pid"], "gpu": worker["gpu"],
                        "status": status, "completed_samples": len(rows), "log": worker["log"]})
        records.extend(rows)
    done = all(w["status"] != "running" for w in workers)
    success = done and all(w["status"] == "completed" for w in workers)
    report = {"status": "completed" if success else "failed_or_stopped" if done else "running",
              "updated_unix": time.time(), "planned_samples": launch["planned_samples"],
              "completed_samples": len(records), "workers": workers,
              "warning": "Partial summaries may contain different question sets at different depths; compare only paired records.",
              "metrics": summarize(records) if records else {}}
    return done, report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--launch", required=True)
    parser.add_argument("--watch", action="store_true")
    args = parser.parse_args()
    path = Path(args.launch)
    launch = json.loads(path.read_text())
    while True:
        done, report = snapshot(launch)
        write_json(path.parent / "progress.json", report)
        if done:
            write_json(path.parent / "summary.json", report)
        print(json.dumps({"status": report["status"], "completed_samples": report["completed_samples"],
                          "planned_samples": report["planned_samples"]}), flush=True)
        if done or not args.watch:
            return
        time.sleep(30)


if __name__ == "__main__":
    main()
