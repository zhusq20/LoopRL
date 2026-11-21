"""Release one owned queue at a durable checkpoint for controlled validation."""
import argparse
import json
from pathlib import Path
import signal
import sys
import time

import psutil

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from looprl.data import write_json
from upgrade_native_2a_queue import checked_process, checkpoint_signature


def wait_owned(processes, timeout):
    # This container's kernel rejects pidfd_open for some Ray children.
    # Poll psutil's PID/create-time identity checks instead of wait_procs.
    deadline = time.monotonic() + timeout
    alive = list(processes)
    while alive:
        remaining = []
        for process in alive:
            try:
                if process.is_running() and process.status() != psutil.STATUS_ZOMBIE:
                    remaining.append(process)
            except psutil.NoSuchProcess:
                pass
        alive = remaining
        if not alive or time.monotonic() >= deadline:
            break
        time.sleep(.25)
    return alive


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--queue", type=Path, required=True)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--boundary", type=int, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    record = dict(source_queue=str(a.queue.resolve()), source_run=str(a.run.resolve()),
                  boundary=a.boundary, started_unix=time.time())
    state = json.loads((a.queue/"status.json").read_text())
    parent = checked_process(state["pid"], a.queue, "advance_native_2a.py")
    driver = checked_process(state["child_pid"], a.run, "run_native_2a.py")
    record.update(queue_pid=parent.pid, driver_pid=driver.pid,
                  queue_argv=parent.cmdline(), driver_argv=driver.cmdline())
    while not checkpoint_signature(a.run, a.boundary):
        if not driver.is_running() or json.loads((a.run/"status.json").read_text())["status"] != "running":
            raise RuntimeError("Run stopped before the requested checkpoint; no processes stopped")
        write_json(a.output, dict(record, status="waiting_for_checkpoint", updated_unix=time.time()))
        time.sleep(10)
    signature = checkpoint_signature(a.run, a.boundary)
    time.sleep(5)
    if checkpoint_signature(a.run, a.boundary) != signature:
        raise RuntimeError("Checkpoint still changing; source left running")
    checked_process(parent.pid, a.queue, "advance_native_2a.py")
    checked_process(driver.pid, a.run, "run_native_2a.py")
    descendants = driver.children(recursive=True)
    record["owned_processes"] = [dict(pid=p.pid, created=p.create_time(), argv=p.cmdline()) for p in descendants]
    record["checkpoint_signature"] = signature
    write_json(a.output, dict(record, status="stopping_after_checkpoint", updated_unix=time.time()))
    parent.terminate()
    driver.send_signal(signal.SIGINT)
    alive = wait_owned([driver] + descendants, timeout=40)
    for process in alive:
        try:
            process.terminate()
        except psutil.NoSuchProcess:
            pass
    alive = wait_owned(alive, timeout=10)
    for process in alive:
        try:
            process.kill()
        except psutil.NoSuchProcess:
            pass
    alive = wait_owned(alive, timeout=5)
    active = [p.pid for p in alive if p.status() != psutil.STATUS_ZOMBIE]
    if active:
        raise RuntimeError(f"Owned processes still alive: {active}")
    final = dict(record, status="stopped_for_speed_validation", updated_unix=time.time())
    write_json(a.output, final)
    for directory in (a.queue, a.run):
        state = json.loads((directory/"status.json").read_text())
        write_json(directory/"status.json", dict(state, status="stopped_for_speed_validation",
            resume_boundary=a.boundary, stop_record=str(a.output.resolve()), updated_unix=time.time()))


if __name__ == "__main__":
    main()
