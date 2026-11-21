"""Run the native 2a queue sequentially, stopping at any failed acceptance gate.

The queue is an ordinary foreground process; launch with nohup for durability.
It owns only its child processes and never resets a shared Ray cluster or GPU.
Existing failed runs are evidence, not silently retried or overwritten.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from looprl.data import write_json


class Queue:
    def __init__(self, output, gpus):
        self.output = output.resolve()
        self.output.mkdir(parents=True, exist_ok=True)
        self.gpus = gpus
        self.stage = "starting"

    def status(self, status, **fields):
        write_json(self.output/"status.json", dict(status=status, stage=self.stage,
            pid=os.getpid(), updated_unix=time.time(), **fields))

    def wait_run(self, run):
        self.stage = "waiting:"+run.name
        while True:
            state = json.loads((run/"status.json").read_text())
            if state["status"] == "completed":
                return
            if state["status"] == "failed":
                raise RuntimeError(f"Prerequisite failed: {run}: {state.get('error')}")
            self.status("running", waiting_on=str(run))
            time.sleep(30)

    def command(self, label, script, argv, gpus=None):
        self.stage = label
        marker = self.output/(label+".json")
        command = [sys.executable, str(ROOT/"scripts"/script), *map(str, argv)]
        if marker.exists():
            previous = json.loads(marker.read_text())
            if previous["argv"] != command or previous.get("returncode") != 0:
                raise RuntimeError(f"Stage already failed or changed: {marker}")
            return
        if gpus:
            # Ray teardown is asynchronous. Wait only for our selected devices;
            # never kill a process just to free them.
            deadline = time.monotonic()+600
            while True:
                inventory = subprocess.check_output(["nvidia-smi", "--query-gpu=index,memory.used",
                    "--format=csv,noheader,nounits"], text=True)
                used = dict(line.split(", ") for line in inventory.strip().splitlines())
                if all(int(used[g]) <= 1024 for g in gpus):
                    break
                self.status("running", waiting_for_gpus=gpus)
                if time.monotonic() > deadline:
                    raise RuntimeError(f"GPUs still occupied: {gpus}")
                time.sleep(15)
        env = dict(os.environ)
        if gpus:
            env["CUDA_VISIBLE_DEVICES"] = ",".join(gpus)
        output_argument = command[command.index("--output")+1] if "--output" in command else None
        log = Path(output_argument).with_suffix(".log") if script == "run_native_2a.py" else self.output/(label+".log")
        log.parent.mkdir(parents=True, exist_ok=True)
        record = dict(argv=command, gpus=gpus, log=str(log), started_unix=time.time())
        with log.open("w") as stream:
            child = subprocess.Popen(command, cwd=ROOT, env=env, stdout=stream,
                stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
            record["child_pid"] = child.pid
            write_json(marker, record)
            while child.poll() is None:
                self.status("running", child_pid=child.pid, log=str(log))
                time.sleep(15)
        record.update(returncode=child.returncode, finished_unix=time.time())
        write_json(marker, record)
        if child.returncode:
            raise RuntimeError(f"Stage failed with exit {child.returncode}: {label}; see {log}")

    def train(self, name, *, arm="stratified", beta=.1, steps=10, prompts=2,
              group=6, resume=None, boundary=None, stop=None, interval=2):
        run = self.output/name
        argv = ["--output", run, "--arm", arm, "--steps", steps, "--prompts", prompts,
                "--group-size", group, "--max-new-tokens", 2048, "--beta", beta,
                "--save-interval", interval, "--batched-decode"]
        if resume is not None:
            argv += ["--resume", resume/"checkpoints", "--resume-step", boundary]
        if stop is not None:
            argv += ["--stop-after", stop]
        self.command(name, "run_native_2a.py", argv, self.gpus[:2] if arm == "fixed" else self.gpus)
        self.accept(run)
        return run

    def accept(self, run):
        self.command("accept-"+run.name, "summarize_native_2a.py",
            ["--run", run, "--output", self.output/("accept-"+run.name+"-report.json")])

    def compare(self, control, resumed, step):
        self.command("compare-"+resumed.name, "compare_native_2a_resume.py",
            ["--control", control, "--resumed", resumed, "--step", step,
             "--output", self.output/("compare-"+resumed.name+"-report.json")])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--acceptance", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--gpus", nargs=4, default=["0", "1", "2", "3"])
    p.add_argument("--through", choices=["gates", "formal-sweep"], default="gates")
    a = p.parse_args()
    if len(set(a.gpus)) != 4:
        p.error("Four distinct GPU IDs are required")
    queue = Queue(a.output, a.gpus)
    try:
        acceptance = a.acceptance.resolve()
        queue.wait_run(acceptance)
        queue.accept(acceptance)
        resumed = queue.train("acceptance-resume", resume=acceptance, boundary=7)
        queue.compare(acceptance, resumed, 9)
        learned = queue.train("learned-acceptance", arm="learned", steps=4)
        learned_resumed = queue.train("learned-resume", arm="learned", steps=4,
                                     resume=learned, boundary=1)
        queue.compare(learned, learned_resumed, 3)
        plan = json.loads((ROOT/"configs/native2a-plan.json").read_text())
        pilots = {}
        for arm in plan["arms"][:2]:
            pilots[arm["name"]] = queue.train(arm["name"]+"-batch-check", arm=arm["arm"],
                beta=arm["beta"], steps=140, prompts=32, group=8, stop=2, interval=1)
        if a.through == "formal-sweep":
            for arm in plan["arms"]:
                prior = pilots.get(arm["name"])
                queue.train(arm["name"], arm=arm["arm"], beta=arm["beta"], steps=140,
                    prompts=32, group=8, resume=prior, boundary=1 if prior else None, interval=20)
        queue.stage = "finished:"+a.through
        queue.status("completed", limitation="No algorithmic benefit claim; held-out selection, adaptive followups and repeated paired evaluation remain separate stages.")
    except BaseException as exc:
        queue.status("failed", error=repr(exc))
        raise


if __name__ == "__main__":
    main()
