"""Run native 2a acceptance stages or an explicitly selected formal sweep.

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
    def __init__(self, output, gpus, attention_implementation="reference", serving_upgrade_report=None,
                 gpu_wait_timeout=600, run_options=None, training_upgrade_report=None):
        self.output = output.resolve()
        self.output.mkdir(parents=True, exist_ok=True)
        self.gpus = gpus
        self.attention_implementation = attention_implementation
        self.serving_upgrade_report = serving_upgrade_report
        self.gpu_wait_timeout = gpu_wait_timeout
        self.run_options = run_options or {}
        self.training_upgrade_report = training_upgrade_report
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
            deadline = time.monotonic()+self.gpu_wait_timeout
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
        if self.attention_implementation != "reference":
            argv += ["--attention-implementation", self.attention_implementation]
        for key,value in self.run_options.items():
            if isinstance(value,bool):
                if value:argv += ["--"+key.replace("_","-")]
            else:
                argv += ["--"+key.replace("_","-"),str(value)]
        if resume is not None:
            argv += ["--resume", resume/"checkpoints", "--resume-step", boundary]
            if self.serving_upgrade_report:
                argv += ["--serving-upgrade-report", self.serving_upgrade_report]
            if self.training_upgrade_report:
                argv += ["--training-upgrade-report", self.training_upgrade_report]
        if stop is not None:
            argv += ["--stop-after", stop]
        count=self.run_options.get("trainer_gpus",1)+self.run_options.get("depth4_replicas",1)
        self.command(name, "run_native_2a.py", argv, self.gpus[:count] if arm == "fixed" else self.gpus)
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
    p.add_argument("--acceptance", type=Path)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--gpus", nargs="+", default=["0", "1", "2", "3"])
    p.add_argument("--through", choices=["gates", "formal-sweep"], default="gates")
    p.add_argument("--direct-formal", action="store_true",
                   help="Start full training directly; retain every in-training numerical check")
    p.add_argument("--arm-names", nargs="+",
                   help="Formal arms for this GPU queue, in the order fixed by the plan")
    p.add_argument("--attention-implementation", choices=["reference", "paged", "paged-fused", "bf16-fast"], default="reference")
    p.add_argument("--precision",choices=["fp32","bf16"],default="fp32")
    p.add_argument("--rollout-correction",choices=["none","tis"],default="none")
    p.add_argument("--tis-clip",type=float,default=2.)
    p.add_argument("--cuda-graph",action="store_true")
    p.add_argument("--max-running-requests",type=int,default=8)
    p.add_argument("--kv-token-capacity",type=int,default=16384)
    p.add_argument("--serving-memory-fraction",type=float,default=.55)
    p.add_argument("--depth4-replicas",type=int,default=1)
    p.add_argument("--serving-replicas",type=int,default=1)
    p.add_argument("--gate-shift",type=float,default=3.)
    p.add_argument("--trainer-gpus", type=int, default=1)
    p.add_argument("--micro-batch-size", type=int, default=1)
    p.add_argument("--reuse-old-log-probs", action="store_true")
    p.add_argument("--recompute", choices=["full", "selective", "selective-mlp", "none"], default="full")
    p.add_argument("--training-upgrade-report", type=Path)
    p.add_argument("--resume-first-run", type=Path)
    p.add_argument("--resume-boundary", type=int)
    p.add_argument("--serving-upgrade-report", type=Path)
    p.add_argument("--gpu-wait-timeout", type=int, default=600)
    a = p.parse_args()
    plan = json.loads((ROOT/"configs/native2a-plan.json").read_text())
    names = {arm["name"] for arm in plan["arms"]}
    if a.arm_names and (not a.direct_formal or not set(a.arm_names) <= names
                        or len(set(a.arm_names)) != len(a.arm_names)):
        p.error("arm-names requires direct-formal and distinct names from the plan")
    selected = [arm for arm in plan["arms"] if not a.arm_names or arm["name"] in a.arm_names]
    if any(arm['arm']=='token_exit' for arm in selected):
        if not a.direct_formal or not all(arm['arm']=='token_exit' for arm in selected):
            p.error('Token-exit arms require a dedicated direct-formal queue')
        required_gpus=a.trainer_gpus+a.serving_replicas
    else:
        required_gpus = (1 if a.direct_formal and all(arm["arm"] == "fixed" for arm in selected) else 3)+a.depth4_replicas-1+a.trainer_gpus
    if a.depth4_replicas < 1:
        p.error("depth4-replicas must be positive")
    if len(a.gpus) != required_gpus or len(set(a.gpus)) != required_gpus:
        p.error(f"{required_gpus} distinct GPU IDs are required")
    if not a.direct_formal and not a.acceptance:
        p.error("acceptance is required unless direct-formal is selected")
    if a.direct_formal and a.acceptance:
        p.error("direct-formal starts from the base checkpoint; do not specify acceptance")
    if a.resume_first_run and (not a.direct_formal or a.resume_boundary is None):
        p.error("resume-first-run requires direct-formal and resume-boundary")
    if (a.resume_boundary is not None or a.serving_upgrade_report or a.training_upgrade_report) and not a.resume_first_run:
        p.error("resume-boundary and serving-upgrade-report require resume-first-run")
    if a.resume_boundary is not None and a.resume_boundary < 0:
        p.error("resume-boundary must be nonnegative")
    if a.gpu_wait_timeout < 1:
        p.error("gpu-wait-timeout must be positive")
    run_options={key:getattr(a,key) for key in ('precision','rollout_correction','tis_clip','cuda_graph',
        'max_running_requests','kv_token_capacity','serving_memory_fraction','depth4_replicas',
        'trainer_gpus','micro_batch_size','reuse_old_log_probs','recompute','serving_replicas','gate_shift')}
    queue = Queue(a.output, a.gpus, a.attention_implementation, a.serving_upgrade_report,
                  a.gpu_wait_timeout,run_options,a.training_upgrade_report)
    try:
        if a.direct_formal:
            write_json(queue.output/"formal-plan.json", dict(plan=plan, selected_arms=selected,
                gpus=a.gpus, started_unix=time.time(),
                execution="Full formal runs; explicit checkpoint resume for the first arm when provided",
                attention_implementation=a.attention_implementation,
                run_options=run_options,
                resume_first_run=str(a.resume_first_run) if a.resume_first_run else None,
                resume_boundary=a.resume_boundary,
                retained_checks="Shared weight versions; explicit TIS and finite sampler/trainer probabilities (or strict sampler error <= 0.001 without TIS); independent grad/no-grad replay error <= 0.001; finite reference and loss; end-of-run checkpoint validation"))
            for index, arm in enumerate(selected):
                queue.train(arm["name"], arm=arm["arm"], beta=arm["beta"],
                    steps=plan["formal_steps"], prompts=plan["formal_prompts_per_step"],
                    group=plan["formal_group_size"], interval=5,
                    **(dict(resume=a.resume_first_run.resolve(), boundary=a.resume_boundary)
                       if index == 0 and a.resume_first_run else {}))
            queue.stage = "finished:formal-sweep"
            queue.status("completed", limitation="Training scan only; held-out selection, adaptive followups and paired evaluation remain separate stages.")
            return
        acceptance = a.acceptance.resolve()
        queue.wait_run(acceptance)
        queue.accept(acceptance)
        resumed = queue.train("acceptance-resume", resume=acceptance, boundary=7)
        queue.compare(acceptance, resumed, 9)
        learned = queue.train("learned-acceptance", arm="learned", steps=4)
        learned_resumed = queue.train("learned-resume", arm="learned", steps=4,
                                     resume=learned, boundary=1)
        queue.compare(learned, learned_resumed, 3)
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
