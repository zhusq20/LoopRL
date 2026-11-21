"""Switch one owned training queue only after a complete checkpoint is durable.

Never edits a running snapshot. The old run and its outputs are retained, and
the replacement gets a new output directory and an explicit serving upgrade.
"""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import psutil

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from looprl.data import sha256, write_json


def checked_process(pid, output, script):
    process=psutil.Process(pid)
    argv=process.cmdline()
    if not any(Path(x).name==script for x in argv) or '--output' not in argv:
        raise RuntimeError(f'PID {pid} is not the expected training process')
    if Path(argv[argv.index('--output')+1]).resolve()!=output.resolve():
        raise RuntimeError(f'PID {pid} belongs to a different output directory')
    return process


def checkpoint_signature(run, step):
    root=run/'checkpoints'
    model=root/f'iter_{step:07d}'
    shards=list(model.glob('*.distcp'))
    paths=[model/'.metadata',root/'rollout'/f'global_dataset_state_dict_{step}.pt',*shards]
    tracker=root/'latest_checkpointed_iteration.txt'
    if not shards or not tracker.is_file() or not all(p.is_file() and p.stat().st_size for p in paths):
        return None
    try:
        if int(tracker.read_text().strip()) < step:
            return None
    except ValueError:
        return None
    return {str(p.relative_to(root)):(p.stat().st_size,p.stat().st_mtime_ns) for p in paths}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-queue',type=Path,required=True)
    p.add_argument('--source-run',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--boundary',type=int,default=4)
    p.add_argument('--validation',type=Path,required=True)
    p.add_argument('--gpus',nargs='+',required=True)
    p.add_argument('--arm-names',nargs='+',required=True)
    args=p.parse_args()
    args.output=args.output.resolve()
    args.output.mkdir(parents=True,exist_ok=False)
    source=args.source_run.resolve();queue=args.source_queue.resolve()
    status_path=args.output/'upgrade-status.json'
    def status(state,**fields):
        write_json(status_path,dict(status=state,pid=os.getpid(),source_run=str(source),
            boundary=args.boundary,updated_unix=time.time(),**fields))
    try:
        validation=json.loads(args.validation.read_text())
        if validation.get('passed') is not True or validation.get('attention_implementation')!='paged-fused':
            raise ValueError('Serving upgrade validation did not pass')
        for name,digest in validation['source_sha256'].items():
            if sha256(ROOT/name)!=digest:
                raise ValueError('Validated source changed before scheduling upgrade')
        old_queue=json.loads((queue/'status.json').read_text())
        parent=checked_process(old_queue['pid'],queue,'advance_native_2a.py')
        driver=checked_process(old_queue['child_pid'],source,'run_native_2a.py')
        write_json(args.output/'upgrade-plan.json',dict(source_queue=str(queue),source_run=str(source),
            queue_pid=parent.pid,driver_pid=driver.pid,queue_created=parent.create_time(),
            driver_created=driver.create_time(),boundary=args.boundary,gpus=args.gpus,
            arm_names=args.arm_names,validation=str(args.validation.resolve()),
            validation_sha256=sha256(args.validation),started_unix=time.time()))
        while not checkpoint_signature(source,args.boundary):
            state=json.loads((source/'status.json').read_text())
            if state['status'] in ('failed','completed') or not driver.is_running():
                raise RuntimeError('Source stopped before the requested durable boundary; no process was terminated')
            status('waiting_for_checkpoint')
            time.sleep(15)
        # The rollout state is written last by slime's checkpoint procedure.
        # Confirm the payload sizes are stable before stopping the producer.
        signature=checkpoint_signature(source,args.boundary)
        time.sleep(5)
        if checkpoint_signature(source,args.boundary)!=signature:
            raise RuntimeError('Checkpoint changed during readiness confirmation')
        protocol=json.loads((source/'protocol.json').read_text())
        preflight=[sys.executable,str(ROOT/'scripts/run_native_2a.py'),
            '--output',str(args.output/'resume-preflight'),'--dry-run',
            '--arm',protocol['arm'],'--beta',str(protocol['beta']),
            '--steps',str(protocol['total_steps']),'--prompts',str(protocol['prompts']),
            '--group-size',str(protocol['group_size']),'--max-new-tokens',str(protocol['max_new_tokens']),
            '--batched-decode','--attention-implementation','paged-fused',
            '--resume',str(source/'checkpoints'),'--resume-step',str(args.boundary),
            '--serving-upgrade-report',str(args.validation.resolve())]
        # Recheck the complete protocol and current source before stopping a
        # productive run. A stale validation report leaves the old run alive.
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=','.join(args.gpus))
        with (args.output/'resume-preflight.log').open('w') as log:
            subprocess.run(preflight,cwd=ROOT,env=env,stdout=log,
                stderr=subprocess.STDOUT,check=True,timeout=120)
        checked_process(parent.pid,queue,'advance_native_2a.py')
        checked_process(driver.pid,source,'run_native_2a.py')
        descendants=driver.children(recursive=True)
        status('stopping_source_after_checkpoint')
        parent.send_signal(signal.SIGTERM)
        driver.send_signal(signal.SIGINT)
        _,alive=psutil.wait_procs([driver]+descendants,timeout=45)
        for process in alive:
            # psutil checks PID identity/create-time before sending a signal.
            try:process.terminate()
            except psutil.NoSuchProcess:pass
        _,alive=psutil.wait_procs(alive,timeout=10)
        if alive:
            raise RuntimeError(f'Owned processes did not stop cleanly: {[p.pid for p in alive]}')
        command=[sys.executable,str(ROOT/'scripts/advance_native_2a.py'),
            '--direct-formal','--output',str(args.output/'training'),
            '--gpus',*args.gpus,'--arm-names',*args.arm_names,
            '--attention-implementation','paged-fused','--resume-first-run',str(source),
            '--resume-boundary',str(args.boundary),'--serving-upgrade-report',str(args.validation.resolve()),
            '--gpu-wait-timeout','3600']
        with (args.output/'training-queue.log').open('w') as log:
            child=subprocess.Popen(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,start_new_session=True)
        status('replacement_launched',replacement_pid=child.pid,argv=command)
    except BaseException as exc:
        status('failed',error=repr(exc))
        raise


if __name__=='__main__':
    main()
