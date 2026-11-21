"""Read every serving gate and compare it with an exported actor checkpoint."""
import argparse
import json
from pathlib import Path
import time
import urllib.request

import torch
from safetensors import safe_open


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--base',type=Path,required=True)
    p.add_argument('--urls',nargs='+',required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--wait-seconds',type=float,default=90.)
    a=p.parse_args()
    with safe_open(str(a.checkpoint/'model.safetensors'),framework='pt') as f:
        expected={n:f.get_tensor('model.early_exit_gate.'+n).float() for n in ('weight','bias')}
    with safe_open(str(a.base/'model.safetensors'),framework='pt') as f:
        changed={n:int((v!=f.get_tensor('model.early_exit_gate.'+n).float()).sum()) for n,v in expected.items()}
    if not sum(changed.values()):raise ValueError('The probe requires an actually updated gate')
    deadline=time.monotonic()+a.wait_seconds
    while True:
        results=[]
        for url in a.urls:
            record=dict(url=url,tensors={},exact_gate_match=True)
            try:
                for name,value in expected.items():
                    request=urllib.request.Request(url.rstrip('/')+'/get_weights_by_name',
                        data=json.dumps(dict(name='model.early_exit_gate.'+name,truncate_size=2048)).encode(),
                        headers={'Content-Type':'application/json'})
                    with urllib.request.urlopen(request,timeout=15) as response:
                        actual=torch.tensor(json.load(response))
                    matched=torch.equal(actual,value)
                    record['exact_gate_match'] &= matched
                    record['tensors'][name]=dict(shape=list(actual.shape),exact_match=matched,
                        max_error=float((actual-value).abs().max()) if actual.shape==value.shape else None)
            except Exception as exc:
                record.update(exact_gate_match=False,error=repr(exc))
            results.append(record)
        passed=all(r['exact_gate_match'] for r in results)
        if passed or time.monotonic()>=deadline:break
        time.sleep(5)
    report=dict(passed=passed,checkpoint=str(a.checkpoint),changed_bf16_elements=changed,engines=results)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps(report),flush=True)
    if not passed:raise SystemExit(1)


if __name__=='__main__':main()
