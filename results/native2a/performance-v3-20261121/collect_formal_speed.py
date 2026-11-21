import ast,json,re,statistics,time,math
from pathlib import Path
root=Path(__file__).resolve().parents[3]; out=root/'results/native2a/performance-v3-20261121'
def readjson(path):return json.loads(path.read_text())
def rows(path):return [json.loads(line) for line in path.open()] if path.exists() else []
def logs(path):
 text=re.sub(r'\x1b\[[0-9;]*m','',path.read_text())
 result={}
 for kind,pattern in [('timing',r'train_metric_utils.py:\d+ - perf (\d+): (\{[^\n]*\})'),('training',r'model.py:\d+ - step (\d+): (\{[^\n]*\})')]:
  result[kind]={int(n):ast.literal_eval(s) for n,s in re.findall(pattern,text)}
 return result
report=dict(updated_unix=time.time(),passed=False,runs_complete=False,scope='Completed full 256-response updates after checkpoint resume; not final experiment acceptance or bitwise resume equivalence.',queues={},limitations=['Before/after batches contain different questions and generated lengths.','Eight GPUs total before and after; stratified queue transfers one serving GPU to data-parallel training.','Reported perf/step_time excludes startup and checkpoint save overhead.'])
for arm,name,dp in [('stratified','stratified-beta0',2),('fixed','fixed4-beta0',1)]:
 old=root/'runs/native2a-bf16-formal-20261121'/arm/name;new=root/'runs/native2a-speed-formal-20261121'/arm/name
 baseline=logs(old.with_suffix('.log'));current=logs(new.with_suffix('.log'))
 steps=sorted(set(current['timing'])&set(current['training']))
 audits={r['step']:r for r in rows(new/'consistency.jsonl')};samples=rows(new/'rollouts.jsonl')
 old_samples=rows(old/'rollouts.jsonl')
 previous={r['metadata']['id'] for r in old_samples if r['step']<5}
 errors=[]
 if steps and steps!=list(range(5,steps[-1]+1)):errors.append('noncontiguous_updates')
 checks=[]
 for step in steps:
  a=audits.get(step,{});batch=[r for r in samples if r['step']==step];m=current['training'][step]
  versions=sorted({v for r in batch for v in r['weight_versions']})
  valid=(a.get('samples')==256 and a.get('data_parallel_size')==dp and a.get('reference_finite') and a.get('grad_no_grad_max_error',math.inf)<=.001 and a.get('old_policy_source')=='detached_training_forward' and len(batch)==256 and len(versions)==1 and all(math.isfinite(v) for v in m.values()) and m['train/grad_norm']>0)
  valid=bool(valid and sorted(r['index'] for r in batch)==list(range(step*256,(step+1)*256)) and not previous.intersection(r['metadata']['id'] for r in batch))
  if not valid:errors.append(f'update_{step}_failed_verification')
  checks.append(dict(step=step,passed=valid,index_min=min(r['index'] for r in batch),index_max=max(r['index'] for r in batch),prompt_groups=len({r['group'] for r in batch}),weight_versions=versions,response_tokens=sum(r['response_length'] for r in batch),consistency=a,metrics=m,timing=current['timing'][step]))
 base_steps=[n for n in baseline['timing'] if n>0]
 mean=lambda data,ids,key:statistics.mean(data[n][key] for n in ids) if ids else None
 keys=['perf/step_time','perf/train_time','perf/train_wait_time','perf/ref_log_probs_time','perf/actor_train_time']
 base={k:mean(baseline['timing'],base_steps,k) for k in keys}
 now={k:mean(current['timing'],steps,k) for k in keys}
 steady=[n for n in steps if n>5]
 state=readjson(new/'status.json') if (new/'status.json').exists() else {}
 def workload(sample_rows,selected,timing):
  selected_rows=[r for r in sample_rows if r['step'] in selected]
  seconds=sum(timing[n]['perf/step_time'] for n in selected)
  tokens=sum(r['response_length'] for r in selected_rows)
  loop_tokens=sum(r['response_length']*r['train_metadata']['ouro_depth'] for r in selected_rows)
  return dict(response_tokens=tokens,response_loop_tokens=loop_tokens,seconds=seconds,response_tokens_per_second=tokens/seconds if seconds else None,response_loop_tokens_per_second=loop_tokens/seconds if seconds else None)
 q=dict(baseline_workload=workload(old_samples,base_steps,baseline['timing']),current_workload=workload(samples,steps,current['timing']),run=str(new.relative_to(root)),status=state,verified_steps=steps,checks=checks,failures=errors,baseline_steps=base_steps,baseline_mean=base,current_mean=now,current_steady_mean={k:mean(current['timing'],steady,k) for k in keys},current_steady_steps=steady,mean_speedup=base['perf/step_time']/now['perf/step_time'] if steps else None)
 report['queues'][arm]=q
report['passed']=all(q['verified_steps'] and not q['failures'] for q in report['queues'].values())
(out/'formal-speed-verification.json').write_text(json.dumps(report,indent=2)+'\n')
for arm,q in report['queues'].items(): print(arm,json.dumps({k:q[k] for k in ['verified_steps','failures','baseline_mean','current_mean','mean_speedup']}))
