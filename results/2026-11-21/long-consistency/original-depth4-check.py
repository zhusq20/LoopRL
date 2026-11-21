import sys,json,time,torch,gc
from pathlib import Path
sys.path.insert(0,'/ossfs/workspace/LoopRL')
from looprl.ouro import OuroAdapter,Rollout
from looprl.data import write_json

torch.set_num_threads(4)
torch.backends.cuda.matmul.allow_tf32=False
torch.backends.cudnn.allow_tf32=False
source='runs/stage0-pilot-20261121/depth4/samples.jsonl'
row=json.loads(Path(source).read_text().splitlines()[0])
a=OuroAdapter.load('/tmp/looprl-models/Ouro-2.6B-Thinking','cuda',dtype=torch.float32)
records=[]
for count in [512,2048]:
 r=Rollout(prompt_ids=row['prompt_ids'],response_ids=row['response_ids'][:count],log_probs=[],depths=[4]*count,text='',terminated=False,seconds=0.,prefill_loop_tokens=4*len(row['prompt_ids']))
 start=time.monotonic()
 with torch.no_grad():
  cached=a.response_log_probs(r,replay_mode='cached')
  full=a.response_log_probs(r,replay_mode='full')
 err=(full-cached).abs()
 record={'tokens':count,'prompt_tokens':len(r.prompt_ids),'depth':4,'precision':'fp32','max_abs_error':float(err.max()),'mean_abs_error':float(err.mean()),'rms_error':float(err.square().mean().sqrt()),'ratio_max_abs_deviation':float(((full-cached).exp()-1).abs().max()),'forward_seconds':time.monotonic()-start}
 assert record['max_abs_error']<0.001,record
 a.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
 a.model.train()
 torch.cuda.reset_peak_memory_stats()
 start=time.monotonic()
 lp=a.response_log_probs(r,replay_mode='full')
 graderr=float((lp.detach()-full).abs().max())
 loss=-lp.mean()
 loss.backward()
 norm=float(torch.nn.utils.clip_grad_norm_(a.model.parameters(),.1))
 assert graderr<.001 and norm>0 and torch.isfinite(torch.tensor(norm))
 record.update(grad_forward_max_abs_error=graderr,grad_norm_before_clip=norm,backward_seconds=time.monotonic()-start,peak_memory_gib=torch.cuda.max_memory_allocated()/1024**3,passed=True)
 records.append(record)
 print(json.dumps(record),flush=True)
 write_json('runs/long-consistency/fp32-long.json',{'source':source,'records':records,'semantics':'Fixed recorded text replay, no task reward or accuracy conclusion.'})
 a.model.zero_grad(set_to_none=True);a.model.eval();del lp,loss,cached,full;gc.collect();torch.cuda.empty_cache()
