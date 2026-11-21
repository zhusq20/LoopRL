import json, time, gc, torch, sys
from pathlib import Path
sys.path.insert(0,'/ossfs/workspace/LoopRL')
from looprl.ouro import OuroAdapter, Rollout
from torch.nn import functional as F
from torch.nn.attention import sdpa_kernel, SDPBackend

torch.set_num_threads(4)
torch.backends.cuda.matmul.allow_tf32=False
torch.backends.cudnn.allow_tf32=False
row=json.loads(Path('runs/stage2a-smoke-20261121-v2/rollouts.jsonl').read_text().splitlines()[2])
r=Rollout(**{k:row[k] for k in Rollout.__dataclass_fields__})
for mode in ['bf16','bf16_head_fp32','bf16_math','fp32']:
 a=OuroAdapter.load('/tmp/looprl-models/Ouro-2.6B-Thinking','cuda',dtype=torch.float32 if mode=='fp32' else torch.bfloat16)
 if mode=='bf16_head_fp32':
  head=a.model.lm_head
  head.forward=lambda x:F.linear(x.float(),head.weight.float())
 started=time.monotonic()
 with torch.no_grad():
  if mode=='bf16_math':
   with sdpa_kernel(SDPBackend.MATH):
    cached=a.response_log_probs(r,replay_mode='cached');full=a.response_log_probs(r)
  else:
   cached=a.response_log_probs(r,replay_mode='cached');full=a.response_log_probs(r)
 err=(cached-full).abs()
 out={'mode':mode,'max_abs':float(err.max()),'mean_abs':float(err.mean()),'rms':float(err.square().mean().sqrt()),'seconds':time.monotonic()-started,'tokens':len(r.response_ids),'depth':r.depths[0]}
 print(json.dumps(out),flush=True)
 with open('runs/long-consistency/precision-grid.jsonl','a') as f:f.write(json.dumps(out)+'\n')
 del a,cached,full,err
 gc.collect();torch.cuda.empty_cache()
