"""Paired native SGLang evaluation with saved per-batch evidence."""
import argparse
import json
import os
from pathlib import Path
import random
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from looprl.data import read_jsonl, sha256, write_json
from looprl.native_2a import append_record, sample_seed
from looprl.rewards import score_answer


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="/tmp/looprl-models/Ouro-2.6B-Thinking")
    p.add_argument("--data", default=str(ROOT/"data/math500.jsonl"))
    p.add_argument("--output", required=True)
    p.add_argument("--depth", type=int, choices=[2, 3, 4], default=4)
    p.add_argument("--mode", choices=["fixed", "learned", "token_exit"], default="fixed")
    p.add_argument("--gate-shift",type=float,default=3.)
    p.add_argument("--limit", type=int, default=32)
    p.add_argument("--repeats", type=int, default=2)
    p.add_argument("--max-new-tokens", type=int, default=8192)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--batched-decode", action="store_true")
    a = p.parse_args()
    gated=a.mode=='token_exit'
    if gated and a.depth!=4:p.error('Token-exit evaluation requires four loop slots')
    if min(a.limit, a.repeats, a.max_new_tokens, a.batch_size) < 1:
        p.error("Evaluation sizes and budgets must be positive")
    output = Path(a.output)
    output.mkdir(parents=True, exist_ok=False)
    from sglang import Engine
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(a.checkpoint, trust_remote_code=True)
    rows = read_jsonl(a.data)
    random.Random(a.seed).shuffle(rows)
    selected = rows[:a.limit]
    tasks = [(row, repeat) for row in selected for repeat in range(a.repeats)]
    protocol = dict(vars(a), data_sha256=sha256(a.data), ids=[r["id"] for r in selected],
                    dtype="float32", temperature=1., top_p=1., backend="sglang_native_ouro",
                    purpose="paired_baseline; do not select training hyperparameters on this test subset")
    write_json(output/"protocol.json", protocol)
    write_json(output/"status.json", dict(status="running", completed=0, planned=len(tasks), pid=os.getpid()))
    depths = [2, 3, 4] if a.mode == "learned" else [a.depth]
    if a.mode == "learned" and len(os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")) != 3:
        raise ValueError("Learned evaluation needs three free GPU IDs")
    engines = {}
    records = []
    started = time.monotonic()
    try:
        for device, depth in enumerate(depths):
            engines[depth] = Engine(model_path=a.checkpoint, trust_remote_code=True, dtype="float32",
                tp_size=1, pp_size=1, base_gpu_id=device, context_length=a.max_new_tokens+2048, max_total_tokens=65536,
                max_running_requests=a.batch_size, mem_fraction_static=.65, attention_backend="torch_native",
                disable_cuda_graph=True, disable_overlap_schedule=True, chunked_prefill_size=-1,
                json_model_override_args=json.dumps({"total_ut_steps": depth, "ouro_batched_decode": a.batched_decode,
                    "ouro_token_exit":gated,"ouro_gate_shift":a.gate_shift}),
                disable_radix_cache=gated,
                return_hidden_states_mode="full" if gated else "last" if a.mode == "learned" else None,
                random_seed=a.seed, log_level="warning")
        for offset in range(0, len(tasks), a.batch_size):
            batch = tasks[offset:offset+a.batch_size]
            texts = [tokenizer.apply_chat_template([dict(role="user", content=r["problem"]+
                "\nPlease put your final answer in \\boxed{}.")], tokenize=False,
                add_generation_prompt=True) for r, _ in batch]
            inputs = [tokenizer.encode(t, add_special_tokens=False) for t in texts]
            params = [dict(temperature=1., top_p=1., max_new_tokens=a.max_new_tokens,
                sampling_seed=sample_seed(a.seed, offset+i, repeat)) for i, (_, repeat) in enumerate(batch)]
            if gated:
                for item in params:item['custom_params']={'ouro_token_exit':True}
            selected_depths = [a.depth]*len(batch)
            logqs = [None]*len(batch)
            if a.mode == "learned":
                distributions = engines[4].generate(input_ids=inputs, return_hidden_states="last",
                    sampling_params=dict(temperature=0., max_new_tokens=1))
                import math
                for i, distribution in enumerate(distributions):
                    logq = distribution["meta_info"]["hidden_states"]
                    if len(logq) != 3:
                        raise ValueError("Checkpoint did not serve a learned three-action depth head")
                    if not all(math.isfinite(v) for v in logq) or abs(sum(math.exp(v) for v in logq)-1) > 1e-5:
                        raise ValueError("Invalid depth action distribution")
                    logqs[i] = logq
                    generator = random.Random(sample_seed(a.seed, offset+i, batch[i][1], "depth"))
                    selected_depths[i] = generator.choices([2, 3, 4], weights=[math.exp(v) for v in logq])[0]
            outputs = [None]*len(batch)
            for depth in depths:
                indices = [i for i,k in enumerate(selected_depths) if k == depth]
                if not indices:
                    continue
                generated = engines[depth].generate(input_ids=[inputs[i] for i in indices],
                    sampling_params=[params[i] for i in indices], return_logprob=True,
                    **({'return_hidden_states':True} if gated else {}))
                for i,result in zip(indices, generated, strict=True):
                    outputs[i] = result
            for (row, repeat), prompt, ids, result, depth, logq in zip(batch, texts, inputs, outputs, selected_depths, logqs, strict=True):
                response = result["text"]
                scoring = "<think>"+response if prompt.rfind("<think>") > prompt.rfind("</think>") else response
                score = score_answer(scoring, row["answer"])
                info = result["meta_info"]
                length = info["completion_tokens"]
                record = dict(id=row["id"], repeat=repeat, depth=depth, mode=a.mode, prompt_tokens=len(ids),
                    depth_log_probs=logq,
                    response=response, gold=row["answer"], score=score, response_tokens=length,
                    loop_tokens=depth*length, executed_loop_tokens=depth*(len(ids)+max(0, length-1))+
                        (4*len(ids) if a.mode == "learned" else 0),
                    truncated=info["finish_reason"]["type"] == "length",
                    meta_info=info)
                if gated:
                    from looprl.native_2b import parse_trace
                    exits,gates=parse_trace(info.pop('hidden_states'),length)
                    record.update(exits=exits,gate_log_probs=gates,depth=sum(exits)/length,
                        loop_tokens=sum(exits),executed_loop_tokens=4*len(ids)+sum(exits[1:]))
                records.append(record)
                append_record(output, "samples.jsonl", record)
            write_json(output/"status.json", dict(status="running", completed=len(records), planned=len(tasks),
                accuracy=sum(r["score"]["reward"] for r in records)/len(records),
                seconds=time.monotonic()-started))
            print(f"mode={a.mode}, depth={a.depth}: {len(records)}/{len(tasks)}", flush=True)
        n = len(records)
        write_json(output/"summary.json", dict(status="completed", samples=n, depth=a.depth, mode=a.mode,
            accuracy=sum(r["score"]["reward"] for r in records)/n,
            truncation_rate=sum(r["truncated"] for r in records)/n,
            mean_response_tokens=sum(r["response_tokens"] for r in records)/n,
            mean_loop_tokens=sum(r["loop_tokens"] for r in records)/n,
            mean_executed_loop_tokens=sum(r["executed_loop_tokens"] for r in records)/n,
            mean_depth=sum(r["depth"] for r in records)/n,
            seconds=time.monotonic()-started))
        write_json(output/"status.json", dict(status="completed", completed=n, planned=n))
    except BaseException as exc:
        write_json(output/"status.json", dict(status="failed", completed=len(records), error=repr(exc)))
        raise
    finally:
        for engine in engines.values():
            engine.shutdown()


if __name__ == "__main__":
    main()
