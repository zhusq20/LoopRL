"""Exercise real SGLang prefill, decode, batching and prefix reuse for Ouro.

The JSON contains token IDs/log probabilities for independent HF or Megatron
teacher forcing; successful generation alone is not a consistency check.
"""

import argparse
import json
import time
from pathlib import Path

from sglang import Engine
from transformers import AutoTokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--depth", type=int, choices=[2, 3, 4], required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--batched-decode", action="store_true")
    parser.add_argument("--dtype", choices=["float32", "bfloat16"], default="float32")
    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, trust_remote_code=True)
    prompts = ["What is 2 + 3?", "What is 2 + 3? Explain briefly."]
    token_ids = [tokenizer.apply_chat_template([dict(role="user", content=text)],
                tokenize=True, add_generation_prompt=True, return_dict=False) for text in prompts]
    engine = Engine(model_path=args.checkpoint, trust_remote_code=True, dtype=args.dtype,
        tp_size=1, pp_size=1, context_length=4096, max_total_tokens=8192,
        max_running_requests=4, mem_fraction_static=0.5, attention_backend="torch_native",
        disable_cuda_graph=True, disable_overlap_schedule=True, chunked_prefill_size=-1,
        json_model_override_args=json.dumps({"total_ut_steps": args.depth, "ouro_batched_decode": args.batched_decode}),
        random_seed=42, log_level="warning")
    try:
        sampling = dict(temperature=0, max_new_tokens=args.max_new_tokens, ignore_eos=True)
        started = time.monotonic()
        outputs = engine.generate(input_ids=token_ids, sampling_params=sampling,
                                   return_logprob=True, logprob_start_len=0)
        repeated = engine.generate(input_ids=token_ids[0], sampling_params=sampling,
                                    return_logprob=True, logprob_start_len=len(token_ids[0]) - 1)
        assert repeated["meta_info"]["cached_tokens"] > 0, "Repeated request must exercise prefix reuse"
        report = dict(depth=args.depth, prompts=prompts, input_ids=token_ids,
                      batched_decode=args.batched_decode, dtype=args.dtype, generation_seconds=time.monotonic()-started,
                      batch_outputs=outputs, repeated_output=repeated)
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2) + "\n")
        print(f"Saved real SGLang output to {output}", flush=True)
    finally:
        engine.shutdown()


if __name__ == "__main__":
    main()
