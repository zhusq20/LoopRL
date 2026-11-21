"""Replay real long responses with native Megatron FP32 gradient accumulation.

Supports torchrun DP and depth-homogeneous packing. Measures the reference,
old-policy/audit, forward/backward and gradient synchronization separately.
Does not change model weights or run concurrent sampling.
"""
import argparse
import json
import os
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace

import torch
import torch.distributed as dist
from megatron.core import parallel_state
from megatron.core.distributed import DistributedDataParallel, DistributedDataParallelConfig
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from slime.backends.megatron_utils.data import DataIterator, get_batch
from slime.backends.megatron_utils.loss import get_log_probs_and_entropy
from slime_plugins.ouro_2a import policy_loss
from slime_plugins.ouro_training import begin_audit, finish_reuse_audit, homogeneous_schedule
from check_native_rollout_logprobs import load_model


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", default="/tmp/looprl-models/Ouro-2.6B-Thinking")
    p.add_argument("--rollouts", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--micro-batch-size", type=int, default=1)
    p.add_argument("--recompute", choices=["full", "selective", "selective-mlp", "none"], default="full")
    p.add_argument("--reuse", action="store_true")
    p.add_argument("--per-depth", type=int, default=4)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--memory-fraction", type=float, default=.8)
    a = p.parse_args()
    rank, world, local = (int(os.environ.get(k, d)) for k, d in [("RANK", "0"), ("WORLD_SIZE", "1"), ("LOCAL_RANK", "0")])
    torch.set_num_threads(4)
    torch.cuda.set_device(local)
    torch.cuda.set_per_process_memory_fraction(a.memory_fraction)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    a.output.parent.mkdir(parents=True, exist_ok=True)
    report = dict(world_size=world, micro_batch_size=a.micro_batch_size,
                  recompute=a.recompute, reuse=a.reuse, runs=[])
    with tempfile.TemporaryDirectory(prefix="ouro-update-bench-") as temp:
        dist.init_process_group("nccl", **({} if world > 1 else
            dict(init_method=f"file://{temp}/init", rank=0, world_size=1)))
        parallel_state.initialize_model_parallel(tensor_model_parallel_size=1, pipeline_model_parallel_size=1)
        model_parallel_cuda_manual_seed(42)
        try:
            all_rows = [json.loads(line) for line in a.rollouts.read_text().splitlines()]
            step = min(r["step"] for r in all_rows)
            rows = []
            for depth in (2, 3, 4):
                choices = [r for r in all_rows if r["step"] == step and r["train_metadata"]["ouro_depth"] == depth]
                choices.sort(key=lambda r: (abs(r["train_metadata"].get("sequence_advantage", 0)) == 0, -len(r["tokens"])))
                rows.extend(choices[:a.per_depth])
            n = len(rows)
            if n == 0 or n % world:
                raise ValueError("Need a nonempty replay batch divisible by DP size")
            selected = list(range(rank, n, world))
            rows = [rows[i] for i in selected]
            model = load_model(a.checkpoint, a.recompute, torch.bfloat16)
            wrapped = DistributedDataParallel(model.config, DistributedDataParallelConfig(
                grad_reduce_in_fp32=True, overlap_grad_reduce=False, use_distributed_optimizer=False), model)
            args = SimpleNamespace(num_steps_per_rollout=1, use_rollout_logprobs=False, kl_coef=0.,
                loss_type="custom_loss", custom_loss_function_path="looprl.native_2a.policy_loss",
                advantage_estimator="grpo", use_tis=True, ouro_rollout_correction="tis", tis_clip=2., tis_clip_low=0.,
                calculate_per_token_loss=False, context_parallel_size=1, rollout_temperature=1., rollout_top_p=1.,
                eps_clip=.2, eps_clip_high=.2, rollout_max_response_len=2048, use_kl_loss=True, kl_loss_coef=.001,
                ouro_output=str(a.output.parent), ouro_reuse_old_log_probs=a.reuse, allgather_cp=False,
                log_probs_chunk_size=-1, use_dynamic_batch_size=False)
            data = dict(tokens=[torch.tensor(r["tokens"], device="cuda", dtype=torch.long) for r in rows],
                metadata=[r["train_metadata"] for r in rows], total_lengths=[len(r["tokens"]) for r in rows],
                response_lengths=[r["response_length"] for r in rows], partition=selected,
                rollout_log_probs=[torch.tensor(r["rollout_log_probs"], device="cuda") for r in rows],
                loss_masks=[torch.ones(r["response_length"], device="cuda") for r in rows],
                advantages=[torch.full((r["response_length"],),r["train_metadata"].get("sequence_advantage",1.),device="cuda") for r in rows])
            schedule = homogeneous_schedule(data["metadata"], data["total_lengths"], a.micro_batch_size)
            data["num_microbatches"] = [len(schedule)]
            batches = []
            iterator = DataIterator(data, schedule)
            for indices in schedule:
                batch = get_batch(iterator, [k for k in data if k != "num_microbatches"], 128, False)
                batch["indices"] = indices
                batches.append(batch)

            def logits(batch):
                return wrapped(batch["tokens"], ouro_depth=batch["metadata"][0]["ouro_depth"],
                    packed_seq_params=batch["packed_seq_params"]).float()

            def probabilities(batch):
                return get_log_probs_and_entropy(logits(batch), args=args,
                    unconcat_tokens=batch["unconcat_tokens"], total_lengths=batch["total_lengths"],
                    response_lengths=batch["response_lengths"], with_entropy=False)[1]["log_probs"]

            def boundary():
                torch.cuda.synchronize()
                dist.barrier()
                return time.monotonic()

            first_gradients = None
            for repeat in range(a.repeats):
                wrapped.zero_grad_buffer()
                wrapped.eval()
                torch.cuda.reset_peak_memory_stats()
                start = boundary()
                with torch.no_grad():
                    for batch in batches:
                        batch["ref_log_probs"] = probabilities(batch)
                ref_done = boundary()
                audits = {}
                with torch.no_grad():
                    seen = set()
                    for batch in reversed(batches):
                        depth = batch["metadata"][0]["ouro_depth"]
                        if not a.reuse or depth not in seen:
                            values = probabilities(batch)
                            batch["log_probs"] = values
                            audits.update(zip(batch["partition"], values))
                            seen.add(depth)
                old_done = boundary()
                wrapped.train()
                if a.reuse:
                    references = [None] * len(rows)
                    for batch in batches:
                        for i, value in zip(batch["indices"], batch["ref_log_probs"], strict=True):
                            references[i] = value
                    data["ref_log_probs"] = references
                    begin_audit(args, data, audits)
                total_loss = 0.
                for batch in batches:
                    value, _ = policy_loss(args, batch, logits(batch), None)
                    (value * world / n).backward()
                    total_loss += float(value.detach()) / n
                backward_done = boundary()
                wrapped.finish_grad_sync()
                if a.reuse:
                    finish_reuse_audit(args)
                done = boundary()
                loss_tensor = torch.tensor(total_loss, device="cuda")
                dist.all_reduce(loss_tensor)
                result = dict(reference_seconds=ref_done-start, old_or_audit_seconds=old_done-ref_done,
                    forward_backward_seconds=backward_done-old_done, sync_seconds=done-backward_done,
                    total_seconds=done-start, peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
                    loss=float(loss_tensor), final_norm_grad_norm=float(model.decoder.final_layernorm.weight.main_grad.norm()))
                gradients = {name: param.main_grad.detach().cpu().clone() for name, param in model.named_parameters()
                             if param.requires_grad and ("layernorm" in name or name.endswith("norm.weight"))}
                if first_gradients is None:
                    first_gradients = gradients
                else:
                    squared_delta = sum((gradients[k]-first_gradients[k]).double().square().sum() for k in gradients)
                    squared_norm = sum(first_gradients[k].double().square().sum() for k in gradients)
                    result["norm_gradient_relative_difference_to_first_repeat"] = float((squared_delta/squared_norm).sqrt())
                report["runs"].append(result)
                if rank == 0:
                    print(json.dumps(result), flush=True)
                    a.output.write_text(json.dumps(report, indent=2)+"\n")
            if rank == 0:
                gradients = {name: param.main_grad.detach().cpu() for name, param in model.named_parameters()
                             if param.requires_grad and ("layernorm" in name or name.endswith("norm.weight"))}
                torch.save(gradients, a.output.with_suffix(".pt"))
        finally:
            parallel_state.destroy_model_parallel()
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
