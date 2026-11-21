"""Isolated Ouro-shaped operator benchmark; does not modify a running model.

Random tensors, fixed cache and one decoder block (without RoPE/cache writes).
This diagnoses precision, batching, and launch overhead, not end-to-end speedup.
"""
import argparse
import gc
import json
import time
from pathlib import Path

import torch
from torch.nn import functional as F


def measure(fn, iterations=30):
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    started = time.perf_counter()
    start.record()
    for _ in range(iterations):
        fn()
    end.record()
    torch.cuda.synchronize()
    return {"wall_ms": (time.perf_counter()-started)*1000/iterations,
            "cuda_event_ms": start.elapsed_time(end)/iterations}


@torch.inference_mode()
def case(batch, length, precision):
    torch.manual_seed(2718)
    torch.set_float32_matmul_precision("high" if precision == "tf32" else "highest")
    dtype = torch.bfloat16 if precision == "bf16" else torch.float32
    def rand(*shape):
        return torch.randn(shape, device="cuda", dtype=dtype) * .02
    hidden = rand(batch, 2048)
    weights = [rand(6144, 2048), rand(2048, 2048), rand(11264, 2048), rand(2048, 5632)]
    norm_weight = torch.ones(2048, device="cuda", dtype=dtype)
    cache_size = max(16384, batch*length)
    key_cache, value_cache = rand(cache_size, 16, 128), rand(cache_size, 16, 128)
    indices = torch.arange(batch*length, device="cuda").reshape(batch, length) % cache_size
    mask = torch.ones(batch, 1, 1, length, device="cuda", dtype=torch.bool)

    def norm(x):
        value = x.float()
        value = value * torch.rsqrt(value.square().mean(-1, keepdim=True) + 1e-6)
        return norm_weight * value.to(x.dtype)

    def forward():
        qkv = F.linear(norm(hidden), weights[0])
        query = qkv[:, :2048].reshape(batch, 16, 128)
        keys = key_cache[indices].permute(0, 2, 1, 3)
        values = value_cache[indices].permute(0, 2, 1, 3)
        attention = F.scaled_dot_product_attention(query.unsqueeze(2), keys, values,
            attn_mask=mask, scale=128**-.5, dropout_p=0.)
        value = hidden + norm(F.linear(attention.squeeze(2).reshape(batch, 2048), weights[1]))
        gate, up = F.linear(norm(value), weights[2]).chunk(2, dim=-1)
        return value + norm(F.linear(F.silu(gate)*up, weights[3]))

    eager = measure(forward)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            forward()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        graph_output = forward()
    graph.replay()
    expected = forward()
    error = (expected.float()-graph_output.float()).abs().max().item()
    graph_time = measure(graph.replay)
    return dict(batch=batch, context=length, precision=precision, eager=eager,
                cuda_graph=graph_time, graph_max_abs_difference=error,
                graph_speedup=eager["wall_ms"]/graph_time["wall_ms"],
                peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    report = {"torch": torch.__version__, "gpu": torch.cuda.get_device_name(),
              "note": __doc__, "cases": []}
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    for precision in ("fp32", "tf32", "bf16"):
        for batch in (1, 8, 16, 32):
            torch.cuda.reset_peak_memory_stats()
            row = case(batch, 2048, precision)
            report["cases"].append(row)
            print(json.dumps(row), flush=True)
            path.write_text(json.dumps(report, indent=2)+"\n")
            gc.collect()
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
