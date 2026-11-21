"""FP32 paged decode without materializing a dense copy of the KV history.

Each split computes FP32 dot products/reductions (no TF32) directly from the
request page table. A second kernel merges the stable softmax partials.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _norm(X, Weight, Out, STRIDE: tl.constexpr, DIM: tl.constexpr,
          EPS: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    col = tl.arange(0, BLOCK)
    x = tl.load(X + row*STRIDE + col, col < DIM, other=0).to(tl.float32)
    variance = tl.sum(x*x, axis=0)/DIM
    value = x*tl.rsqrt(variance+EPS)
    weight = tl.load(Weight+col, col < DIM, other=0).to(tl.float32)
    tl.store(Out+row*DIM+col, value*weight, col < DIM)


def rms_norm(hidden, weight, eps):
    if hidden.dtype != torch.float32 or hidden.ndim != 2 or hidden.stride(-1) != 1:
        raise ValueError("Ouro fused norm requires a two-dimensional FP32 tensor")
    result = torch.empty_like(hidden)
    _norm[(hidden.shape[0],)](hidden, weight, result, hidden.stride(0), hidden.shape[1], eps,
        triton.next_power_of_2(hidden.shape[1]), num_warps=4, enable_fp_fusion=False)
    return result


@triton.jit
def _rope(Q, K, Positions, Cache, QOUT, KOUT, QS: tl.constexpr, KS: tl.constexpr,
          QHEADS: tl.constexpr, KHEADS: tl.constexpr, DIM: tl.constexpr,
          CACHE_STRIDE: tl.constexpr, BLOCK: tl.constexpr):
    row, head = tl.program_id(0), tl.program_id(1)
    i = tl.arange(0, BLOCK)
    position = tl.load(Positions+row)
    cos = tl.load(Cache+position*CACHE_STRIDE+i, i<DIM//2, other=0)
    sin = tl.load(Cache+position*CACHE_STRIDE+DIM//2+i, i<DIM//2, other=0)
    if head < QHEADS:
        a = tl.load(Q+row*QS+head*DIM+i, i<DIM//2, other=0)
        b = tl.load(Q+row*QS+head*DIM+DIM//2+i, i<DIM//2, other=0)
        tl.store(QOUT+row*QHEADS*DIM+head*DIM+i, a*cos-b*sin, i<DIM//2)
        tl.store(QOUT+row*QHEADS*DIM+head*DIM+DIM//2+i, b*cos+a*sin, i<DIM//2)
    else:
        h = head-QHEADS
        a = tl.load(K+row*KS+h*DIM+i, i<DIM//2, other=0)
        b = tl.load(K+row*KS+h*DIM+DIM//2+i, i<DIM//2, other=0)
        tl.store(KOUT+row*KHEADS*DIM+h*DIM+i, a*cos-b*sin, i<DIM//2)
        tl.store(KOUT+row*KHEADS*DIM+h*DIM+DIM//2+i, b*cos+a*sin, i<DIM//2)


def rotary(positions, query, key, cache, head_dim):
    if query.dtype != torch.float32 or key.dtype != torch.float32 or cache.dtype != torch.float32:
        raise ValueError("Ouro fused RoPE requires FP32")
    qout, kout = torch.empty_like(query), torch.empty_like(key)
    qheads, kheads = query.shape[1]//head_dim, key.shape[1]//head_dim
    _rope[(query.shape[0],qheads+kheads)](query,key,positions,cache,qout,kout,
        query.stride(0),key.stride(0),qheads,kheads,head_dim,cache.stride(0),
        triton.next_power_of_2(head_dim//2),num_warps=4,enable_fp_fusion=False)
    return qout,kout


@triton.jit
def _paged_partial(Q, K, V, Pages, Requests, Lengths, Partial, Maxima, Sums,
                   q0: tl.constexpr, q1: tl.constexpr, k0: tl.constexpr, k1: tl.constexpr,
                   v0: tl.constexpr, v1: tl.constexpr, page_stride: tl.constexpr,
                   HEADS: tl.constexpr, KV_HEADS: tl.constexpr, DIM: tl.constexpr,
                   SPLITS: tl.constexpr, SCALE: tl.constexpr, BLOCK: tl.constexpr,
                   WIDTH: tl.constexpr):
    batch, head, split = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    positions = split * BLOCK + tl.arange(0, BLOCK)
    channels = tl.arange(0, WIDTH)
    length = tl.load(Lengths + batch)
    request = tl.load(Requests + batch)
    valid = positions < length
    slots = tl.load(Pages + request * page_stride + positions, valid, other=0)
    kv_head = head // (HEADS // KV_HEADS)
    query = tl.load(Q + batch*q0 + head*q1 + channels, channels < DIM, other=0).to(tl.float32)
    keys = tl.load(K + slots[:, None]*k0 + kv_head*k1 + channels[None, :],
                   valid[:, None] & (channels[None, :] < DIM), other=0).to(tl.float32)
    scores = tl.sum(keys * query[None, :], axis=1) * SCALE
    scores = tl.where(valid, scores, -float("inf"))
    maximum = tl.where(split*BLOCK < length, tl.max(scores, axis=0), 0.)
    probabilities = tl.exp(scores - maximum)
    denominator = tl.sum(probabilities, axis=0)
    values = tl.load(V + slots[:, None]*v0 + kv_head*v1 + channels[None, :],
                     valid[:, None] & (channels[None, :] < DIM), other=0).to(tl.float32)
    numerator = tl.sum(probabilities[:, None] * values, axis=0)
    offset = (batch*HEADS + head)*SPLITS + split
    tl.store(Partial + offset*DIM + channels, numerator, channels < DIM)
    tl.store(Maxima + offset, maximum)
    tl.store(Sums + offset, denominator)


@triton.jit
def _paged_merge(Partial, Maxima, Sums, Output, HEADS: tl.constexpr,
                 DIM: tl.constexpr, SPLITS: tl.constexpr, WIDTH: tl.constexpr,
                 SPLIT_BLOCK: tl.constexpr, o0: tl.constexpr, o1: tl.constexpr):
    batch, head = tl.program_id(0), tl.program_id(1)
    splits = tl.arange(0, SPLIT_BLOCK)
    channels = tl.arange(0, WIDTH)
    offset = (batch*HEADS + head)*SPLITS + splits
    denominator = tl.load(Sums + offset, splits < SPLITS, other=0.)
    maximum = tl.load(Maxima + offset, splits < SPLITS, other=-float("inf"))
    maximum = tl.where(denominator > 0, maximum, -float("inf"))
    factors = tl.exp(maximum - tl.max(maximum, axis=0))
    numerator = tl.load(Partial + offset[:, None]*DIM + channels[None, :],
                        (splits[:, None] < SPLITS) & (channels[None, :] < DIM), other=0.)
    result = tl.sum(numerator*factors[:, None], axis=0) / tl.sum(denominator*factors, axis=0)
    tl.store(Output + batch*o0 + head*o1 + channels, result, channels < DIM)


def paged_decode(query, key_cache, value_cache, pages, requests, lengths, maximum,
                 scale, output=None, workspace=None):
    if query.dtype != torch.float32 or key_cache.dtype != torch.float32 or value_cache.dtype != torch.float32:
        raise ValueError("Ouro paged acceptance kernel requires FP32 Q/K/V")
    batch, heads, dim = query.shape
    if key_cache.shape != value_cache.shape or heads % key_cache.shape[1] or key_cache.shape[2] != dim:
        raise ValueError("Unsupported Ouro paged attention shape")
    block = 256
    splits = triton.cdiv(maximum, block)
    if maximum < 1 or dim > 256 or query.stride(-1) != 1:
        raise ValueError("Invalid Ouro decode dimensions")
    if output is None:
        output = torch.empty_like(query)
    if workspace is None:
        partial = torch.empty((batch, heads, splits, dim), device=query.device, dtype=torch.float32)
        maxima = torch.empty((batch, heads, splits), device=query.device, dtype=torch.float32)
        sums = torch.empty_like(maxima)
    else:
        partial, maxima, sums = workspace
    _paged_partial[(batch, heads, splits)](query, key_cache, value_cache, pages, requests,
        lengths, partial, maxima, sums, query.stride(0), query.stride(1),
        key_cache.stride(0), key_cache.stride(1), value_cache.stride(0), value_cache.stride(1),
        pages.stride(0), heads, key_cache.shape[1], dim, splits, scale, block,
        triton.next_power_of_2(dim), num_warps=8, enable_fp_fusion=False)
    _paged_merge[(batch, heads)](partial, maxima, sums, output, heads, dim, splits,
        triton.next_power_of_2(dim), triton.next_power_of_2(splits),
        output.stride(0), output.stride(1), num_warps=4, enable_fp_fusion=False)
    return output
