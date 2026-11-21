"""Diagnostic FP32 tensor-core linear operations with three-product TF32 dots.

Inputs, outputs, accumulators and gradients remain FP32. This changes rounding
relative to IEEE SGEMM and must pass complete native probability/gradient checks.
It is deliberately separate from the default training implementation.
"""
from types import MethodType

import torch
import triton
import triton.language as tl


@triton.jit
def _gemm(A,B,C,M:tl.constexpr,N:tl.constexpr,K:tl.constexpr,
          A0:tl.constexpr,A1:tl.constexpr,B0:tl.constexpr,B1:tl.constexpr,
          BM:tl.constexpr=128,BN:tl.constexpr=128,BK:tl.constexpr=64):
    rows=tl.program_id(0)*BM+tl.arange(0,BM)
    cols=tl.program_id(1)*BN+tl.arange(0,BN)
    kk=tl.arange(0,BK)
    acc=tl.full((BM,BN),0,tl.float32)
    for start in range(tl.cdiv(K,BK)):
        offsets=start*BK+kk
        a=tl.load(A+rows[:,None]*A0+offsets[None,:]*A1,(rows[:,None]<M)&(offsets[None,:]<K),other=0)
        b=tl.load(B+offsets[:,None]*B0+cols[None,:]*B1,(offsets[:,None]<K)&(cols[None,:]<N),other=0)
        acc=tl.dot(a,b,acc,input_precision='tf32x3')
    tl.store(C+rows[:,None]*N+cols[None,:],acc,(rows[:,None]<M)&(cols[None,:]<N))


def matmul(a,b):
    if a.dtype!=torch.float32 or b.dtype!=torch.float32 or a.ndim!=2 or b.ndim!=2:
        raise ValueError('Ouro FP32x3 GEMM requires FP32 matrices')
    m,k=a.shape
    if b.shape[0]!=k:raise ValueError('Incompatible matrix dimensions')
    n=b.shape[1]
    if min(m,n)<128:
        return a@b
    out=torch.empty((m,n),device=a.device,dtype=torch.float32)
    _gemm[(triton.cdiv(m,128),triton.cdiv(n,128))](a,b,out,m,n,k,*a.stride(),*b.stride(),
                                             num_warps=8,num_stages=3)
    return out


class _Linear(torch.autograd.Function):
    @staticmethod
    def forward(ctx,x,weight):
        ctx.save_for_backward(x,weight)
        return matmul(x.reshape(-1,x.shape[-1]),weight.T).reshape(*x.shape[:-1],weight.shape[0])

    @staticmethod
    def backward(ctx,gradient):
        x,weight=ctx.saved_tensors
        dy=gradient.reshape(-1,gradient.shape[-1])
        dx=matmul(dy,weight).reshape_as(x) if ctx.needs_input_grad[0] else None
        dw=matmul(dy.T,x.reshape(-1,x.shape[-1])) if ctx.needs_input_grad[1] else None
        return dx,dw


def _forward(self,input,weight,**options):
    if (options.get('bias') is not None or options.get('gradient_accumulation_fusion')
            or options.get('allreduce_dgrad') or options.get('sequence_parallel')
            or options.get('grad_output_buffer') is not None or options.get('output_dtype') is not None):
        raise ValueError('Unsupported communication/fusion option for Ouro FP32x3 GEMM')
    return _Linear.apply(input,weight)


def install(model):
    from megatron.core.tensor_parallel.layers import ColumnParallelLinear,RowParallelLinear
    if model.config.tensor_model_parallel_size!=1:
        raise ValueError('Ouro FP32x3 currently requires TP=1')
    for module in model.modules():
        if isinstance(module,(ColumnParallelLinear,RowParallelLinear)):
            module._forward_impl=MethodType(_forward,module)
