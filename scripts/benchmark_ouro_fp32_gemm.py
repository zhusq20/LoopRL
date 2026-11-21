"""Explore full-mantissa FP32 tensor-core GEMM, independently of training."""
import argparse
import json
from pathlib import Path
import torch
import triton
import triton.language as tl


@triton.jit
def gemm(A,B,C,M:tl.constexpr,N:tl.constexpr,K:tl.constexpr,
         A0:tl.constexpr,A1:tl.constexpr,B0:tl.constexpr,B1:tl.constexpr,
         BM:tl.constexpr,BN:tl.constexpr,BK:tl.constexpr):
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


def matmul(a,b,tile=(64,128,32,4,3)):
    m,k=a.shape; _,n=b.shape
    out=torch.empty((m,n),device=a.device,dtype=torch.float32)
    bm,bn,bk,warps,stages=tile
    gemm[(triton.cdiv(m,bm),triton.cdiv(n,bn))](a,b,out,m,n,k,*a.stride(),*b.stride(),bm,bn,bk,
                                           num_warps=warps,num_stages=stages)
    return out


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--tune',action='store_true')
    args=p.parse_args()
    torch.manual_seed(42);torch.backends.cuda.matmul.allow_tf32=False
    result=[]
    for m,n,k in [(32,6144,2048),(2458,6144,2048),(2458,11264,2048),(2458,2048,5632),(5632,2048,2458)]:
        a=torch.randn(m,k,device='cuda')*.02
        b=torch.randn(n,k,device='cuda').T*.02
        tiles=[(64,128,32,4,3)]
        if args.tune:tiles += [(128,128,32,4,3),(64,128,64,4,3),(128,128,64,8,3),(64,256,32,8,3),(128,256,32,8,3)]
        timings=[]
        for tile in tiles:
            matmul(a,b,tile)
            timings.append((triton.testing.do_bench(lambda:matmul(a,b,tile)),tile))
        best_ms,best_tile=min(timings)
        actual=matmul(a,b,best_tile);expected=a@b
        # Check against a double-precision oracle on a representative slice.
        exact=(a[:64].double()@b[:,:64].double()).float()
        row=dict(shape=[m,n,k],fp32_ms=triton.testing.do_bench(lambda:a@b),
                 tf32x3_ms=best_ms,tile=best_tile,tuning=timings,
                 max_difference=float((actual-expected).abs().max()),
                 fp32_oracle_error=float((expected[:64,:64]-exact).abs().max()),
                 tf32x3_oracle_error=float((actual[:64,:64]-exact).abs().max()))
        row['speedup']=row['fp32_ms']/row['tf32x3_ms'];result.append(row)
        args.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(row),flush=True)


if __name__=='__main__':main()
