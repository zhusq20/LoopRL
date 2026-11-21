from types import SimpleNamespace as NS

import pytest
import torch
from torch.nn import functional as F

from slime_plugins.ouro_attention import OuroPagedAttention
from slime_plugins.ouro_kernels import paged_decode


@pytest.mark.parametrize("lengths,heads,kv_heads,dim", [
    ([1, 7, 13], 4, 2, 16), ([257, 511, 2048], 16, 16, 128),
    ([2048, 17, 2049], 4, 1, 64), ([1], 16, 16, 128)])
def test_paged_ragged_decode_matches_fp64_oracle(lengths, heads, kv_heads, dim):
    torch.manual_seed(7)
    batch, maximum = len(lengths), max(lengths)
    n = batch*maximum
    pages = torch.randperm(n, device="cuda").reshape(batch, maximum).int()
    for i, length in enumerate(lengths):
        pages[i, length:] = 2**30
    q = torch.randn(batch, heads, dim, device="cuda")
    k, v = [torch.randn(n, kv_heads, dim, device="cuda") for _ in range(2)]
    actual = paged_decode(q, k, v, pages, torch.arange(batch, device="cuda"),
        torch.tensor(lengths, device="cuda"), maximum, dim**-.5)
    expected = []
    for i, length in enumerate(lengths):
        keys = k[pages[i, :length].long()].transpose(0, 1).repeat_interleave(heads//kv_heads, 0).double()
        values = v[pages[i, :length].long()].transpose(0, 1).repeat_interleave(heads//kv_heads, 0).double()
        weights = (q[i].double().unsqueeze(1)*keys).sum(-1).mul(dim**-.5).softmax(-1)
        expected.append((weights.unsqueeze(-1)*values).sum(1))
    torch.testing.assert_close(actual.double(), torch.stack(expected), atol=3e-6, rtol=3e-5)


@pytest.mark.parametrize("prefixes,lengths", [([0, 9, 31], [7, 5, 1]), ([1, 256, 2048], [1, 1, 1])])
def test_prefill_only_computes_new_queries_with_correct_causal_offset(prefixes, lengths):
    torch.manual_seed(8)
    totals = [p+l for p,l in zip(prefixes,lengths)]
    maximum = max(totals)
    pages = torch.randperm(3*maximum, device="cuda").reshape(3, maximum).int()
    for i,total in enumerate(totals):
        pages[i,total:] = 2**30
    backend = object.__new__(OuroPagedAttention)
    backend.use_sliding_window_kv_pool = False
    backend.req_to_token_pool = NS(req_to_token=pages)
    batch = NS(forward_mode=NS(is_decode=lambda:False), seq_lens_cpu=torch.tensor(totals),
        seq_lens=torch.tensor(totals, device="cuda"), req_pool_indices=torch.arange(3,device="cuda"),
        extend_seq_lens_cpu=lengths, extend_prefix_lens_cpu=prefixes)
    backend.init_forward_metadata(batch)
    q = torch.randn(sum(lengths),4,32,device="cuda")
    k,v = [torch.randn(3*maximum,2,32,device="cuda") for _ in range(2)]
    actual=backend._run_sdpa_forward_extend(q,torch.empty_like(q),k,v,pages,batch.req_pool_indices,
        batch.seq_lens,None,None,scaling=32**-.5,enable_gqa=True,causal=True)
    expected=[];start=0
    for i,(prefix,length,total) in enumerate(zip(prefixes,lengths,totals)):
        keys=k[pages[i,:total].long()].transpose(0,1).repeat_interleave(2,0).double()
        values=v[pages[i,:total].long()].transpose(0,1).repeat_interleave(2,0).double()
        query=q[start:start+length].transpose(0,1).double()
        mask=torch.arange(total,device="cuda")[None,:] <= prefix+torch.arange(length,device="cuda")[:,None]
        weights=(query@keys.transpose(-1,-2)).mul(32**-.5).masked_fill(~mask,-torch.inf).softmax(-1)
        expected.append((weights@values).transpose(0,1));start+=length
    torch.testing.assert_close(actual.double(),torch.cat(expected),atol=3e-6,rtol=3e-5)


def test_fused_fp32_norm_and_rope_preserve_reference_semantics():
    from slime_plugins.ouro_kernels import rms_norm, rotary
    from sglang.srt.layers.rotary_embedding.utils import apply_rotary_emb
    torch.manual_seed(17)
    x=torch.randn(13,2048,device='cuda')
    weight=torch.randn(2048,device='cuda')
    expected=weight*(x*torch.rsqrt(x.square().mean(-1,keepdim=True)+1e-6))
    torch.testing.assert_close(rms_norm(x,weight,1e-6),expected,atol=2e-6,rtol=2e-6)
    qkv=torch.randn(13,6144,device='cuda')
    q,k,_=qkv.chunk(3,dim=-1)
    positions=torch.randint(0,4096,(13,),device='cuda')
    angles=torch.randn(4096,64,device='cuda')
    cache=torch.cat([angles.cos(),angles.sin()],dim=-1)
    actual=rotary(positions,q,k,cache,128)
    cos,sin=cache[positions].chunk(2,dim=-1)
    for output,source in zip(actual,[q,k]):
        reference=apply_rotary_emb(source.reshape(13,16,128),cos,sin,True).reshape_as(source)
        torch.testing.assert_close(output,reference,atol=0,rtol=0)


def test_decode_graph_replays_updated_pages_lengths_and_requests():
    torch.manual_seed(23)
    batch,heads,dim,maximum=3,4,32,1024
    pages=torch.randperm(batch*maximum,device='cuda').reshape(batch,maximum).int()
    q=torch.randn(batch,heads,dim,device='cuda')
    k,v=[torch.randn(batch*maximum,heads,dim,device='cuda') for _ in range(2)]
    requests=torch.arange(batch,device='cuda')
    lengths=torch.tensor([7,1,500],device='cuda')
    out=torch.empty_like(q)
    splits=(maximum+255)//256
    workspace=(torch.empty(batch,heads,splits,dim,device='cuda'),
               torch.empty(batch,heads,splits,device='cuda'),
               torch.empty(batch,heads,splits,device='cuda'))
    def forward():
        paged_decode(q,k,v,pages,requests,lengths,maximum,dim**-.5,out,workspace)
    stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        forward()
    torch.cuda.current_stream().wait_stream(stream)
    graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):forward()
    for new_lengths in ([8,1,501],[700,257,1024]):
        lengths.copy_(torch.tensor(new_lengths,device='cuda'))
        requests.copy_(torch.tensor([2,0,1],device='cuda'))
        q.add_(.01)
        graph.replay()
        expected=paged_decode(q,k,v,pages,requests,lengths,maximum,dim**-.5)
        torch.testing.assert_close(out,expected,atol=0,rtol=0)
