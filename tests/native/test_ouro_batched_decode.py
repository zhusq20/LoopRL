from types import SimpleNamespace as NS

import pytest
import torch
from torch.nn import functional as F
from slime_plugins.ouro_attention import OuroBatchedDecode


@pytest.mark.parametrize("lengths", [[1, 7, 13], [17, 17, 17], [128, 31, 64]])
def test_batched_decode_matches_independent_ragged_sequences(lengths):
    if not torch.cuda.is_available():
        pytest.skip("GPU attention check")
    torch.manual_seed(42)
    backend = object.__new__(OuroBatchedDecode)
    backend.use_sliding_window_kv_pool = False
    req = torch.randint(0, 256, (3, max(lengths)), device="cuda", dtype=torch.int32)
    for i, length in enumerate(lengths):
        req[i, length:] = 2**30  # invalid padded cache locations must never be read
    backend.req_to_token_pool = NS(req_to_token=req)
    batch = NS(forward_mode=NS(is_decode=lambda: True), seq_lens_cpu=torch.tensor(lengths),
        seq_lens=torch.tensor(lengths, device="cuda"), req_pool_indices=torch.arange(3, device="cuda"))
    backend.init_forward_metadata(batch)
    q = torch.randn(3, 4, 16, device="cuda")
    k, v = [torch.randn(256, 2, 16, device="cuda") for _ in range(2)]
    result = backend._run_sdpa_forward_decode(q, torch.empty_like(q), k, v, req,
        batch.req_pool_indices, batch.seq_lens, scaling=.25, enable_gqa=True)
    reference = torch.cat([F.scaled_dot_product_attention(q[i][None, :, None],
        k[req[i, :length].long()].permute(1, 0, 2)[None],
        v[req[i, :length].long()].permute(1, 0, 2)[None], enable_gqa=True, scale=.25).squeeze(2)
        for i,length in enumerate(lengths)])
    torch.testing.assert_close(result, reference, atol=2e-6, rtol=2e-5)
