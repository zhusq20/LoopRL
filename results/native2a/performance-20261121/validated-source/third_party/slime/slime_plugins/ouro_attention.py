"""Reference batched SDPA and opt-in FP32 paged attention for Ouro."""
import torch
from torch.nn import functional as F
from sglang.srt.layers.attention.torch_native_backend import TorchNativeAttnBackend


class OuroBatchedDecode(TorchNativeAttnBackend):
    def init_forward_metadata(self, batch):
        super().init_forward_metadata(batch)
        self.decode_tokens = self.decode_mask = None
        if batch.forward_mode.is_decode():
            lengths = batch.seq_lens_cpu
            if lengths is None:
                raise ValueError("Batched Ouro decode needs CPU length metadata")
            maximum = int(lengths.max())
            mask = torch.arange(maximum, device=batch.seq_lens.device)[None, :] < batch.seq_lens[:, None]
            indices = self.req_to_token_pool.req_to_token[batch.req_pool_indices, :maximum]
            # Padding slots may never have been initialized. Do not read them.
            self.decode_tokens = indices.masked_fill(~mask, 0).long()
            self.decode_mask = mask[:, None, None, :]

    def _run_sdpa_forward_decode(self, query, output, k_cache, v_cache, req_to_token,
            req_pool_indices, seq_lens, encoder_lens=None, scaling=None, enable_gqa=False,
            causal=False, is_cross_attn=False, sliding_window_size=None):
        if encoder_lens is not None or causal or is_cross_attn or sliding_window_size not in (None, -1):
            raise ValueError("Ouro batched decode supports plain causal-decoder history only")
        if self.decode_tokens is None or len(self.decode_tokens) != len(query):
            raise ValueError("Missing or stale decode metadata")
        keys = k_cache[self.decode_tokens].permute(0, 2, 1, 3).to(query.dtype)
        values = v_cache[self.decode_tokens].permute(0, 2, 1, 3).to(query.dtype)
        result = F.scaled_dot_product_attention(query.unsqueeze(2), keys, values,
            attn_mask=self.decode_mask, enable_gqa=enable_gqa, scale=scaling, dropout_p=0.)
        output.copy_(result.squeeze(2))
        return output


class OuroPagedAttention(OuroBatchedDecode):
    """Read paged KV directly on decode; cache prefill metadata once per batch."""

    def init_forward_metadata(self, batch):
        TorchNativeAttnBackend.init_forward_metadata(self, batch)
        if batch.seq_lens_cpu is None:
            raise ValueError("Ouro paged attention needs CPU sequence lengths")
        self.maximum = int(batch.seq_lens_cpu.max())
        self.workspace = None
        self.prefill = []
        self.single_token_extend = False
        if not batch.forward_mode.is_decode():
            lengths = batch.extend_seq_lens_cpu
            prefixes = batch.extend_prefix_lens_cpu
            if lengths is None or prefixes is None:
                raise ValueError("Ouro prefill needs CPU extension metadata")
            self.single_token_extend = all(length == 1 for length in lengths)
            # Index on GPU once, instead of extracting req_pool_indices once per
            # request per physical layer per loop (a device synchronization).
            pages = self.req_to_token_pool.req_to_token[batch.req_pool_indices, :self.maximum]
            start = 0
            for i, (length, prefix) in enumerate(zip(lengths, prefixes, strict=True)):
                total = length + prefix
                mask = None
                if prefix and length > 1:
                    qpos = prefix + torch.arange(length, device=pages.device)
                    mask = torch.arange(total, device=pages.device)[None, :] <= qpos[:, None]
                self.prefill.append((start, length, prefix, pages[i, :total].long(), mask))
                start += length

    def _paged(self, query, output, k_cache, v_cache, pages, requests, lengths, scale):
        from slime_plugins.ouro_kernels import paged_decode
        if self.workspace is None:
            b, h, d = query.shape
            splits = (self.maximum + 255)//256
            self.workspace = (
                torch.empty((b, h, splits, d), device=query.device, dtype=torch.float32),
                torch.empty((b, h, splits), device=query.device, dtype=torch.float32),
                torch.empty((b, h, splits), device=query.device, dtype=torch.float32))
        return paged_decode(query, k_cache, v_cache, pages, requests, lengths,
                            self.maximum, scale, output, self.workspace)

    def _run_sdpa_forward_decode(self, query, output, k_cache, v_cache, req_to_token,
            req_pool_indices, seq_lens, encoder_lens=None, scaling=None, enable_gqa=False,
            causal=False, is_cross_attn=False, sliding_window_size=None):
        if encoder_lens is not None or causal or is_cross_attn or sliding_window_size not in (None, -1):
            raise ValueError("Ouro paged decode supports causal-decoder history only")
        return self._paged(query, output, k_cache, v_cache, req_to_token,
                           req_pool_indices, seq_lens, scaling)

    def _run_sdpa_forward_extend(self, query, output, k_cache, v_cache, req_to_token,
            req_pool_indices, seq_lens, extend_prefix_lens, extend_seq_lens,
            encoder_lens=None, scaling=None, enable_gqa=False, causal=False,
            is_cross_attn=False, sliding_window_size=None):
        if encoder_lens is not None or not causal or is_cross_attn or sliding_window_size not in (None, -1):
            raise ValueError("Ouro prefill supports plain causal attention only")
        if self.single_token_extend:
            return self._paged(query, output, k_cache, v_cache, req_to_token,
                               req_pool_indices, seq_lens, scaling)
        for start, length, prefix, tokens, mask in self.prefill:
            if not length:
                continue
            q = query[start:start+length].transpose(0, 1).unsqueeze(0)
            k = k_cache[tokens].transpose(0, 1).unsqueeze(0)
            v = v_cache[tokens].transpose(0, 1).unsqueeze(0)
            result = F.scaled_dot_product_attention(q, k, v, attn_mask=mask,
                is_causal=prefix == 0, enable_gqa=enable_gqa, scale=scaling, dropout_p=0.)
            output[start:start+length].copy_(result.squeeze(0).transpose(0, 1))
        return output


def install():
    from sglang.srt.layers.attention.attention_registry import ATTENTION_BACKENDS
    original = ATTENTION_BACKENDS["torch_native"]
    if getattr(original, "ouro_dispatch", False):
        return

    def create(runner):
        config = runner.model_config.hf_config
        if config.model_type == "ouro" and getattr(config, "ouro_paged_attention", False):
            return OuroPagedAttention(runner)
        if config.model_type == "ouro" and getattr(config, "ouro_batched_decode", False):
            return OuroBatchedDecode(runner)
        return original(runner)

    create.ouro_dispatch = True
    ATTENTION_BACKENDS["torch_native"] = create
