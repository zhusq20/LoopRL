"""Fixed sequence-depth Ouro serving with shared weights and per-loop KV slots.

Set SGLANG_EXTERNAL_MODEL_PACKAGE=slime_plugins.sglang_models and select the
depth with --json-model-override-args '{"total_ut_steps": 2}'. A serving engine
has one depth; changing depth for a live KV cache is intentionally unsupported.
"""

import torch
from torch import nn

from sglang.srt.layers.linear import QKVParallelLinear, RowParallelLinear
from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.layers.rotary_embedding import get_rope
from sglang.srt.layers.vocab_parallel_embedding import VocabParallelEmbedding
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.models.llama import LlamaForCausalLM, LlamaMLP
from sglang.srt.utils import add_prefix
from sglang.srt.runtime_context import get_parallel
from slime_plugins.ouro_attention import install as install_ouro_attention

install_ouro_attention()


class OuroRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps, fused=False):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps
        self.fused = fused

    def forward(self, hidden):
        if self.fused:
            from slime_plugins.ouro_kernels import rms_norm
            return rms_norm(hidden, self.weight, self.eps)
        value = hidden.float()
        value = value * torch.rsqrt(value.square().mean(-1, keepdim=True) + self.eps)
        return self.weight * value.to(hidden.dtype)


class OuroAttention(nn.Module):
    def __init__(self, config, layer_id, prefix):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = getattr(config, "head_dim", config.hidden_size // self.num_heads)
        self.fused = getattr(config, "ouro_fused_ops", False)
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.qkv_proj = QKVParallelLinear(config.hidden_size, self.head_dim,
            self.num_heads, self.num_kv_heads, bias=False, prefix=add_prefix("qkv_proj", prefix))
        self.o_proj = RowParallelLinear(self.q_size, config.hidden_size,
            bias=False, prefix=add_prefix("o_proj", prefix))
        self.rotary_emb = get_rope(self.head_dim, rotary_dim=self.head_dim,
            max_position=config.max_position_embeddings, base=config.rope_theta,
            rope_scaling=getattr(config, "rope_scaling", None), is_neox_style=True)
        self.attn = nn.ModuleList([
            RadixAttention(self.num_heads, self.head_dim, self.head_dim ** -0.5,
                num_kv_heads=self.num_kv_heads,
                layer_id=loop * config.num_hidden_layers + layer_id,
                prefix=add_prefix(f"attn.{loop}", prefix))
            for loop in range(config.total_ut_steps)
        ])

    def forward(self, positions, hidden, forward_batch, loop):
        qkv, _ = self.qkv_proj(hidden)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        # Keep FP32 acceptance independent of fused-kernel rounding/toolchains.
        # Low-precision performance kernels need their own consistency checks.
        if self.fused:
            from slime_plugins.ouro_kernels import rotary
            q, k = rotary(positions, q, k, self.rotary_emb.cos_sin_cache, self.head_dim)
        else:
            q, k = self.rotary_emb.forward_native(positions, q, k)
        output = self.attn[loop](q, k, v, forward_batch)
        return self.o_proj(output)[0]


class OuroMLP(LlamaMLP):
    def forward(self, hidden, forward_batch=None):
        gate_up, _ = self.gate_up_proj(hidden)
        return self.down_proj(self.act_fn.forward_native(gate_up))[0]


class OuroDecoderLayer(nn.Module):
    def __init__(self, config, layer_id, prefix):
        super().__init__()
        self.self_attn = OuroAttention(config, layer_id, add_prefix("self_attn", prefix))
        self.mlp = OuroMLP(config.hidden_size, config.intermediate_size, config.hidden_act,
                            prefix=add_prefix("mlp", prefix))
        for name in ("input_layernorm", "input_layernorm_2",
                     "post_attention_layernorm", "post_attention_layernorm_2"):
            setattr(self, name, OuroRMSNorm(config.hidden_size, config.rms_norm_eps,
                                         getattr(config, "ouro_fused_ops", False)))

    def forward(self, positions, hidden, forward_batch, loop):
        branch = self.self_attn(positions, self.input_layernorm(hidden), forward_batch, loop)
        hidden = hidden + self.input_layernorm_2(branch)
        branch = self.mlp(self.post_attention_layernorm(hidden))
        return hidden + self.post_attention_layernorm_2(branch)


class OuroModel(nn.Module):
    def __init__(self, config, quant_config=None, prefix=""):
        super().__init__()
        if get_parallel().tp_size != 1 or get_parallel().pp_group.world_size != 1:
            raise ValueError("Initial Ouro serving supports TP=PP=1")
        if quant_config is not None:
            raise ValueError("Ouro numerical acceptance requires unquantized weights")
        depth = config.total_ut_steps
        if type(depth) is not int or depth not in range(1, 5):
            raise ValueError("Ouro total_ut_steps must be an integer in [1, 4]")
        self.config = config
        self.start_layer, self.end_layer = 0, config.num_hidden_layers
        self.embed_tokens = VocabParallelEmbedding(config.vocab_size, config.hidden_size,
                                                   prefix=add_prefix("embed_tokens", prefix))
        self.layers = nn.ModuleList([OuroDecoderLayer(config, i, add_prefix(f"layers.{i}", prefix))
                                     for i in range(config.num_hidden_layers)])
        self.norm = OuroRMSNorm(config.hidden_size, config.rms_norm_eps,
                               getattr(config, "ouro_fused_ops", False))
        self.early_exit_gate = nn.Linear(config.hidden_size, 1)
        self.early_exit_gate.requires_grad_(False)
        if getattr(config, "ouro_learned_depth", False):
            self.sequence_depth_head = nn.Linear(config.hidden_size, 3)
            with torch.no_grad():
                self.sequence_depth_head.weight.zero_()
                self.sequence_depth_head.bias.copy_(torch.tensor([.1, .1, .8]).log())

    def forward(self, input_ids, positions, forward_batch, input_embeds=None, pp_proxy_tensors=None):
        if pp_proxy_tensors is not None:
            raise ValueError("Ouro does not support pipeline parallel serving")
        hidden = self.embed_tokens(input_ids) if input_embeds is None else input_embeds
        for loop in range(self.config.total_ut_steps):
            for layer in self.layers:
                hidden = layer(positions, hidden, forward_batch, loop)
            hidden = self.norm(hidden)
        return hidden


class OuroForCausalLM(LlamaForCausalLM):
    def _init_model(self, config, quant_config=None, prefix=""):
        return OuroModel(config, quant_config, prefix)

    @torch.no_grad()
    def forward(self, *args, **kwargs):
        result = super().forward(*args, **kwargs)
        # A depth4, one-token request with return_hidden_states='last' returns
        # the prompt-conditioned action distribution from the synchronized head.
        if hasattr(self.model, "sequence_depth_head") and getattr(result, "hidden_states", None) is not None:
            result.hidden_states = self.model.sequence_depth_head(result.hidden_states).log_softmax(-1)
        return result

    def forward_split_prefill(self, *args, **kwargs):
        raise ValueError("Split prefill needs recurrent scheduling and is not supported for Ouro")

    def set_eagle3_layers_to_capture(self, *args, **kwargs):
        raise ValueError("Ouro does not support speculative hidden-state capture")

    set_dflash_layers_to_capture = set_eagle3_layers_to_capture

    def load_weights(self, weights):
        # The same path handles initial HF loading and slime's partial weight
        # updates. Unknown names are errors rather than silently skipped tensors.
        parameters = dict(self.named_parameters())
        loaded = set()
        for name, value in weights:
            if name.endswith("rotary_emb.inv_freq"):
                continue
            for target, source, shard in self.stacked_params_mapping:
                if source in name:
                    mapped = name.replace(source, target)
                    parameter = parameters[mapped]
                    parameter.weight_loader(parameter, value, shard)
                    loaded.add(mapped)
                    break
            else:
                parameter = parameters[name]
                loader = getattr(parameter, "weight_loader", default_weight_loader)
                loader(parameter, value)
                loaded.add(name)
        return loaded


EntryClass = OuroForCausalLM
