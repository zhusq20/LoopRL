"""Ouro reference adapter; fixed depth really executes k loops.

The upstream exit_at_step option only selects a readout after executing all
loops. Do not use it to claim compute savings. Calls here set total_ut_steps
temporarily and restore it, or execute decoder loops explicitly for early exit.
This adapter is deliberately single-process/single-sequence (not thread safe).
"""
from contextlib import contextmanager
from dataclasses import dataclass
import importlib
import time

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .objectives import exit_log_probs

MODEL_ID = "ByteDance/Ouro-2.6B-Thinking"
MODEL_REVISION = "f1edd81e7ac41355db670500ceaf204e0f73af68"


def token_log_probs(logits: Tensor, temperature: float = 1., top_p: float = 1.) -> Tensor:
    """Use this SAME distribution in rollout and recomputation.

    Training uses top_p=1 to avoid a moving support across PPO updates. top_p=.7
    is available for paper-style evaluation only.
    """
    if temperature <= 0 or not 0 < top_p <= 1:
        raise ValueError("temperature must be positive and top_p in (0, 1]")
    logits = logits.float() / temperature
    if top_p < 1:
        sorted_logits, order = logits.sort(descending=True, dim=-1)
        cumulative = sorted_logits.softmax(-1).cumsum(-1)
        remove = cumulative - sorted_logits.softmax(-1) >= top_p
        sorted_logits = sorted_logits.masked_fill(remove, -torch.inf)
        logits = torch.empty_like(logits).scatter(-1, order, sorted_logits)
    return logits.log_softmax(-1)


@dataclass
class Rollout:
    prompt_ids: list[int]
    response_ids: list[int]
    log_probs: list[float]
    depths: list[int]
    text: str
    terminated: bool
    seconds: float
    prefill_loop_tokens: int

    @property
    def loop_tokens(self):
        return sum(self.depths)


class DepthPrior(nn.Module):
    def __init__(self, hidden_size: int, choices=(2, 3, 4), initial_deep_prob=.8):
        super().__init__()
        if len(choices) < 2 or not 0 < initial_deep_prob < 1:
            raise ValueError("invalid depth prior")
        self.choices = tuple(choices)
        self.head = nn.Linear(hidden_size, len(choices), dtype=torch.float32)
        nn.init.zeros_(self.head.weight)
        probs = torch.full((len(choices),), (1 - initial_deep_prob) / (len(choices) - 1))
        probs[choices.index(max(choices))] = initial_deep_prob
        with torch.no_grad():
            self.head.bias.copy_(probs.log())

    def forward(self, prompt_hidden: Tensor):
        return self.head(prompt_hidden.float()).log_softmax(-1)


class OuroAdapter:
    def __init__(self, model, tokenizer):
        self.model, self.tokenizer = model, tokenizer
        self.max_loops = model.config.total_ut_steps
        if self.max_loops != 4:
            raise ValueError("This experiment plan requires a four-loop Ouro checkpoint")
        self.cache_class = importlib.import_module(type(model).__module__).UniversalTransformerCache

    @classmethod
    def load(cls, path: str = MODEL_ID, device="cuda", dtype=torch.bfloat16, revision=MODEL_REVISION):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from pathlib import Path
        kwargs = {} if Path(path).is_dir() else {"revision": revision}
        tokenizer = AutoTokenizer.from_pretrained(path, **kwargs)
        model = AutoModelForCausalLM.from_pretrained(
            path, trust_remote_code=True, torch_dtype=dtype,
            attn_implementation="sdpa", **kwargs).to(device).eval()
        return cls(model, tokenizer)

    @property
    def device(self):
        return self.model.device

    @contextmanager
    def depth(self, depth: int):
        if depth not in range(1, self.max_loops + 1):
            raise ValueError("depth must be 1..4")
        old = self.model.model.total_ut_steps
        self.model.model.total_ut_steps = depth
        try:
            yield
        finally:
            self.model.model.total_ut_steps = old

    def prompt_ids(self, problem: str) -> list[int]:
        return self.tokenizer.apply_chat_template(
            [{"role": "user", "content": problem + "\nPlease put your final answer within \\boxed{}."}],
            tokenize=True, add_generation_prompt=True, enable_thinking=True)

    def forward(self, ids: Tensor, depth: int, cache=None, use_cache=False):
        with self.depth(depth):
            return self.model.model(ids, past_key_values=cache, use_cache=use_cache)

    def prompt_feature(self, prompt_ids: list[int]) -> Tensor:
        ids = torch.tensor([prompt_ids], device=self.device)
        _, hidden, _ = self.forward(ids, 4)
        return hidden[-1][:, -1]

    @torch.no_grad()
    def generate(self, prompt_ids: list[int], depth: int, max_new_tokens: int,
                 temperature=1., top_p=1., seed=0) -> Rollout:
        if max_new_tokens < 1:
            raise ValueError("max_new_tokens must be positive")
        generator = torch.Generator(device=self.device).manual_seed(seed)
        ids = torch.tensor([prompt_ids], device=self.device)
        response, logps = [], []
        cache = None
        start = time.monotonic()
        for _ in range(max_new_tokens):
            outputs, hidden, _ = self.forward(ids, depth, cache, use_cache=True)
            cache = outputs.past_key_values
            lp = token_log_probs(self.model.lm_head(hidden[-1][:, -1]), temperature, top_p)
            token = torch.multinomial(lp.exp(), 1, generator=generator)
            value = int(token.item())
            response.append(value)
            logps.append(float(lp[0, value]))
            if value == self.tokenizer.eos_token_id:
                break
            ids = token
        return Rollout(prompt_ids, response, logps, [depth] * len(response),
                       self.tokenizer.decode(response, skip_special_tokens=False),
                       response[-1] == self.tokenizer.eos_token_id, time.monotonic() - start,
                       depth * len(prompt_ids))

    def response_log_probs(self, rollout: Rollout, temperature=1., chunk_size=128, replay_mode="full") -> Tensor:
        """Unpadded teacher forcing; cached mode matches sampler kernel shapes.

        Cached replay retains a differentiable KV graph and is intended for
        short acceptance runs. Full replay is the scalable path, but must pass
        the numerical sampler/recompute check before it is used for training.
        """
        if len(set(rollout.depths)) != 1:
            raise ValueError("Use ragged replay for token-varying depths")
        if replay_mode == "cached":
            if self.model.training and any(getattr(layer, "gradient_checkpointing", False) for layer in self.model.model.layers):
                raise RuntimeError("Gradient checkpointing would discard cached KV")
            ids = torch.tensor([rollout.prompt_ids], device=self.device)
            cache, values = None, []
            for target in rollout.response_ids:
                output, hidden, _ = self.forward(ids, rollout.depths[0], cache, use_cache=True)
                cache = output.past_key_values
                lp = token_log_probs(self.model.lm_head(hidden[-1][:, -1]), temperature)
                values.append(lp[0, target])
                ids = torch.tensor([[target]], device=self.device)
            return torch.stack(values)
        if replay_mode != "full":
            raise ValueError("Unknown replay mode")
        ids = torch.tensor([rollout.prompt_ids + rollout.response_ids[:-1]], device=self.device)
        _, hidden, _ = self.forward(ids, rollout.depths[0])
        states = hidden[-1][0, len(rollout.prompt_ids) - 1:]
        targets = torch.tensor(rollout.response_ids, device=self.device)
        parts = []
        for start in range(0, len(targets), chunk_size):
            logits = self.model.lm_head(states[start:start + chunk_size])
            lp = token_log_probs(logits, temperature)
            parts.append(lp.gather(-1, targets[start:start + chunk_size, None]).squeeze(-1))
        return torch.cat(parts)

    @torch.no_grad()
    def diagnostics(self, rollout: Rollout, chunk_size=64) -> dict:
        """M1/M3 on four-loop histories, selected-token gap, no full-vocab storage."""
        ids = torch.tensor([rollout.prompt_ids + rollout.response_ids[:-1]], device=self.device)
        _, hidden, gates = self.forward(ids, 4)
        offset = len(rollout.prompt_ids) - 1
        selected = torch.tensor(rollout.response_ids, device=self.device)
        hazards = torch.cat(gates[:-1], -1)[0, offset:]
        logq = exit_log_probs(hazards)
        gaps, agreement = [], [0, 0, 0]
        for start in range(0, len(selected), chunk_size):
            end = start + chunk_size
            lps, tops = [], []
            for h in hidden:
                logits = self.model.lm_head(h[0, offset + start:offset + end]).float()
                lps.append(logits.log_softmax(-1).gather(-1, selected[start:end, None]).squeeze(-1))
                tops.append(logits.argmax(-1))
            lp = torch.stack(lps, -1)
            lq = logq[start:end]
            gap = torch.logsumexp(lq + lp, -1) - (lq.exp() * lp).sum(-1)
            gaps.extend(gap.cpu().tolist())
            for k in range(3):
                agreement[k] += int((tops[k] == tops[3]).sum())
        return {"gaps": gaps, "agreement_counts": agreement, "positions": len(selected),
                "mean_native_depth": float((logq.exp() * torch.arange(1, 5, device=self.device)).sum(-1).mean())}

    def _query_loop(self, hidden: Tensor, cache, position: int, loop: int):
        """One query token, its own deep KV remains visible while computing it."""
        base = self.model.model
        positions = torch.tensor([position], device=self.device)
        embedding = base.rotary_emb(hidden, positions[None])
        # Query length 1 sees the complete prefix and itself. No causal mask is
        # required for SDPA's length-1 query (is_causal=False in HF SDPA).
        for layer in base.layers:
            hidden = layer(hidden, attention_mask=None, position_ids=positions[None],
                           past_key_value=cache, use_cache=True, cache_position=positions,
                           position_embeddings=embedding, current_ut=loop)
        return base.norm(hidden)

    def _fill_last(self, cache, exit_depth: int, position: int):
        """Retain loop<=e; deeper readers reuse e. No detach, including update."""
        layers = self.model.config.num_hidden_layers
        for loop in range(exit_depth, self.max_loops):
            for layer in range(layers):
                source = (exit_depth - 1) * layers + layer
                dest = loop * layers + layer
                while len(cache.key_cache) <= dest:
                    cache.key_cache.append(None)
                    cache.value_cache.append(None)
                for entries in (cache.key_cache, cache.value_cache):
                    # Existing counterfactual last position is discarded, only
                    # the actual exit's new KV is retained for future queries.
                    prefix = entries[dest][..., :position, :] if entries[dest] is not None else entries[source][..., :position, :]
                    entries[dest] = torch.cat((prefix, entries[source][..., -1:, :]), dim=-2)

    def ragged_replay(self, prompt_ids: list[int], response_ids: list[int], exits: list[int],
                      gate_shift=0., temperature=1., return_mixture=False):
        """Serial teacher forcing, all 4 readouts per query, ragged prior KV.

        Causal shift: e[j] belongs to the query predicting response[j]. The
        first query is the final prompt token; prompt KV is always dense, so
        e[0] only selects a readout. For j>=1 it controls response[j-1]'s KV.
        This indexing is also used by the early-exit sampler below.
        """
        if not response_ids or len(response_ids) != len(exits) or any(e not in range(1, 5) for e in exits):
            raise ValueError("invalid response/exit trace")
        if self.model.training and any(getattr(layer, "gradient_checkpointing", False) for layer in self.model.model.layers):
            raise RuntimeError("HF gradient checkpointing drops past KV; disable it for ragged replay")
        ids = torch.tensor([prompt_ids], device=self.device)
        outputs, hs, gs = self.forward(ids, 4, use_cache=True)
        cache = outputs.past_key_values
        readouts, hazards, mixtures = [], [], []
        for j, target in enumerate(response_ids):
            if j:
                query = torch.tensor([[response_ids[j - 1]]], device=self.device)
                hidden = self.model.model.embed_tokens(query)
                hs, gs = [], []
                position = len(prompt_ids) + j - 1
                for loop in range(4):
                    hidden = self._query_loop(hidden, cache, position, loop)
                    hs.append(hidden)
                    gs.append(self.model.model.early_exit_gate(hidden))
                self._fill_last(cache, exits[j], position)
            all_lp = torch.stack([token_log_probs(self.model.lm_head(h[:, -1]), temperature)[0] for h in hs])
            hazard = torch.cat([g[:, -1] for g in gs[:3]], -1)[0] - gate_shift
            readouts.append(all_lp[:, target])
            hazards.append(hazard)
            if return_mixture:
                mixtures.append(torch.logsumexp(exit_log_probs(hazard)[:, None] + all_lp, 0))
        result = (torch.stack(readouts)[None], exit_log_probs(torch.stack(hazards))[None])
        return (*result, torch.stack(mixtures)) if return_mixture else result

    @torch.no_grad()
    def generate_ragged(self, prompt_ids: list[int], max_new_tokens: int, gate_shift=0.,
                        temperature=1., seed=0, mode="sample", threshold=.5, top_p=1.) -> Rollout:
        if mode not in ("sample", "q_exit", "confidence"):
            raise ValueError("invalid exit mode")
        gen = torch.Generator(device=self.device).manual_seed(seed)
        outputs, hs, gs = self.forward(torch.tensor([prompt_ids], device=self.device), 4, use_cache=True)
        cache = outputs.past_key_values
        response, depths, logps = [], [], []
        start = time.monotonic()
        for j in range(max_new_tokens):
            survival, logq = 1., 0.
            if j:
                hidden = self.model.model.embed_tokens(torch.tensor([[response[-1]]], device=self.device))
                position = len(prompt_ids) + j - 1
            for loop in range(4):
                if j:
                    hidden = self._query_loop(hidden, cache, position, loop)
                    gate = self.model.model.early_exit_gate(hidden)[:, -1, 0].float() - gate_shift
                    h = hidden[:, -1]
                else:
                    gate = gs[loop][:, -1, 0].float() - gate_shift
                    h = hs[loop][:, -1]
                lam = float(gate.sigmoid()) if loop < 3 else 1.
                lp = None
                if mode == "sample":
                    stop = loop == 3 or float(torch.rand((), device=self.device, generator=gen)) < lam
                elif mode == "q_exit":
                    survival *= 1 - lam
                    stop = survival <= 1 - threshold or loop == 3
                else:
                    # Confidence is measured before nucleus truncation.
                    lp = token_log_probs(self.model.lm_head(h), temperature)
                    stop = float(lp.exp().max()) >= threshold or loop == 3
                if stop:
                    if loop < 3:
                        logq += float(F.logsigmoid(gate))
                    if lp is None or top_p < 1:
                        lp = token_log_probs(self.model.lm_head(h), temperature, top_p)
                    token = int(torch.multinomial(lp.exp(), 1, generator=gen))
                    response.append(token)
                    depths.append(loop + 1)
                    logps.append(float(lp[0, token]) + (logq if mode == "sample" else 0.))
                    if j:
                        self._fill_last(cache, loop + 1, position)
                    break
                logq += float(F.logsigmoid(-gate))
            if response[-1] == self.tokenizer.eos_token_id:
                break
        return Rollout(prompt_ids, response, logps, depths,
                       self.tokenizer.decode(response, skip_special_tokens=False),
                       response[-1] == self.tokenizer.eos_token_id, time.monotonic() - start, 4 * len(prompt_ids))
