from types import SimpleNamespace as NS

import pytest

from looprl.native_serving import context_capacity, configure_token_exit_serving
from looprl.native_2b import parse_generation_trace, parse_trace
from scripts.run_native_2a import validate_resume_protocol


class Tokenizer:
    def apply_chat_template(self, messages, **kwargs):
        assert kwargs == dict(tools=None, tokenize=False, add_generation_prompt=True)
        return messages[0]['content'] + 'tt'

    def encode(self, text, add_special_tokens):
        assert not add_special_tokens
        return list(range(len(text)))


def test_context_counts_chat_template_without_truncating_long_prompt():
    report = context_capacity(Tokenizer(), ['a'*2069, 'short'], 2048, 65536, 65536)
    assert report['context_length'] == 4121
    assert report['required_tokens'] == 4119
    assert report['max_prompt_tokens'] == 2071
    assert report['max_new_tokens'] == 2048
    for model_limit, kv_limit in [(4096, 65536), (65536, 4096)]:
        with pytest.raises(ValueError, match='refusing to truncate'):
            context_capacity(Tokenizer(), ['a'*2069], 2048, model_limit, kv_limit)


def test_pinned_scheduler_does_not_silently_reduce_response_budget():
    from sglang.srt.managers.scheduler import Scheduler
    capacity = context_capacity(Tokenizer(), ['a'*2069], 2048, 65536, 65536)
    worker = NS(max_req_len=capacity['context_length']-1, max_new_tokens_limit=None,
                max_total_num_tokens=65536,kv_shard_widening=1,page_size=1,
                sliding_window_size=None,chunked_prefill_size=-1,
                token_to_kv_pool_allocator=NS(max_new_tokens_for_memory=lambda n,m,**kw:m))
    request = NS(origin_input_ids=[1]*2071,
                 sampling_params=NS(max_new_tokens=2048,min_new_tokens=2048))
    Scheduler.init_req_max_new_tokens(worker,request)
    assert request.sampling_params.max_new_tokens == 2048
    worker.max_req_len = 4119-1
    Scheduler.init_req_max_new_tokens(worker,request)
    assert request.sampling_params.max_new_tokens == 2046


def test_gate_reserves_full_remaining_budget_after_scheduler_decay(monkeypatch):
    from sglang.srt.managers.scheduler_components import new_token_ratio_tracker as tracker
    from sglang.srt.managers.schedule_policy import PrefillAdder
    for key in ('SGLANG_INIT_NEW_TOKEN_RATIO','SGLANG_MIN_NEW_TOKEN_RATIO_FACTOR',
                'SGLANG_CLIP_MAX_NEW_TOKENS_ESTIMATION'):
        monkeypatch.setenv(key, '0.1')
    settings = configure_token_exit_serving(8192)
    assert settings['SGLANG_CLIP_MAX_NEW_TOKENS_ESTIMATION'] == '8192'
    monkeypatch.setattr(tracker, 'get_schedule', lambda: NS(schedule_conservativeness=1.))
    ratio = tracker.NewTokenRatioTracker.from_config()
    for _ in range(10000):
        ratio.decay_step()
    req = NS(sampling_params=NS(max_new_tokens=2048), output_ids=[1]*1000)
    adder = NS(new_token_ratio=ratio.current)
    assert PrefillAdder._get_running_request_total_token_offset(adder, req) == 1048


def test_retracted_trace_is_rejected_even_if_row_count_happens_to_match():
    info = dict(num_retractions=1, completion_tokens=2, hidden_states=[[[4.,-.1]], [3.,-.2]])
    with pytest.raises(ValueError, match='zero KV retractions'):
        parse_generation_trace(info, 2)
    info['num_retractions'] = 0
    assert parse_generation_trace(info, 2) == ([4,3],[-.1,-.2])
    info.pop('num_retractions')
    with pytest.raises(ValueError, match='zero KV retractions'):
        parse_generation_trace(info, 2)


def test_reprefill_trace_cannot_be_flattened_into_valid_gate_actions():
    # Ordinary decode rows followed by a second dense prefill block.
    with pytest.raises(ValueError, match='without retraction'):
        parse_trace([[[4.,-.1]], [3.,-.2], [[0.,0.],[4.,-.3]]], 3)


def test_serving_repair_requires_explicit_upgrade_and_preserves_training():
    before = dict(arm='token_exit', max_new_tokens=2048, total_steps=140, beta=0.)
    normalized = validate_resume_protocol(before, before)
    assert normalized is None
    after = dict(before, serving=dict(context_length=4119,token_exit_kv_reservation={'full':True}))
    with pytest.raises(ValueError, match='protocol'):
        validate_resume_protocol(before, after)
    change = validate_resume_protocol(before, after, allow_serving_upgrade=True)
    assert change['previous']['context_length'] == 4096
    assert change['previous']['token_exit_kv_reservation'] == 'adaptive'
    with pytest.raises(ValueError, match='protocol'):
        validate_resume_protocol(before, dict(after, max_new_tokens=2025), allow_serving_upgrade=True)
