from types import SimpleNamespace

import pytest

from looprl.slime_hooks import priced_rewards


def sample(reward, group, depth):
    return SimpleNamespace(metadata={"looprl": {"depths": [depth] * 2, "beta": .1, "max_new_tokens": 2}},
                           response_length=2, group_index=group, get_reward_value=lambda args: reward)


def test_reward_hook_uses_explicit_groups_and_global_outcome_std():
    samples = [sample(1., 3, 2), sample(0., 8, 2), sample(1., 3, 4), sample(0., 8, 4)]
    raw, advantage = priced_rewards(None, samples)
    assert raw == [1., 0., 1., 0.]
    assert advantage == pytest.approx([.05, .05, -.05, -.05], abs=1e-6)


def test_reward_hook_rejects_missing_trace():
    s = sample(1., 0, 4)
    s.metadata["looprl"]["depths"] = [4]
    with pytest.raises(ValueError):
        priced_rewards(None, [s])
