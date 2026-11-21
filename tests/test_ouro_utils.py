from types import SimpleNamespace

import torch

from looprl.ouro import OuroAdapter, token_log_probs


def test_nucleus_keeps_boundary_token_and_normalizes():
    logits = torch.tensor([[.6, .25, .1, .05]]).log()
    lp = token_log_probs(logits, top_p=.7)
    assert torch.isfinite(lp).tolist() == [[True, True, False, False]]
    torch.testing.assert_close(lp.exp().sum(), torch.tensor(1.))


def test_ragged_fill_preserves_prefix_and_uses_last_exit_with_gradient():
    adapter = object.__new__(OuroAdapter)
    adapter.max_loops = 4
    adapter.model = SimpleNamespace(config=SimpleNamespace(num_hidden_layers=2))
    keys = [torch.full((1, 1, 3, 1), float(i), requires_grad=True) for i in range(8)]
    values = [torch.full((1, 1, 3, 1), float(i + 10), requires_grad=True) for i in range(8)]
    cache = SimpleNamespace(key_cache=list(keys), value_cache=list(values))
    adapter._fill_last(cache, 2, 2)
    for dest in range(4, 8):
        source = 2 + dest % 2
        torch.testing.assert_close(cache.key_cache[dest][..., :2, :], keys[dest][..., :2, :])
        torch.testing.assert_close(cache.key_cache[dest][..., -1:, :], keys[source][..., -1:, :])
    cache.key_cache[6].sum().backward()
    assert keys[2].grad[..., -1, :].item() == 1
    assert keys[6].grad[..., -1, :].item() == 0


def test_ragged_fill_after_real_early_exit_appends_only_one_position():
    adapter = object.__new__(OuroAdapter)
    adapter.max_loops = 4
    adapter.model = SimpleNamespace(config=SimpleNamespace(num_hidden_layers=1))
    cache = SimpleNamespace(key_cache=[torch.ones(1, 1, 4, 1)] + [torch.zeros(1, 1, 3, 1) for _ in range(3)],
                            value_cache=[torch.ones(1, 1, 4, 1)] + [torch.zeros(1, 1, 3, 1) for _ in range(3)])
    adapter._fill_last(cache, 1, 3)
    for loop in (1, 2, 3):
        assert cache.key_cache[loop].shape[-2] == 4
        assert cache.key_cache[loop][0, 0, -1, 0] == 1
        assert cache.key_cache[loop][0, 0, :3, 0].sum() == 0
