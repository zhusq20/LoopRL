"""Exercise real MCore checkpoint metadata with PyTorch Adam's lazy state.

Run in the pinned native environment (without Apex/TE). Full sharded tensor
restore is separately exercised by run_ouro_native_smoke.py --resume.
"""

from types import SimpleNamespace

import pytest
import torch

module = pytest.importorskip("megatron.core.optimizer.distrib_optimizer")
DistributedOptimizer = module.DistributedOptimizer
HAVE_APEX_OR_TE = module.HAVE_APEX_OR_TE


pytestmark = pytest.mark.skipif(HAVE_APEX_OR_TE, reason="PyTorch Adam fallback only")


def wrapper():
    # Metadata serialization needs no process group or CUDA allocation.
    params = [torch.nn.Parameter(torch.ones(2)), torch.nn.Parameter(torch.ones(3))]
    value = object.__new__(DistributedOptimizer)
    value.optimizer = torch.optim.Adam(params, lr=1e-3)
    value.grad_scaler = None
    value.ddp_config = SimpleNamespace(use_megatron_fsdp=False)
    value.config = SimpleNamespace(fp16=False, use_precision_aware_optimizer_no_fp8_or_ds_fp8=False)
    value._optimizer_state_offloader = None
    value.model_param_group_index_map = {p: (0, i) for i, p in enumerate(params)}
    return value, params


def test_fresh_adam_can_build_restore_template():
    value, _ = wrapper()
    saved = value.state_dict()
    assert saved["optimizer"]["param_groups"][0]["step"] == 0
    assert not value.optimizer.state


def test_trained_adam_preserves_step_and_does_not_mutate_state():
    value, params = wrapper()
    for _ in range(2):
        sum(p.square().sum() for p in params).backward()
        value.optimizer.step()
        value.optimizer.zero_grad()
    saved = value.state_dict()
    assert saved["optimizer"]["param_groups"][0]["step"] == 2
    assert all(s["step"].item() == 2 for s in value.optimizer.state.values())
    assert all(torch.count_nonzero(s["exp_avg"]) for s in value.optimizer.state.values())


def test_inconsistent_parameter_steps_still_fail():
    value, params = wrapper()
    sum(p.square().sum() for p in params).backward()
    value.optimizer.step()
    value.optimizer.state[params[1]]["step"].add_(1)
    with pytest.raises(AssertionError, match="Inconsistent PyTorch optimizer steps"):
        value.state_dict()


def test_restored_adam_advances_each_parameter_exactly_once():
    value, params = wrapper()
    sum(p.square().sum() for p in params).backward()
    value.optimizer.step()
    value.optimizer.zero_grad()
    value.load_state_dict(value.state_dict())
    assert value.optimizer.state[params[0]]["step"] is not value.optimizer.state[params[1]]["step"]
    sum(p.square().sum() for p in params).backward()
    value.optimizer.step()
    assert all(s["step"].item() == 2 for s in value.optimizer.state.values())


def test_sharded_parameter_restore_preserves_common_step():
    value, params = wrapper()
    sum(p.square().sum() for p in params).backward()
    value.optimizer.step()
    source = {"param": torch.full_like(params[0], 3),
              "exp_avg": torch.full_like(params[0], 0.7),
              "exp_avg_sq": torch.full_like(params[0], 0.9)}
    value._set_main_param_and_optimizer_states(params[0], source)
    assert value.optimizer.state[params[0]]["step"].item() == 1
    torch.testing.assert_close(params[0], source["param"])
    for key in ["exp_avg", "exp_avg_sq"]:
        torch.testing.assert_close(value.optimizer.state[params[0]][key], source[key])


def test_sharded_restore_rejects_missing_moment():
    value, params = wrapper()
    sum(p.square().sum() for p in params).backward()
    value.optimizer.step()
    with pytest.raises(KeyError, match="exp_avg"):
        value._set_main_param_and_optimizer_states(params[0], {"param": params[0].detach().clone()})
