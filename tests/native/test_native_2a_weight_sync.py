from types import SimpleNamespace as NS

import pytest

from slime.backends.sglang_utils.engine_group import RolloutServer
from slime.backends.sglang_utils.shared_weights import updatable_server


def servers():
    return {f"depth{k}": RolloutServer(server_groups=[NS(engines=[f"engine{k}"],
        all_engines=[f"engine{k}"], num_new_engines=1, num_gpus_per_engine=1,
        gpu_offset=k-2, parallel_config=lambda: {"tp_size": 1})]) for k in (2, 3, 4)}


def test_all_depth_engines_and_reconnect_markers_share_update_group():
    original = servers()
    original["reference"] = RolloutServer(server_groups=[], update_weights=False)
    shared = updatable_server(NS(ouro_shared_depth_engines=True, ouro_arm="stratified"), original)
    assert shared.engines == ["engine2", "engine3", "engine4"]
    assert shared.engine_gpu_counts == [1, 1, 1]
    assert shared.engine_gpu_offsets == [0, 1, 2]
    assert shared.engine_parallel_configs == [{"tp_size": 1}]*3
    assert shared.num_new_engines == 3
    shared.num_new_engines = 0
    assert all(server.num_new_engines == 0 for server in original.values())


def test_ordinary_multi_model_selection_remains_unchanged():
    original = servers()
    assert updatable_server(NS(), original) is original["depth2"]


def test_missing_or_frozen_depth_cannot_silently_skip_weight_sync():
    original = servers()
    original["depth3"].update_weights = False
    with pytest.raises(ValueError, match="exactly"):
        updatable_server(NS(ouro_shared_depth_engines=True, ouro_arm="stratified"), original)


@pytest.mark.parametrize("owner", ["serving", "rollout"])
def test_both_serving_owners_select_all_depths(owner):
    if owner == "serving":
        from slime.ray.serving import ServingCluster
        cls = ServingCluster.__ray_metadata__.modified_class
        select = cls._updatable_server
    else:
        from slime.ray.rollout import RolloutManager
        cls = RolloutManager.__ray_metadata__.modified_class
        select = cls._get_updatable_server
    holder = NS(args=NS(ouro_shared_depth_engines=True, ouro_arm="learned"), servers=servers())
    assert select(holder).engines == ["engine2", "engine3", "engine4"]


def test_first_nccl_broadcast_restores_actor_after_reference_load(monkeypatch):
    import torch
    from slime.backends.megatron_utils import actor as module
    live=torch.tensor([1.])  # Frozen reference currently loaded in the model.
    trained=torch.tensor([2.])
    holder=NS(args=NS(debug_train_only=False,debug_rollout_only=False,
                     rollout_external=True,use_fault_tolerance=False,offload_train=False),
              _active_model_tag='ref',rollout_manager=NS(get_updatable_engines_and_lock=
                  NS(remote=lambda: (['engine'],None,0,[1],[0],[{}]))))
    broadcasts=[]
    def switch(tag):
        assert tag=='actor';live.copy_(trained);holder._active_model_tag=tag
    holder._switch_model=switch
    holder.weight_updater=NS(update_weights=lambda:broadcasts.append(live.clone()))
    monkeypatch.setattr(module.ray,'get',lambda value:value)
    monkeypatch.setattr(module,'print_memory',lambda *args:None)
    update=module.MegatronTrainRayActor.update_weights.__wrapped__
    update(holder)
    assert torch.equal(broadcasts[0],trained)
    # The steady path must not reload an older backup over new live weights.
    live.fill_(3.);update(holder)
    assert broadcasts[1].item()==3.
