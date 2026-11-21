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
