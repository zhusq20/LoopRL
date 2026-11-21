"""Exercise checkpoint comparison on tiny real torch distributed checkpoints."""
import json
from pathlib import Path
import subprocess
import sys

import torch
from torch.distributed.checkpoint import save


def test_read_tensor_payload_preserves_names_shapes_and_values(tmp_path):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]/"scripts"))
    from compare_native_2a_resume import read_tensors
    expected = {"decoder.final_layernorm.weight": torch.tensor([1., 2., 3.]),
                "optimizer.state.exp_avg.decoder.final_layernorm.weight": torch.tensor([.1, -.2, .3])}
    save(expected, checkpoint_id=tmp_path/"checkpoint")
    actual = read_tensors(tmp_path/"checkpoint")
    assert actual.keys() == expected.keys()
    for key in expected:
        torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)
