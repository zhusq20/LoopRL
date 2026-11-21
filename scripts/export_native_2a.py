"""Export a single-rank native Ouro checkpoint for held-out SGLang evaluation."""
import argparse
import json
from pathlib import Path
import shutil
from types import SimpleNamespace

import torch
from torch.distributed.checkpoint import FileSystemReader, load
from torch.distributed.checkpoint.metadata import TensorStorageMetadata
from safetensors import safe_open
from safetensors.torch import save_file
from slime.backends.megatron_utils.megatron_to_hf.ouro import convert_ouro_to_hf


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--step", type=int, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    torch.set_num_threads(8)
    protocol = json.loads((a.run/"protocol.json").read_text())
    base = Path(protocol["checkpoint"])
    config = json.loads((base/"config.json").read_text())
    args = SimpleNamespace(hidden_size=config["hidden_size"], num_attention_heads=config["num_attention_heads"],
                           num_query_groups=config["num_key_value_heads"], kv_channels=config["head_dim"])
    checkpoint = a.run/"checkpoints"/f"iter_{a.step:07d}"
    reader = FileSystemReader(checkpoint)
    md = reader.read_metadata()
    state = {name: torch.empty(item.size, dtype=item.properties.dtype) for name,item in md.state_dict_metadata.items()
             if isinstance(item, TensorStorageMetadata) and not name.startswith("optimizer.")}
    load(state, storage_reader=reader)
    converted = {}
    for name, tensor in state.items():
        if name.startswith("decoder.layers."):
            assert tensor.shape[0] == config["num_hidden_layers"]
            tensors = [(name.replace("decoder.layers.", f"decoder.layers.{i}.", 1), value) for i,value in enumerate(tensor)]
        else:
            tensors = [(name, tensor)]
        for key, value in tensors:
            for target, weights in convert_ouro_to_hf(args, key, value):
                assert target not in converted
                converted[target] = weights.contiguous()
    with safe_open(str(base/"model.safetensors"), framework="pt") as original:
        expected = set(original.keys())
        if protocol["arm"] == "learned":
            expected |= {"model.sequence_depth_head.weight", "model.sequence_depth_head.bias"}
        assert set(converted) == expected, (set(converted)-expected, expected-set(converted))
        for key in original.keys():
            assert list(converted[key].shape) == original.get_slice(key).get_shape()
    a.output.mkdir(parents=True, exist_ok=False)
    for path in base.iterdir():
        if path.is_file() and path.suffix not in (".safetensors", ".bin", ".pt"):
            shutil.copy2(path, a.output/path.name)
    config["ouro_learned_depth"] = protocol["arm"] == "learned"
    (a.output/"config.json").write_text(json.dumps(config, indent=2)+"\n")
    save_file(converted, str(a.output/"model.safetensors"))
    (a.output/"export.json").write_text(json.dumps(dict(source=str(checkpoint), tensors=len(converted),
        protocol=protocol), indent=2)+"\n")
    print(f"Exported {len(converted)} tensors to {a.output}")


if __name__ == "__main__":
    main()
