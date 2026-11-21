"""Native-independent index writer shared by the offline FT conversion tool."""

import json
import os
from typing import Collection, Mapping, Optional

from safetensors import safe_open


def generate_ft_index(
    output_path: str,
    env_params: Mapping,
    module_metadata: Optional[Mapping] = None,
    *,
    expected_ranks: Optional[Collection[str]] = None,
):
    """Index already exported rank shards without changing their tensor layout.

    The original FT schema and filename remain unchanged. Optional module
    metadata describes offline parameter ownership, not an inference backend.
    """
    files = sorted(
        name for name in os.listdir(output_path) if name.endswith(".safetensors")
    )
    if not files:
        raise ValueError(f"No safetensors shards in {output_path}")
    weight_map = {}
    total_size = 0
    for name in files:
        path = os.path.join(output_path, name)
        total_size += os.path.getsize(path)
        with safe_open(path, framework="pt") as checkpoint:
            for key in checkpoint.keys():
                if key in weight_map:
                    raise ValueError(
                        f"Duplicate FT tensor {key} in {weight_map[key]} and {name}"
                    )
                weight_map[key] = name
    if not weight_map:
        raise ValueError("FT export contains no tensors")
    if expected_ranks is not None:
        present = {key.partition(".")[0] for key in weight_map}
        expected = set(expected_ranks)
        if len(expected) != len(expected_ranks) or present != expected:
            raise ValueError(f"Expected FT ranks {sorted(expected)}, got {sorted(present)}")
    index = {
        "metadata": {"total_size": total_size},
        "weight_map": weight_map,
        "is_ft_style_weight": True,
        "__env__params__": dict(env_params),
    }
    if module_metadata is not None:
        index["module_export_metadata"] = dict(module_metadata)
    with open(
        os.path.join(output_path, "model.safetensors.index.json"),
        "w",
        encoding="utf-8",
    ) as output:
        json.dump(index, output, indent=4, ensure_ascii=False)
