"""Native-independent index writer shared by the offline FT conversion tool."""

import json
import os
import stat
import tempfile
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
    head_sources = set()
    total_size = 0
    for name in files:
        path = os.path.join(output_path, name)
        total_size += os.path.getsize(path)
        with safe_open(path, framework="pt") as checkpoint:
            for key in checkpoint.keys():
                if key.endswith(".global.lm_head"):
                    head_sources.add((checkpoint.metadata() or {}).get("ft_lm_head_source"))
                if key in weight_map:
                    raise ValueError(
                        f"Duplicate FT tensor {key} in {weight_map[key]} and {name}"
                    )
                weight_map[key] = name
    if not weight_map:
        raise ValueError("FT export contains no tensors")
    preshuffled = set()
    for sidecar in sorted(os.listdir(output_path)):
        if not sidecar.endswith("-preshuffled.json"):
            continue
        with open(os.path.join(output_path, sidecar), encoding="utf-8") as stream:
            keys = json.load(stream)["moe_weights_preshuffled"]
        prefix = sidecar.removesuffix("preshuffled.json")
        if not isinstance(keys, list) or any(
            not isinstance(key, str)
            or key not in weight_map
            or not weight_map[key].startswith(prefix)
            for key in keys
        ):
            raise ValueError(f"Invalid FT preshuffle metadata in {sidecar}")
        preshuffled.update(keys)
    if expected_ranks is not None:
        present = {key.partition(".")[0] for key in weight_map}
        expected = set(expected_ranks)
        if len(expected) != len(expected_ranks) or present != expected:
            raise ValueError(f"Expected FT ranks {sorted(expected)}, got {sorted(present)}")
    index = {
        "metadata": {"total_size": total_size},
        "weight_map": weight_map,
        "is_ft_style_weight": True,
        "moe_weights_preshuffled": sorted(preshuffled),
        # These are the coordinates consumed by the FT readers. The other
        # conversion options belong to the process, not to the published model.
        "__env__params__": {
            key: env_params[key]
            for key in ("TP_SIZE", "DP_SIZE", "EP_SIZE", "PP_SIZE")
            if key in env_params
        },
    }
    if module_metadata is not None:
        index["module_export_metadata"] = dict(module_metadata)
    if head_sources and head_sources != {None}:
        if len(head_sources) != 1 or not head_sources <= {"embedding", "lm_head"}:
            raise ValueError("FT LM-head provenance is invalid or disagrees between ranks")
        index["ft_lm_head_source"] = head_sources.pop()
    # Readers use the presence of the index as their signal to open it.
    # Stage on the same filesystem so they see either complete generation.
    filename = "model.safetensors.index.json"
    destination = os.path.join(output_path, filename)
    with tempfile.TemporaryDirectory(dir=output_path) as staging:
        staged = os.path.join(staging, filename)
        with open(staged, "w", encoding="utf-8") as output:
            json.dump(index, output, indent=4, ensure_ascii=False)
        try:
            previous_mode = stat.S_IMODE(os.stat(destination).st_mode)
        except FileNotFoundError:
            pass
        else:
            os.chmod(staged, previous_mode)
        os.replace(staged, destination)
