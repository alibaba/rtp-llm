"""Inventory V4.1 safetensors headers without allocating tensor payloads.

Run with ``python -m rtp_llm.tools.checkpoint_inventory CHECKPOINT --output FILE``.
Header hashes identify metadata, not tensor contents; payloads are not hashed.
"""

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

from rtp_llm.config.dsv41_config import V41Config, dsv41_pp_partition
from rtp_llm.utils.safetensors_header import (
    read_safetensors_header,
    unique_json_object as _unique_object,
)

_LAYER = re.compile(r"layers\.(\d+)\.")
_EXPERT = re.compile(r"layers\.(\d+)\.ffn\.experts\.(\d+)\.")
_ENGRAM_TABLE = re.compile(r"layers\.\d+\.engram\.embed\.(weight|scale)$")


def tensor_category(name: str) -> str:
    if name.startswith("mtp."):
        return "draft"
    if name.startswith(("vision.", "aligner.", "image_")):
        return "vision"
    if _ENGRAM_TABLE.fullmatch(name):
        return "engram_host"
    if _EXPERT.match(name):
        return "target_expert"
    if _LAYER.match(name) or name in ("embed.weight", "head.weight", "norm.weight"):
        return "target_other"
    raise ValueError(f"unrecognized V4.1 tensor: {name}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as reader:
        for data in iter(lambda: reader.read(1 << 20), b""):
            digest.update(data)
    return digest.hexdigest()


def inventory_checkpoint(
    checkpoint: Path, include_tensors: bool = False
) -> dict[str, Any]:
    checkpoint = checkpoint.resolve()
    metadata = V41Config.from_path(checkpoint)
    counts = dsv41_pp_partition(metadata.text["num_hidden_layers"], 2)
    index_path = checkpoint / "model.safetensors.index.json"
    with index_path.open(encoding="utf-8") as reader:
        index = json.load(reader, object_pairs_hook=_unique_object)
    weight_map = index["weight_map"]
    totals = Counter()
    stage_other = [0, 0]
    stage_experts = [Counter(), Counter()]
    decode_experts = Counter()
    shards = []
    seen = set()
    host_layers = set()
    for filename in sorted(set(weight_map.values())):
        if Path(filename).name != filename:
            raise ValueError(
                "V4.1 index must refer to shard filenames in the checkpoint directory"
            )
        tensors, shard = read_safetensors_header(checkpoint / filename)
        shard_totals = Counter()
        for name, entry in tensors.items():
            if name in seen or weight_map.get(name) != filename:
                raise ValueError(f"index/header mismatch for {name} in {filename}")
            seen.add(name)
            size = entry["data_offsets"][1] - entry["data_offsets"][0]
            category = tensor_category(name)
            shard_totals[category] += size
            totals[category] += size
            layer = _LAYER.match(name)
            layer_id = int(layer[1]) if layer else None
            if layer_id is not None and layer_id >= sum(counts):
                raise ValueError(f"target tensor outside configured layers: {name}")
            if category == "engram_host":
                host_layers.add(layer_id)
            if category not in ("target_expert", "target_other"):
                continue
            stage = (
                int(layer_id >= counts[0])
                if layer_id is not None
                else int(name != "embed.weight")
            )
            if category == "target_expert":
                expert_id = int(_EXPERT.match(name)[2])
                if expert_id >= metadata.text["n_routed_experts"]:
                    raise ValueError(f"expert outside configured expert count: {name}")
                stage_experts[stage][expert_id] += size
                decode_experts[expert_id] += size
            else:
                stage_other[stage] += size
        shard["categories"] = dict(shard_totals)
        if include_tensors:
            shard["tensors"] = tensors
        shards.append(shard)
    if seen != set(weight_map):
        raise ValueError(
            f"index tensors missing from headers: {sorted(set(weight_map) - seen)[:8]}"
        )
    total = sum(totals.values())
    if index.get("metadata", {}).get("total_size", total) != total:
        raise ValueError("index metadata.total_size does not match shard tensor bytes")
    if totals["engram_host"] != metadata.engram_host_bytes:
        raise ValueError("Engram table bytes do not match configuration")
    if host_layers != set(metadata.text["engram_layer_ids"]):
        raise ValueError("Engram table layers do not match configuration")

    def rank_bytes(ep_size, expert_bytes, other):
        return [
            other
            + sum(expert_bytes[e] for e in metadata.local_expert_ids(ep_size, rank))
            for rank in range(ep_size)
        ]

    asset_paths = {index_path, checkpoint / "config.json"}
    for pattern in (
        "tokenizer*",
        "special_tokens_map.json",
        "encoding/*.py",
        "inference/*.py",
    ):
        asset_paths.update(p for p in checkpoint.glob(pattern) if p.is_file())
    return {
        "schema_version": 1,
        "checkpoint": str(checkpoint),
        "payload_hashes_computed": False,
        "assets_sha256": {
            str(p.relative_to(checkpoint)): _sha256(p) for p in sorted(asset_paths)
        },
        "tensor_count": len(seen),
        "tensor_bytes": total,
        "categories": dict(totals),
        "stored_weight_bytes_per_rank": {
            "decode_dp8ep8": rank_bytes(8, decode_experts, totals["target_other"]),
            "prefill_cep4pp2": [
                rank_bytes(4, stage_experts[s], stage_other[s]) for s in range(2)
            ],
        },
        "pp_stage_layer_counts": counts,
        "shards": shards,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--include-tensors", action="store_true")
    args = parser.parse_args()
    report = inventory_checkpoint(args.checkpoint, args.include_tensors)
    if args.output:
        args.output.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(
            json.dumps(
                {
                    key: report[key]
                    for key in (
                        "tensor_count",
                        "tensor_bytes",
                        "categories",
                        "stored_weight_bytes_per_rank",
                    )
                },
                indent=2,
            )
        )
    else:
        print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
