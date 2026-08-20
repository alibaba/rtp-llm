#!/usr/bin/env python3
"""Truncate DeepSeek-V4-Pro to its first N layers for single-card smoke.

Keeps layers.0..N-1 plus the global tensors (embed/head/norm/hc_head_*),
drops the deeper layers and the mtp.0.* draft model, rewrites config.json
accordingly, and copies the tokenizer/config side files.
"""

import argparse
import json
import os
import shutil
import struct
from collections import defaultdict
from pathlib import Path

import torch  # noqa: F401  (safetensors torch backend)
from safetensors import safe_open
from safetensors.torch import save_file

SRC = Path(os.environ["DSV4_PRO_SRC"])  # full DeepSeek-V4-Pro checkpoint dir
SIDE_FILES = [
    "config.json",
    "configuration.json",
    "generation_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "LICENSE",
]
SHARD_BYTES = 8 << 30  # ~8 GiB per output shard


def keep(name: str, layers: int) -> bool:
    if name.startswith("mtp."):
        return False
    if name.startswith("layers."):
        return int(name.split(".")[1]) < layers
    return True  # embed / head / norm / hc_head_*


def tensor_nbytes(meta: dict) -> int:
    lo, hi = meta["data_offsets"]
    return hi - lo


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--layers", type=int, default=6)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    out = args.out
    out.mkdir(parents=True, exist_ok=True)

    index = json.loads((SRC / "model.safetensors.index.json").read_text())
    weight_map = index["weight_map"]

    # Group kept tensors by source shard so every shard is opened once.
    by_shard: dict[str, list[str]] = defaultdict(list)
    for name, shard in weight_map.items():
        if keep(name, args.layers):
            by_shard[shard].append(name)

    # Per-shard byte sizes from the safetensors headers (for progress only).
    sizes: dict[str, int] = {}
    for shard, names in by_shard.items():
        with open(SRC / shard, "rb") as fh:
            hlen = struct.unpack("<Q", fh.read(8))[0]
            header = json.loads(fh.read(hlen))
        for name in names:
            sizes[name] = tensor_nbytes(header[name])
    total = sum(sizes.values())
    print(
        f"keep {len(sizes)} tensors, {total / 1e9:.1f} GB, from "
        f"{len(by_shard)} source shards"
    )

    new_map: dict[str, str] = {}
    buffer: dict[str, torch.Tensor] = {}
    buffered = 0
    written = 0
    shard_id = 0

    def flush() -> None:
        nonlocal buffer, buffered, shard_id, written
        if not buffer:
            return
        shard_id += 1
        name = f"model-{shard_id:05d}.safetensors"
        save_file(buffer, str(out / name))
        for tensor_name in buffer:
            new_map[tensor_name] = name
        written += buffered
        print(
            f"  wrote {name} ({buffered / 1e9:.1f} GB, "
            f"total {written / 1e9:.1f}/{total / 1e9:.1f} GB)",
            flush=True,
        )
        buffer = {}
        buffered = 0

    for shard in sorted(by_shard):
        with safe_open(str(SRC / shard), framework="pt", device="cpu") as fh:
            for name in sorted(by_shard[shard]):
                buffer[name] = fh.get_tensor(name)
                buffered += sizes[name]
                if buffered >= SHARD_BYTES:
                    flush()
    flush()

    # Rename shards to the canonical model-XXXXX-of-YYYYY scheme.
    final = {}
    for old_id in range(1, shard_id + 1):
        old = f"model-{old_id:05d}.safetensors"
        new = f"model-{old_id:05d}-of-{shard_id:05d}.safetensors"
        (out / old).rename(out / new)
        final[old] = new
    new_map = {k: final[v] for k, v in new_map.items()}

    (out / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {"total_size": total}, "weight_map": new_map}, indent=2)
    )

    config = json.loads((SRC / "config.json").read_text())
    config["num_hidden_layers"] = args.layers
    config["compress_ratios"] = config["compress_ratios"][: args.layers]
    config["num_nextn_predict_layers"] = 0
    (out / "config.json").write_text(json.dumps(config, indent=2))

    for side in SIDE_FILES:
        source = SRC / side
        if side == "config.json" or not source.exists():
            continue
        shutil.copy2(source, out / side)
    print("done:", out)
    print("compress_ratios:", config["compress_ratios"])


if __name__ == "__main__":
    main()
