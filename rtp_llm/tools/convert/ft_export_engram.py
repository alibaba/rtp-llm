"""Copy V4.1's host-only Engram bytes once, outside the rank-local FT index."""

import json
import shutil
from pathlib import Path

from safetensors import safe_open


def export_v41_engram(source, destination):
    source = Path(source).resolve()
    with (source / "config.json").open() as stream:
        raw = json.load(stream)
    layers = {**raw, **raw.get("text_config", {})}.get("engram_layer_ids", [])
    if not layers:
        return
    names = {
        f"layers.{layer}.engram.{suffix}"
        for layer in layers
        for suffix in (
            "embed.weight",
            "embed.scale",
            "wkv.weight",
            "wkv.scale",
            "q_weight",
            "k_weight",
        )
    }
    with (source / "model.safetensors.index.json").open() as stream:
        index = json.load(stream)["weight_map"]
    shards = {}
    for key in sorted(names):
        path = (source / index[key]).resolve()
        if source not in path.parents or path.suffix != ".safetensors":
            raise ValueError(f"Invalid Engram source shard for {key}")
        shards.setdefault(path, []).append(key)

    # The coordinator invokes this once, after all rank workers finish and
    # before publishing the FT index/done marker. Never multiply host tables by TP.
    target = Path(destination) / "v41_engram"
    target.mkdir()
    exported = {}
    for part, (path, keys) in enumerate(shards.items()):
        filename = f"engram-{part:04d}.safetensors"
        with safe_open(str(path), framework="pt") as shard:
            actual = set(shard.keys())
            if not set(keys) <= actual:
                raise ValueError(f"Missing Engram tensors in {path.name}")
        if actual == set(keys):
            shutil.copyfile(path, target / filename)
        else:
            # Mixed shard: rewrite only its small header and copy selected
            # raw frames with bounded host memory; never materialize the table.
            with path.open("rb") as src, (target / filename).open("xb") as dst:
                n = int.from_bytes(src.read(8), "little")
                header = json.loads(src.read(n))
                output, frames, offset = {}, [], 0
                for key in keys:
                    lo, hi = header[key]["data_offsets"]
                    length = hi - lo
                    output[key] = {
                        **header[key],
                        "data_offsets": [offset, offset + length],
                    }
                    frames.append((8 + n + lo, length))
                    offset += length
                data = json.dumps(output, separators=(",", ":")).encode()
                data += b" " * (-len(data) % 8)
                dst.write(len(data).to_bytes(8, "little"))
                dst.write(data)
                for start, length in frames:
                    src.seek(start)
                    while length:
                        chunk = src.read(min(length, 8 << 20))
                        if not chunk:
                            raise OSError(f"Short Engram source frame in {path}")
                        dst.write(chunk)
                        length -= len(chunk)
        exported.update(dict.fromkeys(keys, filename))
    (target / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": exported})
    )
