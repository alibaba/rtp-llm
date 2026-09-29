"""Compare small, named Safetensors payload ranges without loading a model."""

import argparse
import hashlib
import json
from pathlib import Path


KEYS = (
    "language_model.model.layers.0.self_attn.A_log",
    "language_model.model.layers.0.self_attn.f_b_proj.weight",
    "language_model.model.layers.0.mlp.up_proj.weight",
    "language_model.model.layers.1.block_sparse_moe.experts.0.w1.weight_packed",
)


def checksum(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def describe(root):
    config_path = root / "config.json"
    index_path = root / "model.safetensors.index.json"
    index = json.loads(index_path.read_text())
    result = {
        "root": str(root),
        "config_sha256": checksum(config_path),
        "index_sha256": checksum(index_path),
        "shard_count": len(set(index["weight_map"].values())),
        "samples": {},
    }
    for key in KEYS:
        shard = root / index["weight_map"][key]
        with shard.open("rb") as stream:
            header_length = int.from_bytes(stream.read(8), "little")
            header = json.loads(stream.read(header_length))
            tensor = header[key]
            begin, end = tensor["data_offsets"]
            length = min(end - begin, 65536)
            stream.seek(8 + header_length + begin)
            payload = stream.read(length)
            if len(payload) != length:
                raise IOError(f"short read: {shard} {key}")
        result["samples"][key] = {
            "shard": shard.name,
            "dtype": tensor["dtype"],
            "shape": tensor["shape"],
            "sample_bytes": length,
            "sample_sha256": hashlib.sha256(payload).hexdigest(),
        }
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", type=Path, required=True)
    parser.add_argument("--four", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    full, four = describe(args.full), describe(args.four)
    keys = {
        key: full["samples"][key]["sample_sha256"] == four["samples"][key]["sample_sha256"]
        for key in KEYS
    }
    report = {
        "full": full,
        "four_layer": four,
        "sample_equal": keys,
        "all_samples_equal": all(keys.values()),
        "scope": "Only named tensor prefixes were compared; this is not a whole-checkpoint hash.",
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"all_samples_equal": report["all_samples_equal"], "sample_equal": keys}))


if __name__ == "__main__":
    main()
