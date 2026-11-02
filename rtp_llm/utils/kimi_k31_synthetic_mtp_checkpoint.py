"""Build an explicitly untrained DSA draft from a real decoder's exact bytes.

Only the three MTP bridge tensors are synthetic. Target files are read-only;
the output must be a new directory under the caller's explicit fixture root.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import shutil
import struct

from rtp_llm.utils.kimi_k31_fake_checkpoint import fixture_config, _tensor_chunks
from rtp_llm.utils.kimi_k3_mtp_checkpoint import expected_tensors, IGNORED_SUFFIXES, validate_checkpoint

_SIZE = {"BF16": 2, "F32": 4, "U8": 1}
_BRIDGE = {"enorm.weight", "hnorm.weight", "eh_proj.weight"}
_GLOBAL = {
    "embed_tokens.weight": "language_model.model.embed_tokens.weight",
    "shared_head.head.weight": "language_model.lm_head.weight",
    "shared_head.norm.weight": "language_model.model.norm.weight",
}


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as reader:
        for chunk in iter(lambda: reader.read(8 * 1024**2), b""):
            h.update(chunk)
    return h.hexdigest()


def checkpoint_plan(source, layers, seed=3101, source_layer=92):
    root = Path(source).resolve(strict=True)
    original = json.loads((root / "config.json").read_text())
    config = fixture_config(original, layers, seed, mtp=True)
    if source_layer + 1 not in original["text_config"]["dsa_config"]["dsa_layers"]:
        raise ValueError("synthetic draft source must be a real DSA decoder")
    prefix = f"language_model.model.layers.{layers}."
    origin = f"language_model.model.layers.{source_layer}."
    expected = expected_tensors(config["text_config"])
    weight_map = json.loads((root / "model.safetensors.index.json").read_text())["weight_map"]
    headers, stats, items, used = {}, {}, [], set()

    def source_item(destination, name, shape=None, dtype=None):
        if name not in weight_map:
            raise ValueError(f"missing real checkpoint parameter: {name}")
        path = (root / weight_map[name]).resolve(strict=True)
        path.relative_to(root)
        if path not in headers:
            stat = path.stat()
            with path.open("rb") as reader:
                length = struct.unpack("<Q", reader.read(8))[0]
                if not 0 < length <= min(stat.st_size - 8, 64 * 1024**2):
                    raise ValueError(f"invalid safetensors header: {path}")
                header = json.loads(reader.read(length))
            headers[path] = (length + 8, header)
            stats[str(path)] = {"bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}
        base, header = headers[path]
        entry = header[name]
        if shape is not None and (list(shape) != entry["shape"] or dtype != entry["dtype"]):
            raise ValueError(f"real checkpoint parameter contract mismatch: {name}")
        size = math.prod(entry["shape"]) * _SIZE[entry["dtype"]]
        begin, end = entry["data_offsets"]
        if begin < 0 or end - begin != size or base + end > stats[str(path)]["bytes"]:
            raise ValueError(f"invalid or truncated real parameter: {name}")
        used.add(name)
        return dict(name=destination, shape=entry["shape"], dtype=entry["dtype"],
                    bytes=size, source_name=name, source_file=str(path), source_offset=base + begin)

    for name, (shape, dtype) in sorted(expected.items()):
        suffix = name[len(prefix):]
        if suffix in _BRIDGE:
            items.append(dict(name=name, shape=list(shape), dtype=dtype,
                              bytes=math.prod(shape) * _SIZE[dtype], generated=True))
        else:
            items.append(source_item(name, _GLOBAL.get(suffix, origin + suffix), shape, dtype))
    extras = sorted(name for name in weight_map if name.startswith(origin) and name not in used)
    if any(name[len(origin):] not in IGNORED_SUFFIXES for name in extras):
        raise ValueError(f"unmapped decoder parameters: {extras}")
    for name in extras:
        items.append(source_item(prefix + name[len(origin):], name))
    config["rtp_llm_fixture"].update(
        synthetic=True, native=False, trained=False, source_decoder_layer=source_layer,
        checkpoint_nextn_layer=layers, runtime_draft_layer=0,
        accuracy_scope="Synthetic draft state validation only; real target remains unchanged.",
        bridge_rule="Existing SHA256-derived PCG64 per-tensor seed; norm ones; uniform0.05/sqrt(fan_in)",
    )
    return dict(config=config, items=sorted(items, key=lambda x: x["name"]), source_stats=stats,
                source_config_sha256=sha256(root / "config.json"),
                source_index_sha256=sha256(root / "model.safetensors.index.json"),
                source=str(root), payload_bytes=sum(x["bytes"] for x in items))


def write_checkpoint(source, output, layers, *, fixture_root, seed=3101, source_layer=92,
                     shard_bytes=1024**3, copy_bytes=8 * 1024**2):
    if shard_bytes < 1 or copy_bytes < 1:
        raise ValueError("stream capacities must be positive")
    plan = checkpoint_plan(source, layers, seed, source_layer)
    root = Path(fixture_root).resolve(strict=True)
    dest = Path(output).resolve()
    protected = Path("/3fs-data/3fs/mtp_test").resolve()
    if root not in dest.parents or dest == protected or protected in dest.parents:
        raise ValueError("output must be below the explicit permitted fixture root")
    src = Path(plan["source"])
    if dest == src or src in dest.parents or dest in src.parents:
        raise ValueError("synthetic fixture must be separate from the real checkpoint")
    if dest.exists():
        raise FileExistsError(dest)
    if shutil.disk_usage(root).free < plan["payload_bytes"] + 1024**3:
        raise ValueError("insufficient fixture disk capacity")
    groups, group, size = [], [], 0
    for item in plan["items"]:
        if group and size + item["bytes"] > shard_bytes:
            groups.append(group)
            group, size = [], 0
        group.append(item)
        size += item["bytes"]
    if group:
        groups.append(group)
    dest.mkdir()
    (dest / "INCOMPLETE").write_text("Synthetic MTP copy in progress; do not load.\n")
    weight_map, shards, tensor_sources = {}, {}, []
    for number, group in enumerate(groups, 1):
        name = f"model-{number:05d}-of-{len(groups):05d}.safetensors"
        header = {"__metadata__": {"format": "pt", "fixture": "synthetic_dsa_mtp", "trained": "false"}}
        offset = 0
        for item in group:
            header[item["name"]] = dict(shape=item["shape"], dtype=item["dtype"],
                                        data_offsets=[offset, offset + item["bytes"]])
            offset += item["bytes"]
            weight_map[item["name"]] = name
        encoded = json.dumps(header, separators=(",", ":")).encode()
        encoded += b" " * (-len(encoded) % 8)
        digest = hashlib.sha256()
        with (dest / name).open("xb") as writer:
            for chunk in (struct.pack("<Q", len(encoded)), encoded):
                writer.write(chunk)
                digest.update(chunk)
            for item in group:
                per_tensor, written = hashlib.sha256(), 0
                if item.get("generated"):
                    chunks = _tensor_chunks(item["name"], item, seed)
                    reader = None
                else:
                    reader = Path(item["source_file"]).open("rb")
                    reader.seek(item["source_offset"])
                    def source_chunks():
                        remaining = item["bytes"]
                        while remaining:
                            chunk = reader.read(min(copy_bytes, remaining))
                            if not chunk:
                                raise ValueError("source payload ended during copy")
                            remaining -= len(chunk)
                            yield chunk
                    chunks = source_chunks()
                try:
                    for chunk in chunks:
                        writer.write(chunk)
                        digest.update(chunk)
                        per_tensor.update(chunk)
                        written += len(chunk)
                finally:
                    if reader is not None:
                        reader.close()
                if written != item["bytes"]:
                    raise ValueError(f"tensor payload length mismatch: {item['name']}")
                tensor_sources.append(dict(item, payload_sha256=per_tensor.hexdigest()))
        shards[name] = digest.hexdigest()
        if sha256(dest / name) != shards[name]:
            raise ValueError("written fixture shard failed SHA256 verification")
        print(json.dumps(dict(shard=number, shards=len(groups), bytes=offset)), flush=True)
    for name, stat in plan["source_stats"].items():
        current = Path(name).stat()
        if current.st_size != stat["bytes"] or current.st_mtime_ns != stat["mtime_ns"]:
            raise ValueError("real checkpoint changed during extraction")
    if sha256(src / "config.json") != plan["source_config_sha256"] or sha256(src / "model.safetensors.index.json") != plan["source_index_sha256"]:
        raise ValueError("real config or weight index changed during extraction")
    (dest / "config.json").write_text(json.dumps(plan["config"], indent=2) + "\n")
    (dest / "model.safetensors.index.json").write_text(json.dumps(
        dict(metadata=dict(total_size=plan["payload_bytes"]), weight_map=weight_map), indent=2) + "\n")
    auxiliary = {}
    for name in ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "generation_config.json"):
        if (src / name).is_file():
            shutil.copyfile(src / name, dest / name)
            auxiliary[name] = sha256(dest / name)
    checked = validate_checkpoint(dest)
    receipt = dict(plan["config"]["rtp_llm_fixture"], source=plan["source"], seed=seed,
                   source_config_sha256=plan["source_config_sha256"],
                   source_index_sha256=plan["source_index_sha256"], source_stats=plan["source_stats"],
                   tensors=tensor_sources, shards=shards, auxiliary=auxiliary, validation=checked,
                   load_method="fastsafetensors", complete=True)
    (dest / "fixture.json").write_text(json.dumps(receipt, indent=2) + "\n")
    (dest / "INCOMPLETE").unlink()
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--fixture-root", required=True)
    parser.add_argument("--layers", type=int, choices=(4, 93), required=True)
    parser.add_argument("--source-layer", type=int, default=92)
    parser.add_argument("--seed", type=int, default=3101)
    args = parser.parse_args()
    write_checkpoint(args.source, args.output, args.layers, fixture_root=args.fixture_root,
                     source_layer=args.source_layer, seed=args.seed)


if __name__ == "__main__":
    main()
