"""Read-only metadata checks and opt-in layout validation for K3 checkpoints.

The smoke uses metadata mode for published shared checkpoints. Explicit preflight
also validates safetensors headers and byte layout without loading tensors.
Source identity and model semantics are checked separately by the smoke worker.
"""

import argparse
import hashlib
import json
import math
import os
import re
import stat
import struct
import subprocess
from pathlib import Path, PurePosixPath

MAX_JSON_BYTES = 128 * 1024**2
DTYPE_BITS = {
    "BOOL": 8,
    "U8": 8,
    "I8": 8,
    "F8_E4M3": 8,
    "F8_E5M2": 8,
    "F8_E4M3FN": 8,
    "F8_E4M3FNUZ": 8,
    "F8_E5M2FNUZ": 8,
    "I16": 16,
    "U16": 16,
    "F16": 16,
    "BF16": 16,
    "I32": 32,
    "U32": 32,
    "F32": 32,
    "I64": 64,
    "U64": 64,
    "F64": 64,
}


def _object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def _json(data):
    return json.loads(data, object_pairs_hook=_object)


def _integer(value):
    return isinstance(value, int) and not isinstance(value, bool)


def _filesystem_type(path):
    values = subprocess.check_output(
        ["findmnt", "-T", str(path), "-n", "-o", "FSTYPE"], text=True
    ).split()
    if len(set(values)) != 1:
        raise ValueError(f"Ambiguous filesystem for {path}")
    return values[0]


def checkpoint_root(
    checkpoint,
    *,
    local_data_root=None,
    allow_hf3fs_root=None,
    allow_checkpoint_root=None,
):
    """Resolve the source only within one explicitly selected storage root."""
    roots = [local_data_root, allow_hf3fs_root, allow_checkpoint_root]
    if sum(root is not None for root in roots) != 1:
        raise ValueError("Select exactly one checkpoint storage root")
    selected = next(root for root in roots if root is not None)
    source, root = Path(checkpoint), Path(selected)
    if not source.is_absolute() or not root.is_absolute():
        raise ValueError("Checkpoint and storage root must be absolute paths")
    source, root = source.resolve(strict=True), root.resolve(strict=True)
    if root == Path(root.anchor) or not root.is_dir() or not source.is_dir():
        raise ValueError("Checkpoint storage root must be a specific directory")
    if not source.is_relative_to(root):
        raise ValueError(f"Checkpoint escapes approved storage root: {source}")
    if local_data_root is not None:
        for path in (root, source):
            if not re.match(r"^/(?:ssd|data[0-9]*)/", str(path)):
                raise ValueError(f"Not a local data path: {path}")
            if _filesystem_type(path) not in {"ext4", "xfs", "btrfs"}:
                raise ValueError(f"Not a local data filesystem: {path}")
    elif allow_hf3fs_root is not None:
        for path in (root, source):
            if _filesystem_type(path) != "fuse.hf3fs":
                raise ValueError(f"Expected fuse.hf3fs for {path}")
    return source, root


def _file(source, root, relative):
    name = PurePosixPath(relative)
    if (
        not isinstance(relative, str)
        or not relative
        or "\\" in relative
        or "\x00" in relative
        or name.is_absolute()
        or ".." in name.parts
        or name.as_posix() != relative
    ):
        raise ValueError(f"Unsafe checkpoint file name: {relative!r}")
    path = (source / relative).resolve(strict=True)
    if not path.is_relative_to(source) or not path.is_relative_to(root):
        raise ValueError(f"Checkpoint file escapes approved directory: {relative}")
    if not stat.S_ISREG(path.stat().st_mode):
        raise ValueError(f"Checkpoint entry is not a regular file: {relative}")
    return path


def _read_json(path):
    if not 0 < path.stat().st_size <= MAX_JSON_BYTES:
        raise ValueError(f"Invalid JSON file size: {path.name}")
    raw = path.read_bytes()
    value = _json(raw)
    if not isinstance(value, dict) or not value:
        raise ValueError(f"Expected nonempty JSON object: {path.name}")
    return value, hashlib.sha256(raw).hexdigest()


def _tensor_layout(name, value, payload_size):
    if not isinstance(value, dict) or set(value) != {"dtype", "shape", "data_offsets"}:
        raise ValueError(f"Invalid tensor description: {name}")
    dtype, shape, offsets = value["dtype"], value["shape"], value["data_offsets"]
    if not isinstance(dtype, str) or dtype not in DTYPE_BITS:
        raise ValueError(f"Unsupported safetensors dtype: {name}: {dtype}")
    if not isinstance(shape, list) or any(not _integer(n) or n < 0 for n in shape):
        raise ValueError(f"Invalid tensor shape: {name}")
    if (
        not isinstance(offsets, list)
        or len(offsets) != 2
        or any(not _integer(n) for n in offsets)
        or not 0 <= offsets[0] <= offsets[1] <= payload_size
    ):
        raise ValueError(f"Invalid tensor data offsets: {name}")
    size = math.prod(shape) * DTYPE_BITS[dtype] // 8
    if offsets[1] - offsets[0] != size:
        raise ValueError(f"Tensor shape/dtype byte size mismatch: {name}")
    return offsets[0], offsets[1]


def read_shard(path):
    """Validate a safetensors shard without reading its weight payload."""
    with path.open("rb") as stream:
        before = os.fstat(stream.fileno())
        raw_length = stream.read(8)
        if len(raw_length) != 8:
            raise ValueError(f"Truncated safetensors header length: {path.name}")
        length = struct.unpack("<Q", raw_length)[0]
        if not 0 < length <= min(MAX_JSON_BYTES, before.st_size - 8):
            raise ValueError(f"Invalid safetensors header length: {path.name}")
        raw_header = stream.read(length)
        if len(raw_header) != length or not raw_header.startswith(b"{"):
            raise ValueError(f"Invalid safetensors header: {path.name}")
        header = _json(raw_header)
        if not isinstance(header, dict):
            raise ValueError(f"Invalid safetensors header object: {path.name}")
        metadata = header.pop("__metadata__", {})
        if not isinstance(metadata, dict) or any(
            not isinstance(k, str) or not isinstance(v, str)
            for k, v in metadata.items()
        ):
            raise ValueError(f"Invalid safetensors metadata: {path.name}")
        if not header or any(not name for name in header):
            raise ValueError(f"Empty safetensors tensor names: {path.name}")
        payload_size = before.st_size - 8 - length
        spans = sorted(
            _tensor_layout(name, value, payload_size) for name, value in header.items()
        )
        end = 0
        for start, stop in spans:
            if start != end:
                raise ValueError(
                    f"Overlapping or gapped safetensors payload: {path.name}"
                )
            end = stop
        if end != payload_size:
            raise ValueError(f"Unindexed safetensors payload bytes: {path.name}")
        after = os.fstat(stream.fileno())
        if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
        ):
            raise ValueError(f"Checkpoint changed during validation: {path.name}")
    return set(header), {
        "file": path.name,
        "file_bytes": before.st_size,
        "tensor_bytes": payload_size,
        "tensor_count": len(header),
        "header_sha256": hashlib.sha256(raw_header).hexdigest(),
    }


def preflight(
    checkpoint,
    *,
    load_method="fastsafetensors",
    expected_shards=None,
    check_layout=True,
    **roots,
):
    if load_method != "fastsafetensors":
        raise ValueError("K3 smoke requires fastsafetensors")
    if expected_shards is not None and (
        not _integer(expected_shards) or expected_shards < 1
    ):
        raise ValueError("Expected shard count must be a positive integer")
    source, root = checkpoint_root(checkpoint, **roots)
    _, config_hash = _read_json(_file(source, root, "config.json"))
    index, index_hash = _read_json(_file(source, root, "model.safetensors.index.json"))
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise ValueError("Checkpoint index has no weight_map")
    shards = {}
    for name, relative in weight_map.items():
        if (
            not name
            or not isinstance(relative, str)
            or not relative.endswith(".safetensors")
        ):
            raise ValueError("Invalid checkpoint weight_map entry")
        shards.setdefault(relative, set()).add(name)
    if expected_shards is not None and len(shards) != expected_shards:
        raise ValueError(f"Expected {expected_shards} shards, found {len(shards)}")
    reports = []
    for relative, expected in sorted(shards.items()):
        path = _file(source, root, relative)
        if check_layout:
            names, report = read_shard(path)
            if names != expected:
                raise ValueError(f"Index/header tensor membership mismatch: {relative}")
        else:
            # Publication owns full validation for shared checkpoints. Only
            # check the manifest and that its files exist; do not reopen headers.
            size = path.stat().st_size
            if size <= 0:
                raise ValueError(f"Empty checkpoint shard: {relative}")
            report = {"file_bytes": size}
        report["file"] = relative
        reports.append(report)
    size = sum(item["tensor_bytes"] for item in reports) if check_layout else None
    file_size = sum(item["file_bytes"] for item in reports)
    metadata = index.get("metadata", {})
    if not isinstance(metadata, dict):
        raise ValueError("Invalid checkpoint index metadata")
    accounting = "not-declared" if check_layout else "not-checked"
    if check_layout and "total_size" in metadata:
        declared = metadata["total_size"]
        if not _integer(declared) or declared not in {size, file_size}:
            raise ValueError(
                "Index total_size disagrees with tensor and shard file sizes"
            )
        # HF indexes count tensor bytes; the native MTP exporter counts the
        # complete shard files. Both must exactly match the validated layout.
        accounting = "tensor-data-bytes" if declared == size else "shard-file-bytes"
    return {
        "passed": True,
        "checkpoint": str(source),
        "approved_root": str(root),
        "load_method": load_method,
        "config_sha256": config_hash,
        "index_sha256": index_hash,
        "shard_count": len(reports),
        "tensor_count": len(weight_map),
        "tensor_bytes": size,
        "file_bytes": file_size,
        "index_total_size_accounting": accounting,
        "shards": reports,
        "validation": (
            "complete-index-and-header-layout" if check_layout else "index-and-files"
        ),
        "weight_payload_checksums_verified": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=["metadata", "preflight"])
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--load-method", default="fastsafetensors")
    parser.add_argument("--expected-shards", type=int)
    roots = parser.add_mutually_exclusive_group(required=True)
    roots.add_argument("--local-data-root")
    roots.add_argument("--allow-hf3fs-root")
    roots.add_argument("--allow-checkpoint-root")
    args = vars(parser.parse_args())
    args["check_layout"] = args.pop("operation") == "preflight"
    print(json.dumps(preflight(**args), indent=2))


if __name__ == "__main__":
    main()
