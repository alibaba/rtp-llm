"""Validated safetensors metadata without mapping or loading tensor payloads."""

import hashlib
import json
import math
import struct
from pathlib import Path
from typing import Any

_DTYPE_BYTES = {
    "BOOL": 1,
    "I8": 1,
    "U8": 1,
    "F8_E4M3": 1,
    "F8_E5M2": 1,
    "F8_E8M0": 1,
    "I16": 2,
    "U16": 2,
    "F16": 2,
    "BF16": 2,
    "I32": 4,
    "U32": 4,
    "F32": 4,
    "I64": 8,
    "U64": 8,
    "F64": 8,
}


def unique_json_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def read_safetensors_header(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    stat = path.stat()
    with path.open("rb") as reader:
        prefix = reader.read(8)
        if len(prefix) != 8:
            raise ValueError(f"{path.name}: truncated safetensors length")
        size = struct.unpack("<Q", prefix)[0]
        if size < 2 or size > min(100_000_000, stat.st_size - 8):
            raise ValueError(f"{path.name}: invalid safetensors header size {size}")
        raw = reader.read(size)
    header = json.loads(raw, object_pairs_hook=unique_json_object)
    if not isinstance(header, dict):
        raise ValueError(f"{path.name}: header must be a JSON object")
    tensors = {name: entry for name, entry in header.items() if name != "__metadata__"}
    spans = []
    for name, entry in tensors.items():
        shape, dtype, offsets = entry["shape"], entry["dtype"], entry["data_offsets"]
        if not isinstance(shape, list) or any(
            type(n) is not int or n < 0 for n in shape
        ):
            raise ValueError(f"{path.name}:{name}: invalid tensor shape")
        if dtype not in _DTYPE_BYTES:
            raise ValueError(f"{path.name}:{name}: unsupported dtype {dtype}")
        if (
            not isinstance(offsets, list)
            or len(offsets) != 2
            or any(type(n) is not int or n < 0 for n in offsets)
        ):
            raise ValueError(f"{path.name}:{name}: invalid data offsets")
        start, end = offsets
        if end - start != math.prod(shape) * _DTYPE_BYTES[dtype]:
            raise ValueError(f"{path.name}:{name}: shape/dtype/byte count mismatch")
        spans.append((start, end, name))
    cursor = 0
    for start, end, name in sorted(spans):
        if start != cursor:
            raise ValueError(
                f"{path.name}:{name}: overlapping or noncontiguous data offsets"
            )
        cursor = end
    if cursor != stat.st_size - 8 - size:
        raise ValueError(f"{path.name}: tensor offsets do not cover the file payload")
    return tensors, {
        "name": path.name,
        "file_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "header_bytes": size,
        "header_sha256": hashlib.sha256(raw).hexdigest(),
        "tensor_count": len(tensors),
        "tensor_bytes": cursor,
    }
