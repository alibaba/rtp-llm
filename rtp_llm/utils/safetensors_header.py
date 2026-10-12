"""Validated safetensors metadata without mapping or loading tensor payloads."""

import hashlib
import json
import math
import os
import struct
from dataclasses import dataclass
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


def safetensors_file_identity(stat) -> tuple[int, ...]:
    return (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)


@dataclass(frozen=True)
class SafetensorsSpan:
    path: str
    name: str
    shape: tuple[int, ...]
    dtype: str
    offset: int
    size: int
    file_identity: tuple[int, ...]
    header_sha256: str

    def check_source(self):
        if safetensors_file_identity(Path(self.path).stat()) != self.file_identity:
            raise RuntimeError("Checkpoint changed after metadata admission")

    def read_into(self, destination, *, chunk_bytes: int = 8 * 1024 * 1024):
        """Read only this admitted tensor into caller storage, with bounded I/O."""
        if type(chunk_bytes) is not int or chunk_bytes <= 0:
            raise ValueError("Tensor read chunk size must be a positive integer")
        target = memoryview(destination).cast("B")
        if target.readonly or target.nbytes != self.size:
            raise ValueError(
                "Tensor destination must be writable and match the admitted bytes"
            )
        with open(self.path, "rb", buffering=0) as source:
            if (
                safetensors_file_identity(os.fstat(source.fileno()))
                != self.file_identity
            ):
                raise RuntimeError("Checkpoint changed before tensor read")
            source.seek(self.offset)
            copied = 0
            while copied < self.size:
                count = source.readinto(
                    target[copied : min(copied + chunk_bytes, self.size)]
                )
                if not count:
                    raise ValueError("Checkpoint ended during tensor read")
                copied += count
            if (
                safetensors_file_identity(os.fstat(source.fileno()))
                != self.file_identity
            ):
                raise RuntimeError("Checkpoint changed during tensor read")
        self.check_source()


def named_safetensors_spans(
    checkpoint, expected: dict[str, tuple[tuple[int, ...], str]]
) -> dict[str, SafetensorsSpan]:
    """Admit exact names/shapes/dtypes from headers; never read tensor payloads."""
    if not expected:
        return {}
    root = Path(checkpoint).resolve()
    with (root / "model.safetensors.index.json").open() as reader:
        weight_map = json.load(reader, object_pairs_hook=unique_json_object)[
            "weight_map"
        ]
    spans, headers = {}, {}
    for name, (shape, dtype) in expected.items():
        filename = weight_map[name]
        if (
            not isinstance(filename, str)
            or not filename
            or Path(filename).name != filename
        ):
            raise ValueError("Safetensors shard must be in the checkpoint directory")
        path = (root / filename).resolve()
        if path.parent != root:
            raise ValueError("Safetensors shard must be in the checkpoint directory")
        if path not in headers:
            before = safetensors_file_identity(path.stat())
            tensors, header = read_safetensors_header(path)
            if before != safetensors_file_identity(path.stat()):
                raise RuntimeError("Checkpoint changed while reading headers")
            headers[path] = tensors, header, before
        tensors, header, identity = headers[path]
        entry = tensors[name]
        if tuple(entry["shape"]) != tuple(shape) or entry["dtype"] != dtype:
            raise ValueError(f"{name}: shape/dtype disagrees with configuration")
        begin, end = entry["data_offsets"]
        spans[name] = SafetensorsSpan(
            str(path),
            name,
            tuple(shape),
            dtype,
            8 + header["header_bytes"] + begin,
            end - begin,
            identity,
            header["header_sha256"],
        )
    return spans
