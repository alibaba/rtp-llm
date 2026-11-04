# SPDX-License-Identifier: Apache-2.0
"""Inspect immutable checkpoint metadata for the distributed shard publisher.

Imports do not initialize CUDA. The caller supplies a native backend and trusted
ExpectedLayout. Checkpoint inputs must remain read-only throughout inspection
and loading; stat tokens detect ordinary replacement/truncation, not all
in-place modification. A same-tick write can preserve ctime; restoring mtime
then leaves the token unchanged. These tokens do not prove content immutability.
Allocation and per-host budget checks belong to the shard publisher. Optional
cache advice is exclusively for operator-confirmed private immutable copies;
it never establishes that the OS actually reclaimed memory.
"""
from __future__ import annotations

from dataclasses import dataclass
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path, PurePosixPath

from .engram_manifest import ExpectedLayout, Limits

_MAX_JSON_BYTES = 32 << 20


class CheckpointError(ValueError):
    def __init__(self, code):
        self.code = code
        super().__init__(code)


@dataclass(frozen=True)
class FileSnapshot:
    relative_path: str
    stat_token: tuple[int, ...]
    header_sha256: str


@dataclass(frozen=True)
class TensorPlan:
    name: str
    relative_path: str
    shape: tuple[int, int]
    source_dtype: str
    data_range: tuple[int, int]


@dataclass(frozen=True)
class LayerPlan:
    layer_id: int
    tensors: tuple[TensorPlan, TensorPlan]


@dataclass(frozen=True)
class CheckpointPlan:
    root: str
    expected_layout: ExpectedLayout
    index: FileSnapshot
    files: tuple[FileSnapshot, ...]
    layers: tuple[LayerPlan, ...]

def _json(data):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise CheckpointError("DUPLICATE_JSON_KEY")
            result[key] = value
        return result

    def constant(_):
        raise CheckpointError("INVALID_JSON")

    try:
        value = json.loads(data, object_pairs_hook=pairs, parse_constant=constant)
    except (UnicodeError, ValueError, RecursionError) as error:
        if isinstance(error, CheckpointError):
            raise
        raise CheckpointError("INVALID_JSON") from None
    if not isinstance(value, dict):
        raise CheckpointError("INVALID_JSON")
    return value


def _parts(relative, *, tensor_file=True):
    if (not isinstance(relative, str) or not relative or len(relative) > 4096
            or "\\" in relative or any(ord(c) < 32 for c in relative)
            or PurePosixPath(relative).is_absolute()
            or any(part in ("", ".", "..") for part in relative.split("/"))
            or (tensor_file and not relative.endswith(".safetensors"))):
        raise CheckpointError("UNSAFE_INDEX_PATH")
    return relative.split("/")


def _path(root, relative, *, tensor_file=True):
    path = root
    for part in _parts(relative, tensor_file=tensor_file):
        path = path / part
        if path.is_symlink():
            raise CheckpointError("UNSAFE_INDEX_PATH")
    try:
        path.resolve(strict=True).relative_to(root)
        if not path.is_file():
            raise CheckpointError("NOT_A_FILE")
    except (OSError, ValueError):
        raise CheckpointError("UNREADABLE_CHECKPOINT_FILE") from None
    return path


def _stat_token(s):
    return (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns)


def _stat(path):
    return _stat_token(path.stat())


def _same_snapshot(root, snapshot, *, index=False):
    path = _path(root, snapshot.relative_path, tensor_file=not index)
    if _stat(path) != snapshot.stat_token:
        raise CheckpointError("CHECKPOINT_CHANGED")
    return path


def _header(path, relative):
    before = _stat(path)
    with path.open("rb") as source:
        prefix = source.read(8)
        if len(prefix) != 8:
            raise CheckpointError("INVALID_SAFETENSORS_HEADER")
        length = int.from_bytes(prefix, "little")
        if not 0 < length <= _MAX_JSON_BYTES or length > before[2] - 8:
            raise CheckpointError("INVALID_SAFETENSORS_HEADER")
        raw = source.read(length)
    if len(raw) != length or _stat(path) != before:
        raise CheckpointError("CHECKPOINT_CHANGED")
    return _json(raw), 8 + length, FileSnapshot(relative, before, hashlib.sha256(raw).hexdigest())


def _read_exact(source, size):
    """Read a bounded byte span, handling short reads without accepting EOF."""
    pieces, remaining = [], size
    while remaining:
        piece = source.read(remaining)
        if not piece:
            raise CheckpointError("SHORT_READ")
        pieces.append(piece)
        remaining -= len(piece)
    return b"".join(pieces)


@contextmanager
def _open_tensor_source(root, snapshot):
    """Bind the checked header and streamed bytes to the same file descriptor."""
    path = _same_snapshot(root, snapshot)
    with path.open("rb", buffering=0) as source:
        if _stat_token(os.fstat(source.fileno())) != snapshot.stat_token:
            raise CheckpointError("CHECKPOINT_CHANGED")
        length = int.from_bytes(_read_exact(source, 8), "little")
        if not 0 < length <= _MAX_JSON_BYTES or length > snapshot.stat_token[2] - 8:
            raise CheckpointError("CHECKPOINT_CHANGED")
        header = _read_exact(source, length)
        if hashlib.sha256(header).hexdigest() != snapshot.header_sha256:
            raise CheckpointError("CHECKPOINT_CHANGED")
        del header
        _same_snapshot(root, snapshot)
        yield source, 8 + length
        if _stat_token(os.fstat(source.fileno())) != snapshot.stat_token:
            raise CheckpointError("CHECKPOINT_CHANGED")
    _same_snapshot(root, snapshot)


def _cache_page_size(enabled):
    if type(enabled) is not bool:
        raise CheckpointError("INVALID_ARGUMENT")
    if not enabled:
        return None
    if (not callable(getattr(os, "posix_fadvise", None))
            or not hasattr(os, "POSIX_FADV_DONTNEED")):
        raise CheckpointError("CACHE_ADVICE_UNSUPPORTED")
    try:
        page_size = os.sysconf("SC_PAGESIZE")
    except (AttributeError, OSError, ValueError) as error:
        raise CheckpointError("CACHE_ADVICE_UNSUPPORTED") from error
    if type(page_size) is not int or page_size <= 0:
        raise CheckpointError("CACHE_ADVICE_UNSUPPORTED")
    return page_size


def _discard_consumed_pages(source, start, end, page_size):
    """Advise only complete pages inside [start, end); return the cursor."""
    first = ((start + page_size - 1) // page_size) * page_size
    last = end // page_size * page_size
    if last > first:
        try:
            os.posix_fadvise(source.fileno(), first, last - first, os.POSIX_FADV_DONTNEED)
        except OSError as error:
            raise CheckpointError("CACHE_ADVICE_FAILED") from error
        return last
    return start


def inspect_checkpoint(checkpoint_dir, expected_layout: ExpectedLayout):
    """Inspect exact table metadata without allocating or applying host budgets.

    Local padded allocation sizes are checked by publish_shard after partitioning.
    safetensors/torch imports occur only during this explicitly requested call.
    """
    from safetensors import safe_open

    if not isinstance(expected_layout, ExpectedLayout) or expected_layout.backend != "fabric_egm":
        raise CheckpointError("INVALID_LAYOUT")
    dim = expected_layout.head_dim
    if dim < 32 or dim & (dim - 1):
        raise CheckpointError("UNSUPPORTED_LOOKUP_WIDTH")
    if len(expected_layout.layers) * 2 > Limits().max_tensors:
        raise CheckpointError("MANIFEST_LIMIT")
    root = Path(checkpoint_dir).resolve(strict=True)
    index_path = _path(root, "model.safetensors.index.json", tensor_file=False)
    token = _stat(index_path)
    if token[2] > _MAX_JSON_BYTES:
        raise CheckpointError("INDEX_TOO_LARGE")
    index_bytes = index_path.read_bytes()
    if _stat(index_path) != token:
        raise CheckpointError("CHECKPOINT_CHANGED")
    index_snapshot = FileSnapshot(index_path.name, token, hashlib.sha256(index_bytes).hexdigest())
    weight_map = _json(index_bytes).get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise CheckpointError("INVALID_INDEX")
    for name, relative in weight_map.items():
        if not isinstance(name, str) or not name:
            raise CheckpointError("INVALID_INDEX")
        _parts(relative)
    expected = {}
    for layer, rows in expected_layout.layers:
        for role, columns in (("weight", expected_layout.head_dim),
                              ("scale", expected_layout.head_dim // 32)):
            name = f"layers.{layer}.engram.embed.{role}"
            if name not in weight_map:
                raise CheckpointError("MISSING_TENSOR")
            expected[name] = (rows, columns)
    files, tensors = [], {}
    for relative in sorted({weight_map[name] for name in expected}):
        path = _path(root, relative)
        header, data_start, snapshot = _header(path, relative)
        with safe_open(path, framework="pt", device="cpu") as reader:
            for name, shape in expected.items():
                if weight_map[name] != relative:
                    continue
                raw = header.get(name)
                if not isinstance(raw, dict) or set(raw) != {"dtype", "shape", "data_offsets"}:
                    raise CheckpointError("INVALID_TENSOR_METADATA")
                allowed = ("F8_E4M3",) if name.endswith(".weight") else ("U8", "F8_E8M0")
                if raw["dtype"] not in allowed:
                    raise CheckpointError("UNSUPPORTED_DTYPE")
                if (not isinstance(raw["shape"], list)
                        or any(type(n) is not int for n in raw["shape"])
                        or tuple(raw["shape"]) != shape):
                    raise CheckpointError("SHAPE_MISMATCH")
                bounds = raw["data_offsets"]
                if (not isinstance(bounds, list) or len(bounds) != 2
                        or any(type(n) is not int for n in bounds)
                        or not 0 <= bounds[0] <= bounds[1] <= snapshot.stat_token[2] - data_start
                        or bounds[1] - bounds[0] != shape[0] * shape[1]):
                    raise CheckpointError("INVALID_TENSOR_RANGE")
                view = reader.get_slice(name)
                if tuple(view.get_shape()) != shape or view.get_dtype() != raw["dtype"]:
                    raise CheckpointError("TENSOR_METADATA_MISMATCH")
                tensors[name] = TensorPlan(name, relative, shape, raw["dtype"], tuple(bounds))
                del view
        _same_snapshot(root, snapshot)
        files.append(snapshot)
    layers = []
    for layer, _ in expected_layout.layers:
        prefix = f"layers.{layer}.engram.embed."
        pair = (tensors[prefix + "weight"], tensors[prefix + "scale"])
        layers.append(LayerPlan(layer, pair))
    _same_snapshot(root, index_snapshot, index=True)
    return CheckpointPlan(str(root), expected_layout, index_snapshot, tuple(files), tuple(layers))
