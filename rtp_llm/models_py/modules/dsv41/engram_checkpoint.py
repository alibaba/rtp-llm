# SPDX-License-Identifier: Apache-2.0
"""Inspect and load immutable local checkpoints into an unpublished EGM owner.

Imports do not initialize CUDA. The caller supplies a native backend and trusted
ExpectedLayout. Checkpoint inputs must remain read-only throughout inspection
and loading; stat tokens detect ordinary replacement/truncation, not all
in-place modification. A same-tick write can preserve ctime; restoring mtime
then leaves the token unchanged. These tokens do not prove content immutability.
The budget covers padded native backing, NOT total RSS:
file page-cache residency and Python/header overhead are separate. Optional
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
from types import MappingProxyType
from uuid import uuid4

from .engram_manifest import (ExpectedLayout, LAYOUT_ABI, Limits, compute_dataset_id,
                             parse_manifest, validate_for_attach)
from .engram_storage import CleanupPendingError
from .engram_capacity import (CapacityError, DEFAULT_MAX_ALLOCATION_BYTES,
                              HARD_MAX_ALLOCATION_BYTES, checked_allocation_bytes,
                              validate_capacity)

MAX_ALLOCATION_BYTES = HARD_MAX_ALLOCATION_BYTES
MAX_LOAD_CHUNK_BYTES = 64 << 20
_MAX_JSON_BYTES = 32 << 20
_DISCARD_INTERVAL_BYTES = 64 << 20


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
    target_offset: int


@dataclass(frozen=True)
class LayerPlan:
    layer_id: int
    logical_bytes: int
    allocation_bytes: int
    tensors: tuple[TensorPlan, TensorPlan]


@dataclass(frozen=True)
class CheckpointPlan:
    root: str
    expected_layout: ExpectedLayout
    allocation_granularity: int
    budget_bytes: int
    max_allocation_bytes: int
    index: FileSnapshot
    files: tuple[FileSnapshot, ...]
    layers: tuple[LayerPlan, ...]

    @property
    def physical_bytes(self):
        return sum(layer.allocation_bytes for layer in self.layers)


def _positive(value, maximum=None):
    if type(value) is not int or value <= 0 or (maximum is not None and value > maximum):
        raise CheckpointError("INVALID_ARGUMENT")
    return value


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


def _document(layout, generation):
    return dict(schema_version=1, model_revision=layout.model_revision,
                generation=generation, layout_abi=LAYOUT_ABI, hash_abi=layout.hash_abi,
                backend="fabric_egm", state="READY", tensors=[])


def _tensor_metadata(tensor, layer, allocation_id, incarnation, granularity, digest):
    rows, columns = tensor.shape
    return dict(name=tensor.name, shape=list(tensor.shape),
                dtype="float8_e4m3fn" if tensor.name.endswith(".weight") else "float8_e8m0fnu",
                storage_dtype="uint8", row_bytes=columns, quant_block=32,
                shards=[dict(shard_id=str(uuid4()), row_begin=0, row_end=rows,
                             logical_bytes=rows * columns, byte_offset=tensor.target_offset,
                             allocation_bytes=layer.allocation_bytes, allocation_granularity=granularity,
                             owner_id="engram-checkpoint-owner", owner_incarnation=incarnation,
                             allocation_id=allocation_id, handle_ref="capability-" + allocation_id,
                             sha256=digest)])


def _validated_document(document, layout):
    document["dataset_id"] = compute_dataset_id(document)
    data = json.dumps(document, sort_keys=True, separators=(",", ":")).encode()
    manifest = parse_manifest(data)
    validate_for_attach(manifest, layout, document["dataset_id"], document["generation"])
    return data, manifest


def inspect_checkpoint(checkpoint_dir, expected_layout: ExpectedLayout, *,
                       allocation_granularity, budget_bytes,
                       max_allocation_bytes=DEFAULT_MAX_ALLOCATION_BYTES):
    """Validate every expected table and budget without constructing any Owner.

    Obtain granularity through native.host_numa_granularity(numa, device).
    load_checkpoint re-queries it and re-inspects this immutable plan before use.
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
    granularity = _positive(allocation_granularity, MAX_ALLOCATION_BYTES)
    try:
        validate_capacity(max_allocation_bytes, budget_bytes)
    except CapacityError:
        raise CheckpointError("INVALID_ARGUMENT") from None
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
                offset = 0 if name.endswith(".weight") else shape[0] * expected_layout.head_dim
                tensors[name] = TensorPlan(name, relative, shape, raw["dtype"], tuple(bounds), offset)
                del view
        _same_snapshot(root, snapshot)
        files.append(snapshot)
    layers = []
    for layer, _ in expected_layout.layers:
        prefix = f"layers.{layer}.engram.embed."
        pair = (tensors[prefix + "weight"], tensors[prefix + "scale"])
        logical = sum(t.shape[0] * t.shape[1] for t in pair)
        try:
            padded = checked_allocation_bytes(logical, granularity, max_allocation_bytes)
        except CapacityError as error:
            raise CheckpointError(error.code) from None
        layers.append(LayerPlan(layer, logical, padded, pair))
    if sum(layer.allocation_bytes for layer in layers) > budget_bytes:
        raise CheckpointError("BUDGET_EXCEEDED")
    _same_snapshot(root, index_snapshot, index=True)
    # Exercise the exact final schema/serialized-size limits before any Owner.
    # UUIDs and digests have fixed lengths; placeholders contain no capabilities.
    draft, incarnation = _document(expected_layout, str(uuid4())), str(uuid4())
    for layer in layers:
        allocation_id = str(uuid4())
        for tensor in layer.tensors:
            draft["tensors"].append(_tensor_metadata(
                tensor, layer, allocation_id, incarnation, granularity, "sha256:" + "0" * 64))
    _validated_document(draft, expected_layout)
    return CheckpointPlan(str(root), expected_layout, granularity, budget_bytes,
                          max_allocation_bytes, index_snapshot, tuple(files), tuple(layers))


class LoadedGeneration:
    """Owns unpublished/ready allocations; close requires all consumers stopped.

    Retain this object until all consumers have drained; this is not distributed
    reference counting. private_handles contains capabilities: transmit through
    an authenticated channel, never log them. No network publication occurs here.
    """
    def __init__(self):
        self._owners = []
        self._handles = {}
        self._state = "LOADING"
        self.manifest_bytes = None
        self.manifest = None
        self.physical_bytes = 0

    def __repr__(self):
        return f"LoadedGeneration(state={self._state}, allocations={len(self._owners)})"

    @property
    def private_handles(self):
        if self._state != "READY":
            raise CheckpointError("GENERATION_NOT_READY")
        return MappingProxyType(self._handles)

    def close(self):
        if self._state == "CLOSED":
            return
        self._state = "CLOSING"
        failures = []
        for owner in reversed(self._owners):
            try:
                owner.close()
            except BaseException as error:
                failures.append(error)
        if failures:
            self._state = "CLEANUP_PENDING"
            raise CleanupPendingError("checkpoint owner cleanup remains pending", self) from failures[0]
        self._owners.clear()
        self._handles.clear()
        self._state = "CLOSED"


def load_checkpoint(plan: CheckpointPlan, *, native, device, numa, chunk_bytes=8 << 20,
                    discard_file_cache=False):
    """Stream exact tensor bytes, hash them, then export the whole generation.

    Raw reads avoid full-file tensor mappings; chunk_bytes still does not bound
    OS page cache. discard_file_cache is an explicit operator assertion that
    plan.files belong to private immutable checkpoint copies. It advises those
    files before allocation and consumed complete pages during loading. It does
    not touch the index, unrelated paths or global caches; callers must not
    enable it for shared source files. Advice failure aborts loading;
    success is only a hint, not a guarantee of eviction or sufficient memory.
    No Owner exists until metadata, native granularity, budgets and limits pass.
    """
    import torch

    if not isinstance(plan, CheckpointPlan) or type(device) is not int or device < 0 or type(numa) is not int or numa < 0:
        raise CheckpointError("INVALID_ARGUMENT")
    page_size = _cache_page_size(discard_file_cache)
    _positive(chunk_bytes, min(MAX_LOAD_CHUNK_BYTES, native.max_load_chunk_bytes))
    if any(t.shape[1] > chunk_bytes for layer in plan.layers for t in layer.tensors):
        raise CheckpointError("CHUNK_SMALLER_THAN_ROW")
    granularity = native.host_numa_granularity(numa, device)
    fresh = inspect_checkpoint(plan.root, plan.expected_layout, allocation_granularity=granularity,
                               budget_bytes=plan.budget_bytes,
                               max_allocation_bytes=plan.max_allocation_bytes)
    if fresh != plan:
        raise CheckpointError("CHECKPOINT_OR_GRANULARITY_CHANGED")
    if any(layer.allocation_bytes > native.max_allocation_bytes for layer in plan.layers):
        raise CheckpointError("ALLOCATION_LIMIT")
    dtypes = {"F8_E4M3": getattr(torch, "float8_e4m3fn", None), "U8": torch.uint8,
              "F8_E8M0": getattr(torch, "float8_e8m0fnu", None)}
    if any(dtypes[t.source_dtype] is None for layer in plan.layers for t in layer.tensors):
        raise CheckpointError("UNSUPPORTED_TORCH_DTYPE")
    root = Path(plan.root)
    if page_size is not None:
        for snapshot in plan.files:
            with _open_tensor_source(root, snapshot) as (source, _):
                _discard_consumed_pages(source, 0, snapshot.stat_token[2], page_size)
    result = LoadedGeneration()
    result.physical_bytes = plan.physical_bytes
    snapshots = {snapshot.relative_path: snapshot for snapshot in plan.files}
    generation, incarnation = str(uuid4()), str(uuid4())
    document = _document(plan.expected_layout, generation)
    allocation_owners = []
    try:
        for layer in plan.layers:
            owner = native.Owner(size=layer.logical_bytes, numa=numa, device=device)
            result._owners.append(owner)
            info = owner.info()
            if (info["allocation_bytes"], info["granularity"]) != (layer.allocation_bytes, granularity):
                raise CheckpointError("NATIVE_ALLOCATION_MISMATCH")
            allocation_id = str(uuid4())
            allocation_owners.append((allocation_id, owner))
            for tensor in layer.tensors:
                digest = hashlib.sha256()
                rows, columns = tensor.shape
                rows_per_chunk = chunk_bytes // columns
                snapshot = snapshots[tensor.relative_path]
                with _open_tensor_source(root, snapshot) as (source, data_start):
                    start = data_start + tensor.data_range[0]
                    end = data_start + tensor.data_range[1]
                    source.seek(start)
                    consumed, discarded = start, start
                    for begin in range(0, rows, rows_per_chunk):
                        length = min(rows_per_chunk, rows - begin) * columns
                        payload = _read_exact(source, length)
                        digest.update(payload)
                        owner.load(payload, offset=tensor.target_offset + begin * columns)
                        del payload
                        consumed += length
                        if page_size is not None and consumed - discarded >= _DISCARD_INTERVAL_BYTES:
                            discarded = _discard_consumed_pages(source, discarded, consumed, page_size)
                    if consumed != end:
                        raise CheckpointError("INVALID_TENSOR_RANGE")
                    if page_size is not None:
                        _discard_consumed_pages(source, discarded, consumed, page_size)
                document["tensors"].append(_tensor_metadata(
                    tensor, layer, allocation_id, incarnation, granularity, "sha256:" + digest.hexdigest()))
        # Final generation-wide consistency check before any handle is exported.
        _same_snapshot(root, plan.index, index=True)
        for snapshot in plan.files:
            _same_snapshot(root, snapshot)
        manifest_bytes, manifest = _validated_document(document, plan.expected_layout)
        for allocation_id, owner in allocation_owners:
            result._handles[allocation_id] = owner.export_handle()
        result.manifest_bytes, result.manifest = manifest_bytes, manifest
        result._state = "READY"
        return result
    except BaseException as original:
        try:
            result.close()
        except BaseException:
            raise CleanupPendingError("checkpoint load failed and cleanup remains pending", result) from original
        raise
