# SPDX-License-Identifier: Apache-2.0
"""Checkpoint inspection, bounded row loading and shard publication."""
from __future__ import annotations

import base64
import hashlib
import json
import logging
import os
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from .manifest import PAIRED_SHARD_SCHEMA, ExpectedLayout, Limits
from .placement import FabricBacking, PlacementPlan
from .storage import BootstrapError, _load_native

_MAX_JSON_BYTES = 32 << 20


class CheckpointError(ValueError):
    def __init__(self, code):
        self.code = code
        super().__init__(code)


@dataclass(frozen=True)
class FileSnapshot:
    relative_path: str
    stat_token: tuple[int, ...]
    header_sha256: str | None = None


@dataclass(frozen=True)
class TensorPlan:
    name: str
    relative_path: str
    shape: tuple[int, int]
    data_range: tuple[int, int]


@dataclass(frozen=True)
class LayerPlan:
    layer_id: int
    tensors: tuple[TensorPlan, TensorPlan]


@dataclass(frozen=True)
class CheckpointPlan:
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
    if (
        not isinstance(relative, str)
        or not relative
        or len(relative) > 4096
        or "\\" in relative
        or any(ord(c) < 32 for c in relative)
        or PurePosixPath(relative).is_absolute()
        or any(part in ("", ".", "..") for part in relative.split("/"))
        or (tensor_file and not relative.endswith(".safetensors"))
    ):
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
    return (
        _json(raw),
        8 + length,
        FileSnapshot(relative, before, hashlib.sha256(raw).hexdigest()),
    )


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
    if not callable(getattr(os, "posix_fadvise", None)) or not hasattr(
        os, "POSIX_FADV_DONTNEED"
    ):
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
            os.posix_fadvise(
                source.fileno(), first, last - first, os.POSIX_FADV_DONTNEED
            )
        except OSError as error:
            raise CheckpointError("CACHE_ADVICE_FAILED") from error
        return last
    return start


def inspect_checkpoint(checkpoint_dir, expected_layout: ExpectedLayout):
    """Inspect exact table metadata without allocating or applying host budgets.

    Local padded allocation sizes are checked by ShardPublisher after partitioning.
    safetensors/torch imports occur only during this explicitly requested call.
    """
    from safetensors import safe_open

    if not isinstance(expected_layout, ExpectedLayout):
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
    index_snapshot = FileSnapshot(index_path.name, token)
    weight_map = _json(index_bytes).get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise CheckpointError("INVALID_INDEX")
    for name, relative in weight_map.items():
        if not isinstance(name, str) or not name:
            raise CheckpointError("INVALID_INDEX")
        _parts(relative)
    expected = {}
    for layer, rows in expected_layout.layers:
        for role, columns in (
            ("weight", expected_layout.head_dim),
            ("scale", expected_layout.head_dim // 32),
        ):
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
                if not isinstance(raw, dict) or set(raw) != {
                    "dtype",
                    "shape",
                    "data_offsets",
                }:
                    raise CheckpointError("INVALID_TENSOR_METADATA")
                allowed = (
                    ("F8_E4M3",) if name.endswith(".weight") else ("U8", "F8_E8M0")
                )
                if raw["dtype"] not in allowed:
                    raise CheckpointError("UNSUPPORTED_DTYPE")
                if (
                    not isinstance(raw["shape"], list)
                    or any(type(n) is not int for n in raw["shape"])
                    or tuple(raw["shape"]) != shape
                ):
                    raise CheckpointError("SHAPE_MISMATCH")
                bounds = raw["data_offsets"]
                if (
                    not isinstance(bounds, list)
                    or len(bounds) != 2
                    or any(type(n) is not int for n in bounds)
                    or not 0
                    <= bounds[0]
                    <= bounds[1]
                    <= snapshot.stat_token[2] - data_start
                    or bounds[1] - bounds[0] != shape[0] * shape[1]
                ):
                    raise CheckpointError("INVALID_TENSOR_RANGE")
                view = reader.get_slice(name)
                if tuple(view.get_shape()) != shape or view.get_dtype() != raw["dtype"]:
                    raise CheckpointError("TENSOR_METADATA_MISMATCH")
                tensors[name] = TensorPlan(name, relative, shape, tuple(bounds))
                del view
        _same_snapshot(root, snapshot)
        files.append(snapshot)
    layers = []
    for layer, _ in expected_layout.layers:
        prefix = f"layers.{layer}.engram.embed."
        pair = (tensors[prefix + "weight"], tensors[prefix + "scale"])
        layers.append(LayerPlan(layer, pair))
    _same_snapshot(root, index_snapshot, index=True)
    return CheckpointPlan(index_snapshot, tuple(files), tuple(layers))


DEFAULT_CHUNK_BYTES = 64 << 20
# Allocations must outlive every consumer; the process keeps them here.
_LOGGER = logging.getLogger(__name__)


def _check_deadline(deadline):
    if deadline is not None and time.monotonic() >= deadline:
        raise ValueError("LOCAL_SHARD_PUBLISH_TIMEOUT")


def _reclaim_legacy_tables(deadline):
    """Retire orphaned local-backend tables before allocating their replacement.

    Reuse the local backend's UID namespace, creation lock and reader leases.
    Its nonblocking scan skips live/busy tables; completing it does not imply
    that every old allocation was removed. No online lookup invokes this path.
    """
    from rtp_llm.models_py.modules.dsv41.engram import reclaim_stale_shared_tables

    _check_deadline(deadline)
    started = time.monotonic()
    error_class = "none"
    try:
        reclaim_stale_shared_tables()
    except Exception as error:
        # Cleanup is best effort, as in the local backend. Report only a safe
        # type, never the exception text (which may contain filesystem paths).
        error_class = type(error).__name__[:64]
    try:
        _LOGGER.log(
            logging.INFO if error_class == "none" else logging.WARNING,
            "engram event=legacy_shared_table_cleanup phase=publish status=%s "
            "duration_ms=%.3f error_class=%s",
            "completed" if error_class == "none" else "failed",
            (time.monotonic() - started) * 1000,
            error_class,
        )
    except Exception:
        # A diagnostic handler must not turn optional cleanup into a failure.
        pass
    _check_deadline(deadline)


def load_rows(
    owner,
    root,
    tensor,
    snapshot,
    start,
    end,
    columns,
    *,
    discard_file_cache=False,
    chunk_bytes=DEFAULT_CHUNK_BYTES,
    deadline=None,
):
    """Read exact FP8/scale bytes without creating full-file tensor mappings."""
    if not 0 <= start < end <= tensor.shape[0] or columns != tensor.shape[1]:
        raise ValueError("invalid shard row range")
    if not columns <= chunk_bytes <= DEFAULT_CHUNK_BYTES:
        raise ValueError("chunk must hold a row and fit the native load ceiling")

    _check_deadline(deadline)
    page_size = _cache_page_size(discard_file_cache)
    rows_per_chunk = chunk_bytes // columns
    offset = 0
    with _open_tensor_source(root, snapshot) as (source, data_start):
        consumed = data_start + tensor.data_range[0] + start * columns
        expected_end = data_start + tensor.data_range[0] + end * columns
        if expected_end > data_start + tensor.data_range[1]:
            raise ValueError("shard exceeds source tensor extent")
        source.seek(consumed)
        discarded = consumed
        for row in range(start, end, rows_per_chunk):
            _check_deadline(deadline)
            length = min(rows_per_chunk, end - row) * columns
            payload = _read_exact(source, length)
            _check_deadline(deadline)
            owner.load(payload, offset)
            _check_deadline(deadline)
            offset += length
            consumed += length
            del payload
            if page_size is not None and consumed - discarded >= DEFAULT_CHUNK_BYTES:
                discarded = _discard_consumed_pages(
                    source, discarded, consumed, page_size
                )
        if page_size is not None:
            _discard_consumed_pages(source, discarded, consumed, page_size)
    _check_deadline(deadline)
    if offset != (end - start) * columns:
        raise RuntimeError("shard row range has an unexpected byte count")


def checkpoint_samples(checkpoint, manifest):
    """Read only the reference rows used to verify Fabric mappings."""
    import torch
    from safetensors import safe_open

    root = Path(checkpoint).resolve(strict=True)
    index = json.loads((root / "model.safetensors.index.json").read_text())[
        "weight_map"
    ]
    result = {}
    for tensor in manifest.tensors:
        name, row_count = tensor.name, tensor.shape[0]
        rows = sorted(
            {0, row_count // 2, row_count - 1}
            | {
                row
                for part in tensor.shards
                for row in (part.row_begin, part.row_end - 1)
            }
        )
        relative = Path(index[name])
        if relative.is_absolute() or ".." in relative.parts:
            raise CheckpointError("UNSAFE_CHECKPOINT_PATH")
        path = (root / relative).resolve(strict=True)
        if not path.is_relative_to(root):
            raise CheckpointError("UNSAFE_CHECKPOINT_PATH")
        with safe_open(path, framework="pt", device="cpu") as source:
            view = source.get_slice(name)
            payload = b"".join(
                view[row : row + 1].contiguous().view(torch.uint8).numpy().tobytes()
                for row in rows
            )
        result[name] = dict(
            row_ids=rows, sha256="sha256:" + hashlib.sha256(payload).hexdigest()
        )
    return result


def _expected_layout(layout, model_revision, hash_abi):
    from rtp_llm.models_py.modules.dsv41.engram_distributed.manifest import (
        ExpectedLayout,
    )

    return ExpectedLayout(
        layers=tuple(zip(layout.layer_ids, layout.num_embeddings)),
        head_dim=layout.head_dim,
        model_revision=model_revision,
        hash_abi=hash_abi,
    )


def hash_state_digest(hash_state):
    """Bind the actual tokenizer mapping and all lookup hash constants.

    Algorithm semantics are versioned independently of file names. Constants
    remain on CPU in NgramHashState; computing this digest must not pull a model
    tensor from the GPU or require remote checkpoint table files.
    """
    import torch

    layout = hash_state.layout
    metadata = dict(
        algorithm="dsv41-xor-int64-boundary-cumsum-v1",
        layer_ids=list(layout.layer_ids),
        num_embeddings=list(layout.num_embeddings),
        head_dim=layout.head_dim,
        max_ngram_size=layout.max_ngram_size,
        n_heads=layout.n_heads,
        compressed_vocab_size=layout.compressed_vocab_size,
        pad_token_id=layout.pad_token_id,
        compressed_pad_id=hash_state.pad_id,
    )
    digest = hashlib.sha256(
        json.dumps(
            metadata, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("ascii")
    )
    for name, tensor in (
        ("token_map", hash_state.token_map),
        ("multipliers", hash_state.multipliers),
        ("primes", hash_state.primes),
        ("offsets", layout.offsets),
    ):
        if tensor.device.type != "cpu" or tensor.dtype != torch.int64:
            raise BootstrapError("HASH_CONSTANTS_MUST_BE_CPU_INT64")
        descriptor = json.dumps(
            [name, list(tensor.shape)], separators=(",", ":")
        ).encode("ascii")
        digest.update(len(descriptor).to_bytes(8, "little"))
        digest.update(descriptor)
        digest.update(
            tensor.detach().contiguous().numpy().astype("<i8", copy=False).tobytes()
        )
    return "sha256:" + digest.hexdigest()


class ShardPublisher:
    """Own local Fabric backing until explicit drained retirement or worker exit."""

    def __init__(self):
        self._owners = {}

    def publish(
        self,
        checkpoint,
        model_revision,
        host_id,
        generation,
        shard,
        shards,
        *,
        device=0,
        numa=0,
        memory_kind="host",
        gpu_devices=None,
        gpu_reserve_bytes=32 << 30,
        max_allocation_bytes=None,
        budget_bytes=None,
        release_input_page_cache=False,
        deadline=None,
    ):
        """Allocate one shard here, fill it from ``checkpoint`` and describe it.

        ``deadline`` is checked before/after every I/O and load chunk. Blocking
        OS/driver calls cannot be interrupted here; a process watchdog is still
        needed for a genuinely stuck syscall. No extra work enters online lookup.
        """
        _check_deadline(deadline)
        from rtp_llm.models_py.modules.dsv41.engram import EngramLayout, NgramHashState
        from rtp_llm.models_py.modules.dsv41.engram_distributed.config import (
            HARD_MAX_ALLOCATION_BYTES,
            HARD_MAX_TOTAL_ALLOCATION_BYTES,
            validate_capacity,
        )

        if type(shard) is not int or type(shards) is not int or not 0 <= shard < shards:
            raise ValueError("invalid shard index")
        checkpoint = Path(checkpoint).resolve(strict=True)
        config = json.loads((checkpoint / "config.json").read_text())
        config = {**config, **config.get("text_config", {})}
        layout = EngramLayout(config)
        hash_abi = hash_state_digest(NgramHashState(layout, str(checkpoint)))
        native = _load_native()
        backing = FabricBacking.resolve(
            native,
            layout.layer_ids,
            memory_kind=memory_kind,
            device=device,
            numa=numa,
            gpu_devices=gpu_devices,
        )
        granularity = backing.granularity
        if max_allocation_bytes is None:
            max_allocation_bytes = HARD_MAX_ALLOCATION_BYTES
        if budget_bytes is None:
            budget_bytes = HARD_MAX_TOTAL_ALLOCATION_BYTES
        validate_capacity(max_allocation_bytes, budget_bytes)
        load_chunk = min(DEFAULT_CHUNK_BYTES, native.max_load_chunk_bytes)
        plan = inspect_checkpoint(
            checkpoint, _expected_layout(layout, model_revision, hash_abi)
        )
        snapshots = {snapshot.relative_path: snapshot for snapshot in plan.files}
        physical = PlacementPlan.build(
            plan,
            shard,
            shards,
            granularity,
            {layer: backing.nodes for layer in layout.layer_ids},
        )
        physical.validate_capacity(
            min(max_allocation_bytes, native.max_allocation_bytes), budget_bytes
        )
        backing.validate_capacity(physical, gpu_reserve_bytes)
        _check_deadline(deadline)
        _reclaim_legacy_tables(deadline)
        page_size = _cache_page_size(release_input_page_cache)
        if page_size is not None:
            for snapshot in plan.files:
                with _open_tensor_source(checkpoint, snapshot) as (source, _):
                    _discard_consumed_pages(
                        source, 0, snapshot.stat_token[2], page_size
                    )
        owners, layers, local_bytes = [], [], 0
        try:
            for layer_plan in plan.layers:
                rows, head_dim = layer_plan.tensors[0].shape
                segments = {}
                for tensor in layer_plan.tensors:
                    _check_deadline(deadline)
                    columns = tensor.shape[1]
                    role = tensor.name.rsplit(".", 1)[-1]
                    for allocation in physical.for_tensor(tensor.name):
                        node = allocation.location
                        start, end = allocation.row_begin, allocation.row_end
                        _check_deadline(deadline)
                        allocation_bytes = allocation.size_bytes
                        local_bytes += allocation_bytes
                        target = backing.target_device(allocation)
                        owner = backing.allocate(allocation)
                        owners.append(owner)
                        load_rows(
                            owner,
                            checkpoint,
                            tensor,
                            snapshots[tensor.relative_path],
                            start,
                            end,
                            columns,
                            discard_file_cache=release_input_page_cache,
                            chunk_bytes=load_chunk,
                            deadline=deadline,
                        )
                        segment = segments.setdefault(
                            (start, end, node),
                            dict(row_begin=start, row_end=end, location=node),
                        )
                        segment[role] = dict(
                            handle=base64.b64encode(owner.export_handle()).decode(
                                "ascii"
                            ),
                            allocation_bytes=allocation_bytes,
                        )
                        _LOGGER.info(
                            "engram event=backing_allocation layer=%d tensor=%s shard=%d "
                            "location=%s device=%d row_begin=%d row_end=%d physical_bytes=%d memory=%s",
                            layer_plan.layer_id,
                            role,
                            shard,
                            node,
                            target,
                            start,
                            end,
                            allocation_bytes,
                            memory_kind,
                        )
                layers.append(
                    dict(
                        layer_id=layer_plan.layer_id,
                        rows=rows,
                        head_dim=head_dim,
                        segments=list(segments.values()),
                    )
                )
            if local_bytes != physical.total_bytes:
                raise RuntimeError("NUMA_SPLIT_CHANGED_PHYSICAL_FOOTPRINT")
            _same_snapshot(checkpoint, plan.index, index=True)
        except BaseException as original:
            # A partial shard is useless: hand the memory back before unwinding.
            failed = []
            for owner in reversed(owners):
                try:
                    owner.close()
                except BaseException:
                    failed.append(owner)
            if failed:
                self._owners.setdefault(generation, []).extend(reversed(failed))
                raise RuntimeError("LOCAL_SHARD_CLEANUP_INCOMPLETE") from original
            raise
        document = dict(
            schema_version=PAIRED_SHARD_SCHEMA,
            memory_kind=memory_kind,
            granularity=granularity,
            shard=shard,
            shards=shards,
            generation=generation,
            model_revision=model_revision,
            hash_abi=hash_abi,
            host_id=host_id,
            layers=layers,
        )
        self._owners.setdefault(generation, []).extend(owners)
        return document

    def release(self, generation):
        """Test-only retirement AFTER all local/remote consumers drain and exit.

        Never use as an in-process serving restart: remote graphs may hold pointers.
        Failed native closes retain their references so cleanup can be retried.
        """
        owners = self._owners.pop(generation, [])
        failures, retained = [], []
        for owner in reversed(owners):
            try:
                owner.close()
            except Exception as error:  # native close is best effort at exit
                failures.append(error)
                retained.append(owner)
        if retained:
            self._owners[generation] = list(reversed(retained))
        return failures


shard_publisher = ShardPublisher()
