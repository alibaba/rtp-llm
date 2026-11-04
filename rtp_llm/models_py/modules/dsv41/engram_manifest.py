# SPDX-License-Identifier: Apache-2.0
"""S1a metadata validation only; no CUDA, RPC, allocation or tensor imports.

Hashes below authenticate declared metadata against an expected dataset identity;
they do not verify the bytes of an actual allocation. Owner loading and consumer
GPU reads must perform that verification separately.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass, field
from typing import Mapping
from uuid import UUID

LAYOUT_ABI = "engram-fp8-e8m0-v1"
_MAX_INTEGER = (1 << 63) - 1
_SHA = re.compile(r"sha256:[0-9a-f]{64}\Z")
_NAME = re.compile(r"layers\.(0|[1-9][0-9]*)\.engram\.embed\.(weight|scale)\Z")


class ManifestError(ValueError):
    """Stable code and field path; never include untrusted field values."""

    def __init__(self, code: str, path: str):
        self.code, self.path = code, path
        super().__init__(f"{code} at {path}")


@dataclass(frozen=True)
class Limits:
    max_manifest_bytes: int = 1 << 20
    max_tensors: int = 1024
    max_allocation_bytes: int = _MAX_INTEGER
    max_total_allocation_bytes: int = _MAX_INTEGER

    def __post_init__(self):
        for name, value in asdict(self).items():
            if type(value) is not int or not 0 < value <= _MAX_INTEGER:
                raise ManifestError("INVALID_LIMIT", name)


@dataclass(frozen=True)
class ExpectedLayout:
    """Trusted model contract: (layer ID, rows) pairs, not manifest-derived."""

    layers: tuple[tuple[int, int], ...]
    head_dim: int
    model_revision: str
    hash_abi: str
    layout_abi: str = LAYOUT_ABI
    backend: str = "fabric_egm"

    def __post_init__(self):
        try:
            layers = tuple(tuple(pair) for pair in self.layers)
        except TypeError:
            raise ManifestError("INVALID_LAYOUT", "expected.layers") from None
        if not layers or any(len(pair) != 2 for pair in layers):
            raise ManifestError("INVALID_LAYOUT", "expected.layers")
        for layer, rows in layers:
            _integer(layer, "expected.layer", minimum=0)
            _integer(rows, "expected.rows")
        if len({pair[0] for pair in layers}) != len(layers):
            raise ManifestError("DUPLICATE_LAYER", "expected.layers")
        _integer(self.head_dim, "expected.head_dim")
        if self.head_dim % 32 or self.layout_abi != LAYOUT_ABI:
            raise ManifestError("UNSUPPORTED_LAYOUT", "expected")
        if self.backend not in ("local_shm", "fabric_egm"):
            raise ManifestError("UNSUPPORTED_BACKEND", "expected.backend")
        _text(self.model_revision, "expected.model_revision")
        _digest(self.hash_abi, "expected.hash_abi")
        object.__setattr__(self, "layers", tuple(sorted(layers)))


@dataclass(frozen=True)
class AllocationSlice:
    shard_id: str
    row_begin: int
    row_end: int
    logical_bytes: int
    byte_offset: int
    allocation_bytes: int
    allocation_granularity: int
    owner_id: str
    owner_incarnation: str
    allocation_id: str
    handle_ref: str = field(repr=False)
    sha256: str


@dataclass(frozen=True)
class TensorSpec:
    name: str
    shape: tuple[int, int]
    dtype: str
    storage_dtype: str
    row_bytes: int
    quant_block: int
    shards: tuple[AllocationSlice, ...]


@dataclass(frozen=True)
class Manifest:
    schema_version: int
    dataset_id: str
    model_revision: str
    generation: str
    layout_abi: str
    hash_abi: str
    backend: str
    state: str
    tensors: tuple[TensorSpec, ...]


@dataclass(frozen=True)
class ValidatedManifest:
    manifest: Manifest
    physical_bytes: int
    allocation_count: int
    manifest_digest: str


def _integer(value, path, minimum=1, maximum=_MAX_INTEGER):
    if type(value) is not int or not minimum <= value <= maximum:
        raise ManifestError("INVALID_INTEGER", path)
    return value


def _text(value, path):
    if not isinstance(value, str) or not 0 < len(value) <= 256:
        raise ManifestError("INVALID_TEXT", path)
    if any(ord(c) < 32 or 0xD800 <= ord(c) <= 0xDFFF for c in value):
        raise ManifestError("INVALID_TEXT", path)
    return value


def _digest(value, path):
    if not isinstance(value, str) or not _SHA.fullmatch(value):
        raise ManifestError("INVALID_DIGEST", path)
    return value


def _uuid(value, path):
    try:
        if not isinstance(value, str) or str(UUID(value)) != value:
            raise ValueError()
    except (ValueError, AttributeError):
        raise ManifestError("INVALID_UUID", path) from None
    return value


def _object(value, cls, path):
    if not isinstance(value, dict) or set(value) != set(cls.__dataclass_fields__):
        raise ManifestError("INVALID_FIELDS", path)
    return value


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False).encode("ascii")


def _hash(value):
    return "sha256:" + hashlib.sha256(_canonical(value)).hexdigest()


def _metadata(document, limits):
    """Validate structure and ranges, without trusting dataset_id or READY."""
    d = _object(document, Manifest, "manifest")
    _integer(d["schema_version"], "schema_version")
    if d["schema_version"] != 1 or d["layout_abi"] != LAYOUT_ABI:
        raise ManifestError("UNSUPPORTED_SCHEMA", "manifest")
    _digest(d["dataset_id"], "dataset_id")
    _digest(d["hash_abi"], "hash_abi")
    _text(d["model_revision"], "model_revision")
    _uuid(d["generation"], "generation")
    if d["backend"] not in ("local_shm", "fabric_egm"):
        raise ManifestError("UNSUPPORTED_BACKEND", "backend")
    if d["state"] not in ("LOADING", "VALIDATING", "READY", "DRAINING", "RETIRED", "FAILED"):
        raise ManifestError("INVALID_STATE", "state")
    if not isinstance(d["tensors"], list) or not 0 < len(d["tensors"]) <= limits.max_tensors:
        raise ManifestError("TENSOR_LIMIT", "tensors")
    tensors, names, shard_ids, owners, allocations, intervals = [], set(), set(), set(), {}, {}
    for i, raw in enumerate(d["tensors"]):
        path = f"tensors[{i}]"
        t = _object(raw, TensorSpec, path)
        name = _text(t["name"], path + ".name")
        match = _NAME.fullmatch(name)
        if not match:
            raise ManifestError("INVALID_TENSOR_NAME", path + ".name")
        if name in names:
            raise ManifestError("DUPLICATE_TENSOR", path + ".name")
        names.add(name)
        shape = t["shape"]
        if not isinstance(shape, list) or len(shape) != 2:
            raise ManifestError("INVALID_SHAPE", path + ".shape")
        rows, columns = (_integer(x, path + ".shape") for x in shape)
        role = match.group(2)
        dtype = "float8_e4m3fn" if role == "weight" else "float8_e8m0fnu"
        if t["dtype"] != dtype or t["storage_dtype"] != "uint8":
            raise ManifestError("UNSUPPORTED_DTYPE", path)
        if _integer(t["quant_block"], path + ".quant_block") != 32:
            raise ManifestError("UNSUPPORTED_QUANTIZATION", path)
        if role == "weight" and columns % 32:
            raise ManifestError("INVALID_SHAPE", path + ".shape")
        if _integer(t["row_bytes"], path + ".row_bytes") != columns:
            raise ManifestError("INVALID_STRIDE", path + ".row_bytes")
        if rows > _MAX_INTEGER // columns:
            raise ManifestError("SIZE_OVERFLOW", path + ".shape")
        logical_bytes = rows * columns
        shards = t["shards"]
        if not isinstance(shards, list) or not 1 <= len(shards) <= 64:
            raise ManifestError("UNSUPPORTED_SHARDING", path + ".shards")
        multi = len(shards) > 1
        parts, shard_rows = [], None
        for j, raw_shard in enumerate(shards):
            spath = f"{path}.shards[{j}]"
            s = _object(raw_shard, AllocationSlice, spath)
            for key in ("shard_id", "owner_incarnation", "allocation_id"):
                _uuid(s[key], spath + "." + key)
            for key in ("row_begin", "byte_offset"):
                _integer(s[key], spath + "." + key, minimum=0)
            for key in ("row_end", "logical_bytes", "allocation_bytes", "allocation_granularity"):
                _integer(s[key], spath + "." + key)
            _text(s["owner_id"], spath + ".owner_id")
            _text(s["handle_ref"], spath + ".handle_ref")
            _digest(s["sha256"], spath + ".sha256")
            if s["shard_id"] in shard_ids:
                raise ManifestError("DUPLICATE_SHARD", spath)
            shard_ids.add(s["shard_id"])
            if not multi and (s["row_begin"], s["row_end"], s["logical_bytes"]) != (
                    0, rows, logical_bytes):
                raise ManifestError("INCOMPLETE_RANGE", path)
            if not 0 <= s["row_begin"] < s["row_end"] <= rows:
                raise ManifestError("INVALID_RANGE", spath)
            if s["logical_bytes"] != (s["row_end"] - s["row_begin"]) * columns:
                raise ManifestError("INVALID_LOGICAL_BYTES", spath)
            size, offset, granularity = s["allocation_bytes"], s["byte_offset"], s["allocation_granularity"]
            if size > limits.max_allocation_bytes or size % granularity:
                raise ManifestError("INVALID_ALLOCATION_SIZE", spath)
            if multi:
                # Stitched shards are separate allocations mapped at offset 0.
                if offset != 0 or size % columns:
                    raise ManifestError("INVALID_ALLOCATION_SIZE", spath)
                local_rows = size // columns
                if s["row_end"] - s["row_begin"] > local_rows:
                    raise ManifestError("OUT_OF_BOUNDS", spath)
                if shard_rows is None:
                    shard_rows = local_rows
                elif local_rows != shard_rows:
                    raise ManifestError("UNEVEN_SHARDS", spath)
            else:
                # Tensor offsets need not be VMM-granularity aligned: map the
                # entire allocation at map offset 0, then address subranges.
                if offset > size or logical_bytes > size - offset:
                    raise ManifestError("OUT_OF_BOUNDS", spath)
            owner = (s["owner_id"], s["owner_incarnation"])
            owners.add(owner)
            aid = s["allocation_id"]
            metadata = (size, granularity, *owner, s["handle_ref"])
            if aid in allocations and allocations[aid] != metadata:
                raise ManifestError("ALLOCATION_CONFLICT", spath)
            allocations[aid] = metadata
            intervals.setdefault(aid, []).append((offset, offset + s["logical_bytes"]))
            parts.append(s)
        parts.sort(key=lambda s: s["row_begin"])
        if multi:
            cursor = 0
            for s in parts:
                if s["row_begin"] != cursor:
                    raise ManifestError("INCOMPLETE_RANGE", path)
                cursor = s["row_end"]
            if cursor != rows:
                raise ManifestError("INCOMPLETE_RANGE", path)
        tensors.append(TensorSpec(name, tuple(shape), dtype, "uint8", columns, 32,
                                  tuple(AllocationSlice(**s) for s in parts)))
    if len(owners) != 1 and not any(len(t.shards) > 1 for t in tensors):
        raise ManifestError("MULTIPLE_OWNERS", "tensors")
    total = sum(metadata[0] for metadata in allocations.values())
    if total > limits.max_total_allocation_bytes:
        raise ManifestError("TOTAL_ALLOCATION_LIMIT", "tensors")
    for ranges in intervals.values():
        ordered = sorted(ranges)
        if any(left[1] > right[0] for left, right in zip(ordered, ordered[1:])):
            raise ManifestError("OVERLAPPING_TENSORS", "tensors")
    by_name = {t.name: t for t in tensors}
    for t in tensors:
        prefix = t.name.rsplit(".", 1)[0]
        weight, scale = by_name.get(prefix + ".weight"), by_name.get(prefix + ".scale")
        if weight is None or scale is None or scale.shape != (weight.shape[0], weight.shape[1] // 32):
            raise ManifestError("INCOMPLETE_WEIGHT_SCALE", "tensors")
        if len(weight.shards) != len(scale.shards):
            raise ManifestError("INCOMPLETE_WEIGHT_SCALE", "tensors")
        if len(weight.shards) > 1:
            for ws, ss in zip(weight.shards, scale.shards):
                if (ws.row_begin, ws.row_end) != (ss.row_begin, ss.row_end):
                    raise ManifestError("INCOMPLETE_WEIGHT_SCALE", "tensors")
                if ws.allocation_bytes // weight.row_bytes != ss.allocation_bytes // scale.row_bytes:
                    raise ManifestError("UNEVEN_SHARDS", "tensors")
    result = dict(d, tensors=tuple(sorted(tensors, key=lambda t: t.name)))
    return Manifest(**result), total, len(allocations)


def _content_digest(manifest):
    tensors = [{"name": t.name, "shape": t.shape, "dtype": t.dtype,
                "storage_dtype": t.storage_dtype, "row_bytes": t.row_bytes,
                "quant_block": t.quant_block,
                "sha256": (t.shards[0].sha256 if len(t.shards) == 1
                           else [s.sha256 for s in t.shards])}
               for t in manifest.tensors]
    return _hash({"model_revision": manifest.model_revision, "layout_abi": manifest.layout_abi,
                  "hash_abi": manifest.hash_abi, "tensors": tensors})


def compute_dataset_id(document: Mapping, limits: Limits = Limits()) -> str:
    """Hash semantic/content metadata; dataset_id may be omitted in this draft.

    Owner/generation/allocation/transport changes do not affect this identity.
    sha256 fields must come from trusted byte hashing, not arbitrary labels.
    """
    if not isinstance(document, Mapping):
        raise ManifestError("INVALID_FIELDS", "manifest")
    metadata, _, _ = _metadata(dict(document, dataset_id="sha256:" + "0" * 64), limits)
    return _content_digest(metadata)


def parse_manifest(data: bytes, limits: Limits = Limits()) -> Manifest:
    if not isinstance(data, bytes) or len(data) > limits.max_manifest_bytes:
        raise ManifestError("INPUT_LIMIT", "manifest")

    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ManifestError("DUPLICATE_KEY", "manifest")
            result[key] = value
        return result

    def constant(_):
        raise ManifestError("INVALID_JSON", "manifest")

    try:
        document = json.loads(data.decode("utf-8"), object_pairs_hook=pairs, parse_constant=constant)
    except (UnicodeError, ValueError, RecursionError) as exc:
        if isinstance(exc, ManifestError):
            raise
        raise ManifestError("INVALID_JSON", "manifest") from None
    manifest, _, _ = _metadata(document, limits)
    if _content_digest(manifest) != manifest.dataset_id:
        raise ManifestError("DATASET_DIGEST_MISMATCH", "dataset_id")
    return manifest


def manifest_digest(manifest: Manifest) -> str:
    """Digest the immutable publication, including generation and references."""
    return _hash(asdict(manifest))


def validate_for_attach(manifest: Manifest, expected_layout: ExpectedLayout,
                        expected_dataset_id: str, expected_generation: str,
                        limits: Limits = Limits()) -> ValidatedManifest:
    _digest(expected_dataset_id, "expected_dataset_id")
    _uuid(expected_generation, "expected_generation")
    # Revalidate even manually constructed dataclasses; caller-supplied limits
    # may also be tighter than the parser's limits.
    if not isinstance(manifest, Manifest) or not isinstance(expected_layout, ExpectedLayout):
        raise ManifestError("INVALID_FIELDS", "attach")
    try:
        serialized = _canonical(asdict(manifest))
    except (TypeError, ValueError, RecursionError):
        raise ManifestError("INVALID_FIELDS", "manifest") from None
    if len(serialized) > limits.max_manifest_bytes:
        raise ManifestError("INPUT_LIMIT", "manifest")
    document = json.loads(serialized)
    manifest, total, count = _metadata(document, limits)
    if _content_digest(manifest) != manifest.dataset_id or manifest.dataset_id != expected_dataset_id:
        raise ManifestError("DATASET_MISMATCH", "dataset_id")
    if manifest.generation != expected_generation:
        raise ManifestError("GENERATION_MISMATCH", "generation")
    if manifest.state != "READY":
        raise ManifestError("NOT_READY", "state")
    if manifest.backend != expected_layout.backend:
        raise ManifestError("BACKEND_MISMATCH", "backend")
    if (manifest.model_revision, manifest.hash_abi, manifest.layout_abi) != (
            expected_layout.model_revision, expected_layout.hash_abi, expected_layout.layout_abi):
        raise ManifestError("MODEL_CONTRACT_MISMATCH", "manifest")
    expected = {}
    for layer, rows in expected_layout.layers:
        for role, columns in (("weight", expected_layout.head_dim), ("scale", expected_layout.head_dim // 32)):
            expected[f"layers.{layer}.engram.embed.{role}"] = (rows, columns)
    if {t.name: t.shape for t in manifest.tensors} != expected:
        raise ManifestError("INCOMPLETE_GENERATION", "tensors")
    return ValidatedManifest(manifest, total, count, manifest_digest(manifest))
