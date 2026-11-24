# SPDX-License-Identifier: Apache-2.0
"""Build the in-memory mapping layout from paired shard publications."""
from __future__ import annotations

import base64
import re
from dataclasses import asdict, dataclass

from .placement import DEFAULT_BLOCK, MAX_LOCAL_SEGMENTS, plan_shards, segment_rows

LAYOUT_ABI = "engram-fp8-e8m0-v1"
PAIRED_SHARD_SCHEMA = 4
_MAX_INTEGER = (1 << 63) - 1
_SHA = re.compile(r"sha256:[0-9a-f]{64}\Z")


class ManifestError(ValueError):
    """Stable code and field path; never include untrusted field values."""

    def __init__(self, code: str, path: str):
        self.code, self.path = code, path
        super().__init__(f"{code} at {path}")


@dataclass(frozen=True)
class Limits:
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
        if self.head_dim % DEFAULT_BLOCK or self.layout_abi != LAYOUT_ABI:
            raise ManifestError("UNSUPPORTED_LAYOUT", "expected")
        if self.backend not in ("fabric_egm", "fabric_gpu", "cuda_ipc"):
            raise ManifestError("UNSUPPORTED_BACKEND", "expected.backend")
        _text(self.model_revision, "expected.model_revision")
        _digest(self.hash_abi, "expected.hash_abi")
        object.__setattr__(self, "layers", tuple(sorted(layers)))


@dataclass(frozen=True)
class AllocationSlice:
    row_begin: int
    row_end: int
    logical_bytes: int
    allocation_bytes: int
    allocation_granularity: int
    host_id: str
    allocation_id: str


@dataclass(frozen=True)
class TensorSpec:
    name: str
    shape: tuple[int, int]
    shards: tuple[AllocationSlice, ...]

    @property
    def row_bytes(self):
        return self.shape[1]


@dataclass(frozen=True)
class Manifest:
    model_revision: str
    generation: str
    layout_abi: str
    hash_abi: str
    backend: str
    tensors: tuple[TensorSpec, ...]


@dataclass(frozen=True)
class ParsedManifest:
    manifest: Manifest
    physical_bytes: int
    allocation_count: int


@dataclass(frozen=True)
class ValidatedManifest(ParsedManifest):
    layout: ExpectedLayout


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


def bind_manifest(parsed, expected_layout, expected_generation, limits):
    """Apply this worker's limits and model contract to the parsed layout."""
    manifest = parsed.manifest
    if len(manifest.tensors) > limits.max_tensors:
        raise ManifestError("TENSOR_LIMIT", "tensors")
    if parsed.physical_bytes > limits.max_total_allocation_bytes:
        raise ManifestError("TOTAL_ALLOCATION_LIMIT", "tensors")
    if any(
        part.allocation_bytes > limits.max_allocation_bytes
        for tensor in manifest.tensors
        for part in tensor.shards
    ):
        raise ManifestError("INVALID_ALLOCATION_SIZE", "tensors")
    return _validate_contract(
        manifest,
        parsed.physical_bytes,
        parsed.allocation_count,
        expected_layout,
        expected_generation,
    )


def _validate_contract(manifest, total, count, expected_layout, expected_generation):
    if not isinstance(expected_layout, ExpectedLayout):
        raise ManifestError("INVALID_FIELDS", "attach")
    if manifest.generation != expected_generation:
        raise ManifestError("GENERATION_MISMATCH", "generation")
    if manifest.backend != expected_layout.backend:
        raise ManifestError("BACKEND_MISMATCH", "backend")
    if (manifest.model_revision, manifest.hash_abi, manifest.layout_abi) != (
        expected_layout.model_revision,
        expected_layout.hash_abi,
        expected_layout.layout_abi,
    ):
        raise ManifestError("MODEL_CONTRACT_MISMATCH", "manifest")
    expected = {}
    for layer, rows in expected_layout.layers:
        for role, columns in (
            ("weight", expected_layout.head_dim),
            ("scale", expected_layout.head_dim // DEFAULT_BLOCK),
        ):
            expected[f"layers.{layer}.engram.embed.{role}"] = (rows, columns)
    if {t.name: t.shape for t in manifest.tensors} != expected:
        raise ManifestError("INCOMPLETE_GENERATION", "tensors")
    return ValidatedManifest(manifest, total, count, expected_layout)


def _layer_records(document):
    layers = document.get("layers")
    if type(layers) is not list or not 0 < len(layers) <= Limits().max_tensors // 2:
        raise ManifestError("TENSOR_LIMIT", "layers")
    records = {}
    for layer in layers:
        if type(layer) is not dict or layer.keys() != {
            "layer_id",
            "rows",
            "head_dim",
            "segments",
        }:
            raise ManifestError("INVALID_FIELDS", "layer")
        layer_id = _integer(layer["layer_id"], "layer_id", minimum=0)
        if layer_id in records:
            raise ManifestError("DUPLICATE_LAYER", "layers")
        rows = _integer(layer["rows"], "rows")
        dim = _integer(layer["head_dim"], "head_dim")
        records[layer_id] = (rows, dim, layer["segments"])
    return records


def _handle_bytes(value):
    if type(value) is not str or not 0 < len(value) <= 1024:
        raise ValueError("invalid capability encoding")
    try:
        raw = base64.b64decode(value, validate=True)
    except (ValueError, TypeError):
        raise ValueError("invalid capability encoding") from None
    if not 0 < len(raw) <= 768:
        raise ValueError("empty or oversized capability")
    return raw


def prepare_manifest(documents, shards, generation, model_revision, hash_abi):
    """Build the allocation list for the generation established by startup."""
    from .config import validate_count

    validate_count(shards)
    if type(documents) is not list or len(documents) != shards:
        raise ValueError("one descriptor per storage host required")
    fields = {
        "schema_version",
        "shard",
        "shards",
        "generation",
        "model_revision",
        "hash_abi",
        "host_id",
        "memory_kind",
        "granularity",
        "layers",
    }
    if any(type(d) is not dict or d.keys() != fields for d in documents):
        raise ManifestError("INVALID_FIELDS", "descriptor")
    if any(
        d["schema_version"] != PAIRED_SHARD_SCHEMA
        or type(d["schema_version"]) is not int
        for d in documents
    ):
        raise ManifestError("UNSUPPORTED_SCHEMA", "descriptor")
    indices = [
        _integer(d["shard"], "shard", minimum=0, maximum=shards - 1) for d in documents
    ]
    if len(set(indices)) != shards:
        raise ValueError("descriptor shard indices are incomplete or duplicated")
    documents = sorted(documents, key=lambda d: d["shard"])
    hosts = [_text(d["host_id"], "host_id") for d in documents]
    if len(set(hosts)) != shards:
        raise ValueError("storage host IDs must be distinct")

    first = documents[0]
    granularity = _integer(first["granularity"], "granularity")
    memory_kind = first["memory_kind"]
    if memory_kind not in ("host", "gpu"):
        raise ManifestError("UNSUPPORTED_BACKEND", "memory_kind")
    if hash_abi is None:
        hash_abi = first["hash_abi"]
    _text(model_revision, "model_revision")
    _digest(hash_abi, "hash_abi")
    for d in documents:
        if (
            type(d["shards"]) is not int
            or d["shards"] != shards
            or d["generation"] != generation
            or d["model_revision"] != model_revision
            or d["hash_abi"] != hash_abi
        ):
            raise ValueError("descriptor generation/count/revision/hash mismatch")
        if (
            d["memory_kind"] != memory_kind
            or _integer(d["granularity"], "granularity") != granularity
        ):
            raise ValueError("descriptor memory kind or granularity mismatch")

    layout = _layer_records(first)
    plans = {
        layer: plan_shards(rows, dim, shards, granularity=granularity)
        for layer, (rows, dim, _) in layout.items()
    }
    pieces = {(layer, role): [] for layer in plans for role in ("weight", "scale")}
    handles, seen = {}, set()
    total = 0
    max_location = (1 << 31) - 1 if memory_kind == "gpu" else 65535
    for d in documents:
        layers = layout if d is first else _layer_records(d)
        if layers.keys() != plans.keys():
            raise ValueError("descriptor layer sets differ")
        for layer, (rows, dim, segments) in layers.items():
            plan = plans[layer]
            if (rows, dim) != (plan.total_rows, plan.dim):
                raise ValueError("descriptor tensor shape mismatch")
            if (
                type(segments) is not list
                or not 1 <= len(segments) <= MAX_LOCAL_SEGMENTS
            ):
                raise ManifestError("UNSUPPORTED_SHARDING", "segments")
            if any(
                type(s) is not dict
                or s.keys() != {"location", "row_begin", "row_end", "weight", "scale"}
                for s in segments
            ):
                raise ManifestError("INVALID_FIELDS", "segment")
            locations = tuple(
                _integer(s["location"], "location", minimum=0, maximum=max_location)
                for s in segments
            )
            if len(set(locations)) != len(locations):
                raise ValueError("duplicate physical segment location")
            geometry = segment_rows(plan, d["shard"], locations)
            for index, (segment, (_, begin, end, padded)) in enumerate(
                zip(segments, geometry)
            ):
                if (
                    _integer(segment["row_begin"], "row_begin", minimum=0) != begin
                    or _integer(segment["row_end"], "row_end") != end
                ):
                    raise ValueError(
                        "descriptor geometry does not match storage host index"
                    )
                for role, width in (("weight", dim), ("scale", dim // plan.block)):
                    allocation = segment[role]
                    if type(allocation) is not dict or allocation.keys() != {
                        "handle",
                        "allocation_bytes",
                    }:
                        raise ManifestError("INVALID_FIELDS", role)
                    size = _integer(allocation["allocation_bytes"], "allocation_bytes")
                    if size != padded * width:
                        raise ValueError(
                            "descriptor geometry does not match physical allocation size"
                        )
                    handle = _handle_bytes(allocation["handle"])
                    if handle in seen:
                        raise ValueError("duplicate capability")
                    seen.add(handle)
                    aid = f'{d["shard"]}/{layer}/{index}/{role}'
                    handles[aid] = handle
                    pieces[layer, role].append(
                        AllocationSlice(
                            row_begin=begin,
                            row_end=end,
                            logical_bytes=(end - begin) * width,
                            allocation_bytes=size,
                            allocation_granularity=granularity,
                            host_id=d["host_id"],
                            allocation_id=aid,
                        )
                    )
                    total += size
    if total > _MAX_INTEGER:
        raise ManifestError("TOTAL_ALLOCATION_LIMIT", "tensors")
    tensors = tuple(
        TensorSpec(
            f"layers.{layer}.engram.embed.{role}",
            (plan.total_rows, width),
            tuple(pieces[layer, role]),
        )
        for layer, plan in sorted(plans.items())
        for role, width in (("weight", plan.dim), ("scale", plan.dim // plan.block))
    )
    manifest = Manifest(
        generation=generation,
        model_revision=model_revision,
        hash_abi=hash_abi,
        layout_abi=LAYOUT_ABI,
        backend="fabric_gpu" if memory_kind == "gpu" else "fabric_egm",
        tensors=tensors,
    )
    return ParsedManifest(manifest, total, len(handles)), handles
