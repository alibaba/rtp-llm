"""L20 cache deltas carried by the existing PP named-tensor transport.

This is a synchronous prefill protocol. The scheduler supplies trusted admission
identities and expected spans; transport tickets must complete before import.
It does not implement PP messaging, prefix restore or forward-local selections.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Mapping, Sequence

import torch

from rtp_llm.models.dsv41_kv_cache import dsv41_page_layouts

if TYPE_CHECKING:
    from rtp_llm.models_py.modules.dsv4.v41_cache import V41PagedCache

_VERSION = 1
_SOURCE = 20
_TAGS = ("hca_kv", "indexer_kv")
_HEADER = "v41_delta_header"
_FINGERPRINT = "v41_cache_fingerprint"
_SPANS = "v41_delta_spans"
_ROWS = ("v41_l20_global_delta", "v41_l20_index_delta")
_WIRE_KEYS = frozenset((_HEADER, _FINGERPRINT, _SPANS, *_ROWS))
_INT64_MAX = (1 << 63) - 1


def _integer(value: int, name: str, minimum: int = 0) -> None:
    if type(value) is not int or not minimum <= value <= _INT64_MAX:
        raise ValueError(f"V4.1 {name} must be an integer in [{minimum}, {_INT64_MAX}]")


@dataclass(frozen=True)
class V41L20Contract:
    checkpoint_sha256: str
    config_sha256: str
    tokenizer_sha256: str
    max_sequence_length: int

    def __post_init__(self):
        for digest in (
            self.checkpoint_sha256,
            self.config_sha256,
            self.tokenizer_sha256,
        ):
            if (
                not isinstance(digest, str)
                or len(digest) != 64
                or any(c not in "0123456789abcdef" for c in digest)
            ):
                raise ValueError("V4.1 identities must be canonical SHA256 hex strings")
        _integer(self.max_sequence_length, "max_sequence_length", 1)

    def canonical_bytes(self) -> bytes:
        layouts = dsv41_page_layouts(0)
        manifest = {
            "schema": "rtp_llm.v41.l20_delta",
            "version": _VERSION,
            "checkpoint_sha256": self.checkpoint_sha256,
            "config_sha256": self.config_sha256,
            "tokenizer_sha256": self.tokenizer_sha256,
            "max_sequence_length": self.max_sequence_length,
            "source_layer": _SOURCE,
            "tokens_per_block": 256,
            "kernel_tokens_per_block": 256,
            "cp_size": 4,
            "cp_cache_sharded": False,
            "replay": "exact",
            "regions": [
                {
                    "tag": tag,
                    "entries": layouts[tag].entries,
                    "payload_bytes": layouts[tag].payload_bytes_per_entry,
                    "scale_bytes": layouts[tag].scale_bytes_per_entry,
                    "scale_offset": layouts[tag].scale_plane_offset,
                    "stride": layouts[tag].stride_bytes,
                    "codec": "e2m1_rne_low_nibble_first",
                    "scales": (
                        "e4m3_group16_satfinite"
                        if tag == "hca_kv"
                        else "ue8m0_group32_ceil"
                    ),
                }
                for tag in _TAGS
            ],
        }
        return json.dumps(
            manifest, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("ascii")

    @property
    def fingerprint(self) -> bytes:
        return hashlib.sha256(self.canonical_bytes()).digest()


@dataclass(frozen=True)
class V41CacheIdentity:
    request_id: int
    generation: int
    cache_incarnation: int

    def __post_init__(self):
        _integer(self.request_id, "request_id", -(1 << 63))
        _integer(self.generation, "generation")
        _integer(self.cache_incarnation, "cache_incarnation")


@dataclass(frozen=True)
class V41L20Span:
    identity: V41CacheIdentity
    chunk_id: int
    start: int
    count: int

    def __post_init__(self):
        if not isinstance(self.identity, V41CacheIdentity):
            raise ValueError("V4.1 span requires a cache identity")
        _integer(self.chunk_id, "chunk_id")
        _integer(self.start, "delta start")
        _integer(self.count, "delta count", 1)
        _integer(self.start + self.count, "delta end", 1)

    def wire_values(self) -> tuple[int, ...]:
        return (
            self.identity.request_id,
            self.identity.generation,
            self.identity.cache_incarnation,
            self.chunk_id,
            self.start,
            self.count,
        )


def _spans(
    spans: Sequence[V41L20Span], contract: V41L20Contract
) -> tuple[V41L20Span, ...]:
    result = tuple(spans)
    if any(not isinstance(s, V41L20Span) for s in result):
        raise ValueError("V4.1 invalid delta span")
    if len({s.identity.request_id for s in result}) != len(result):
        raise ValueError("V4.1 delta batch has duplicate requests")
    if any(s.start + s.count > contract.max_sequence_length for s in result):
        raise ValueError("V4.1 delta exceeds the sequence limit")
    _integer(sum(s.count for s in result), "delta rows")
    return result


def _cp_rank(rank: int) -> None:
    _integer(rank, "CP rank")
    if rank >= 4:
        raise ValueError("V4.1 L20 deltas require a CEP4 stage partner")


def _cache_pair(caches: Mapping[str, V41PagedCache]) -> dict[str, V41PagedCache]:
    if set(caches) != set(_TAGS):
        raise ValueError("V4.1 L20 transfer requires exactly hca_kv and indexer_kv")
    result = dict(caches)
    layouts = dsv41_page_layouts(0)
    for tag, cache in result.items():
        if cache.tag != tag or cache.layout != layouts[tag]:
            raise ValueError("V4.1 L20 cache tag or geometry mismatch")
    first, second = (result[tag].raw for tag in _TAGS)
    if first.device != second.device:
        raise ValueError("V4.1 L20 regions must share a device")
    if _overlap(first, second):
        raise ValueError("V4.1 L20 cache regions must not overlap")
    return result


def _overlap(a: torch.Tensor, b: torch.Tensor) -> bool:
    if a.device != b.device or not a.numel() or not b.numel():
        return False
    # All protocol tensors have nonnegative strides; bounds include storage gaps.
    a_end = (
        a.data_ptr()
        + (1 + sum((n - 1) * s for n, s in zip(a.shape, a.stride()))) * a.element_size()
    )
    b_end = (
        b.data_ptr()
        + (1 + sum((n - 1) * s for n, s in zip(b.shape, b.stride()))) * b.element_size()
    )
    return a.data_ptr() < b_end and b.data_ptr() < a_end


def _tables(
    blocks: Mapping[str, Sequence[int]], caches: Mapping[str, V41PagedCache]
) -> dict[str, tuple[int, ...]]:
    if set(blocks) != set(_TAGS):
        raise ValueError("V4.1 L20 block tables must cover both regions")
    result = {tag: tuple(blocks[tag]) for tag in _TAGS}
    for tag, pages in result.items():
        if any(
            type(p) is not int or not 0 <= p < caches[tag].raw.shape[0] for p in pages
        ):
            raise ValueError("V4.1 L20 block table has an invalid physical page")
        if len(set(pages)) != len(pages):
            raise ValueError("V4.1 L20 writable block table aliases itself")
    if len(result[_TAGS[0]]) != len(result[_TAGS[1]]):
        raise ValueError("V4.1 L20 regions must cover the same logical pages")
    return result


def _slots(span: V41L20Span, pages: tuple[int, ...]) -> list[int]:
    if span.start + span.count > len(pages) * 256:
        raise ValueError("V4.1 L20 delta exceeds its local allocation")
    return [
        pages[i // 256] * 256 + i % 256
        for i in range(span.start, span.start + span.count)
    ]


def export_l20_delta(
    contract: V41L20Contract,
    cp_rank: int,
    round_id: int,
    spans: Sequence[V41L20Span],
    caches: Mapping[str, V41PagedCache],
    block_tables: Mapping[V41CacheIdentity, Mapping[str, Sequence[int]]],
    *,
    backend: str = "triton",
) -> dict[str, torch.Tensor]:
    """Gather completed producer rows; retain the result through PP send tickets."""
    _cp_rank(cp_rank)
    _integer(round_id, "round_id")
    spans = _spans(spans, contract)
    caches = _cache_pair(caches)
    if backend not in ("eager", "triton"):
        raise ValueError("V4.1 invalid transfer backend")
    positions = {tag: [] for tag in _TAGS}
    occupied = {tag: set() for tag in _TAGS}
    for span in spans:
        if span.identity not in block_tables:
            raise ValueError("V4.1 missing producer block table")
        tables = _tables(block_tables[span.identity], caches)
        for tag in _TAGS:
            if occupied[tag].intersection(tables[tag]):
                raise ValueError(
                    "V4.1 producer requests overlap in their page allocations"
                )
            occupied[tag].update(tables[tag])
            positions[tag].extend(_slots(span, tables[tag]))
    result = {
        _HEADER: torch.tensor(
            [
                _VERSION,
                _SOURCE,
                cp_rank,
                round_id,
                len(spans),
                sum(s.count for s in spans),
            ],
            dtype=torch.int64,
        ),
        _FINGERPRINT: torch.tensor(list(contract.fingerprint), dtype=torch.uint8),
    }
    # PP serializes zero-element tensors as undefined, so omit all empty fields.
    if spans:
        result[_SPANS] = torch.tensor(
            [s.wire_values() for s in spans], dtype=torch.int64
        )
        for tag, key in zip(_TAGS, _ROWS):
            slots = torch.tensor(
                positions[tag], dtype=torch.int64, device=caches[tag].raw.device
            )
            result[key] = caches[tag].gather(slots, backend=backend)
    return result


def _host_tensor(
    tensors: Mapping[str, torch.Tensor],
    key: str,
    shape: tuple[int, ...],
    dtype: torch.dtype,
) -> torch.Tensor:
    value = tensors.get(key)
    if (
        not isinstance(value, torch.Tensor)
        or value.device.type != "cpu"
        or value.dtype != dtype
        or tuple(value.shape) != shape
        or not value.is_contiguous()
    ):
        raise ValueError(f"V4.1 {key} must be contiguous CPU {dtype} {shape}")
    return value


@dataclass
class _Replica:
    identity: V41CacheIdentity
    tables: dict[str, tuple[int, ...]]
    visible_rows: int = 0
    next_chunk: int = 0
    cancelled: bool = False


class V41L20ReplicaReceiver:
    """Single-threaded, one-in-flight CEP4 receiver with prefix reuse disabled.

    Admission generations are globally increasing within this receiver session,
    supplied by trusted scheduler control on both stages. This fences recycled
    request IDs without retaining an unbounded set of retired request IDs.
    """

    def __init__(
        self,
        contract: V41L20Contract,
        cp_rank: int,
        caches: Mapping[str, V41PagedCache],
        *,
        backend: str = "triton",
    ):
        _cp_rank(cp_rank)
        if backend not in ("eager", "triton"):
            raise ValueError("V4.1 invalid transfer backend")
        self.contract = contract
        self.cp_rank = cp_rank
        self.caches = _cache_pair(caches)
        self.backend = backend
        self._requests: dict[int, _Replica] = {}
        self._owners: dict[tuple[str, int], V41CacheIdentity] = {}
        self._last_generation = -1
        self._next_round = 0
        self._failed = False
        self._drained = True

    def _state(self, identity: V41CacheIdentity) -> _Replica:
        state = self._requests.get(identity.request_id)
        if state is None or state.identity != identity:
            raise ValueError("V4.1 unknown or stale replica identity")
        return state

    def _usable(self) -> None:
        if self._failed:
            raise RuntimeError(
                "V4.1 L20 receiver failed; recreate the session after draining"
            )

    def _reserve(
        self, identity: V41CacheIdentity, tables: dict[str, tuple[int, ...]]
    ) -> None:
        for tag, pages in tables.items():
            if any(self._owners.get((tag, p), identity) != identity for p in pages):
                raise ValueError("V4.1 replica allocations overlap another request")
        self._owners.update(
            {(tag, p): identity for tag, pages in tables.items() for p in pages}
        )

    def begin_request(
        self, identity: V41CacheIdentity, blocks: Mapping[str, Sequence[int]]
    ) -> None:
        self._usable()
        if identity.request_id in self._requests:
            raise ValueError("V4.1 request already has an allocated replica")
        if identity.generation <= self._last_generation:
            raise ValueError("V4.1 admission generation must increase")
        tables = _tables(blocks, self.caches)
        self._reserve(identity, tables)
        self._requests[identity.request_id] = _Replica(identity, tables)
        self._last_generation = identity.generation

    def extend_request(
        self, identity: V41CacheIdentity, blocks: Mapping[str, Sequence[int]]
    ) -> None:
        self._usable()
        state = self._state(identity)
        if state.cancelled:
            raise ValueError("V4.1 replica is cancelled")
        tables = _tables(blocks, self.caches)
        if any(
            tables[tag][: len(state.tables[tag])] != state.tables[tag] for tag in _TAGS
        ):
            raise ValueError(
                "V4.1 replica pages may only be appended; prefix relocation requires restoration"
            )
        self._reserve(identity, tables)
        state.tables = tables

    def cancel(self, identity: V41CacheIdentity) -> None:
        self._state(identity).cancelled = True

    def release(self, identity: V41CacheIdentity) -> None:
        """Call only after external PP tickets/consumers have also been drained."""
        state = self._state(identity)
        if not self._drained:
            raise RuntimeError(
                "V4.1 replica writes have not drained; allocation cannot be released"
            )
        for tag, pages in state.tables.items():
            for page in pages:
                del self._owners[(tag, page)]
        del self._requests[identity.request_id]

    def require_ready(
        self, identity: V41CacheIdentity, rows: int, chunk_id: int
    ) -> None:
        self._usable()
        _integer(rows, "required rows", 1)
        _integer(chunk_id, "required chunk")
        state = self._state(identity)
        if (
            state.cancelled
            or state.visible_rows != rows
            or state.next_chunk != chunk_id + 1
        ):
            raise ValueError("V4.1 L20 replica is not current for this chunk")

    def apply(
        self, tensors: Mapping[str, torch.Tensor], expected_spans: Sequence[V41L20Span]
    ) -> None:
        """Validate the complete batch before writing either region, then commit."""
        self._usable()
        expected = _spans(expected_spans, self.contract)
        header = _host_tensor(tensors, _HEADER, (6,), torch.int64).tolist()
        version, source, cp_rank, round_id, span_count, row_count = header
        if (version, source, cp_rank, round_id, span_count, row_count) != (
            _VERSION,
            _SOURCE,
            self.cp_rank,
            self._next_round,
            len(expected),
            sum(s.count for s in expected),
        ):
            raise ValueError(
                "V4.1 delta envelope disagrees with the receiver/dispatch round"
            )
        fingerprint = _host_tensor(tensors, _FINGERPRINT, (32,), torch.uint8)
        if bytes(fingerprint.tolist()) != self.contract.fingerprint:
            raise ValueError("V4.1 L20 cache compatibility fingerprint mismatch")
        keys = _WIRE_KEYS if expected else frozenset((_HEADER, _FINGERPRINT))
        if set(tensors).intersection(_WIRE_KEYS) != keys:
            raise ValueError("V4.1 delta has missing or unexpected payload fields")
        if not expected:
            self._next_round += 1
            return
        wire_spans = _host_tensor(tensors, _SPANS, (len(expected), 6), torch.int64)
        if wire_spans.tolist() != [list(s.wire_values()) for s in expected]:
            raise ValueError("V4.1 delta spans disagree with trusted dispatch metadata")
        positions = {tag: [] for tag in _TAGS}
        states = []
        for span in expected:
            state = self._state(span.identity)
            if (
                state.cancelled
                or state.next_chunk != span.chunk_id
                or state.visible_rows != span.start
            ):
                raise ValueError(
                    "V4.1 stale, duplicate, missing or cancelled L20 delta"
                )
            for tag in _TAGS:
                positions[tag].extend(_slots(span, state.tables[tag]))
            states.append(state)
        slots = {}
        for tag, key in zip(_TAGS, _ROWS):
            cache = self.caches[tag]
            rows = tensors[key]
            width = (
                cache.layout.payload_bytes_per_entry
                + cache.layout.scale_bytes_per_entry
            )
            if (
                not isinstance(rows, torch.Tensor)
                or rows.dtype != torch.uint8
                or rows.device != cache.raw.device
                or tuple(rows.shape) != (row_count, width)
                or not rows.is_contiguous()
            ):
                raise ValueError(
                    "V4.1 packed delta rows have invalid dtype/device/shape/stride"
                )
            if any(_overlap(rows, other.raw) for other in self.caches.values()):
                raise ValueError("V4.1 delta payload aliases a replica cache")
            slots[tag] = torch.tensor(
                positions[tag], dtype=torch.int64, device=cache.raw.device
            )
        device = self.caches[_TAGS[0]].raw.device
        if self.backend == "triton" and device.type != "cuda":
            raise ValueError("V4.1 Triton import requires CUDA caches")
        self._drained = False
        try:
            for tag, key in zip(_TAGS, _ROWS):
                self.caches[tag].scatter(tensors[key], slots[tag], backend=self.backend)
            if device.type == "cuda":
                torch.cuda.current_stream(device).synchronize()
            self._drained = True
        except Exception:
            self._failed = True
            # Drain queued writes even when the second region fails. Never publish readiness.
            if device.type == "cuda":
                torch.cuda.current_stream(device).synchronize()
            self._drained = True
            raise
        for state, span in zip(states, expected):
            state.visible_rows += span.count
            state.next_chunk += 1
        self._next_round += 1
