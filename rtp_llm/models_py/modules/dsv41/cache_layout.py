"""Typed V4.1 cache geometry and complete memory-checkpoint selection.

These descriptors describe the compact byte layouts. They do not select a GPU
reader or imply that a page has been written merely because it is allocated.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from enum import Enum
from functools import cached_property
from typing import Iterable, Optional, Sequence, Tuple

TARGET_LAYERS = 40
DRAFT_LAYERS = (40, 41, 42)
GLOBAL_OWNERS = (2, 8, 14, 20)
INDEX_QUERY_OWNERS = (2, 8, 14, 20, 24, 28, 32, 36)
PAIR_OWNERS = (2, 8, 14)
SWA_WINDOW = 128


def _identity(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class LayerSources:
    layer: int
    ratio: int
    global_owner: Optional[int]
    index_k_owner: Optional[int]
    topk_owner: Optional[int]

    @property
    def writes_global(self) -> bool:
        return self.layer == self.global_owner

    @property
    def writes_index_k(self) -> bool:
        return self.layer == self.index_k_owner

    @property
    def scores_queries(self) -> bool:
        return self.layer == self.topk_owner


def layer_sources(layer: int) -> LayerSources:
    if not 0 <= layer < TARGET_LAYERS + len(DRAFT_LAYERS):
        raise ValueError(f"invalid V4.1 layer: {layer}")
    if layer < 2 or layer in DRAFT_LAYERS:
        return LayerSources(layer, 0, None, None, None)
    if layer < 20:
        owner = 2 + ((layer - 2) // 6) * 6
        return LayerSources(layer, 2, owner, owner, owner)
    topk_owner = 20 + ((layer - 20) // 4) * 4
    return LayerSources(layer, 1, 20, 20, topk_owner)


def visible_global_entries(layer: int, query_position: int) -> int:
    ratio = layer_sources(layer).ratio
    if ratio == 0 or query_position < 0:
        raise ValueError("global visibility requires a global layer and valid query")
    return (query_position + 1) // ratio


class CacheRegion(str, Enum):
    SWA = "swa"
    GLOBAL = "global"
    INDEX_K = "index_k"


@dataclass(frozen=True, order=True)
class RegionSlot:
    region: CacheRegion
    owner_layer: int


@dataclass(frozen=True)
class RegionEncoding:
    head_dim: int
    payload_dtype: str
    payload_bits: int
    scale_dtype: str
    group_size: int

    @property
    def payload_bytes(self) -> int:
        return self.head_dim * self.payload_bits // 8

    @property
    def scale_bytes(self) -> int:
        return self.head_dim // self.group_size

    @property
    def entry_bytes(self) -> int:
        return self.payload_bytes + self.scale_bytes


ENCODINGS = {
    CacheRegion.SWA: RegionEncoding(512, "fp8_e4m3", 8, "ue8m0", 32),
    CacheRegion.GLOBAL: RegionEncoding(512, "fp4_e2m1", 4, "fp8_e4m3", 16),
    CacheRegion.INDEX_K: RegionEncoding(128, "fp4_e2m1", 4, "ue8m0", 32),
}


@dataclass(frozen=True)
class RegionPage:
    slot: RegionSlot
    ratio: int
    encoding: RegionEncoding
    entries: int
    token_block_size: int
    alignment: int
    cp_size: int

    @property
    def page_stride_bytes(self) -> int:
        # Whole SWA pages are byte-sliced under CP, including alignment padding.
        alignment = math.lcm(self.alignment, self.cp_size)
        natural = self.entries * self.encoding.entry_bytes
        return ((natural + alignment - 1) // alignment) * alignment

    @property
    def prefill_shard_bytes(self) -> int:
        if self.slot.region == CacheRegion.SWA:
            return self.page_stride_bytes // self.cp_size
        return self.page_stride_bytes

    def swa_byte_slice(self, rank: int) -> Tuple[int, int]:
        if self.slot.region != CacheRegion.SWA:
            raise ValueError("byte slicing applies only to SWA pages")
        if not 0 <= rank < self.cp_size:
            raise ValueError("CP rank is outside the layout")
        size = self.prefill_shard_bytes
        return rank * size, (rank + 1) * size

    def paged_location(self, compressed_position: int) -> Tuple[int, int, int]:
        """Return (CP rank, virtual block, local row) for a global/index entry."""
        if self.slot.region == CacheRegion.SWA:
            raise ValueError("SWA uses a ring, not compressed paged positions")
        if compressed_position < 0:
            raise ValueError("compressed position must be non-negative")
        token = compressed_position * self.ratio
        physical_block, offset = divmod(token, self.token_block_size)
        virtual_block, rank = divmod(physical_block, self.cp_size)
        return rank, virtual_block, offset // self.ratio


@dataclass(frozen=True)
class PairStateSpec:
    owner_layer: int
    snapshots: int
    head_dim: int = 512
    partial_dtype: str = "float32"
    position_dtype: str = "int64"

    @property
    def partial_elements(self) -> int:
        # Each snapshot holds unnormalized partial KV and per-channel scores.
        return self.snapshots * 2 * self.head_dim


@dataclass(frozen=True)
class CacheLayout:
    token_block_size: int = 128
    cp_size: int = 8
    speculative_tokens: int = 5
    draft_enabled: bool = True
    page_alignment: int = 512
    version: int = 1

    def __post_init__(self) -> None:
        if self.version != 1:
            raise ValueError("unsupported V4.1 cache layout version")
        if self.token_block_size not in (128, 256):
            raise ValueError("V4.1 token blocks must contain 128 or 256 tokens")
        if self.cp_size not in (1, 8):
            raise ValueError("V4.1 supports CP8; CP1 is for local component probes")
        if self.speculative_tokens < 0:
            raise ValueError("speculative token slack must be non-negative")
        if self.page_alignment < 512 or self.page_alignment % 512:
            raise ValueError("compact index pages require 512-byte alignment")

    @property
    def reuse_unit(self) -> int:
        return self.token_block_size * self.cp_size

    @property
    def swa_entries(self) -> int:
        # Preserve all 128 history rows plus speculative writes, then CP-align.
        alignment = math.lcm(2, self.cp_size)
        count = SWA_WINDOW + self.speculative_tokens
        return ((count + alignment - 1) // alignment) * alignment

    @cached_property
    def pages(self) -> Tuple[RegionPage, ...]:
        pages = []
        layers = TARGET_LAYERS + (len(DRAFT_LAYERS) if self.draft_enabled else 0)
        for layer in range(layers):
            pages.append(self._page(CacheRegion.SWA, layer, 0, self.swa_entries))
        for owner in GLOBAL_OWNERS:
            ratio = layer_sources(owner).ratio
            for region in (CacheRegion.GLOBAL, CacheRegion.INDEX_K):
                pages.append(
                    self._page(region, owner, ratio, self.token_block_size // ratio)
                )
        return tuple(pages)

    def _page(
        self, region: CacheRegion, owner: int, ratio: int, entries: int
    ) -> RegionPage:
        return RegionPage(
            RegionSlot(region, owner),
            ratio,
            ENCODINGS[region],
            entries,
            self.token_block_size,
            self.page_alignment,
            self.cp_size,
        )

    @cached_property
    def pair_states(self) -> Tuple[PairStateSpec, ...]:
        # Initial state and one snapshot after every anchor/proposal verify row.
        return tuple(
            PairStateSpec(owner, self.speculative_tokens + 2) for owner in PAIR_OWNERS
        )

    @cached_property
    def required_slots(self) -> frozenset[RegionSlot]:
        return frozenset(page.slot for page in self.pages)

    @cached_property
    def global_slots(self) -> frozenset[RegionSlot]:
        return frozenset(
            slot for slot in self.required_slots if slot.region != CacheRegion.SWA
        )

    @cached_property
    def fingerprint(self) -> str:
        return _identity(
            {
                "config": asdict(self),
                "pages": [asdict(page) for page in self.pages],
                "pair_states": [asdict(state) for state in self.pair_states],
                "sources": [asdict(layer_sources(i)) for i in range(TARGET_LAYERS)],
            }
        )


@dataclass(frozen=True)
class CacheIdentity:
    model_revision: str
    layout_fingerprint: str
    replay_fingerprint: str

    def __post_init__(self) -> None:
        if not all(asdict(self).values()):
            raise ValueError(
                "cache identity requires model, layout and replay identity"
            )

    @property
    def fingerprint(self) -> str:
        return _identity(asdict(self))


@dataclass(frozen=True)
class GlobalBlock:
    ordinal: int
    identity: CacheIdentity
    valid_slots: frozenset[RegionSlot]
    copy_complete: bool = False


@dataclass(frozen=True)
class MemoryCheckpoint:
    materialized_end: int
    identity: CacheIdentity
    valid_slots: frozenset[RegionSlot]
    target_swa_start: int
    draft_swa_start: Optional[int]
    replay_floor: int
    history_ready: bool = False
    copy_complete: bool = False
    backing_protected: bool = False

    def is_complete(self, layout: CacheLayout, identity: CacheIdentity) -> bool:
        end = self.materialized_end
        required_start = max(0, end - SWA_WINDOW)
        return (
            0 < end <= 1048576
            and end % layout.reuse_unit == 0
            and self.identity == identity
            and identity.layout_fingerprint == layout.fingerprint
            and self.valid_slots == layout.required_slots
            and 0 <= self.target_swa_start <= required_start
            and end - self.target_swa_start <= layout.swa_entries
            and (
                not layout.draft_enabled
                or (
                    self.draft_swa_start is not None
                    and 0 <= self.draft_swa_start <= required_start
                    and end - self.draft_swa_start <= layout.swa_entries
                )
            )
            and 0 <= self.replay_floor <= self.target_swa_start
            and (
                not layout.draft_enabled
                or (
                    self.draft_swa_start is not None
                    and self.replay_floor <= self.draft_swa_start
                )
            )
            and self.history_ready
            and self.copy_complete
        )


def select_complete_checkpoint(
    layout: CacheLayout,
    identity: CacheIdentity,
    global_blocks: Sequence[GlobalBlock],
    checkpoints: Iterable[MemoryCheckpoint],
    requested_end: int,
) -> Optional[MemoryCheckpoint]:
    """Match a continuous owner chain, then its latest complete SWA checkpoint."""
    if requested_end < 0:
        raise ValueError("requested prefix end must be non-negative")
    if identity.layout_fingerprint != layout.fingerprint:
        raise ValueError("cache identity does not describe this layout")
    ordered = sorted(global_blocks, key=lambda block: block.ordinal)
    if len({block.ordinal for block in ordered}) != len(ordered):
        raise ValueError("duplicate global block ordinal")
    count = 0
    for block in ordered:
        if (
            block.ordinal != count
            or block.identity != identity
            or not block.copy_complete
            or block.valid_slots != layout.global_slots
        ):
            break
        count += 1
    end = min(requested_end, count * layout.reuse_unit)
    candidates = [
        checkpoint
        for checkpoint in checkpoints
        if checkpoint.materialized_end <= end
        and checkpoint.is_complete(layout, identity)
    ]
    return max(candidates, key=lambda item: item.materialized_end, default=None)
