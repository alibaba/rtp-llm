"""Byte-exact CP cache packing with strided storage and explicit slot lifetimes."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import torch

SLOT_BYTES = 8  # int64 slot ID prefix
HALO_ROWS_PER_TAIL = 4  # = overlap * ratio for the CSA ratio-4 compressor
CSA_RATIO = 4
HCA_RATIO = 128

# Compression-only byte model constants. SWA uses its separate BF16 gather;
# CSA boundary exchange carries hidden rows and their positions.
HIDDEN = 4096
HIDDEN_BYTES = 2
POSITION_BYTES = 8
SWA_GATHER_BYTES_PER_LAYER = 1 << 20
CSA_HALO_BYTES_PER_LAYER = (
    2 * HALO_ROWS_PER_TAIL * (HIDDEN * HIDDEN_BYTES + POSITION_BYTES)
)  # 65600


# ---------------------------------------------------------------------------
# Row layouts
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RowLayout:
    """One entry's byte layout; pages store separate data/scale regions plus optional padding."""

    name: str
    data_bytes: int
    scale_bytes: int
    compress_ratio: int  # 1 for the SWA (uncompressed) cache

    @property
    def entry_bytes(self) -> int:
        return self.data_bytes + self.scale_bytes

    @property
    def packed_row_bytes(self) -> int:
        return SLOT_BYTES + self.entry_bytes

    def check(self) -> None:
        if self.data_bytes <= 0 or self.scale_bytes <= 0:
            raise ValueError(f"{self.name}: empty regions")
        if self.compress_ratio <= 0:
            raise ValueError(f"{self.name}: bad ratio")


MAIN_KV_LAYOUT = RowLayout("main_kv", 576, 8, CSA_RATIO)
INDEXER_LAYOUT = RowLayout("indexer", 128, 4, CSA_RATIO)
SWA_KV_LAYOUT = RowLayout("swa_kv", 576, 8, 1)
HCA_KV_LAYOUT = RowLayout("hca_kv", 576, 8, HCA_RATIO)


# ---------------------------------------------------------------------------
# Ownership geometry (frozen zigzag)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CPOwnership:
    """Zigzag geometry for a padded chunk divisible by 2 * cp_size."""

    cp_size: int
    chunk_len: int

    def __post_init__(self) -> None:
        if self.cp_size <= 0:
            raise ValueError("cp_size must be positive")
        if self.chunk_len % (2 * self.cp_size) != 0:
            raise ValueError(
                f"chunk_len {self.chunk_len} not a multiple of 2*cp_size "
                f"{2 * self.cp_size}"
            )

    @property
    def local_rows(self) -> int:
        return self.chunk_len // self.cp_size

    @property
    def segment_len(self) -> int:
        return self.chunk_len // (2 * self.cp_size)

    def owned_segments(self, rank: int) -> Tuple[int, ...]:
        if not 0 <= rank < self.cp_size:
            raise ValueError(f"rank {rank} outside 0..{self.cp_size - 1}")
        return (rank, 2 * self.cp_size - 1 - rank)

    def local_to_global(self, rank: int, dtype=torch.int64) -> torch.Tensor:
        """[local_rows] global row of each local row, production zigzag order."""
        seg = self.segment_len
        i = torch.arange(self.local_rows, dtype=dtype)
        half = self.local_rows // 2
        first = i < half
        return torch.where(
            first,
            rank * seg + i,
            self.chunk_len - (rank + 1) * seg + (i - half),
        )

    def owned_global_rows(self, rank: int, dtype=torch.int64) -> torch.Tensor:
        return self.local_to_global(rank, dtype)

    def global_to_owner(self, dtype=torch.int64) -> torch.Tensor:
        """[chunk_len] owning rank of every global row."""
        seg = self.segment_len
        g = torch.arange(self.chunk_len, dtype=dtype)
        seg_id = g // seg
        return torch.where(seg_id < self.cp_size, seg_id, 2 * self.cp_size - 1 - seg_id)

    def raw_intervals(
        self, rank: int, chunk_start: int = 0
    ) -> Tuple[Tuple[int, int], ...]:
        """Return both disjoint half-open global intervals; do not assume contiguous ownership."""
        seg = self.segment_len
        return tuple(
            (chunk_start + s * seg, chunk_start + (s + 1) * seg)
            for s in self.owned_segments(rank)
        )

    # ---- tails / halo ----------------------------------------------------

    def owned_tail_rows(self, rank: int, tail_rows: int) -> torch.Tensor:
        """Publish both owned tails with fixed count; skip the exchange entirely for CP1."""
        seg = self.segment_len
        out: List[torch.Tensor] = []
        for s in self.owned_segments(rank):
            end = (s + 1) * seg
            start = end - tail_rows
            if start < 0:
                raise ValueError("tail_rows exceeds segment length")
            out.append(torch.arange(start, end, dtype=torch.int64))
        return torch.cat(out)

    def halo_rows_for(self, rank: int, tail_rows: int) -> torch.Tensor:
        """Import the preceding tail for each owned segment except the sequence's first."""
        seg = self.segment_len
        out: List[torch.Tensor] = []
        for s in self.owned_segments(rank):
            start = s * seg
            if start == 0:
                continue
            out.append(torch.arange(start - tail_rows, start, dtype=torch.int64))
        if not out:
            return torch.empty(0, dtype=torch.int64)
        return torch.cat(out)

    def halo_source_rank(self, rank: int) -> Dict[int, int]:
        """rank -> number of halo rows it must publish for this rank."""
        needs = self.halo_rows_for(rank, HALO_ROWS_PER_TAIL)
        if needs.numel() == 0:
            return {}
        owners = self.global_to_owner()[needs]
        counts: Dict[int, int] = {}
        for o in owners.tolist():
            counts[o] = counts.get(o, 0) + 1
        return counts

    def external_halo_sources(self, rank: int) -> Dict[int, int]:
        """Return only cross-rank halo rows; adjacent self-owned segments need no exchange."""
        return {src: n for src, n in self.halo_source_rank(rank).items() if src != rank}


# ---------------------------------------------------------------------------
# Slot validity / dtype contracts
# ---------------------------------------------------------------------------


class PoisonedSlotError(ValueError):
    """Raised when a slot that must be real is a sentinel or out of range."""


def slot_is_real(slot: int) -> bool:
    return slot >= 0


def _require_int_slots(slots: torch.Tensor, what: str = "slots") -> None:
    if not isinstance(slots, torch.Tensor):
        raise ValueError(f"{what} must be a tensor, got {type(slots).__name__}")
    if slots.dtype not in (torch.int32, torch.int64):
        raise ValueError(
            f"{what} must be an integer dtype (int32/int64), got {slots.dtype} "
            "— fractional slot ids must never be silently truncated"
        )
    if slots.dim() != 1:
        raise ValueError(f"{what} must be 1-D, got {tuple(slots.shape)}")


def validate_pack_slots(slots: torch.Tensor) -> torch.Tensor:
    """Select real slots; -1 padding must never be published."""
    _require_int_slots(slots)
    return torch.nonzero(slots >= 0, as_tuple=False).reshape(-1)


# ---------------------------------------------------------------------------
# Split-page pool views and region normalization
# ---------------------------------------------------------------------------


def split_region_views(
    pool_raw: torch.Tensor,
    layout: RowLayout,
    *,
    num_blocks: int,
    entries_per_block: int,
    page_stride_bytes: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return storage-sharing [blocks, entries, bytes] data/scale views with page strides."""
    layout.check()
    if pool_raw.dtype != torch.uint8:
        raise ValueError(f"raw pool must be uint8, got {pool_raw.dtype}")
    if not pool_raw.is_contiguous():
        raise ValueError("raw pool must be contiguous")
    entry_bytes = layout.entry_bytes
    region_bytes = entries_per_block * entry_bytes
    page = region_bytes if page_stride_bytes is None else int(page_stride_bytes)
    if page < region_bytes:
        raise ValueError(
            f"page stride {page} smaller than data+scale region {region_bytes}"
        )
    need = int(num_blocks) * page
    if pool_raw.numel() < need:
        raise ValueError(
            f"raw pool has {pool_raw.numel()} bytes, need >= {need} "
            f"({num_blocks} pages x {page})"
        )
    # Slice bytes, not the leading dimension: callers may pass an entire
    # contiguous cache tensor with more pages than this active view needs.
    # view preserves the parent storage; no scatter destination is copied.
    pool = pool_raw.view(-1)[:need].view(int(num_blocks), page)
    data = pool[:, : entries_per_block * layout.data_bytes].view(
        int(num_blocks), int(entries_per_block), layout.data_bytes
    )
    scale = pool[
        :,
        entries_per_block * layout.data_bytes : entries_per_block * entry_bytes,
    ].view(int(num_blocks), int(entries_per_block), layout.scale_bytes)
    return data, scale


def _normalize_region(
    region: torch.Tensor,
    layout_bytes: int,
    num_blocks: int,
    entries_per_block: int,
    name: str,
    layout_name: str,
) -> torch.Tensor:
    """Accept a real ``[blocks, entries, bytes]`` region (any page stride) or
    a contiguous legacy flat form, and return the 3-D region."""
    if not isinstance(region, torch.Tensor) or region.dtype != torch.uint8:
        raise ValueError(
            f"{layout_name}.{name} pool must be a uint8 tensor, got "
            f"{getattr(region, 'dtype', type(region).__name__)}"
        )
    total = int(num_blocks) * int(entries_per_block)
    if region.dim() == 3:
        # Strided views with inter-page gaps are the production form.
        if tuple(region.shape) != (
            int(num_blocks),
            int(entries_per_block),
            int(layout_bytes),
        ):
            raise ValueError(
                f"{layout_name}.{name} region shape {tuple(region.shape)} != "
                f"({num_blocks}, {entries_per_block}, {layout_bytes})"
            )
        byte_stride = region.stride(2)
        entry_stride = region.stride(1)
        page_stride = region.stride(0)
        page_span = (int(entries_per_block) - 1) * entry_stride + int(layout_bytes)
        # Metadata-only guard: preserve page padding, but reject zero-stride
        # expansion and overlapping entries/pages before any write occurs.
        # Production bytes within an entry are contiguous by writer contract.
        if (
            byte_stride != 1
            or entry_stride < int(layout_bytes)
            or page_stride < page_span
        ):
            raise ValueError(
                f"{layout_name}.{name} region has overlapping or unsupported "
                f"strides {region.stride()}; require contiguous bytes, disjoint "
                "entries and disjoint pages"
            )
        return region
    if region.dim() == 1 and region.numel() == total * layout_bytes:
        if not region.is_contiguous():
            raise ValueError(
                f"{layout_name}.{name} flat region must be contiguous "
                "(pass the strided [blocks, entries, bytes] views otherwise)"
            )
        return region.view(int(num_blocks), int(entries_per_block), layout_bytes)
    if region.dim() == 2 and tuple(region.shape) == (total, layout_bytes):
        if not region.is_contiguous():
            raise ValueError(
                f"{layout_name}.{name} flat region must be contiguous "
                "(pass the strided [blocks, entries, bytes] views otherwise)"
            )
        return region.view(int(num_blocks), int(entries_per_block), layout_bytes)
    raise ValueError(
        f"{layout_name}.{name} region must be [blocks, entries, {layout_bytes}] "
        f"(strided ok) or a contiguous flat form of the same size; got shape "
        f"{tuple(region.shape)}"
    )


def _agree_devices(named: Sequence[Tuple[str, torch.Tensor]]) -> torch.device:
    device = named[0][1].device
    for n, t in named[1:]:
        if t.device != device:
            raise ValueError(
                f"device mismatch: {n} on {t.device} vs {named[0][0]} on {device}"
            )
    return device


# ---------------------------------------------------------------------------
# Pack / scatter
# ---------------------------------------------------------------------------


def _check_slots_in_range(slots: torch.Tensor, total_rows: int, what: str) -> None:
    """Checked-path validation: ONE host sync on success, details only on
    failure.  Callers with pre-validated plans use ``check=False`` and pay
    no host conversions at all."""
    if slots.numel() == 0:
        return
    bad = (slots < 0) | (slots >= total_rows)
    if bool(bad.any().item()):  # the single sync of the checked path
        vals = slots[bad].tolist()
        negative = [v for v in vals if v < 0]
        if negative:
            raise PoisonedSlotError(f"{what} called with sentinel slot {min(negative)}")
        raise PoisonedSlotError(f"{what} slot {max(vals)} >= pool rows {total_rows}")


def pack_rows(
    pool_data: torch.Tensor,
    pool_scale: torch.Tensor,
    slots: torch.Tensor,
    layout: RowLayout,
    *,
    num_blocks: int,
    entries_per_block: int,
    check: bool = True,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Pack [slot i64 little-endian, data, scale] rows from the strided cache.

    check=False avoids host synchronization and requires a validated plan."""
    layout.check()
    _require_int_slots(slots)
    total_rows = int(num_blocks) * int(entries_per_block)
    data_region = _normalize_region(
        pool_data,
        layout.data_bytes,
        num_blocks,
        entries_per_block,
        "data",
        layout.name,
    )
    scale_region = _normalize_region(
        pool_scale,
        layout.scale_bytes,
        num_blocks,
        entries_per_block,
        "scale",
        layout.name,
    )
    _agree_devices(
        [("slots", slots), ("data pool", pool_data), ("scale pool", pool_scale)]
        + ([("out", out)] if out is not None else [])
    )
    if check:
        _check_slots_in_range(slots, total_rows, "pack")

    n = int(slots.numel())
    row_bytes = layout.packed_row_bytes
    if out is None:
        out = torch.empty(n, row_bytes, dtype=torch.uint8, device=slots.device)
    else:
        if out.dtype != torch.uint8 or tuple(out.shape) != (n, row_bytes):
            raise ValueError(
                f"out must be uint8 [{n},{row_bytes}], got {out.dtype} "
                f"{tuple(out.shape)}"
            )
    slots64 = slots.to(torch.int64).contiguous()
    out[:, :SLOT_BYTES] = slots64.view(torch.uint8).reshape(n, SLOT_BYTES)
    blk = (slots64 // entries_per_block).to(torch.int64)
    off = (slots64 % entries_per_block).to(torch.int64)
    out[:, SLOT_BYTES : SLOT_BYTES + layout.data_bytes] = data_region[blk, off]
    out[:, SLOT_BYTES + layout.data_bytes :] = scale_region[blk, off]
    return out


def scatter_packed_rows(
    pool_data: torch.Tensor,
    pool_scale: torch.Tensor,
    packed: torch.Tensor,
    layout: RowLayout,
    *,
    num_blocks: int,
    entries_per_block: int,
    check: bool = True,
) -> torch.Tensor:
    """Scatter in place through real cache strides; a flattened copy would lose writes.

    Checked mode rejects invalid or duplicate slots; unchecked mode requires a valid plan.
    """
    layout.check()
    row_bytes = layout.packed_row_bytes
    if packed.dim() != 2 or packed.size(1) != row_bytes:
        raise ValueError(
            f"packed rows must be [N,{row_bytes}], got {tuple(packed.shape)}"
        )
    if packed.dtype != torch.uint8:
        raise ValueError(f"packed rows must be uint8, got {packed.dtype}")
    total_rows = int(num_blocks) * int(entries_per_block)
    data_region = _normalize_region(
        pool_data,
        layout.data_bytes,
        num_blocks,
        entries_per_block,
        "data",
        layout.name,
    )
    scale_region = _normalize_region(
        pool_scale,
        layout.scale_bytes,
        num_blocks,
        entries_per_block,
        "scale",
        layout.name,
    )
    _agree_devices(
        [("packed", packed), ("data pool", pool_data), ("scale pool", pool_scale)]
    )
    slots = (
        packed[:, :SLOT_BYTES]
        .contiguous()
        .view(torch.int64)
        .reshape(-1)
        .to(torch.int64)
    )
    if check and slots.numel():
        in_range = ((slots >= 0) & (slots < total_rows)).all()
        _, counts = torch.unique(slots, return_counts=True)
        unique = (counts <= 1).all()
        if not bool((in_range & unique).item()):  # the single sync
            vals = slots.tolist()
            negative = [v for v in vals if v < 0]
            if negative:
                raise PoisonedSlotError("received packed row with sentinel slot")
            big = [v for v in vals if v >= total_rows]
            if big:
                raise PoisonedSlotError(
                    f"received packed slot {max(big)} >= pool rows {total_rows}"
                )
            dups = sorted({v for v in vals if vals.count(v) > 1})
            raise ValueError(
                f"conflicting duplicate destination slots in one scatter: "
                f"{dups[:8]}"
            )
    blk = slots // entries_per_block
    off = slots % entries_per_block
    data_region[blk, off] = packed[:, SLOT_BYTES : SLOT_BYTES + layout.data_bytes]
    scale_region[blk, off] = packed[:, SLOT_BYTES + layout.data_bytes :]
    return slots


def round_trip_check(
    pool_data: torch.Tensor,
    pool_scale: torch.Tensor,
    packed: torch.Tensor,
    layout: RowLayout,
    *,
    num_blocks: int,
    entries_per_block: int,
) -> bool:
    """Byte-identity diagnostic (host sync; fixtures/debug only)."""
    slots = packed[:, :SLOT_BYTES].contiguous().view(torch.int64).reshape(-1)
    back = pack_rows(
        pool_data,
        pool_scale,
        slots,
        layout,
        num_blocks=num_blocks,
        entries_per_block=entries_per_block,
        check=False,
    )
    return bool(torch.equal(back, packed))


# ---------------------------------------------------------------------------
# Fixed-count replication plan
# ---------------------------------------------------------------------------


class CPWritePathNotEligible(Exception):
    """The chunk cannot use packed-row replication; caller must fall back."""


@dataclass(frozen=True)
class ReplicationPlan:
    """Fixed per-chunk row count; reject geometry that would vary the all-gather shape."""

    cp_size: int
    chunk_len: int
    layout: RowLayout
    published_rows_per_rank: int
    compressed_rows_global: int
    contiguous_suffix_global: int = 0

    @property
    def gathered_rows(self) -> int:
        return self.published_rows_per_rank * self.cp_size


def plan_replication(
    ownership: CPOwnership, layout: RowLayout, chunk_start: int = 0
) -> ReplicationPlan:
    """Require aligned chunk starts and segments so compressed publish counts agree."""
    if layout.compress_ratio > 1 and int(chunk_start) % layout.compress_ratio != 0:
        raise CPWritePathNotEligible(
            f"{layout.name}: chunk_start {chunk_start} not aligned to "
            f"ratio {layout.compress_ratio}"
        )
    if layout.compress_ratio == 1:
        return ReplicationPlan(
            cp_size=ownership.cp_size,
            chunk_len=ownership.chunk_len,
            layout=layout,
            published_rows_per_rank=ownership.local_rows,
            compressed_rows_global=ownership.chunk_len,
        )
    seg = ownership.segment_len
    r = layout.compress_ratio
    if seg % r != 0:
        raise CPWritePathNotEligible(
            f"{layout.name}: segment {seg} not divisible by ratio {r}"
        )
    per_rank = ownership.local_rows // r
    # global compressed rows fully inside the chunk
    global_rows = ownership.chunk_len // r
    leftover = ownership.chunk_len % r
    return ReplicationPlan(
        cp_size=ownership.cp_size,
        chunk_len=ownership.chunk_len,
        layout=layout,
        published_rows_per_rank=per_rank,
        compressed_rows_global=global_rows,
        contiguous_suffix_global=leftover,
    )


def local_compressed_slots(
    ownership: CPOwnership,
    rank: int,
    layout: RowLayout,
    compress_slot_of_global: Dict[int, int],
    *,
    chunk_start: int = 0,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Map owned compression boundaries to slots in absolute-position order; reject gaps."""
    r = layout.compress_ratio
    if r > 1 and int(chunk_start) % r != 0:
        raise CPWritePathNotEligible(
            f"{layout.name}: chunk_start {chunk_start} not aligned to ratio {r}"
        )
    if r == 1:
        rel = ownership.local_to_global(rank)
    else:
        boundaries = torch.arange(r - 1, ownership.chunk_len, r, dtype=torch.int64)
        owners = ownership.global_to_owner()[boundaries]
        rel = boundaries[owners == rank]
    glob = rel + int(chunk_start)
    slots = []
    for g in glob.tolist():
        if g not in compress_slot_of_global:
            raise PoisonedSlotError(f"no slot mapped for compressed row {g}")
        slots.append(int(compress_slot_of_global[g]))
    return torch.tensor(slots, dtype=torch.int64, device=device)


# ---------------------------------------------------------------------------
# First-port scope (compressor-only) and its byte model
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FirstPortScope:
    """Pack compressed caches with CSA halos; preserve the separate BF16 SWA gather."""

    packed_layouts: Tuple[RowLayout, ...] = (MAIN_KV_LAYOUT, INDEXER_LAYOUT)
    unchanged_layouts: Tuple[RowLayout, ...] = (SWA_KV_LAYOUT,)
    csa_halo_exchange: bool = True
    hca_halo_exchange: bool = False


FIRST_PORT_SCOPE = FirstPortScope()


def plan_first_port(
    ownership: CPOwnership, chunk_start: int = 0
) -> Dict[str, ReplicationPlan]:
    """Fixed-count plans for the first-port packed layouts (fail closed per
    layout on ineligible geometry)."""
    return {
        layout.name: plan_replication(ownership, layout, chunk_start=chunk_start)
        for layout in FIRST_PORT_SCOPE.packed_layouts
    }


def compression_only_input_bytes(
    *,
    layers: int,
    csa: int,
    hca: int,
    local_rows: int = 1024,
) -> Dict[str, int]:
    """Estimate per-rank CP input buffer bytes, not wire traffic or latency."""
    main = (csa * (local_rows // CSA_RATIO) + hca * (local_rows // HCA_RATIO)) * (
        MAIN_KV_LAYOUT.packed_row_bytes
    )
    indexer = csa * (local_rows // CSA_RATIO) * INDEXER_LAYOUT.packed_row_bytes
    swa = layers * SWA_GATHER_BYTES_PER_LAYER
    csa_halo = csa * CSA_HALO_BYTES_PER_LAYER
    total = swa + main + indexer + csa_halo
    return {
        "swa": swa,
        "main": main,
        "indexer": indexer,
        "csa_halo": csa_halo,
        "total": total,
        "collectives": layers + 4 * csa + hca,
    }


COMPRESSION_ONLY_STAGES = {
    0: dict(
        layers=22, csa=10, hca=10, baseline_input_MiB=162.0, expected_total=25645952
    ),
    1: dict(
        layers=21, csa=11, hca=10, baseline_input_MiB=171.0, expected_total=24850368
    ),
}


# ---------------------------------------------------------------------------
# Slot lifetime (overlap / back-to-back chunks)
# ---------------------------------------------------------------------------


@dataclass
class SlotLifetimeGuard:
    """Prevent slot reuse until every owning chunk is released.

    Snapshot slots and match completion tokens; callers must establish device completion.
    """

    _inflight: Dict[int, Tuple[int, ...]] = field(default_factory=dict)
    _fences: Dict[int, object] = field(default_factory=dict)
    _replicated: set = field(default_factory=set)

    def begin_chunk(self, chunk_id: int, slots: torch.Tensor) -> None:
        _require_int_slots(slots)
        if chunk_id in self._inflight:
            raise RuntimeError(f"chunk {chunk_id} already in flight")
        snapshot = tuple(
            sorted({int(v) for v in slots.detach().reshape(-1).to("cpu").tolist()})
        )
        for cid, owner_snapshot in self._inflight.items():
            overlap = set(owner_snapshot) & set(snapshot)
            if overlap:
                raise RuntimeError(
                    f"chunk {chunk_id} reuses slots still in flight from "
                    f"chunk {cid}: {sorted(overlap)[:8]}"
                )
        self._inflight[chunk_id] = snapshot

    def mark_replicated(self, chunk_id: int, fence: Optional[object] = None) -> None:
        if chunk_id not in self._inflight:
            raise RuntimeError(f"chunk {chunk_id} not in flight")
        self._replicated.add(chunk_id)
        if fence is not None:
            self._fences[chunk_id] = fence

    def end_chunk(self, chunk_id: int, fence: Optional[object] = None) -> None:
        if chunk_id in self._inflight and chunk_id not in self._replicated:
            raise RuntimeError(
                f"chunk {chunk_id} ended before its rows were replicated"
            )
        recorded = self._fences.get(chunk_id)
        if recorded is not None and fence is not recorded:
            raise RuntimeError(
                f"chunk {chunk_id} end must carry the completion fence token "
                "recorded at mark_replicated"
            )
        self._inflight.pop(chunk_id, None)
        self._replicated.discard(chunk_id)
        self._fences.pop(chunk_id, None)

    @property
    def inflight(self) -> Sequence[int]:
        return sorted(self._inflight)
