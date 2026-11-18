"""Forward-owned CP readback with one physical copy of an aliased prefix.

The native cache remains sharded. The packed temporary bank stores a common
prefix followed by independent request tails; the ordinary BF16 attention
provider consumes its decoded bank through remapped integer indices.
"""

from dataclasses import dataclass
from functools import lru_cache

import torch


@dataclass(frozen=True)
class SharedGlobalBank:
    packed: torch.Tensor
    prefix_rows: int
    counts: tuple[int, ...]
    tail_offsets: tuple[int, ...]

    @classmethod
    def create(cls, packed, prefix_rows, counts):
        counts = tuple(counts)
        if (type(prefix_rows) is not int or prefix_rows <= 0 or len(counts) < 2
                or any(type(n) is not int or n < prefix_rows for n in counts)
                or not isinstance(packed, torch.Tensor) or packed.dtype != torch.uint8
                or packed.ndim != 2 or packed.shape[1] != 288 or not packed.is_contiguous()):
            raise ValueError("shared global readback requires contiguous 288-byte rows and a proven prefix")
        offsets, total = [], prefix_rows
        for count in counts:
            offsets.append(total)
            total += count - prefix_rows
        if packed.shape[0] != total:
            raise ValueError("shared global readback storage differs from its logical layout")
        return cls(packed, prefix_rows, counts, tuple(offsets))

    def materialize(self, request, count):
        """Legacy Tensor ABI, independently writable from the immutable bank."""
        if (type(request) is not int or not 0 <= request < len(self.counts)
                or type(count) is not int or not 0 <= count <= self.counts[request]):
            raise ValueError("shared global request range is outside its published bank")
        prefix = min(count, self.prefix_rows)
        tail = max(0, count - self.prefix_rows)
        if not tail:
            return self.packed[:prefix].clone()
        begin = self.tail_offsets[request]
        return torch.cat((self.packed[:prefix], self.packed[begin:begin + tail]))


@dataclass(frozen=True)
class SharedGlobalSlice:
    bank: SharedGlobalBank
    request: int


def global_rows(value, count, *, stats=None):
    if isinstance(value, SharedGlobalSlice):
        if stats is not None:
            stats["legacy_materializations"] += 1
        return value.bank.materialize(value.request, count)
    return value[:count]


def compact_offsets(counts, prefix_rows):
    """Storage offsets for prefix-once then all request tails."""
    counts = tuple(counts)
    if (type(prefix_rows) is not int or prefix_rows <= 0 or len(counts) < 2
            or len(set(counts)) != 1 or any(type(n) is not int or n < prefix_rows for n in counts)):
        raise ValueError("compact readback requires a shared prefix and uniform complete requests")
    offsets, total = [], prefix_rows
    for count in counts:
        offsets.append(total)
        total += count - prefix_rows
    return tuple(offsets), total


@lru_cache(maxsize=1)
def _kernels():
    import triton
    import triton.language as tl

    @triton.jit
    def assemble_shared_global_indices(
        metadata, positions, selected, indices, lengths, position_stride,
        selected_row_stride, selected_col_stride,
        TOPK: tl.constexpr, RATIO: tl.constexpr, WINDOW: tl.constexpr,
        WIDTH: tl.constexpr, BLOCK: tl.constexpr, FIELDS: tl.constexpr = 6,
    ):
        # Same stable global-then-SWA compaction as graph_cache_kernels.
        # Ten-field metadata additionally removes holes between SWA pieces.
        row = tl.program_id(0).to(tl.int64)
        column = tl.arange(0, BLOCK)
        position = tl.load(positions + row * position_stride)
        count = tl.load(metadata + row * FIELDS + 1)
        swa_base = tl.load(metadata + row * FIELDS + 2)
        swa_first = tl.load(metadata + row * FIELDS + 3)
        prefix = tl.load(metadata + row * FIELDS + 4)
        tail_base = tl.load(metadata + row * FIELDS + 5)
        swa_position = position - WINDOW + 1 + column - TOPK
        swa_live = (column >= TOPK) & (column < TOPK + WINDOW) & (swa_position >= swa_first)
        values = tl.where(swa_live, swa_base + swa_position - swa_first, -1)
        if FIELDS == 10:
            stop0 = tl.load(metadata + row * FIELDS + 6)
            base1 = tl.load(metadata + row * FIELDS + 7)
            first1 = tl.load(metadata + row * FIELDS + 8)
            stop1 = tl.load(metadata + row * FIELDS + 9)
            in0 = (swa_position >= swa_first) & (swa_position < stop0)
            in1 = (swa_position >= first1) & (swa_position < stop1)
            mapped = tl.where(in0, swa_base + swa_position - swa_first,
                              base1 + swa_position - first1)
            values = tl.where((column >= TOPK) & (column < TOPK + WINDOW) & (in0 | in1), mapped, -1)
        if TOPK:
            entry = tl.load(selected + row * selected_row_stride + column * selected_col_stride,
                            column < TOPK, other=-1)
            live = (column < TOPK) & (column < (position + 1) // RATIO) & (entry >= 0) & (entry < count)
            mapped = tl.where(entry < prefix, entry, tail_base + entry - prefix)
            values = tl.where(live, mapped, values)
        values = values.to(tl.int32)
        live = values >= 0
        rank = tl.cumsum(live.to(tl.int32))
        length = tl.sum(live.to(tl.int32), 0)
        destination = tl.where(live, rank - 1, length + column - rank)
        tl.store(indices + row * WIDTH + destination, values, column < TOPK + WINDOW)
        tl.store(indices + row * WIDTH + column, -1, (column >= TOPK + WINDOW) & (column < WIDTH))
        tl.store(lengths + row, length)

    return triton, assemble_shared_global_indices


def assemble_indices(metadata, positions, selected, ratio, window, width):
    """Existing BF16 ABI with a shared-global and optional two-span SWA map."""
    rows = positions.numel()
    if not (
        positions.is_cuda and positions.ndim == 1 and positions.dtype == torch.int64
        and metadata.ndim == 2 and metadata.shape[0] == rows and metadata.shape[1] in (6, 10)
        and metadata.is_contiguous()
        and metadata.dtype == torch.int64 and metadata.device == positions.device
    ):
        return None
    topk, strides = 0, (0, 0)
    if selected is not None:
        if not (selected.ndim == 2 and selected.shape[0] == rows and ratio > 0
                and selected.device == positions.device and selected.dtype in (torch.int32, torch.int64)):
            return None
        topk, strides = selected.shape[1], selected.stride()
    if not 0 < window <= topk + window <= width:
        return None
    indices = torch.empty((rows, 1, width), dtype=torch.int32, device=positions.device)
    lengths = torch.empty(rows, dtype=torch.int32, device=positions.device)
    if rows:
        triton, kernel = _kernels()
        kernel[(rows,)](metadata, positions, selected, indices, lengths, positions.stride(0),
                       *strides, topk, ratio, window, width, triton.next_power_of_2(width),
                       FIELDS=metadata.shape[1])
    return indices, lengths
