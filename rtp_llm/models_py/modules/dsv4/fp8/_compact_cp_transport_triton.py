"""Default-off compact-CP byte transport; no compressor arithmetic or collectives.
Private unchecked-value leaves: caller retains range/uniqueness/logical-ID gates.
Uses actual split-page byte strides, including storage offsets and page padding.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


def _regions(data, scales, slots):
    if data.ndim != 3 or scales.ndim != 3 or slots.ndim != 1:
        raise ValueError("compact transport expects 3D split regions and 1D slots")
    if data.shape[:2] != scales.shape[:2] or min(data.shape[:2]) <= 0:
        raise ValueError("compact transport pool dimensions")
    db, sb = data.shape[2], scales.shape[2]
    if (db, sb) not in ((128, 4), (576, 8)):
        raise ValueError("unsupported split-row format")
    if data.dtype != torch.uint8 or scales.dtype != torch.uint8:
        raise ValueError("split regions must be raw uint8")
    if slots.dtype not in (torch.int32, torch.int64) or not slots.is_contiguous():
        raise ValueError("slots must be contiguous integers")
    if not (data.device == scales.device == slots.device):
        raise ValueError("compact transport devices differ")
    for tensor, width in ((data, db), (scales, sb)):
        if (
            tensor.stride(2) != 1
            or tensor.stride(1) < width
            or tensor.stride(0) < (tensor.shape[1] - 1) * tensor.stride(1) + width
        ):
            raise ValueError("compact transport invalid byte strides")
    return db, sb, data.shape[1]


@triton.jit
def _pack_compact_bytes(
    data,
    scales,
    slots,
    logical_rows,
    output,
    start,
    data_s0,
    data_s1,
    scale_s0,
    scale_s1,
    ENTRIES: tl.constexpr,
    DB: tl.constexpr,
    SB: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    cols = tl.arange(0, BLOCK)
    slot = tl.load(slots + row).to(tl.int64)
    block, entry = slot // ENTRIES, slot % ENTRIES
    identity = (tl.load(logical_rows + row).to(tl.int64) + start).to(tl.uint64)
    # Bound shifts even on lanes later discarded by the byte selector.
    prefix = ((identity >> ((cols & 7) * 8)) & 255).to(tl.uint8)
    values = tl.load(
        data + block * data_s0 + entry * data_s1 + cols - 8,
        mask=(cols >= 8) & (cols < 8 + DB),
        other=0,
    )
    sf = tl.load(
        scales + block * scale_s0 + entry * scale_s1 + cols - 8 - DB,
        mask=(cols >= 8 + DB) & (cols < 8 + DB + SB),
        other=0,
    )
    result = tl.where(cols < 8, prefix, tl.where(cols < 8 + DB, values, sf))
    tl.store(output + row * (8 + DB + SB) + cols, result, mask=cols < 8 + DB + SB)


@triton.jit
def _scatter_compact_bytes(
    data,
    scales,
    packed,
    receiver_slots,
    data_s0,
    data_s1,
    scale_s0,
    scale_s1,
    ENTRIES: tl.constexpr,
    DB: tl.constexpr,
    SB: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    cols = tl.arange(0, BLOCK)
    slot = tl.load(receiver_slots + row).to(tl.int64)
    block, entry = slot // ENTRIES, slot % ENTRIES
    value = tl.load(
        packed + row * (8 + DB + SB) + 8 + cols, mask=cols < DB + SB, other=0
    )
    tl.store(data + block * data_s0 + entry * data_s1 + cols, value, mask=cols < DB)
    tl.store(
        scales + block * scale_s0 + entry * scale_s1 + cols - DB,
        value,
        mask=(cols >= DB) & (cols < DB + SB),
    )


def pack_compact_bytes(data, scales, slots, logical_rows, start):
    db, sb, entries = _regions(data, scales, slots)
    if (
        logical_rows.shape != slots.shape
        or logical_rows.dtype not in (torch.int32, torch.int64)
        or not logical_rows.is_contiguous()
        or logical_rows.device != data.device
    ):
        raise ValueError("logical row metadata mismatch")
    if type(start) is not int or not 0 <= start <= 2**63 - 4096:
        raise ValueError("absolute chunk start outside int64 range")
    out = torch.empty(
        (slots.numel(), 8 + db + sb), dtype=torch.uint8, device=data.device
    )
    if slots.numel():
        _pack_compact_bytes[(slots.numel(),)](
            data,
            scales,
            slots,
            logical_rows,
            out,
            start,
            data.stride(0),
            data.stride(1),
            scales.stride(0),
            scales.stride(1),
            ENTRIES=entries,
            DB=db,
            SB=sb,
            BLOCK=triton.next_power_of_2(8 + db + sb),
            num_warps=4,
        )
    return out


def scatter_compact_bytes(data, scales, packed, receiver_slots):
    db, sb, entries = _regions(data, scales, receiver_slots)
    if (
        packed.dtype != torch.uint8
        or packed.shape != (receiver_slots.numel(), 8 + db + sb)
        or packed.device != data.device
        or not packed.is_contiguous()
    ):
        raise ValueError("packed receive buffer mismatch")
    if receiver_slots.numel():
        _scatter_compact_bytes[(receiver_slots.numel(),)](
            data,
            scales,
            packed,
            receiver_slots,
            data.stride(0),
            data.stride(1),
            scales.stride(0),
            scales.stride(1),
            ENTRIES=entries,
            DB=db,
            SB=sb,
            BLOCK=triton.next_power_of_2(db + sb),
            num_warps=4,
        )
