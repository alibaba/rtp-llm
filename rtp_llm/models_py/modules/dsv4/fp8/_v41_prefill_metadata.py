"""Device-only V4.1 prefill slot and scoring-bound metadata.

Slots preserve the physical-owner/kernel-page distinction and STATE's
intra-block ring slices. Prefill producers supply nonnegative positions
and valid request IDs; invalid positions are skipped before table loads.
"""

from __future__ import annotations

import os

import torch
import triton
import triton.language as tl


def _enabled():
    return os.environ.get("DSV41_FUSED_PREFILL_METADATA", "1") != "0"


def _integer_vector(value, rows, device):
    return (
        value.device == device
        and value.ndim == 1
        and value.numel() == rows
        and value.dtype in (torch.int32, torch.int64)
        and value.stride(0) > 0
    )


@triton.jit
def _prefill_slots_kernel(
    positions,
    requests,
    table,
    seq_ends,
    out,
    ROWS: tl.constexpr,
    REQUESTS: tl.constexpr,
    COLS: tl.constexpr,
    POS_STRIDE: tl.constexpr,
    REQ_STRIDE: tl.constexpr,
    TABLE_STRIDE: tl.constexpr,
    END_STRIDE: tl.constexpr,
    EB: tl.constexpr,
    TPB: tl.constexpr,
    OWNER_TPB: tl.constexpr,
    RATIO: tl.constexpr,
    CP: tl.constexpr,
    RANK: tl.constexpr,
    STATE: tl.constexpr,
    HAS_ENDS: tl.constexpr,
    TILE: tl.constexpr,
):
    row = tl.program_id(0) * TILE + tl.arange(0, TILE)
    pos = tl.load(positions + row * POS_STRIDE, row < ROWS, other=0).to(tl.int64)
    req = tl.load(requests + row * REQ_STRIDE, row < ROWS, other=0).to(tl.int64)
    valid = (row < ROWS) & (pos >= 0) & (req >= 0) & (req < REQUESTS)
    pos = tl.maximum(pos, 0)
    if STATE:
        raw_column = pos // TPB
        column = raw_column % COLS
        offset = pos % (EB * CP)
        valid &= offset // EB == RANK
        offset %= EB
        if HAS_ENDS:
            seq_end = tl.load(seq_ends + req * END_STRIDE, valid, other=0).to(tl.int64)
            effective_end = tl.minimum((raw_column + 1) * TPB, seq_end)
            valid &= pos + EB * CP >= effective_end
    else:
        owner_block = pos // OWNER_TPB
        column = owner_block // CP * (OWNER_TPB // TPB) + pos % OWNER_TPB // TPB
        offset = pos % TPB // RATIO
        valid &= (owner_block % CP == RANK) & (column < COLS)
        valid &= (pos + 1) % RATIO == 0
    block = tl.load(table + req * TABLE_STRIDE + column, valid, other=0).to(tl.int64)
    slot = tl.where(valid & (block > 0), block * EB + offset, -1)
    tl.store(out + row, slot, row < ROWS)


def try_slot_mapping(
    positions,
    requests,
    table,
    entries_per_block,
    tokens_per_block,
    ratio,
    cp_size=1,
    cp_rank=0,
    *,
    owner_tokens_per_block=None,
    state=False,
    seq_ends=None,
):
    """Return int64 slots, or None before launch for unsupported metadata."""
    if not (
        _enabled()
        and positions.is_cuda
        and torch.version.hip is None
        and positions.ndim == 1
        and positions.numel() > 0
        and _integer_vector(positions, positions.numel(), positions.device)
        and _integer_vector(requests, positions.numel(), positions.device)
        and table.device == positions.device
        and table.ndim == 2
        and table.dtype in (torch.int32, torch.int64)
        and table.shape[0] > 0
        and table.shape[1] > 0
        and table.stride(1) == 1
        and table.stride(0) >= table.shape[1]
        and entries_per_block > 0
        and tokens_per_block > 0
        and ratio in (1, 2)
        and cp_size > 0
        and 0 <= cp_rank < cp_size
    ):
        return None
    owner_tpb = owner_tokens_per_block or tokens_per_block
    if owner_tpb <= 0 or owner_tpb % tokens_per_block:
        return None
    if seq_ends is not None and not _integer_vector(
        seq_ends, table.shape[0], positions.device
    ):
        return None
    out = torch.empty(positions.shape, dtype=torch.int64, device=positions.device)
    _prefill_slots_kernel[(triton.cdiv(positions.numel(), 256),)](
        positions,
        requests,
        table,
        seq_ends if seq_ends is not None else positions,
        out,
        positions.numel(),
        table.shape[0],
        table.shape[1],
        positions.stride(0),
        requests.stride(0),
        table.stride(0),
        seq_ends.stride(0) if seq_ends is not None else 1,
        entries_per_block,
        tokens_per_block,
        owner_tpb,
        ratio,
        cp_size,
        cp_rank,
        state,
        seq_ends is not None,
        256,
    )
    return out


@triton.jit
def _prefill_bounds_kernel(
    positions,
    bounds,
    ROWS: tl.constexpr,
    STRIDE: tl.constexpr,
    WIDTH: tl.constexpr,
    RATIO: tl.constexpr,
    TILE: tl.constexpr,
):
    row = tl.program_id(0) * TILE + tl.arange(0, TILE)
    pos = tl.load(positions + row * STRIDE, row < ROWS, other=-1).to(tl.int64)
    length = tl.minimum(tl.maximum(pos + 1, 0) // RATIO, WIDTH).to(tl.int32)
    tl.store(bounds + row, 0, row < ROWS)
    tl.store(bounds + ROWS + row, length, row < ROWS)


def try_score_bounds(positions, width, ratio):
    """Build reusable int32 starts/ends directly from device positions."""
    if not (
        _enabled()
        and positions.is_cuda
        and torch.version.hip is None
        and positions.ndim == 1
        and positions.numel() > 0
        and _integer_vector(positions, positions.numel(), positions.device)
        and 0 < width < 2**31
        and ratio in (1, 2)
    ):
        return None
    bounds = torch.empty(
        (2, positions.numel()), dtype=torch.int32, device=positions.device
    )
    _prefill_bounds_kernel[(triton.cdiv(positions.numel(), 256),)](
        positions,
        bounds,
        positions.numel(),
        positions.stride(0),
        width,
        ratio,
        256,
    )
    return bounds.unbind(0)
