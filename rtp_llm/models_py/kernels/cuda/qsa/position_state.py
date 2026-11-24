"""Fixed-shape masked stores for the model's int32 MRoPE coordinate ring."""

import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice


@triton.jit
def _store(
    pool, slots, values, count: tl.constexpr, stride: tl.constexpr, BLOCK: tl.constexpr
):
    row = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    slot = tl.load(slots + row, row < count, -1)
    for axis in tl.static_range(3):
        value = tl.load(values + row * stride + axis, row < count, 0)
        tl.store(pool + slot * 3 + axis, value, (row < count) & (slot >= 0))


def store_positions(pool, slots, values):
    _store[(triton.cdiv(slots.numel(), 128),)](
        pool,
        slots,
        values,
        slots.numel(),
        values.stride(0),
        BLOCK=128,
    )


@triton.jit
def _position_axis(pair, H_PAIRS: tl.constexpr, W_PAIRS: tl.constexpr):
    return tl.where(
        (pair % 3 == 1) & (pair < 3 * H_PAIRS),
        1,
        tl.where((pair % 3 == 2) & (pair < 3 * W_PAIRS), 2, 0),
    )


@triton.jit
def group_position_angle(
    pool,
    slots,
    in_current,
    current,
    frequency,
    row,
    pair,
    H_PAIRS: tl.constexpr,
    W_PAIRS: tl.constexpr,
):
    axis = _position_axis(pair, H_PAIRS, W_PAIRS)
    fresh = tl.load(in_current + row)
    slot = tl.load(slots + row)
    new = tl.load(current + row * 3 + axis, mask=fresh, other=0)
    saved = tl.load(pool + slot * 3 + axis, mask=~fresh, other=0)
    position = tl.where(fresh, new, saved).to(tl.float32)
    return position * tl.load(frequency + pair)


@triton.jit
def _store_pair_coefficients(cos, sin, row, pair, angle):
    c = libdevice.cos(angle)
    s = libdevice.sin(angle)
    tl.store(cos + row * 64 + pair, c)
    tl.store(cos + row * 64 + pair + 32, c)
    tl.store(sin + row * 64 + pair, s)
    tl.store(sin + row * 64 + pair + 32, s)


@triton.jit
def _query_position_rope(
    positions,
    frequency,
    cos,
    sin,
    STRIDE: tl.constexpr,
    H_PAIRS: tl.constexpr,
    W_PAIRS: tl.constexpr,
):
    row = tl.program_id(0)
    pair = tl.arange(0, 32)
    axis = _position_axis(pair, H_PAIRS, W_PAIRS)
    coordinate = tl.load(positions + row * STRIDE + axis).to(tl.float32)
    angle = coordinate * tl.load(frequency + pair)
    _store_pair_coefficients(cos, sin, row, pair, angle)


@triton.jit
def _group_position_rope(
    pool,
    slots,
    in_current,
    current,
    frequency,
    cos,
    sin,
    H_PAIRS: tl.constexpr,
    W_PAIRS: tl.constexpr,
):
    row = tl.program_id(0)
    pair = tl.arange(0, 32)
    angle = group_position_angle(
        pool, slots, in_current, current, frequency, row, pair, H_PAIRS, W_PAIRS
    )
    _store_pair_coefficients(cos, sin, row, pair, angle)


def query_position_rope(positions, frequency, h_pairs, w_pairs):
    rows = positions.shape[0]
    cos = torch.empty((rows, 64), dtype=torch.bfloat16, device=positions.device)
    sin = torch.empty_like(cos)
    _query_position_rope[(rows,)](
        positions,
        frequency,
        cos,
        sin,
        STRIDE=positions.stride(0),
        H_PAIRS=h_pairs,
        W_PAIRS=w_pairs,
        num_warps=1,
        enable_fp_fusion=False,
    )
    return cos, sin


def group_position_rope(pool, slots, in_current, current, frequency, h_pairs, w_pairs):
    rows = slots.numel()
    cos = torch.empty((rows, 64), dtype=torch.bfloat16, device=pool.device)
    sin = torch.empty_like(cos)
    _group_position_rope[(rows,)](
        pool,
        slots,
        in_current,
        current,
        frequency,
        cos,
        sin,
        H_PAIRS=h_pairs,
        W_PAIRS=w_pairs,
        num_warps=1,
        enable_fp_fusion=False,
    )
    return cos, sin


@triton.jit
def _position_geometry(
    POSITIONS,
    PREFIXES,
    CU,
    TABLE,
    REQUESTS,
    LOGICAL,
    ENDS,
    VALID,
    SLOTS,
    UNDO,
    GROUP_SLOTS,
    GROUP_CURRENT,
    CURRENT_POSITIONS,
    N: tl.constexpr,
    BATCH: tl.constexpr,
    WIDTH: tl.constexpr,
    POOL_BLOCKS: tl.constexpr,
    PAGE: tl.constexpr,
    ROWS: tl.constexpr,
    REQUEST_BLOCK: tl.constexpr,
):
    row = tl.program_id(0) * ROWS + tl.arange(0, ROWS)
    seq = tl.arange(0, REQUEST_BLOCK)
    cut = tl.load(CU + seq + 1, seq < BATCH, other=0)
    request = tl.minimum(
        tl.sum(
            ((row[:, None] >= cut[None, :]) & (seq[None, :] < BATCH)).to(tl.int32), 1
        ),
        BATCH - 1,
    )
    prefix = tl.load(PREFIXES + request)
    begin = tl.load(CU + request)
    finish = tl.load(CU + request + 1)
    logical = prefix + row - begin
    end = prefix + finish - begin - 1
    page = tl.maximum(logical, 0) // PAGE
    block = tl.load(TABLE + request * WIDTH + tl.minimum(page, WIDTH - 1)).to(tl.int64)
    slot = tl.minimum(tl.maximum(block, 0), POOL_BLOCKS - 1) * 8 + (logical & 7)
    valid = (
        (row < tl.load(CU + BATCH))
        & (logical >= 0)
        & (page < WIDTH)
        & (block > 0)
        & (block < POOL_BLOCKS)
    )
    write = valid & (logical + 8 > tl.minimum((page + 1) * PAGE - 1, end))
    start = logical - (logical & 3)
    source = tl.minimum(tl.maximum(begin + start - prefix, 0), N - 1)
    group_page = tl.maximum(start, 0) // PAGE
    group_block = tl.load(
        TABLE + request * WIDTH + tl.minimum(group_page, WIDTH - 1)
    ).to(tl.int64)
    group_slot = tl.minimum(tl.maximum(group_block, 0), POOL_BLOCKS - 1) * 8 + (
        start & 7
    )
    tl.store(REQUESTS + row, request, row < N)
    tl.store(LOGICAL + row, logical, row < N)
    tl.store(ENDS + row, end, row < N)
    tl.store(VALID + row, valid, row < N)
    tl.store(SLOTS + row, slot, row < N)
    tl.store(UNDO + row, tl.where(write, slot, -1), row < N)
    tl.store(GROUP_SLOTS + row, group_slot, row < N)
    tl.store(GROUP_CURRENT + row, start >= prefix, row < N)
    for axis in tl.static_range(3):
        v = tl.load(POSITIONS + source * 3 + axis)
        tl.store(CURRENT_POSITIONS + row * 3 + axis, v, row < N)


def position_geometry(positions, prefixes, cu, table, pool_blocks, page_size):
    rows = positions.shape[0]
    batch = prefixes.numel()
    if not (
        1 <= rows <= 128
        and 1 <= batch <= 32
        and table.ndim == 2
        and table.shape[0] == batch
        and table.shape[1] > 0
        and cu.shape == (batch + 1,)
        and positions.shape == (rows, 3)
        and all(t.is_contiguous() for t in (positions, prefixes, cu, table))
    ):
        return None
    requests = torch.empty(rows, device=positions.device, dtype=torch.int64)
    logical, ends, slots, undo, group_slots = (
        torch.empty_like(requests) for _ in range(5)
    )
    valid = torch.empty(rows, device=positions.device, dtype=torch.bool)
    in_current = torch.empty_like(valid)
    current = torch.empty_like(positions)
    _position_geometry[(triton.cdiv(rows, 16),)](
        positions,
        prefixes,
        cu,
        table,
        requests,
        logical,
        ends,
        valid,
        slots,
        undo,
        group_slots,
        in_current,
        current,
        N=rows,
        BATCH=batch,
        WIDTH=table.shape[1],
        POOL_BLOCKS=pool_blocks,
        PAGE=page_size,
        ROWS=16,
        REQUEST_BLOCK=triton.next_power_of_2(batch),
        num_warps=4,
    )
    return (
        requests,
        logical,
        ends,
        valid,
        slots,
        undo,
        (group_slots, in_current, current),
    )
