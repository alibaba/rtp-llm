"""Fixed-shape masked stores for the model's int32 MRoPE coordinate ring."""

import torch
import triton
import triton.language as tl


@triton.jit
def _target_slots(
    prefix,
    cu,
    table,
    undo,
    history,
    current,
    visible_lengths,
    compressed_lengths,
    state_table,
    kv_table,
    ROWS: tl.constexpr,
    BATCH: tl.constexpr,
    POOL: tl.constexpr,
    COLS: tl.constexpr,
    TABLE_STRIDE: tl.constexpr,
    REQUESTS: tl.constexpr,
    Q: tl.constexpr,
    Q_TILE: tl.constexpr,
    STATE_COLS: tl.constexpr,
    KV_COLS: tl.constexpr,
    STATE_STRIDES: tl.constexpr,
    KV_STRIDES: tl.constexpr,
    COL_TILE: tl.constexpr,
):
    row = tl.program_id(0)
    req = tl.arange(0, REQUESTS)
    ends = tl.load(cu + req + 1, req < BATCH, other=2147483647)
    owner = tl.minimum(
        tl.sum(((row >= ends) & (req < BATCH)).to(tl.int32), 0), BATCH - 1
    )
    begin = tl.load(cu + owner).to(tl.int64)
    end = tl.load(cu + owner + 1).to(tl.int64)
    base = tl.load(prefix + owner).to(tl.int64)
    logical = base + row - begin
    page = tl.maximum(logical, 0) // 128
    block = tl.load(table + owner * TABLE_STRIDE + tl.minimum(page, COLS - 1))
    last = tl.minimum(base + end - begin - 1, (page + 1) * 128 - 1)
    valid = (row < tl.load(cu + BATCH)) & (logical >= 0)
    valid &= (page < COLS) & (block > 0) & (block < POOL)
    write = valid & (logical + 8 > last)
    tl.store(undo + row, tl.where(write, block.to(tl.int64) * 8 + (logical & 7), -1))

    start = logical & -4
    src = tl.minimum(tl.maximum(begin + start - base, 0), ROWS - 1)
    page = tl.minimum(tl.maximum(start, 0) // 128, COLS - 1)
    block = tl.load(table + owner * TABLE_STRIDE + page).to(tl.int64)
    slot = tl.minimum(tl.maximum(block, 0), POOL - 1) * 8 + (start & 7)
    tl.store(history + row, slot)
    tl.store(current + row, tl.where(start >= base, src, -1))

    if Q:
        # One program per request also produces the lengths shared by all QSA layers.
        if row % Q == 0:
            request = row // Q
            active = True
            if COL_TILE:
                cols = tl.arange(0, COL_TILE)
                state = tl.load(
                    state_table + request * STATE_STRIDES[0] + cols * STATE_STRIDES[1],
                    cols < STATE_COLS,
                    other=0,
                )
                kv = tl.load(
                    kv_table + request * KV_STRIDES[0] + cols * KV_STRIDES[1],
                    cols < KV_COLS,
                    other=0,
                )
                active = tl.sum(((state != 0) | (kv != 0)).to(tl.int32), 0) > 0
            step = tl.arange(0, Q_TILE)
            length = tl.where(active, tl.load(prefix + request) + step + 1, 0)
            tl.store(visible_lengths + row + step, length, step < Q)
            tl.store(compressed_lengths + row + step, length >> 2, step < Q)


def target_position_slots(
    prefix, cu, table, rows, pool_blocks, *, lengths=None, padding_tables=None
):
    undo = torch.empty(rows, dtype=torch.int64, device=table.device)
    history = torch.empty_like(undo)
    current = torch.empty_like(undo)
    visible, compressed = lengths if lengths is not None else (None, None)
    query_len = visible.shape[1] if visible is not None else 0
    state, kv = padding_tables if padding_tables is not None else (None, None)
    state_cols = state.shape[1] if state is not None else 0
    kv_cols = kv.shape[1] if kv is not None else 0
    _target_slots[(rows,)](
        prefix,
        cu,
        table,
        undo,
        history,
        current,
        ROWS=rows,
        BATCH=prefix.numel(),
        POOL=pool_blocks,
        COLS=table.shape[1],
        TABLE_STRIDE=table.stride(0),
        REQUESTS=triton.next_power_of_2(prefix.numel()),
        visible_lengths=visible,
        compressed_lengths=compressed,
        state_table=state,
        kv_table=kv,
        Q=query_len,
        Q_TILE=triton.next_power_of_2(query_len) if query_len else 1,
        STATE_COLS=state_cols,
        KV_COLS=kv_cols,
        STATE_STRIDES=state.stride() if state is not None else (),
        KV_STRIDES=kv.stride() if kv is not None else (),
        COL_TILE=triton.next_power_of_2(max(state_cols, kv_cols)) if state_cols else 0,
        num_warps=1,
    )
    return undo, history, current


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
