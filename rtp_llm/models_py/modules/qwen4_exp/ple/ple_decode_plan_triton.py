"""Read-only S=1 PLE cache plan and token-history materialization."""

from typing import NamedTuple

import torch
import triton
import triton.language as tl


class DecodeCachePlan(NamedTuple):
    state_read: torch.Tensor
    state_write: torch.Tensor
    context_write: torch.Tensor
    history: torch.Tensor


@triton.jit
def _decode_block(
    Table,
    row,
    page,
    live,
    COLS: tl.constexpr,
    ROW_STRIDE: tl.constexpr,
    COL_STRIDE: tl.constexpr,
    POOL_ROWS: tl.constexpr,
):
    safe = tl.minimum(tl.maximum(page, 0), COLS - 1)
    block = tl.load(
        Table + row.to(tl.int64) * ROW_STRIDE + safe * COL_STRIDE,
        live,
        other=0,
    ).to(tl.int64)
    valid = (page >= 0) & (page < COLS) & (block > 0) & (block < POOL_ROWS)
    return tl.where(valid, block, 0), valid


@triton.jit
def _decode_cache_plan(
    Lengths,
    Ids,
    StateTable,
    ContextTable,
    ContextPool,
    Blocks,
    History,
    Bad,
    B: tl.constexpr,
    S_COLS: tl.constexpr,
    S_ROW: tl.constexpr,
    S_COL: tl.constexpr,
    C_COLS: tl.constexpr,
    C_ROW: tl.constexpr,
    C_COL: tl.constexpr,
    S_ROWS: tl.constexpr,
    C_ROWS: tl.constexpr,
    BLOCK: tl.constexpr,
    GRAPH_MODE: tl.constexpr,
):
    tile = tl.program_id(0)
    row = tile.to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    live = row < B
    length = tl.load(Lengths + row, live, other=0).to(tl.int64)
    if GRAPH_MODE:
        length -= 1  # The replay buffer is int32 cached length plus one.
    # Clamp before subtraction so even INT64_MIN cannot wrap to a live page.
    read_page = (tl.maximum(length, 1) - 1) // 64
    write_page = tl.maximum(length, 0) // 64
    if GRAPH_MODE:
        # Match the original Graph floor-divide and dummy block contract.
        read_page = tl.where(length > 0, read_page, -1)
        write_page = tl.where(length >= 0, write_page, -1)
    sr, ok_sr = _decode_block(
        StateTable, row, read_page, live, S_COLS, S_ROW, S_COL, S_ROWS
    )
    sw, ok_sw = _decode_block(
        StateTable, row, write_page, live, S_COLS, S_ROW, S_COL, S_ROWS
    )
    cr, ok_cr = _decode_block(
        ContextTable, row, read_page, live, C_COLS, C_ROW, C_COL, C_ROWS
    )
    cw, ok_cw = _decode_block(
        ContextTable, row, write_page, live, C_COLS, C_ROW, C_COL, C_ROWS
    )
    read_history = live
    if not GRAPH_MODE:
        read_history = live & (length > 0) & ok_sr & ok_sw & ok_cr & ok_cw
    # In Graph, each lookup maps invalid ids to row zero independently. In
    # particular, history must read the dummy row, which need not be zeroed.
    first = tl.load(ContextPool + cr * 2, read_history, other=0)
    second = tl.load(ContextPool + cr * 2 + 1, read_history, other=0)
    token = tl.load(Ids + row, live, other=0).to(tl.int64)
    tl.store(Blocks + row, sr, live)
    tl.store(Blocks + B + row, sw, live)
    tl.store(Blocks + 2 * B + row, cw, live)
    tl.store(History + row * 3, first, live)
    tl.store(History + row * 3 + 1, second, live)
    tl.store(History + row * 3 + 2, token, live)
    if not GRAPH_MODE:
        # Each CTA owns one verdict; no atomics, initialization or grid barrier.
        tl.store(Bad + tile, tl.max((live & ~read_history).to(tl.int32), 0))


def try_plan_decode_cache(
    lengths,
    ids,
    state_table,
    context_table,
    state_pool,
    context_pool,
    page_size,
    *,
    graph_mode=False,
):
    """Plan PLE cache accesses without writing persistent state.

    Eager takes raw lengths (possibly pinned CPU) and returns None on invalid
    metadata so the original checks still report the error. Graph takes the
    device length-plus-one replay buffer and uses the existing dummy block
    mapping independently for every lookup, without a host verdict readback.
    """
    if graph_mode:
        length_supported = lengths.dtype == torch.int32 and lengths.device == ids.device
    else:
        length_supported = lengths.dtype in (torch.int32, torch.int64) and (
            lengths.device == ids.device
            or (lengths.device.type == "cpu" and lengths.is_pinned())
        )
    if not (
        page_size == 64
        and ids.is_cuda
        and torch.version.hip is None
        and ids.ndim == lengths.ndim == 1
        and ids.dtype in (torch.int32, torch.int64)
        and length_supported
        and ids.is_contiguous()
        and lengths.is_contiguous()
        and 0 < ids.numel() == lengths.numel()
        and all(
            table.ndim == 2
            and table.shape[0] == ids.numel()
            and table.shape[1] > 0
            and table.dtype == torch.int32
            for table in (state_table, context_table)
        )
        and state_pool.ndim == 3
        and state_pool.shape[1:] == (9, 10240)
        and state_pool.dtype == torch.bfloat16
        and context_pool.ndim == 2
        and context_pool.shape[1] == 2
        and context_pool.dtype == torch.int64
        and context_pool.is_contiguous()
        and min(state_pool.shape[0], context_pool.shape[0]) > 0
        and all(
            tensor.device == ids.device
            for tensor in (state_table, context_table, state_pool, context_pool)
        )
        and torch.cuda.get_device_capability(ids.device) == (9, 0)
        and (graph_mode or not torch.cuda.is_current_stream_capturing())
    ):
        return None
    batch = ids.numel()
    # Retain the small-batch geometry; larger batches add independent CTAs
    # instead of growing one CTA's register footprint with the whole batch.
    block = min(triton.next_power_of_2(batch), 128)
    tiles = triton.cdiv(batch, block)
    blocks = torch.empty((3, batch), device=ids.device, dtype=torch.int64)
    history = torch.empty((batch, 3), device=ids.device, dtype=torch.int64)
    bad = None if graph_mode else torch.empty(tiles, device=ids.device, dtype=torch.int32)
    device_lengths = (
        lengths if graph_mode else lengths.to(device=ids.device, non_blocking=True)
    )
    _decode_cache_plan[(tiles,)](
        device_lengths,
        ids,
        state_table,
        context_table,
        context_pool,
        blocks,
        history,
        bad,
        B=batch,
        S_COLS=state_table.shape[1],
        S_ROW=state_table.stride(0),
        S_COL=state_table.stride(1),
        C_COLS=context_table.shape[1],
        C_ROW=context_table.stride(0),
        C_COL=context_table.stride(1),
        S_ROWS=state_pool.shape[0],
        C_ROWS=context_pool.shape[0],
        BLOCK=block,
        GRAPH_MODE=graph_mode,
        num_warps=1 if block <= 64 else 4,
    )
    if bad is not None and bool((bad if tiles == 1 else bad.any()).item()):
        return None
    return DecodeCachePlan(blocks[0], blocks[1], blocks[2], history)
