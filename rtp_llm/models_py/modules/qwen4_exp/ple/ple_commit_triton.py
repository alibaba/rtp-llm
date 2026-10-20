"""Fused PLE commit destination verdict and host-mirror payload."""

import torch
import triton
import triton.language as tl


@triton.jit
def _commit_block_verdict_kernel(
    TABLE,
    PAGES,
    OUTPUT,
    B: tl.constexpr,
    COLUMNS: tl.constexpr,
    STRIDE_ROW: tl.constexpr,
    STRIDE_COL: tl.constexpr,
    POOL_ROWS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.arange(0, BLOCK)
    live = row < B
    page = tl.load(PAGES + row, live, other=0)
    safe_page = tl.minimum(tl.maximum(page, 0), COLUMNS - 1)
    block = tl.load(
        TABLE + row.to(tl.int64) * STRIDE_ROW + safe_page * STRIDE_COL, live, other=0
    ).to(tl.int64)
    tl.store(OUTPUT, tl.sum((live & (page < 0)).to(tl.int32), 0) > 0)
    tl.store(OUTPUT + 1, tl.sum((live & (page >= COLUMNS)).to(tl.int32), 0) > 0)
    tl.store(OUTPUT + 2, tl.sum((live & (block <= 0)).to(tl.int32), 0) > 0)
    tl.store(OUTPUT + 3, tl.sum((live & (block >= POOL_ROWS)).to(tl.int32), 0) > 0)
    tl.store(OUTPUT + 4 + row, block, live)


def resolve_commit_blocks(table: torch.Tensor, pages: torch.Tensor, pool_rows: int):
    """Return [four ordered error flags, B block IDs], or None if unsupported.

    This is a device-only operation. The caller must inspect the flags before
    reading or writing persistent pool rows. Clamped lookups never validate an
    invalid logical page.
    """
    if not (
        table.is_cuda
        and pages.is_cuda
        and table.device == pages.device
        and table.ndim == 2
        and table.dtype == torch.int32
        and pages.ndim == 1
        and pages.dtype == torch.int64
        and pages.is_contiguous()
        and 0 < pages.numel() <= 1024
        and table.shape[0] == pages.numel()
        and table.shape[1] > 0
        and pool_rows > 0
    ):
        return None
    result = torch.empty(4 + pages.numel(), dtype=torch.int64, device=pages.device)
    _commit_block_verdict_kernel[(1,)](
        table,
        pages,
        result,
        pages.numel(),
        table.shape[1],
        table.stride(0),
        table.stride(1),
        pool_rows,
        triton.next_power_of_2(pages.numel()),
        num_warps=4,
    )
    return result


@triton.jit
def _write_pair_verdict(
    TABLE,
    OUTPUT,
    row,
    live,
    position,
    page,
    COLS: tl.constexpr,
    ROW_STRIDE: tl.constexpr,
    COL_STRIDE: tl.constexpr,
    POOL: tl.constexpr,
    FLAG: tl.constexpr,
    IDS: tl.constexpr,
):
    safe_page = tl.minimum(page, COLS - 1)
    block = tl.load(
        TABLE + row.to(tl.int64) * ROW_STRIDE + safe_page * COL_STRIDE, live, other=0
    ).to(tl.int64)
    tl.store(OUTPUT + FLAG, tl.sum((live & (position < 0)).to(tl.int32), 0) > 0)
    tl.store(OUTPUT + FLAG + 1, tl.sum((live & (page >= COLS)).to(tl.int32), 0) > 0)
    tl.store(OUTPUT + FLAG + 2, tl.sum((live & (block <= 0)).to(tl.int32), 0) > 0)
    tl.store(OUTPUT + FLAG + 3, tl.sum((live & (block >= POOL)).to(tl.int32), 0) > 0)
    tl.store(OUTPUT + IDS + row, block, live)


@triton.jit
def _commit_pair_verdict_kernel(
    PREFIX,
    ACCEPT,
    STATE_TABLE,
    CTX_TABLE,
    OUTPUT,
    B: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    S_COLS: tl.constexpr,
    S_ROW: tl.constexpr,
    S_COL: tl.constexpr,
    C_COLS: tl.constexpr,
    C_ROW: tl.constexpr,
    C_COL: tl.constexpr,
    S_POOL: tl.constexpr,
    C_POOL: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.arange(0, BLOCK)
    live = row < B
    position = tl.load(PREFIX + row, live, other=0).to(tl.int64)
    position += tl.load(ACCEPT + row, live, other=1).to(tl.int64) - 1
    # Negative positions are rejected before either block list is consumed.
    page = tl.maximum(position, 0) // PAGE_SIZE
    _write_pair_verdict(
        STATE_TABLE,
        OUTPUT,
        row,
        live,
        position,
        page,
        S_COLS,
        S_ROW,
        S_COL,
        S_POOL,
        0,
        8,
    )
    _write_pair_verdict(
        CTX_TABLE,
        OUTPUT,
        row,
        live,
        position,
        page,
        C_COLS,
        C_ROW,
        C_COL,
        C_POOL,
        4,
        8 + B,
    )


def resolve_commit_pair(
    state_table, ctx_table, prefixes, accepted, page_size, state_rows, ctx_rows
):
    """Compute both tag-local destination verdicts, without accessing pools.

    Returns eight ordered flags followed by state IDs and context IDs. Accept
    lengths must already be validated; the owner inspects state flags before
    context flags, then checks duplicate destinations before gathering undo.
    """
    tables = (state_table, ctx_table)
    vectors = (prefixes, accepted)
    if not (
        torch.version.hip is None
        and all(t.is_cuda and t.device == accepted.device for t in tables + vectors)
        and all(t.ndim == 2 and t.dtype == torch.int32 for t in tables)
        and all(
            t.ndim == 1 and t.dtype == torch.int64 and t.is_contiguous()
            for t in vectors
        )
        and 0 < accepted.numel() <= 1024
        and prefixes.numel()
        == state_table.shape[0]
        == ctx_table.shape[0]
        == accepted.numel()
        and min(
            state_table.shape[1], ctx_table.shape[1], state_rows, ctx_rows, page_size
        )
        > 0
    ):
        return None
    batch = accepted.numel()
    output = torch.empty(8 + 2 * batch, device=accepted.device, dtype=torch.int64)
    _commit_pair_verdict_kernel[(1,)](
        prefixes,
        accepted,
        state_table,
        ctx_table,
        output,
        batch,
        page_size,
        state_table.shape[1],
        *state_table.stride(),
        ctx_table.shape[1],
        *ctx_table.stride(),
        state_rows,
        ctx_rows,
        triton.next_power_of_2(batch),
        num_warps=4,
    )
    return output


@triton.jit
def _read_blocks_kernel(
    TABLE,
    PAGES,
    OUTPUT,
    ROWS: tl.constexpr,
    COLUMNS: tl.constexpr,
    STRIDE_ROW: tl.constexpr,
    STRIDE_COL: tl.constexpr,
    POOL_ROWS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    page = tl.load(PAGES + row, row < ROWS, other=0).to(tl.int64)
    safe_page = tl.minimum(tl.maximum(page, 0), COLUMNS - 1)
    block = tl.load(
        TABLE + row.to(tl.int64) * STRIDE_ROW + safe_page * STRIDE_COL,
        row < ROWS,
        other=0,
    ).to(tl.int64)
    valid = (page >= 0) & (page < COLUMNS) & (block > 0) & (block < POOL_ROWS)
    tl.store(OUTPUT + row, tl.where(valid, block, 0), row < ROWS)


def resolve_read_blocks(table, pages, pool_rows):
    """Graph read-only lookup: invalid pages/blocks select reserved dummy zero.

    This is deliberately separate from commit validation. It never validates
    write destinations or changes the host verdict required before publication.
    """
    if not (
        table.is_cuda
        and pages.is_cuda
        and table.device == pages.device
        and table.ndim == 2
        and table.dtype in (torch.int32, torch.int64)
        and pages.ndim == 1
        and pages.dtype in (torch.int32, torch.int64)
        and pages.is_contiguous()
        and 0 < pages.numel() <= 1024
        and table.shape[0] == pages.numel()
        and table.shape[1] > 0
        and pool_rows > 0
        and torch.version.hip is None
    ):
        return None
    out = torch.empty(pages.shape, dtype=torch.int64, device=pages.device)
    _read_blocks_kernel[(triton.cdiv(pages.numel(), 128),)](
        table,
        pages,
        out,
        ROWS=pages.numel(),
        COLUMNS=table.shape[1],
        STRIDE_ROW=table.stride(0),
        STRIDE_COL=table.stride(1),
        POOL_ROWS=pool_rows,
        BLOCK=128,
        num_warps=4,
    )
    return out


@triton.jit
def _target_read_state_kernel(
    PREFIX,
    IDS,
    STATE_TABLE,
    CTX_TABLE,
    STATE_POOL,
    CTX_POOL,
    STATE_OUT,
    HISTORY,
    S_COLS: tl.constexpr,
    C_COLS: tl.constexpr,
    S_ROW: tl.constexpr,
    S_COL: tl.constexpr,
    C_ROW: tl.constexpr,
    C_COL: tl.constexpr,
    S_POOL: tl.constexpr,
    C_POOL: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    STATE_WIDTH: tl.constexpr,
    CONTEXT: tl.constexpr,
    QUERY: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row, tile = tl.program_id(0), tl.program_id(1)
    prefix = tl.load(PREFIX + row).to(tl.int64)
    page = (prefix - 1) // PAGE_SIZE
    sp = tl.minimum(tl.maximum(page, 0), S_COLS - 1)
    cp = tl.minimum(tl.maximum(page, 0), C_COLS - 1)
    sb = tl.load(STATE_TABLE + row * S_ROW + sp * S_COL).to(tl.int64)
    cb = tl.load(CTX_TABLE + row * C_ROW + cp * C_COL).to(tl.int64)
    sb = tl.where(
        (prefix > 0) & (page >= 0) & (page < S_COLS) & (sb > 0) & (sb < S_POOL), sb, 0
    )
    cb = tl.where(
        (prefix > 0) & (page >= 0) & (page < C_COLS) & (cb > 0) & (cb < C_POOL), cb, 0
    )
    col = tile * BLOCK + tl.arange(0, BLOCK)
    value = tl.load(STATE_POOL + sb * STATE_WIDTH + col, col < STATE_WIDTH, other=0)
    tl.store(STATE_OUT + row * STATE_WIDTH + col, value, col < STATE_WIDTH)
    if tile == 0:
        h = tl.arange(0, BLOCK)
        previous = tl.load(CTX_POOL + cb * CONTEXT + h, h < CONTEXT, other=0)
        current = tl.load(
            IDS + row * QUERY + h - CONTEXT,
            (h >= CONTEXT) & (h < CONTEXT + QUERY),
            other=0,
        )
        tl.store(
            HISTORY + row * (CONTEXT + QUERY) + h,
            tl.where(h < CONTEXT, previous, current),
            h < CONTEXT + QUERY,
        )


def gather_target_read_state(
    prefixes, ids, state_table, ctx_table, state_pool, ctx_pool, page_size
):
    """Gather independent tag-local read state and candidate history for Graph.

    Persistent pools are read-only. Every invalid page/block retains the prior
    reserved-row-zero behavior; commit validation stays with its owner.
    """
    tensors = (prefixes, ids, state_table, ctx_table, state_pool, ctx_pool)
    if not (
        all(t.is_cuda and t.device == prefixes.device for t in tensors)
        and torch.version.hip is None
        and state_pool.dtype == torch.bfloat16
        and ctx_pool.dtype == torch.int64
        and prefixes.dtype == ids.dtype == torch.int64
        and state_table.dtype == ctx_table.dtype == torch.int32
        and prefixes.ndim == ids.ndim == 1
        and prefixes.is_contiguous()
        and ids.is_contiguous()
        and state_pool.ndim == 3
        and ctx_pool.ndim == 2
        and state_pool.is_contiguous()
        and ctx_pool.is_contiguous()
        and state_table.ndim == ctx_table.ndim == 2
        and 0 < prefixes.numel() <= 32
        and ids.numel() % prefixes.numel() == 0
        and 0 < ids.numel() // prefixes.numel() <= 32
        and state_table.shape[0] == ctx_table.shape[0] == prefixes.numel()
        and min(
            state_table.shape[1],
            ctx_table.shape[1],
            state_pool.shape[0],
            ctx_pool.shape[0],
        )
        > 0
        and min(state_pool.shape[1:]) > 0
        and 0 < ctx_pool.shape[1] <= 32
        and page_size > 0
    ):
        return None
    batch = prefixes.numel()
    query = ids.numel() // batch
    context = ctx_pool.shape[1]
    width = state_pool.shape[1] * state_pool.shape[2]
    state = torch.empty(
        (batch, *state_pool.shape[1:]), device=prefixes.device, dtype=state_pool.dtype
    )
    history = torch.empty(
        (batch, context + query), device=prefixes.device, dtype=ctx_pool.dtype
    )
    _target_read_state_kernel[(batch, triton.cdiv(width, 1024))](
        prefixes,
        ids,
        state_table,
        ctx_table,
        state_pool,
        ctx_pool,
        state,
        history,
        S_COLS=state_table.shape[1],
        C_COLS=ctx_table.shape[1],
        S_ROW=state_table.stride(0),
        S_COL=state_table.stride(1),
        C_ROW=ctx_table.stride(0),
        C_COL=ctx_table.stride(1),
        S_POOL=state_pool.shape[0],
        C_POOL=ctx_pool.shape[0],
        PAGE_SIZE=page_size,
        STATE_WIDTH=width,
        CONTEXT=context,
        QUERY=query,
        BLOCK=1024,
        num_warps=4,
    )
    return state, history[:, :context], history
