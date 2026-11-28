"""PLE fixed-state cache reads and decode writes."""

import torch
import triton
import triton.language as tl


@triton.jit
def _store_decode_rows(
    POOL, IDS, VALUES, WIDTH: tl.constexpr, STRIDE: tl.constexpr, BLOCK: tl.constexpr
):
    row = tl.program_id(0)
    column = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    block = tl.load(IDS + row)
    value = tl.load(VALUES + row * WIDTH + column, column < WIDTH, other=0)
    # Block zero belongs to the allocator, never to a padding request. Even
    # writing its previous value races when multiple dummy rows share it.
    tl.store(POOL + block * STRIDE + column, value, (block > 0) & (column < WIDTH))


def store_decode_rows(pool, blocks, values):
    """Store fixed-size request rows, skipping reserved block zero in Graph."""
    if pool.is_cuda and torch.version.hip is None:
        values = values.reshape(blocks.numel(), -1).contiguous()
        _store_decode_rows[(blocks.numel(), triton.cdiv(values.shape[1], 256))](
            pool,
            blocks,
            values,
            WIDTH=values.shape[1],
            STRIDE=pool.stride(0),
            BLOCK=256,
        )
    else:
        live = blocks > 0
        pool.index_copy_(0, blocks[live], values[live].to(pool.dtype))


@triton.jit(do_not_specialize=["ROWS"])
def _read_blocks_kernel(
    TABLE,
    PAGES,
    OUTPUT,
    ROWS,
    COLUMNS: tl.constexpr,
    STRIDE_ROW: tl.constexpr,
    STRIDE_COL: tl.constexpr,
    POOL_ROWS,
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
