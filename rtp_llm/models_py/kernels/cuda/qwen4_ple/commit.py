"""Fused PLE commit destination verdict and host-mirror payload."""

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


@triton.jit
def _store_decode_state(
    STATE_POOL,
    CTX_POOL,
    STATE_BLOCKS,
    CTX_BLOCKS,
    STATE,
    HISTORY,
    STATE_ROWS: tl.constexpr,
    STATE_WIDTH: tl.constexpr,
    STATE_BATCH_STRIDE: tl.constexpr,
    STATE_ROW_STRIDE: tl.constexpr,
    STATE_POOL_BLOCK_STRIDE: tl.constexpr,
    STATE_POOL_ROW_STRIDE: tl.constexpr,
    CTX_SIZE: tl.constexpr,
    CTX_BLOCK: tl.constexpr,
    CTX_POOL_STRIDE: tl.constexpr,
    HISTORY_ROW_STRIDE: tl.constexpr,
    HISTORY_TOKEN_STRIDE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    batch = tl.program_id(0).to(tl.int64)
    tile = tl.program_id(1)
    col = tile * BLOCK + tl.arange(0, BLOCK)
    state_row = col // STATE_WIDTH
    state_col = col % STATE_WIDTH
    block = tl.load(STATE_BLOCKS + batch)
    mask = (block > 0) & (col < STATE_ROWS * STATE_WIDTH)
    value = tl.load(
        STATE + batch * STATE_BATCH_STRIDE + state_row * STATE_ROW_STRIDE + state_col,
        mask,
        other=0,
    )
    tl.store(
        STATE_POOL
        + block * STATE_POOL_BLOCK_STRIDE
        + state_row * STATE_POOL_ROW_STRIDE
        + state_col,
        value,
        mask,
    )
    if tile == 0:
        # Drop the oldest n-gram token, using the full history's row stride.
        ctx_col = tl.arange(0, CTX_BLOCK)
        ctx_block = tl.load(CTX_BLOCKS + batch)
        ctx_mask = (ctx_block > 0) & (ctx_col < CTX_SIZE)
        token = tl.load(
            HISTORY + batch * HISTORY_ROW_STRIDE + (ctx_col + 1) * HISTORY_TOKEN_STRIDE,
            ctx_mask,
            other=0,
        )
        tl.store(CTX_POOL + ctx_block * CTX_POOL_STRIDE + ctx_col, token, ctx_mask)


def try_store_decode_state(
    state_pool, ctx_pool, state_blocks, ctx_blocks, new_state, history
):
    """Commit both cache regions using validated decode destinations."""
    if not (
        state_pool.is_cuda
        and torch.version.hip is None
        and all(
            x.device == state_pool.device
            for x in (ctx_pool, state_blocks, ctx_blocks, new_state, history)
        )
        and state_pool.ndim == new_state.ndim == 3
        and ctx_pool.ndim == history.ndim == 2
        and state_pool.dtype == new_state.dtype == torch.bfloat16
        and ctx_pool.dtype == history.dtype == torch.int64
        and state_pool.shape[1:] == new_state.shape[1:]
        and 0 < new_state.shape[0] == history.shape[0]
        and state_pool.shape[1] * state_pool.shape[2] > 0
        and 0 < ctx_pool.shape[1] == history.shape[1] - 1 <= 32
        and state_pool.stride(-1) == new_state.stride(-1) == ctx_pool.stride(-1) == 1
        and all(
            blocks.shape == (new_state.shape[0],)
            and blocks.dtype == torch.int64
            and blocks.is_contiguous()
            for blocks in (state_blocks, ctx_blocks)
        )
    ):
        return False
    block = 2048
    _store_decode_state[
        (
            new_state.shape[0],
            triton.cdiv(new_state.shape[1] * new_state.shape[2], block),
        )
    ](
        state_pool,
        ctx_pool,
        state_blocks,
        ctx_blocks,
        new_state,
        history,
        STATE_ROWS=new_state.shape[1],
        STATE_WIDTH=new_state.shape[2],
        STATE_BATCH_STRIDE=new_state.stride(0),
        STATE_ROW_STRIDE=new_state.stride(1),
        STATE_POOL_BLOCK_STRIDE=state_pool.stride(0),
        STATE_POOL_ROW_STRIDE=state_pool.stride(1),
        CTX_SIZE=ctx_pool.shape[1],
        CTX_BLOCK=triton.next_power_of_2(ctx_pool.shape[1]),
        CTX_POOL_STRIDE=ctx_pool.stride(0),
        HISTORY_ROW_STRIDE=history.stride(0),
        HISTORY_TOKEN_STRIDE=history.stride(1),
        BLOCK=block,
        num_warps=4,
    )
    return True


@triton.jit(do_not_specialize=["length"])
def _store_prefill_state(
    STATE_POOL,
    CTX_POOL,
    NORMED,
    HISTORY,
    STATE_BLOCKS,
    CTX_BLOCKS,
    length,
    PAGE_SIZE: tl.constexpr,
    STATE_ROWS: tl.constexpr,
    STATE_WIDTH: tl.constexpr,
    STATE_POOL_STRIDE: tl.constexpr,
    CTX_SIZE: tl.constexpr,
    CTX_BLOCK: tl.constexpr,
    CTX_POOL_STRIDE: tl.constexpr,
    HISTORY_STRIDE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    page = tl.program_id(0).to(tl.int64)
    tile = tl.program_id(1)
    end = tl.minimum((page + 1) * PAGE_SIZE, length)
    sb = tl.load(STATE_BLOCKS + page).to(tl.int64)
    cb = tl.load(CTX_BLOCKS + page).to(tl.int64)
    live = (sb > 0) & (cb > 0)
    col = tile * BLOCK + tl.arange(0, BLOCK)
    mask = live & (col < STATE_ROWS * STATE_WIDTH)
    value = tl.load(NORMED + (end - STATE_ROWS) * STATE_WIDTH + col, mask, other=0)
    tl.store(STATE_POOL + sb * STATE_POOL_STRIDE + col, value, mask)
    if tile == 0:
        ctx_col = tl.arange(0, CTX_BLOCK)
        ctx_mask = live & (ctx_col < CTX_SIZE)
        token = tl.load(HISTORY + (end + ctx_col) * HISTORY_STRIDE, ctx_mask, other=0)
        tl.store(CTX_POOL + cb * CTX_POOL_STRIDE + ctx_col, token, ctx_mask)


def try_store_prefill_state(
    state_pool, ctx_pool, normed, history, state_blocks, ctx_blocks, page_size
):
    """Commit one request's page suffixes using validated prefill destinations."""
    if not (
        state_pool.is_cuda
        and torch.version.hip is None
        and all(
            x.device == state_pool.device
            for x in (ctx_pool, normed, history, state_blocks, ctx_blocks)
        )
        and state_pool.ndim == normed.ndim == 3
        and ctx_pool.ndim == history.ndim == 2
        and state_pool.dtype == normed.dtype == torch.bfloat16
        and ctx_pool.dtype == history.dtype == torch.int64
        and normed.shape[0] == history.shape[0] == 1
        and 0 < state_pool.shape[2] == normed.shape[2]
        and 0 < state_pool.shape[1] <= min(normed.shape[1], page_size)
        and 0 < ctx_pool.shape[1] == history.shape[1] - normed.shape[1] <= 32
        and state_pool.stride()[1:] == (normed.shape[2], 1)
        and normed.is_contiguous()
        and ctx_pool.stride(1) == 1
    ):
        return False
    length = normed.shape[1]
    pages = triton.cdiv(length, page_size)
    if not all(
        blocks.shape == (pages,)
        and blocks.dtype in (torch.int32, torch.int64)
        and blocks.is_contiguous()
        for blocks in (state_blocks, ctx_blocks)
    ):
        return False
    state_rows, width = state_pool.shape[1:]
    block = 2048
    _store_prefill_state[(pages, triton.cdiv(state_rows * width, block))](
        state_pool,
        ctx_pool,
        normed,
        history,
        state_blocks,
        ctx_blocks,
        length,
        PAGE_SIZE=page_size,
        STATE_ROWS=state_rows,
        STATE_WIDTH=width,
        STATE_POOL_STRIDE=state_pool.stride(0),
        CTX_SIZE=ctx_pool.shape[1],
        CTX_BLOCK=triton.next_power_of_2(ctx_pool.shape[1]),
        CTX_POOL_STRIDE=ctx_pool.stride(0),
        HISTORY_STRIDE=history.stride(1),
        BLOCK=block,
        num_warps=4,
    )
    return True


@triton.jit(do_not_specialize=["B"])
def _commit_block_verdict_kernel(
    TABLE,
    PAGES,
    OUTPUT,
    B,
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
    bad_position = tl.sum((live & (position < 0)).to(tl.int32), 0) > 0
    bad_page = tl.sum((live & (page >= COLS)).to(tl.int32), 0) > 0
    bad_unallocated = tl.sum((live & (block <= 0)).to(tl.int32), 0) > 0
    bad_pool = tl.sum((live & (block >= POOL)).to(tl.int32), 0) > 0
    tl.store(OUTPUT + FLAG, bad_position)
    tl.store(OUTPUT + FLAG + 1, bad_page)
    tl.store(OUTPUT + FLAG + 2, bad_unallocated)
    tl.store(OUTPUT + FLAG + 3, bad_pool)
    # Clamped storage keeps speculative pool gathers in bounds; every clamped
    # value trips one of the verdict flags above and aborts before pool writes.
    safe_block = tl.minimum(tl.maximum(block, 0), POOL - 1)
    tl.store(OUTPUT + IDS + row, safe_block, live)
    return safe_block, bad_position | bad_page | bad_unallocated | bad_pool


@triton.jit(do_not_specialize=["B"])
def _commit_pair_verdict_kernel(
    PREFIX,
    ACCEPT,
    STATE_TABLE,
    CTX_TABLE,
    OUTPUT,
    B,
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
def _graph_page_block(
    table,
    row,
    page,
    cols: tl.constexpr,
    row_stride: tl.constexpr,
    col_stride: tl.constexpr,
    pool_rows: tl.constexpr,
):
    valid_page = (page >= 0) & (page < cols)
    block = tl.load(table + row * row_stride + page * col_stride, valid_page, other=0)
    return tl.where((block > 0) & (block < pool_rows), block, 0).to(tl.int64)


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
    STATE_DEST,
    CTX_DEST,
    S_COLS: tl.constexpr,
    C_COLS: tl.constexpr,
    S_ROW: tl.constexpr,
    S_COL: tl.constexpr,
    C_ROW: tl.constexpr,
    C_COL: tl.constexpr,
    S_POOL: tl.constexpr,
    C_POOL: tl.constexpr,
    S_STRIDE: tl.constexpr,
    C_STRIDE: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    STATE_WIDTH: tl.constexpr,
    CONTEXT: tl.constexpr,
    QUERY: tl.constexpr,
    BLOCK: tl.constexpr,
    DECODE_EOS: tl.constexpr,
):
    row, tile = tl.program_id(0), tl.program_id(1)
    prefix = tl.load(PREFIX + row).to(tl.int64)
    if DECODE_EOS is not None:
        prefix -= 1
    page = (prefix - 1) // PAGE_SIZE
    page = tl.where(prefix > 0, page, -1)
    sb = _graph_page_block(STATE_TABLE, row, page, S_COLS, S_ROW, S_COL, S_POOL)
    col = tile * BLOCK + tl.arange(0, BLOCK)
    read_live = prefix > 0 if DECODE_EOS is not None else True
    value = tl.load(
        STATE_POOL + sb * S_STRIDE + col, (col < STATE_WIDTH) & read_live, other=0
    )
    tl.store(STATE_OUT + row * STATE_WIDTH + col, value, col < STATE_WIDTH)
    if tile == 0:
        h = tl.arange(0, BLOCK)
        cb = _graph_page_block(CTX_TABLE, row, page, C_COLS, C_ROW, C_COL, C_POOL)
        if DECODE_EOS is not None:
            dest = tl.where(prefix > 0, prefix // PAGE_SIZE, -1)
            sw = _graph_page_block(STATE_TABLE, row, dest, S_COLS, S_ROW, S_COL, S_POOL)
            cw = _graph_page_block(CTX_TABLE, row, dest, C_COLS, C_ROW, C_COL, C_POOL)
            tl.store(STATE_DEST + row, sw)
            tl.store(CTX_DEST + row, cw)
        previous = tl.load(
            CTX_POOL + cb * C_STRIDE + h,
            (h < CONTEXT) & read_live,
            other=0 if DECODE_EOS is None else DECODE_EOS,
        )
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


def _gather_read_state(
    prefixes,
    ids,
    state_table,
    ctx_table,
    state_pool,
    ctx_pool,
    page_size,
    *,
    decode_eos
):
    decode = decode_eos is not None
    tensors = (prefixes, ids, state_table, ctx_table, state_pool, ctx_pool)
    if not (
        all(t.is_cuda and t.device == prefixes.device for t in tensors)
        and torch.version.hip is None
        and state_pool.dtype == torch.bfloat16
        and ctx_pool.dtype == torch.int64
        and ids.dtype == torch.int64
        and prefixes.dtype == (torch.int32 if decode else torch.int64)
        and state_table.dtype == ctx_table.dtype == torch.int32
        and prefixes.ndim == ids.ndim == 1
        and prefixes.is_contiguous()
        and ids.is_contiguous()
        and state_pool.ndim == 3
        and ctx_pool.ndim == 2
        and state_pool.stride()[1:] == (state_pool.shape[2], 1)
        and ctx_pool.stride(1) == 1
        and state_table.ndim == ctx_table.ndim == 2
        and 0 < prefixes.numel() <= (2048 if decode else 32)
        and ids.numel() % prefixes.numel() == 0
        and 0 < ids.numel() // prefixes.numel() <= (1 if decode else 32)
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
    state_dest = torch.empty_like(ids) if decode else None
    ctx_dest = torch.empty_like(ids) if decode else None
    _target_read_state_kernel[(batch, triton.cdiv(width, 1024))](
        prefixes,
        ids,
        state_table,
        ctx_table,
        state_pool,
        ctx_pool,
        state,
        history,
        state_dest,
        ctx_dest,
        S_COLS=state_table.shape[1],
        C_COLS=ctx_table.shape[1],
        S_ROW=state_table.stride(0),
        S_COL=state_table.stride(1),
        C_ROW=ctx_table.stride(0),
        C_COL=ctx_table.stride(1),
        S_POOL=state_pool.shape[0],
        C_POOL=ctx_pool.shape[0],
        S_STRIDE=state_pool.stride(0),
        C_STRIDE=ctx_pool.stride(0),
        PAGE_SIZE=page_size,
        STATE_WIDTH=width,
        CONTEXT=context,
        QUERY=query,
        BLOCK=1024,
        DECODE_EOS=decode_eos,
        num_warps=4,
    )
    return state, history, state_dest, ctx_dest


def gather_target_read_state(
    prefixes, ids, state_table, ctx_table, state_pool, ctx_pool, page_size
):
    """Gather tag-local state and candidate history for target-verify Graph."""
    result = _gather_read_state(
        prefixes,
        ids,
        state_table,
        ctx_table,
        state_pool,
        ctx_pool,
        page_size,
        decode_eos=None,
    )
    if result is None:
        return None
    state, history, _, _ = result
    return state, history[:, : ctx_pool.shape[1]], history


def prepare_decode_graph(
    sequence_plus_one, ids, state_table, ctx_table, state_pool, ctx_pool, page_size, eos
):
    """Gather Decode inputs and resolve both write destinations from live metadata."""
    return _gather_read_state(
        sequence_plus_one,
        ids,
        state_table,
        ctx_table,
        state_pool,
        ctx_pool,
        page_size,
        decode_eos=eos,
    )
