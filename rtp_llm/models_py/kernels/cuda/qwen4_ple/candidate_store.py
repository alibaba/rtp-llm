"""Publish PLE verification candidates into the reserved state blocks."""

import torch
import triton
import triton.language as tl


@triton.jit
def _store_candidate_states_kernel(
    INITIAL, CANDIDATE, CONTEXT, IDS, PREFIXES,
    STATE_TABLE, CONTEXT_TABLE, STATE_POOL, CONTEXT_POOL,
    PAGE_SIZE: tl.constexpr, TABLE_WIDTH: tl.constexpr,
    STATE_ROWS: tl.constexpr, CONTEXT_ROWS: tl.constexpr,
    LENGTH: tl.constexpr, WIDTH: tl.constexpr, CTX: tl.constexpr,
    I0: tl.constexpr, I1: tl.constexpr, I2: tl.constexpr,
    C0: tl.constexpr, C1: tl.constexpr, C2: tl.constexpr,
    H0: tl.constexpr, H1: tl.constexpr, T0: tl.constexpr, T1: tl.constexpr,
    ST0: tl.constexpr, ST1: tl.constexpr, CT0: tl.constexpr, CT1: tl.constexpr,
    SP0: tl.constexpr, SP1: tl.constexpr, SP2: tl.constexpr,
    CP0: tl.constexpr, CP1: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row, step, tile = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    prefix = tl.load(PREFIXES + row)
    slot = prefix // PAGE_SIZE + step
    allocated = (prefix > 0) & (slot < TABLE_WIDTH)
    state_block = tl.load(STATE_TABLE + row * ST0 + slot * ST1, allocated, 0).to(tl.int64)
    ctx_block = tl.load(CONTEXT_TABLE + row * CT0 + slot * CT1, allocated, 0).to(tl.int64)
    if (state_block > 0) & (state_block < STATE_ROWS) & (ctx_block > 0) & (ctx_block < CONTEXT_ROWS):
        index = tile * BLOCK + tl.arange(0, BLOCK)
        history_row, column = index // WIDTH, index % WIDTH
        position = history_row + step + 1
        valid = history_row < LENGTH
        previous = tl.load(
            INITIAL + row * I0 + position * I1 + column * I2,
            valid & (position < LENGTH),
            0,
        )
        candidate = tl.load(
            CANDIDATE + row * C0 + (position - LENGTH) * C1 + column * C2,
            valid & (position >= LENGTH),
            0,
        )
        tl.store(
            STATE_POOL + state_block * SP0 + history_row * SP1 + column * SP2,
            tl.where(position < LENGTH, previous, candidate),
            valid,
        )
        if tile == 0:
            offset = tl.arange(0, BLOCK)
            position = offset + step + 1
            previous_ctx = tl.load(
                CONTEXT + row * H0 + position * H1,
                (offset < CTX) & (position < CTX),
                0,
            )
            candidate_ctx = tl.load(
                IDS + row * T0 + (position - CTX) * T1,
                (offset < CTX) & (position >= CTX),
                0,
            )
            tl.store(
                CONTEXT_POOL + ctx_block * CP0 + offset * CP1,
                tl.where(position < CTX, previous_ctx, candidate_ctx),
                offset < CTX,
            )


def store_candidate_states(
    initial, candidate, context, ids, state_pool, context_pool,
    prefixes, page_size, state_table, context_table,
):
    """Write each verification step in the same logical slots as GDN."""
    batch, length, width = initial.shape
    steps = candidate.shape[1]
    table_width = min(state_table.shape[1], context_table.shape[1])
    if initial.is_cuda:
        state_table = state_table.to(initial.device, non_blocking=True)
        context_table = context_table.to(initial.device, non_blocking=True)
        _store_candidate_states_kernel[(batch, steps, triton.cdiv(length * width, 512))](
            initial, candidate, context, ids, prefixes,
            state_table, context_table, state_pool, context_pool,
            page_size, table_width, state_pool.shape[0], context_pool.shape[0],
            length, width, context.shape[1],
            *initial.stride(), *candidate.stride(), *context.stride(), *ids.stride(),
            *state_table.stride(), *context_table.stride(),
            *state_pool.stride(), *context_pool.stride(), 512,
        )
        return

    positions = prefixes[:, None] // page_size + torch.arange(steps)
    slots = positions.clamp(max=table_width - 1).long()
    state_blocks = state_table.gather(1, slots).long()
    ctx_blocks = context_table.gather(1, slots).long()
    live = (
        (prefixes[:, None] > 0) & (positions < table_width)
        & (state_blocks > 0) & (state_blocks < state_pool.shape[0])
        & (ctx_blocks > 0) & (ctx_blocks < context_pool.shape[0])
    )
    states = torch.cat((initial, candidate), dim=1).unfold(1, length, 1)[:, 1:]
    contexts = torch.cat((context, ids), dim=1).unfold(1, context.shape[1], 1)[:, 1:]
    state_pool.index_copy_(0, state_blocks[live], states.transpose(2, 3)[live])
    context_pool.index_copy_(0, ctx_blocks[live], contexts[live])
