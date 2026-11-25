"""Select accepted PLE state and immutable undo directly from their owners.

Like vLLM's accepted-offset Mamba state copies, index the accepted window in
the kernel instead of materializing a concatenated history. RTP's prepare,
cross-rank consensus, tentative commit and rollback boundaries stay separate.
The caller must validate accept lengths and physical destinations first.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _prepare_values_kernel(
    ACCEPT,
    INITIAL,
    CANDIDATE,
    CONTEXT,
    IDS,
    STATE_POOL,
    CTX_POOL,
    STATE_BLOCKS,
    CTX_BLOCKS,
    VALUES,
    CTX_VALUES,
    UNDO,
    CTX_UNDO,
    LENGTH: tl.constexpr,
    WIDTH: tl.constexpr,
    CTX: tl.constexpr,
    I0: tl.constexpr,
    I1: tl.constexpr,
    I2: tl.constexpr,
    C0: tl.constexpr,
    C1: tl.constexpr,
    C2: tl.constexpr,
    H0: tl.constexpr,
    H1: tl.constexpr,
    T0: tl.constexpr,
    T1: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row, tile = tl.program_id(0), tl.program_id(1)
    accepted = tl.load(ACCEPT + row).to(tl.int64)
    index = tile * BLOCK + tl.arange(0, BLOCK)
    live = index < LENGTH * WIDTH
    position = accepted + index // WIDTH
    column = index % WIDTH
    previous = tl.load(
        INITIAL + row * I0 + position * I1 + column * I2,
        live & (position < LENGTH),
        other=0,
    )
    candidate = tl.load(
        CANDIDATE + row * C0 + (position - LENGTH) * C1 + column * C2,
        live & (position >= LENGTH),
        other=0,
    )
    tl.store(
        VALUES + row * LENGTH * WIDTH + index,
        tl.where(position < LENGTH, previous, candidate),
        live,
    )
    state_block = tl.load(STATE_BLOCKS + row).to(tl.int64)
    original = tl.load(STATE_POOL + state_block * LENGTH * WIDTH + index, live, other=0)
    tl.store(UNDO + row * LENGTH * WIDTH + index, original, live)
    if tile == 0:
        offset = tl.arange(0, BLOCK)
        context_position = accepted + offset
        previous_context = tl.load(
            CONTEXT + row * H0 + context_position * H1,
            (offset < CTX) & (context_position < CTX),
            other=0,
        )
        candidate_context = tl.load(
            IDS + row * T0 + (context_position - CTX) * T1,
            (offset < CTX) & (context_position >= CTX),
            other=0,
        )
        tl.store(
            CTX_VALUES + row * CTX + offset,
            tl.where(context_position < CTX, previous_context, candidate_context),
            offset < CTX,
        )
        ctx_block = tl.load(CTX_BLOCKS + row).to(tl.int64)
        original_context = tl.load(
            CTX_POOL + ctx_block * CTX + offset, offset < CTX, other=0
        )
        tl.store(CTX_UNDO + row * CTX + offset, original_context, offset < CTX)


def prepare_commit_values(
    initial,
    candidate,
    context,
    ids,
    state_pool,
    ctx_pool,
    accepted,
    state_blocks,
    ctx_blocks,
):
    """Return (state, context, undo state, undo context), or None for fallback.

    No persistent writes or host synchronization occur here. All four outputs
    own their storage; undo never aliases either pool or selected values.
    The transaction owner checks all destinations (including duplicates) before
    calling this function and synchronizes before publishing its prepare plan.
    """
    tensors = (
        initial,
        candidate,
        context,
        ids,
        state_pool,
        ctx_pool,
        accepted,
        state_blocks,
        ctx_blocks,
    )
    if not (
        torch.version.hip is None
        and all(t.is_cuda and t.device == accepted.device for t in tensors)
        and initial.dtype == candidate.dtype == state_pool.dtype == torch.bfloat16
        and context.dtype == ids.dtype == ctx_pool.dtype == torch.int64
        and accepted.dtype == state_blocks.dtype == ctx_blocks.dtype == torch.int64
        and accepted.ndim == state_blocks.ndim == ctx_blocks.ndim == 1
        and accepted.is_contiguous()
        and state_blocks.is_contiguous()
        and ctx_blocks.is_contiguous()
        and initial.ndim == candidate.ndim == state_pool.ndim == 3
        and context.ndim == ids.ndim == ctx_pool.ndim == 2
        and 0 < accepted.numel() <= 128
        and state_blocks.numel() == ctx_blocks.numel() == accepted.numel()
        and initial.shape[0]
        == candidate.shape[0]
        == context.shape[0]
        == ids.shape[0]
        == accepted.numel()
        and initial.shape[2] == candidate.shape[2] == state_pool.shape[2]
        and initial.shape[1:] == state_pool.shape[1:]
        and context.shape[1] == ctx_pool.shape[1]
        and candidate.shape[1] == ids.shape[1]
        and 0 < candidate.shape[1] <= 32
        and min(initial.shape[1:]) > 0
        and 0 < context.shape[1] <= 32
        and state_pool.is_contiguous()
        and ctx_pool.is_contiguous()
    ):
        return None
    batch, length, width = initial.shape
    values = torch.empty(initial.shape, dtype=initial.dtype, device=initial.device)
    ctx_values = torch.empty(context.shape, dtype=context.dtype, device=context.device)
    undo, ctx_undo = torch.empty_like(values), torch.empty_like(ctx_values)
    _prepare_values_kernel[(batch, triton.cdiv(length * width, 512))](
        accepted,
        initial,
        candidate,
        context,
        ids,
        state_pool,
        ctx_pool,
        state_blocks,
        ctx_blocks,
        values,
        ctx_values,
        undo,
        ctx_undo,
        length,
        width,
        context.shape[1],
        *initial.stride(),
        *candidate.stride(),
        *context.stride(),
        *ids.stride(),
        512,
        num_warps=4,
    )
    return values, ctx_values, undo, ctx_undo


@triton.jit
def _predicated_apply_commit_kernel(
    VERDICT,
    STATE_POOL,
    STATE_BLOCKS,
    STATE_VALUES,
    CTX_POOL,
    CTX_BLOCKS,
    CTX_VALUES,
    LENGTH: tl.constexpr,
    WIDTH: tl.constexpr,
    CTX: tl.constexpr,
    BLOCK: tl.constexpr,
):
    if tl.load(VERDICT) != 0:
        return
    row, tile = tl.program_id(0), tl.program_id(1)
    index = tile * BLOCK + tl.arange(0, BLOCK)
    live = index < LENGTH * WIDTH
    state_block = tl.load(STATE_BLOCKS + row).to(tl.int64)
    value = tl.load(STATE_VALUES + row * LENGTH * WIDTH + index, live, other=0)
    tl.store(STATE_POOL + state_block * LENGTH * WIDTH + index, value, live)
    if tile == 0:
        offset = tl.arange(0, BLOCK)
        ctx_live = offset < CTX
        ctx_block = tl.load(CTX_BLOCKS + row).to(tl.int64)
        ctx_value = tl.load(CTX_VALUES + row * CTX + offset, ctx_live, other=0)
        tl.store(CTX_POOL + ctx_block * CTX + offset, ctx_value, ctx_live)


def can_apply_commit_values_predicated(
    state_pool,
    state_blocks,
    state_values,
    ctx_pool,
    ctx_blocks,
    ctx_values,
) -> bool:
    """Return True iff the predicated Triton commit kernel supports these tensors."""
    tensors = (
        state_pool,
        state_blocks,
        state_values,
        ctx_pool,
        ctx_blocks,
        ctx_values,
    )
    device = state_pool.device
    return bool(
        torch.version.hip is None
        and device.type == "cuda"
        and all(t.is_cuda and t.device == device for t in tensors)
        and state_pool.dtype == state_values.dtype == torch.bfloat16
        and ctx_pool.dtype == ctx_values.dtype == torch.int64
        and state_blocks.dtype == ctx_blocks.dtype == torch.int64
        and state_blocks.ndim == ctx_blocks.ndim == 1
        and state_pool.ndim == state_values.ndim == 3
        and ctx_pool.ndim == ctx_values.ndim == 2
        and 0 < state_blocks.numel() <= 128
        and state_blocks.numel()
        == ctx_blocks.numel()
        == state_values.shape[0]
        == ctx_values.shape[0]
        and state_pool.shape[1:] == state_values.shape[1:]
        and ctx_pool.shape[1] == ctx_values.shape[1]
        and min(state_values.shape[1:]) > 0
        and 0 < ctx_values.shape[1] <= 512
        and all(t.is_contiguous() for t in tensors)
    )


def apply_commit_values_predicated(
    verdict_flag,
    state_pool,
    state_blocks,
    state_values,
    ctx_pool,
    ctx_blocks,
    ctx_values,
) -> bool:
    """Write prepared state/ctx rows only when ``verdict_flag[0] == 0``.

    Guarantees zero persistent pool writes whenever the GPU verdict tripped
    (invalid accept_len, out-of-range/unallocated block, or duplicate target).
    """
    if not (
        isinstance(verdict_flag, torch.Tensor)
        and verdict_flag.is_cuda
        and verdict_flag.device == state_pool.device
        and verdict_flag.dtype == torch.int64
        and verdict_flag.numel() >= 1
        and can_apply_commit_values_predicated(
            state_pool,
            state_blocks,
            state_values,
            ctx_pool,
            ctx_blocks,
            ctx_values,
        )
    ):
        return False
    batch, length, width = state_values.shape
    _predicated_apply_commit_kernel[(batch, triton.cdiv(length * width, 512))](
        verdict_flag,
        state_pool,
        state_blocks,
        state_values,
        ctx_pool,
        ctx_blocks,
        ctx_values,
        length,
        width,
        ctx_values.shape[1],
        512,
        num_warps=4,
    )
    return True
