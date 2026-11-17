# SPDX-License-Identifier: Apache-2.0
"""Qwen4-Max single-token short convolution with direct Decode cache writeback."""

import torch
import triton
import triton.language as tl

from .conv import _store_ple_output


@triton.jit
def _ple_decode_conv_commit(
    Gated,
    Normed,
    Hidden,
    Initial,
    Weight,
    Output,
    StatePool,
    CtxPool,
    StateBlocks,
    CtxBlocks,
    History,
    STATE_STRIDE: tl.constexpr,
    CTX_STRIDE: tl.constexpr,
    HISTORY_STRIDE: tl.constexpr,
    HISTORY_TOKEN_STRIDE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    tile = tl.program_id(0)
    row = tl.program_id(1).to(tl.int64)
    col = tile * BLOCK + tl.arange(0, BLOCK)
    offset = row * 32768 + col
    initial = Initial + row * 9 * 32768 + col
    state_block = tl.load(StateBlocks + row)
    dest = StatePool + state_block * STATE_STRIDE + col
    acc = tl.full((BLOCK,), 0, tl.float32)
    for tap in tl.static_range(3):
        past = tl.load(initial + tap * 3 * 32768)
        w = tl.load(Weight + col * 4 + tap).to(tl.float32)
        acc = tl.fma(past.to(tl.float32), w, acc)
        if tap > 0:
            tl.store(dest + (tap * 3 - 1) * 32768, past, state_block > 0)
    current = tl.load(Normed + offset)
    w = tl.load(Weight + col * 4 + 3).to(tl.float32)
    acc = tl.fma(current.to(tl.float32), w, acc)
    _store_ple_output(acc, Gated, Output, Hidden, offset, True, True)
    if state_block > 0:
        tl.store(dest + 8 * 32768, current)
        # The convolution already wrote the two retained dilation taps.
        for state_row in tl.static_range(1, 9):
            if state_row % 3:
                past = tl.load(initial + state_row * 32768)
                tl.store(dest + (state_row - 1) * 32768, past)
    if tile == 0:
        ctx_block = tl.load(CtxBlocks + row)
        if ctx_block > 0:
            token = tl.arange(0, 2)
            previous = tl.load(
                History + row * HISTORY_STRIDE + (token + 1) * HISTORY_TOKEN_STRIDE
            )
            tl.store(CtxPool + ctx_block * CTX_STRIDE + token, previous)


def try_decode_epilogue(
    gated,
    normed,
    hidden,
    conv_buffer,
    conv_weight,
    state_pool,
    ctx_pool,
    state_write_blocks,
    ctx_write_blocks,
    history,
):
    """Return the PLE residual output and write the valid cache destinations.

    The Decode prepare kernel owns the old-state snapshot and resolves both
    destinations before any writes. Each positive destination is independent.
    """
    if not (
        hidden.is_cuda
        and torch.version.hip is None
        and hidden.ndim == 3
        and hidden.shape[0] > 0
        and hidden.shape[1:] == (1, 32768)
        and torch.cuda.get_device_capability(hidden.device) in ((10, 0), (10, 3))
    ):
        return None
    batch = hidden.shape[0]
    if not (
        gated.shape == normed.shape == hidden.shape
        and conv_buffer.shape == (batch, 9, 32768)
        and conv_weight.shape == (32768, 1, 4)
        and state_pool.ndim == 3
        and state_pool.shape[1:] == (9, 32768)
        and state_pool.dtype == torch.bfloat16
        and state_pool.stride()[1:] == (32768, 1)
        and conv_buffer.untyped_storage().data_ptr()
        != state_pool.untyped_storage().data_ptr()
        and ctx_pool.ndim == 2
        and ctx_pool.shape[1] == 2
        and ctx_pool.dtype == torch.int64
        and ctx_pool.stride(1) == 1
        and history.shape == (batch, 3)
        and history.dtype == torch.int64
        and all(
            x.shape == (batch,) and x.dtype == torch.int64 and x.is_contiguous()
            for x in (state_write_blocks, ctx_write_blocks)
        )
        and all(
            x.device == hidden.device
            for x in (
                state_pool,
                ctx_pool,
                state_write_blocks,
                ctx_write_blocks,
                history,
            )
        )
        and all(
            x.device == hidden.device
            and x.dtype == torch.bfloat16
            and x.is_contiguous()
            and not x.requires_grad
            for x in (gated, normed, hidden, conv_buffer, conv_weight)
        )
    ):
        return None

    output = torch.empty_like(hidden)
    block = 256
    _ple_decode_conv_commit[(32768 // block, batch)](
        gated,
        normed,
        hidden,
        conv_buffer,
        conv_weight,
        output,
        state_pool,
        ctx_pool,
        state_write_blocks,
        ctx_write_blocks,
        history,
        STATE_STRIDE=state_pool.stride(0),
        CTX_STRIDE=ctx_pool.stride(0),
        HISTORY_STRIDE=history.stride(0),
        HISTORY_TOKEN_STRIDE=history.stride(1),
        BLOCK=block,
        num_warps=4 if batch < 128 else 2,
    )
    return output
