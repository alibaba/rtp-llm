# SPDX-License-Identifier: Apache-2.0
"""SM90 BF16 single-token PLE gate, convolution and history roll."""

import math

import torch
import triton
import triton.language as tl

from .ple_gate_triton import _gate_branch, try_ple_gate


@triton.jit
def _decode_output(gated, acc, residual_ptr, valid, ADD_RESIDUAL: tl.constexpr):
    conv = acc.to(tl.bfloat16).to(tl.float32)
    activated = (conv * tl.sigmoid(conv)).to(tl.bfloat16)
    # Materialize the rounded PLE result before adding the outer residual.
    # This also prevents contracting either producer multiply into the add.
    out = tl.inline_asm_elementwise(
        "add.rn.bf16 $0, $1, $2;",
        constraints="=h,h,h",
        args=[gated, activated],
        dtype=tl.bfloat16,
        is_pure=True,
        pack=1,
    )
    if ADD_RESIDUAL:
        residual = tl.load(residual_ptr, valid, other=0).to(tl.float32)
        out = (out.to(tl.float32) + residual).to(tl.bfloat16)
    return out


@triton.jit
def _ple_decode_gate_conv(
    Key,
    Value,
    Hidden,
    GK,
    GQ,
    GC,
    History,
    Weight,
    Output,
    NextHistory,
    H: tl.constexpr,
    WIDTH: tl.constexpr,
    EPS: tl.constexpr,
    KEY_STRIDE: tl.constexpr,
    VALUE_STRIDE: tl.constexpr,
    HIDDEN_STRIDE: tl.constexpr,
    HISTORY_BATCH_STRIDE: tl.constexpr,
    HISTORY_ROW_STRIDE: tl.constexpr,
    ADD_RESIDUAL: tl.constexpr,
    BLOCK: tl.constexpr,
):
    batch = tl.program_id(0).to(tl.int64)
    col = tl.arange(0, BLOCK)
    valid = col < H
    hc_col = tl.program_id(1) * H + col
    gated, normed = _gate_branch(
        Key, Value, Hidden, GK, GQ, GC, None, None, batch, col, hc_col, valid,
        H, EPS, KEY_STRIDE, VALUE_STRIDE, HIDDEN_STRIDE,
    )

    # Kernel size 4 and dilation 3 select history rows 0, 3 and 6.
    previous = History + batch * HISTORY_BATCH_STRIDE + hc_col
    next_row = NextHistory + batch * 9 * WIDTH + hc_col
    weight = Weight + hc_col * 4
    acc = tl.full((BLOCK,), 0, tl.float32)
    for position in tl.static_range(9):
        history = tl.load(previous + position * HISTORY_ROW_STRIDE, valid, other=0)
        if position % 3 == 0:
            tap = tl.load(weight + position // 3, valid, other=0).to(tl.float32)
            acc = tl.fma(history.to(tl.float32), tap, acc)
        if position > 0:
            tl.store(next_row + (position - 1) * WIDTH, history, valid)
    tap = tl.load(weight + 3, valid, other=0).to(tl.float32)
    acc = tl.fma(normed.to(tl.float32), tap, acc)
    out = _decode_output(
        gated, acc, Hidden + batch * HIDDEN_STRIDE + hc_col, valid, ADD_RESIDUAL
    )
    tl.store(Output + batch * WIDTH + hc_col, out, valid)
    tl.store(next_row + 8 * WIDTH, normed, valid)


@triton.jit
def _ple_decode_conv_roll(
    Gated,
    Normed,
    Hidden,
    History,
    Weight,
    Output,
    WIDTH: tl.constexpr,
    BLOCK: tl.constexpr,
    ADD_RESIDUAL: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    col = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    valid = col < WIDTH
    history = History + row * 9 * WIDTH + col
    weight = Weight + col * 4
    offset = row * WIDTH + col
    acc = tl.full((BLOCK,), 0, tl.float32)
    for tap in tl.static_range(3):
        value = tl.load(history + 3 * tap * WIDTH, valid, other=0).to(tl.float32)
        filt = tl.load(weight + tap, valid, other=0).to(tl.float32)
        acc = tl.fma(value, filt, acc)
    normed = tl.load(Normed + offset, valid, other=0)
    filt = tl.load(weight + 3, valid, other=0).to(tl.float32)
    acc = tl.fma(normed.to(tl.float32), filt, acc)
    gated = tl.load(Gated + offset, valid, other=0)
    out = _decode_output(gated, acc, Hidden + offset, valid, ADD_RESIDUAL)
    tl.store(Output + offset, out, valid)

    # The CTA owns all nine history rows for its channel tile. Read the
    # convolution taps first, then roll old rows in ascending order.
    for pos in tl.static_range(8):
        prev = tl.load(history + (pos + 1) * WIDTH, valid, other=0)
        tl.store(history + pos * WIDTH, prev, valid)
    tl.store(history + 8 * WIDTH, normed, valid)


def try_ple_decode_fusion(
    key, value, hidden, norm_key, norm_query, norm_conv, history, weight, eps, dilation,
    *,
    add_residual=False,
    reuse_history=False,
):
    """Return the output and rolled history, optionally reusing a private input."""
    width, hidden_size, state_len = 10240, 2560, 9
    if not (
        hidden.is_cuda
        and torch.version.hip is None
        and hidden.ndim == 3
        and hidden.shape[0] > 0
        and hidden.shape[1:] == (1, width)
        and torch.cuda.get_device_capability(hidden.device) == (9, 0)
        and dilation == 3
        and math.isfinite(eps)
        and eps >= 0
    ):
        return None
    batch = hidden.shape[0]
    if batch > 1 and not hidden.is_contiguous():
        return None
    tensors = (
        (hidden, (batch, 1, width)),
        (key, (batch, 1, width)),
        (value, (batch, 1, hidden_size)),
        (norm_key, (width,)),
        (norm_query, (width,)),
        (norm_conv, (width,)),
        (history, (batch, state_len, width)),
        (weight, (width, 1, 4)),
    )
    if not weight.is_contiguous() or not all(
        x.shape == shape
        and x.device == hidden.device
        and x.dtype == torch.bfloat16
        and x.stride(-1) == 1
        and not x.requires_grad
        for x, shape in tensors
    ):
        return None

    output = torch.empty_like(hidden, memory_format=torch.contiguous_format)
    # At larger batches the gate's full-width reduction and the memory-bound
    # roll favor different tiles. Use the existing gate with a 512-channel
    # convolution CTA, as in vLLM's decode layout.
    if reuse_history and batch >= 16 and history.is_contiguous():
        gates = try_ple_gate(key, value, hidden, norm_key, norm_query, norm_conv, eps)
        if gates is not None:
            block = 512
            _ple_decode_conv_roll[(batch, triton.cdiv(width, block))](
                *gates, hidden, history, weight, output,
                WIDTH=width, BLOCK=block, ADD_RESIDUAL=add_residual, num_warps=8,
            )
            return output, history

    next_history = history.new_empty((batch, state_len, width))
    _ple_decode_gate_conv[(batch, width // hidden_size)](
        key,
        value,
        hidden,
        norm_key,
        norm_query,
        norm_conv,
        history,
        weight,
        output,
        next_history,
        H=hidden_size,
        WIDTH=width,
        EPS=eps,
        KEY_STRIDE=key.stride(0),
        VALUE_STRIDE=value.stride(0),
        HIDDEN_STRIDE=hidden.stride(0),
        HISTORY_BATCH_STRIDE=history.stride(0),
        HISTORY_ROW_STRIDE=history.stride(1),
        ADD_RESIDUAL=add_residual,
        BLOCK=triton.next_power_of_2(hidden_size),
        num_warps=4,
    )
    return output, next_history
