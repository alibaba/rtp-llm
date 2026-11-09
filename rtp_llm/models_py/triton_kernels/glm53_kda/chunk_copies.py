"""Fuse layout copies around KDA BLAS without changing its operands or math."""

import os

import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice

_ENABLED = os.environ.get("GLM53_KDA_FUSED_CHUNK_COPY", "0") == "1"
_PREFETCH_STATE = os.environ.get("GLM53_KDA_STATE_PREFETCH", "0") == "1"


@triton.jit
def bf16_bits(value):
    bits = value.to(tl.uint32, bitcast=True)
    rounded = (bits + 32767 + (bits >> 16 & 1) >> 16).to(tl.uint16)
    rounded = tl.where(
        value != value, value.to(tl.bfloat16).to(tl.uint16, bitcast=True), rounded
    )
    return rounded.to(tl.bfloat16, bitcast=True)


@triton.jit
def _prepare(
    state,
    u,
    k,
    stored,
    bias,
    transposed,
    H: tl.constexpr,
    L: tl.constexpr,
    US: tl.constexpr,
    KS: tl.constexpr,
    TILE: tl.constexpr,
):
    head = tl.program_id(0).to(tl.int64)
    tile = tl.program_id(1).to(tl.int64)
    if tile < 16:
        i = tile * 1024 + tl.arange(0, 1024).to(tl.int64)
        value = tl.load(state + head * 16384 + i)
        tl.store(stored + head * 16384 + i, bf16_bits(value))
    else:
        part = tile - 16
        t = part // (128 // TILE) * TILE + tl.arange(0, TILE).to(tl.int64)
        d = part % (128 // TILE) * TILE + tl.arange(0, TILE).to(tl.int64)
        uv = tl.load(u + t[:, None] * US + head * 128 + d[None, :], t[:, None] < L, 0)
        fp32 = (uv.to(tl.uint16, bitcast=True).to(tl.uint32) << 16).to(
            tl.float32, bitcast=True
        )
        tl.store(
            bias + head * L * 128 + t[:, None] * 128 + d[None, :], fp32, t[:, None] < L
        )
        if L != 1:
            kv = tl.load(
                k + t[:, None] * KS + head * 128 + d[None, :], t[:, None] < L, 0
            )
            tl.store(
                transposed + head * 128 * L + d[None, :] * L + t[:, None],
                kv,
                t[:, None] < L,
            )


@triton.jit
def _finish(
    value,
    bf16,
    output,
    H: tl.constexpr,
    L: tl.constexpr,
    OS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    i = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
    h, remainder = (i // (L * 128), i % (L * 128))
    t, d = (remainder // 128, remainder % 128)
    converted = bf16_bits(tl.load(value + i, h < H, 0))
    tl.store(bf16 + i, converted, h < H)
    tl.store(output + t * OS + h * 128 + d, converted, h < H)


def _prepare_supported(state, u, k):
    return not (
        not _ENABLED
        or torch.version.hip is not None
        or any(not t.is_cuda or t.requires_grad for t in (state, u, k))
        or any(t.device != state.device for t in (u, k))
        or state.dtype != torch.float32
        or not state.is_contiguous()
        or state.ndim != 3
        or state.shape[1:] != (128, 128)
        or u.ndim != 3
        or u.shape != k.shape
        or u.shape[1:] != (state.shape[0], 128)
        or not 0 < u.shape[0] <= 256
        or not 0 < state.shape[0] <= 128
        or any(
            t.dtype != torch.bfloat16
            or t.stride(2) != 1
            or t.stride(1) != 128
            or t.stride(0) < t.shape[1] * 128
            for t in (u, k)
        )
    )


def prepare_chunk_copies(state, u, k):
    """Return owned BLAS inputs, or None for unsupported layouts."""
    if not _prepare_supported(state, u, k):
        return None
    tile = 16 if state.shape[0] <= 16 else 32
    length, heads, _ = u.shape
    stored = torch.empty_like(state, dtype=torch.bfloat16)
    bias = torch.empty((heads, length, 128), device=u.device, dtype=torch.float32)
    transposed = (
        k.permute(1, 2, 0)
        if length == 1
        else torch.empty((heads, 128, length), device=k.device, dtype=k.dtype)
    )
    _prepare[heads, 16 + triton.cdiv(length, tile) * (128 // tile)](
        state,
        u,
        k,
        stored,
        bias,
        transposed,
        heads,
        length,
        u.stride(0),
        k.stride(0),
        tile,
        num_warps=4,
        enable_fp_fusion=False,
        enable_reflect_ftz=False,
    )
    return (stored, bias, transposed)


def finish_chunk_copies(value, output):
    """Convert to BLAS BF16 layout and scatter into the owned v_new tensor."""
    if (
        not _ENABLED
        or torch.version.hip is not None
        or any(not t.is_cuda or t.requires_grad for t in (value, output))
        or value.device != output.device
        or value.dtype != torch.float32
        or output.dtype != torch.bfloat16
        or not value.is_contiguous()
        or value.ndim != 3
        or value.shape[2] != 128
        or output.shape != (value.shape[1], value.shape[0], 128)
        or output.stride(2) != 1
        or output.stride(1) != 128
        or output.stride(0) < value.shape[0] * 128
    ):
        return None
    heads, length, _ = value.shape
    bf16 = torch.empty_like(value, dtype=torch.bfloat16)
    _finish[triton.cdiv(value.numel(), 1024),](
        value,
        bf16,
        output,
        heads,
        length,
        output.stride(0),
        1024,
        num_warps=4,
        enable_fp_fusion=False,
        enable_reflect_ftz=False,
    )
    return bf16


@triton.jit
def _update_prepare(
    state,
    delta,
    gate,
    updated,
    u,
    k,
    stored,
    bias,
    transposed,
    H: tl.constexpr,
    L: tl.constexpr,
    US: tl.constexpr,
    KS: tl.constexpr,
    TILE: tl.constexpr,
    STATE_BLOCK: tl.constexpr,
):
    head = tl.program_id(0).to(tl.int64)
    tile = tl.program_id(1).to(tl.int64)
    if tile < 16384 // STATE_BLOCK:
        rows = tile * (STATE_BLOCK // 128) + tl.arange(0, STATE_BLOCK // 128).to(
            tl.int64
        )
        cols = tl.arange(0, 128).to(tl.int64)
        i = head * 16384 + rows[:, None] * 128 + cols[None, :]
        decay = libdevice.exp2(tl.load(gate + head * 128 + rows).to(tl.float32))
        value = tl.load(state + i)
        increment = tl.load(delta + i)
        result = libdevice.add_rn(libdevice.mul_rn(value, decay[:, None]), increment)
        tl.store(updated + i, result)
        tl.store(stored + i, bf16_bits(result))
    else:
        part = tile - 16384 // STATE_BLOCK
        t = part // (128 // TILE) * TILE + tl.arange(0, TILE).to(tl.int64)
        d = part % (128 // TILE) * TILE + tl.arange(0, TILE).to(tl.int64)
        uv = tl.load(u + t[:, None] * US + head * 128 + d[None, :], t[:, None] < L, 0)
        fp32 = (uv.to(tl.uint16, bitcast=True).to(tl.uint32) << 16).to(
            tl.float32, bitcast=True
        )
        tl.store(
            bias + head * L * 128 + t[:, None] * 128 + d[None, :], fp32, t[:, None] < L
        )
        if L != 1:
            kv = tl.load(
                k + t[:, None] * KS + head * 128 + d[None, :], t[:, None] < L, 0
            )
            tl.store(
                transposed + head * 128 * L + d[None, :] * L + t[:, None],
                kv,
                t[:, None] < L,
            )


def update_prepare_chunk(state, delta, gate, u, k, output):
    """Write a checkpoint and prepare the following chunk, or return None.

    Only the current sequence's next chunk is eligible. Every BLAS operand
    retains the original layout, and FP32 update operations round separately.
    """
    if not _PREFETCH_STATE or not _prepare_supported(state, u, k) or output is None:
        return None
    if (
        any(
            not t.is_cuda
            or not t.is_contiguous()
            or t.requires_grad
            or t.device != state.device
            for t in (delta, gate, output)
        )
        or delta.shape != state.shape
        or output.shape != state.shape
        or delta.dtype != torch.float32
        or output.dtype != torch.float32
        or gate.shape != state.shape[:2]
        or gate.dtype not in (torch.float32, torch.bfloat16)
    ):
        return None
    begin = output.data_ptr()
    end = begin + output.numel() * output.element_size()
    for value in (state, delta, gate, u, k):
        source = value.data_ptr()
        # u/k may have gaps between token rows. Include those gaps in the
        # overlap check; adjacent checkpoint views of the same storage are valid.
        span = (
            sum(
                (size - 1) * stride for size, stride in zip(value.shape, value.stride())
            )
            + 1
        )
        if begin < source + span * value.element_size() and source < end:
            return None
    heads = state.shape[0]
    length = u.shape[0]
    stored = torch.empty_like(state, dtype=torch.bfloat16)
    bias = torch.empty((heads, length, 128), device=u.device, dtype=torch.float32)
    transposed = (
        k.permute(1, 2, 0)
        if length == 1
        else torch.empty((heads, 128, length), device=k.device, dtype=k.dtype)
    )
    tile = 16 if heads <= 16 else 32
    _update_prepare[heads, 32 + triton.cdiv(length, tile) * (128 // tile)](
        state,
        delta,
        gate,
        output,
        u,
        k,
        stored,
        bias,
        transposed,
        heads,
        length,
        u.stride(0),
        k.stride(0),
        tile,
        512,
        num_warps=4,
        enable_fp_fusion=False,
        enable_reflect_ftz=False,
    )
    return output, (stored, bias, transposed)
