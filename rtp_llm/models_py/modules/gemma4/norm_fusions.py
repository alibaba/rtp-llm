"""Experimental BF16 residual/router fusions with original rounding boundaries."""

import torch
import triton
import triton.language as tl

from rtp_llm.models_py.modules.gemma4.elementwise import _reference_reduce_width


@triton.jit
def _input_value(X, B, row, col, D: tl.constexpr, XS: tl.constexpr, BS: tl.constexpr,
                 PRE_SUM: tl.constexpr):
    value = tl.load(X + row * XS + col, mask=col < D, other=0).to(tl.float32)
    if PRE_SUM:
        other = tl.load(B + row * BS + col, mask=col < D, other=0).to(tl.float32)
        value = (value + other).to(tl.bfloat16).to(tl.float32)
    return value


@triton.jit
def _norm_fusion_kernel(X, B, W, Y, XS: tl.constexpr, BS: tl.constexpr,
                        D: tl.constexpr, EPS: tl.constexpr, SCALAR: tl.constexpr,
                        MODE: tl.constexpr, BLOCK: tl.constexpr,
                        REDUCE_WIDTH: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    lane = tl.arange(0, REDUCE_WIDTH)
    acc0 = tl.full((REDUCE_WIDTH,), 0, tl.float32)
    acc1 = tl.full((REDUCE_WIDTH,), 0, tl.float32)
    acc2 = tl.full((REDUCE_WIDTH,), 0, tl.float32)
    acc3 = tl.full((REDUCE_WIDTH,), 0, tl.float32)
    for index in range(tl.cdiv(D, REDUCE_WIDTH * 4)):
        base = (lane + index * REDUCE_WIDTH) * 4
        a = _input_value(X, B, row, base, D, XS, BS, MODE == 1)
        b = _input_value(X, B, row, base + 1, D, XS, BS, MODE == 1)
        c = _input_value(X, B, row, base + 2, D, XS, BS, MODE == 1)
        d = _input_value(X, B, row, base + 3, D, XS, BS, MODE == 1)
        acc0 = acc0 + a * a
        acc1 = acc1 + b * b
        acc2 = acc2 + c * c
        acc3 = acc3 + d * d
    total = ((acc0 + acc1) + acc2) + acc3
    if REDUCE_WIDTH > 32:
        total = tl.sum(tl.reshape(total, (REDUCE_WIDTH // 32, 32)), axis=0)
    variance = tl.sum(total, axis=0) * (1.0 / D)
    inverse = tl.rsqrt(variance + EPS)
    col = tl.arange(0, BLOCK)
    value = _input_value(X, B, row, col, D, XS, BS, MODE == 1)
    normalized = value * inverse
    weight = tl.load(W + col, mask=col < D, other=0).to(tl.float32)
    if MODE != 2:
        normalized = normalized * weight
    normalized = normalized.to(tl.bfloat16).to(tl.float32)
    if MODE == 0:
        residual = tl.load(B + row * BS + col, mask=col < D, other=0).to(tl.float32)
        normalized = normalized + residual
    elif MODE == 2:
        normalized = (normalized * weight).to(tl.bfloat16).to(tl.float32)
        normalized = normalized * SCALAR
    tl.store(Y + row * D + col, normalized, mask=col < D)


@triton.jit
def _residual_scale_kernel(X, B, W, Y, COUNT, BLOCK: tl.constexpr):
    index = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(X + index, mask=index < COUNT, other=0).to(tl.float32)
    b = tl.load(B + index, mask=index < COUNT, other=0).to(tl.float32)
    scale = tl.load(W).to(tl.float32)
    summed = (x + b).to(tl.bfloat16).to(tl.float32)
    tl.store(Y + index, summed * scale, mask=index < COUNT)


def _rows_supported(x):
    return (
        x.is_cuda and x.dtype == torch.bfloat16 and x.dim() == 2
        and x.numel() > 0 and x.stride(1) == 1 and x.stride(0) > 0
        and 128 <= x.shape[1] <= 16384 and x.shape[1] % 4 == 0
    )


def _norm_fusion(x, other, weight, eps, mode, scalar=1.0):
    if (
        not _rows_supported(x) or eps <= 0
        or weight.device != x.device or not weight.is_contiguous()
        or weight.shape != (x.shape[1],)
        or weight.dtype not in (torch.bfloat16, torch.float32)
        or (mode == 2 and weight.dtype != torch.bfloat16)
        or (mode != 2 and (
            not _rows_supported(other) or other.shape != x.shape
            or other.device != x.device
        ))
    ):
        return None
    output = torch.empty(x.shape, dtype=x.dtype, device=x.device)
    _norm_fusion_kernel[(x.shape[0],)](
        x, other if mode != 2 else x, weight, output,
        x.stride(0), other.stride(0) if mode != 2 else x.stride(0),
        x.shape[1], eps, scalar, mode, triton.next_power_of_2(x.shape[1]),
        _reference_reduce_width(x.shape[1], x.shape[0]),
        num_warps=4 if x.shape[1] <= 2048 else 8,
        enable_fp_fusion=False,
    )
    return output


def norm_add(x, weight, residual, eps=1e-6):
    return _norm_fusion(x, residual, weight, eps, 0)


def add_norm(x, other, weight, eps=1e-6):
    return _norm_fusion(x, other, weight, eps, 1)


def router_norm(x, learned_scale, scalar, eps=1e-6):
    return _norm_fusion(x, x, learned_scale, eps, 2, scalar)


def residual_scale(x, residual, scale):
    if (
        not _rows_supported(x) or not _rows_supported(residual)
        or residual.shape != x.shape or residual.device != x.device
        or not x.is_contiguous() or not residual.is_contiguous()
        or scale.device != x.device or scale.dtype != torch.bfloat16
        or scale.numel() != 1 or not scale.is_contiguous()
    ):
        return None
    output = torch.empty_like(x)
    _residual_scale_kernel[(triton.cdiv(x.numel(), 1024),)](
        x, residual, scale, output, x.numel(), 1024,
        num_warps=4, enable_fp_fusion=False,
    )
    return output
