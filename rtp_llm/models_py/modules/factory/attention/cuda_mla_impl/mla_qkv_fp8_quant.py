"""One-launch ordinary E4M3 quantization of MLA Prefill Q, K and V."""

import math

import torch
import triton
import triton.language as tl

from . import mla_fp8_kernels


@triton.jit
def _quantize_operand(
    X,
    Y,
    block,
    N,
    SHAPE: tl.constexpr,
    STRIDES,
    CONTIGUOUS: tl.constexpr,
    INV_SCALE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = block.to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    N = N.to(tl.int64)
    if (SHAPE[0] * SHAPE[1]) % 8 == 0:
        # N = tokens * heads * features. Preserve its proven alignment without
        # specializing the token count; otherwise the tail mask scalarizes IO.
        N = tl.multiple_of(N, 8)
    if CONTIGUOUS:
        source = offsets
    else:
        # Only head/feature geometry is specialized. Token counts and strides
        # remain runtime values, with wide products for large prefix views.
        token = offsets // (SHAPE[0] * SHAPE[1])
        head = (offsets // SHAPE[1]) % SHAPE[0]
        feature = offsets % SHAPE[1]
        source = token * STRIDES[0] + head * STRIDES[1] + feature * STRIDES[2]
    values = tl.load(X + source, offsets < N, other=0).to(tl.float32) * INV_SCALE
    # Exactly the original quantize_fp8 expression, including NaN preservation
    # and saturating infinities instead of generating E4M3 NaNs on overflow.
    values = tl.where(
        values != values, values, tl.minimum(tl.maximum(values, -448.0), 448.0)
    )
    tl.store(Y + offsets, values, offsets < N)


@triton.jit(
    do_not_specialize=[
        "Q_N",
        "K_N",
        "V_N",
        "Q_BLOCKS",
        "K_BLOCKS",
        "Q_STRIDES",
        "K_STRIDES",
        "V_STRIDES",
    ]
)
def _quantize_qkv(
    Q,
    K,
    V,
    OQ,
    OK,
    OV,
    Q_N,
    K_N,
    V_N,
    Q_BLOCKS,
    K_BLOCKS,
    Q_SHAPE: tl.constexpr,
    K_SHAPE: tl.constexpr,
    V_SHAPE: tl.constexpr,
    Q_STRIDES,
    K_STRIDES,
    V_STRIDES,
    Q_CONTIGUOUS: tl.constexpr,
    K_CONTIGUOUS: tl.constexpr,
    V_CONTIGUOUS: tl.constexpr,
    Q_INV_SCALE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    block = tl.program_id(0)
    # Partition CTAs, not elements: each CTA has one operand and one uniform
    # branch. Unequal Q/KV lengths need neither concatenation nor padded work.
    if block < Q_BLOCKS:
        _quantize_operand(
            Q, OQ, block, Q_N, Q_SHAPE, Q_STRIDES, Q_CONTIGUOUS, Q_INV_SCALE, BLOCK
        )
    elif block < Q_BLOCKS + K_BLOCKS:
        _quantize_operand(
            K, OK, block - Q_BLOCKS, K_N, K_SHAPE, K_STRIDES, K_CONTIGUOUS, 1.0, BLOCK
        )
    else:
        _quantize_operand(
            V,
            OV,
            block - Q_BLOCKS - K_BLOCKS,
            V_N,
            V_SHAPE,
            V_STRIDES,
            V_CONTIGUOUS,
            1.0,
            BLOCK,
        )


def quantize_qkv_fp8(q, k, v, q_scale=1.0):
    """Quantize three [tokens, heads, features] operands without staging.

    Q uses q_scale; expanded K/V use the existing unit scale. Each operand may
    have independent lengths, strides and floating dtype. Return three dense
    E4M3 tensors; there is no change to compressed KV-cache quantization.
    """
    if not math.isfinite(q_scale) or q_scale <= 0:
        raise ValueError("FP8 MLA scale must be finite and positive")
    inputs = (q, k, v)
    for x in inputs:
        if not x.is_cuda or x.device != q.device:
            raise ValueError("MLA Q/K/V must share one CUDA device")
        if x.ndim != 3 or x.shape[1] <= 0 or x.shape[2] <= 0:
            raise ValueError(
                "MLA Q/K/V require [tokens, heads, features] with positive head/feature sizes"
            )
        if x.dtype not in (torch.bfloat16, torch.float16, torch.float32):
            raise TypeError("FP8 MLA quantizer requires floating activations")
    if mla_fp8_kernels._FP8_DIAGNOSTICS:
        for x, scale, name in zip(
            inputs, (q_scale, 1.0, 1.0), ("prefill_q", "prefill_k", "prefill_v")
        ):
            mla_fp8_kernels.observe_fp8_input(x, scale, name)
    outputs = tuple(
        torch.empty_like(
            x, dtype=torch.float8_e4m3fn, memory_format=torch.contiguous_format
        )
        for x in inputs
    )
    block = 1024
    sizes = tuple(x.numel() for x in inputs)
    blocks = tuple(triton.cdiv(n, block) for n in sizes)
    if sum(blocks):
        _quantize_qkv[(sum(blocks),)](
            *inputs,
            *outputs,
            *sizes,
            blocks[0],
            blocks[1],
            *(tuple(x.shape[1:]) for x in inputs),
            *(tuple(x.stride()) for x in inputs),
            *(x.is_contiguous() for x in inputs),
            1.0 / q_scale,
            block,
        )
    return outputs
