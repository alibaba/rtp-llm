"""DSv4 E2M1/UE8M0 representation for the FP4 indexer.

The FP4 numerical contract follows RTP-LLM revision
5416c5edbb25877f08ac43d012b9b7c21f625abc's fused_norm_rope_v2.cuh
and main_norm_rope.cuh: normalized Hadamard-128, groups of 32,
midpoint ties toward the smaller magnitude, and positive zero.
Pages contain all 64-byte payloads followed by all four-byte scales.
"""

import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice


@triton.jit
def hadamard128(values):
    dims = tl.arange(0, 128)
    for step in tl.static_range(7):
        other = tl.gather(values, dims ^ (1 << step), axis=0)
        values = tl.where((dims & (1 << step)) != 0, other - values, values + other)
    return values * 0.08838834764831845


@triton.jit
def e2m1_code(values):
    magnitude = tl.minimum(tl.abs(values), 6.0)
    code = (magnitude > 0.25).to(tl.uint8)
    code += (magnitude > 0.75).to(tl.uint8)
    code += (magnitude > 1.25).to(tl.uint8)
    code += (magnitude > 1.75).to(tl.uint8)
    code += (magnitude > 2.5).to(tl.uint8)
    code += (magnitude > 3.5).to(tl.uint8)
    code += (magnitude > 5.0).to(tl.uint8)
    return code | (((values < 0) & (code != 0)).to(tl.uint8) << 3)


@triton.jit
def e2m1_value(code):
    magnitude = code & 7
    value = tl.where(
        magnitude < 4,
        magnitude.to(tl.float32) * 0.5,
        tl.where(
            magnitude == 4,
            2.0,
            tl.where(magnitude == 5, 3.0, tl.where(magnitude == 6, 4.0, 6.0)),
        ),
    )
    return tl.where((code & 8) != 0, -value, value)


@triton.jit
def quantize128(values):
    grouped = tl.reshape(values, (4, 32))
    maximum = tl.maximum(tl.max(tl.abs(grouped), axis=1), 1.0e-4)
    raw_scale = maximum / 6.0
    bits = raw_scale.to(tl.int32, bitcast=True)
    exponent = ((bits >> 23) & 255) + ((bits & 0x7FFFFF) != 0).to(tl.int32)
    exponent = tl.minimum(tl.maximum(exponent, 1), 254)
    inverse = ((254 - exponent) << 23).to(tl.float32, bitcast=True)
    codes = e2m1_code(grouped * inverse[:, None])
    low, high = tl.split(tl.reshape(codes, (64, 2)))
    return low | (high << 4), exponent.to(tl.uint8)


@triton.jit
def store_rotated_key(values, cache, block_base, slot, page_size):
    packed, scales = quantize128(hadamard128(values))
    tl.store(cache + block_base + slot * 64 + tl.arange(0, 64), packed)
    tl.store(cache + block_base + page_size * 64 + slot * 4 + tl.arange(0, 4), scales)


@triton.jit
def _quantize_rows(
    X, Q, S, ROWS: tl.constexpr, ROTATE: tl.constexpr, ROUND_BF16: tl.constexpr
):
    row = tl.program_id(0)
    values = tl.load(X + row * 128 + tl.arange(0, 128)).to(tl.float32)
    if ROTATE:
        values = hadamard128(values)
    if ROUND_BF16:
        values = values.to(tl.bfloat16).to(tl.float32)
    packed, scales = quantize128(values)
    tl.store(Q + row * 64 + tl.arange(0, 64), packed)
    tl.store(S + row * 4 + tl.arange(0, 4), scales)


def quantize_rows(values, *, rotate=False, round_bf16=False):
    if values.shape[-1] != 128 or not values.is_contiguous():
        raise ValueError("FP4 quantization requires contiguous rows of 128 values")
    rows = values.numel() // 128
    packed = torch.empty(
        (*values.shape[:-1], 64), dtype=torch.uint8, device=values.device
    )
    scales = torch.empty(
        (*values.shape[:-1], 4), dtype=torch.uint8, device=values.device
    )
    if rows:
        _quantize_rows[(rows,)](
            values,
            packed,
            scales,
            rows,
            rotate,
            round_bf16,
            num_warps=4,
            enable_fp_fusion=False,
        )
    return packed, scales


@triton.jit
def _dequantize_rows(Q, S, OUT, ROWS: tl.constexpr, TILE: tl.constexpr):
    rows = tl.program_id(0) * TILE + tl.arange(0, TILE)
    dims = tl.arange(0, 128)
    packed = tl.load(
        Q + rows[:, None] * 64 + dims[None, :] // 2, rows[:, None] < ROWS, 0
    )
    codes = (packed >> ((dims[None, :] & 1) * 4)) & 15
    scales = tl.load(
        S + rows[:, None] * 4 + dims[None, :] // 32, rows[:, None] < ROWS, 0
    ).to(tl.int32)
    values = e2m1_value(codes) * (scales << 23).to(tl.float32, bitcast=True)
    tl.store(OUT + rows[:, None] * 128 + dims[None, :], values, rows[:, None] < ROWS)


def dequantize_rows(packed, scales, *, dtype=torch.bfloat16):
    if packed.shape[-1] != 64 or scales.shape != (*packed.shape[:-1], 4):
        raise ValueError("FP4 rows require 64 payload bytes and four UE8M0 bytes")
    if packed.dtype != torch.uint8 or scales.dtype != torch.uint8:
        raise ValueError("FP4 payloads and scales must be uint8")
    rows = packed.numel() // 64
    out = torch.empty((*packed.shape[:-1], 128), dtype=dtype, device=packed.device)
    if rows:
        _dequantize_rows[(triton.cdiv(rows, 16),)](
            packed.contiguous(), scales.contiguous(), out, rows, 16
        )
    return out


@triton.jit
def _quantize_q(
    Q,
    W,
    FREQ,
    OUT,
    WEIGHTS,
    PACKED,
    SCALES,
    HEADS: tl.constexpr,
    FREQ_STRIDE: tl.constexpr,
    FREQ_GROUP: tl.constexpr,
    WEIGHT_SCALE: tl.constexpr,
    RETURN_PACKED: tl.constexpr,
    RECONSTRUCT: tl.constexpr,
):
    row = tl.program_id(0)
    dims = tl.arange(0, 128)
    values = tl.load(Q + row * 128 + dims).to(tl.float32)
    rope = dims >= 64
    partner = tl.load(Q + row * 128 + (dims ^ 1)).to(tl.float32)
    freq_row = (row // HEADS) // FREQ_GROUP
    freq_dim = tl.maximum((dims - 64) // 2, 0)
    real = tl.load(FREQ + freq_row * FREQ_STRIDE + freq_dim * 2, rope, 1.0)
    imag = tl.load(FREQ + freq_row * FREQ_STRIDE + freq_dim * 2 + 1, rope, 0.0)
    # Fix the FP4 product/FMA order before BF16 so output modes quantize alike.
    real_product = libdevice.mul_rn(values, real)
    imag_product = libdevice.mul_rn(partner, imag)
    rotated = tl.where(
        (dims & 1) != 0,
        libdevice.fma_rn(partner, imag, real_product),
        libdevice.fma_rn(values, real, -imag_product),
    )
    values = tl.where(rope, rotated, values).to(tl.bfloat16).to(tl.float32)
    values = hadamard128(values).to(tl.bfloat16).to(tl.float32)
    packed, scales = quantize128(values)
    if RETURN_PACKED:
        tl.store(PACKED + row * 64 + tl.arange(0, 64), packed)
        tl.store(SCALES + row * 4 + tl.arange(0, 4), scales)
    if RECONSTRUCT:
        bytes_ = tl.gather(packed, dims // 2, axis=0)
        codes = (bytes_ >> ((dims & 1) * 4)) & 15
        scale = tl.gather(scales, dims // 32, axis=0).to(tl.int32)
        dequantized = e2m1_value(codes) * (scale << 23).to(tl.float32, bitcast=True)
        tl.store(OUT + row * 128 + dims, dequantized)
    weight = tl.load(W + row).to(tl.float32)
    tl.store(WEIGHTS + row, weight * WEIGHT_SCALE)


def quantize_q_rope(
    q,
    weights,
    freqs,
    rope_dim,
    *,
    weight_scale=1.0,
    return_packed=False,
    reconstruct=True,
):
    if not reconstruct and not return_packed:
        raise ValueError("FP4 query quantization requires an output representation")
    if q.ndim != 4 or q.shape[-1] != 128 or rope_dim != 64:
        raise ValueError("FP4 indexer Q requires [B,S,H,128] and RoPE dimension 64")
    if q.dtype != torch.bfloat16 or weights.shape != q.shape[:-1]:
        raise ValueError("FP4 indexer Q requires BF16 values and matching weights")
    q, weights = q.contiguous(), weights.contiguous()
    freq = torch.view_as_real(freqs).reshape(-1, 64).contiguous()
    b, s, h, _ = q.shape
    tokens = b * s
    if freq.shape[0] == tokens:
        group = 1
    elif s == 1 and freq.shape[0] == b:
        group = 1
    else:
        raise ValueError("FP4 indexer requires one RoPE row per query token")
    out = torch.empty_like(q) if reconstruct else None
    folded = torch.empty_like(weights, dtype=torch.float32)
    packed = (
        torch.empty((tokens * h, 64), device=q.device, dtype=torch.uint8)
        if return_packed
        else None
    )
    scales = (
        torch.empty((tokens * h, 4), device=q.device, dtype=torch.uint8)
        if return_packed
        else None
    )
    if tokens * h:
        _quantize_q[(tokens * h,)](
            q,
            weights,
            freq,
            out,
            folded,
            packed,
            scales,
            h,
            64,
            group,
            weight_scale,
            return_packed,
            reconstruct,
            num_warps=1,
            enable_fp_fusion=False,
        )
    if return_packed:
        return out, folded, packed, scales
    return out, folded


def quantize_q_rope_packed(q, weights, freqs, rope_dim, *, weight_scale=1.0):
    """Return native Q payload/scales without a reconstructed BF16 allocation."""
    _, folded, packed, scales = quantize_q_rope(
        q,
        weights,
        freqs,
        rope_dim,
        weight_scale=weight_scale,
        return_packed=True,
        reconstruct=False,
    )
    return packed.view(*q.shape[:-1], 64), folded, scales.view(*q.shape[:-1], 4)
