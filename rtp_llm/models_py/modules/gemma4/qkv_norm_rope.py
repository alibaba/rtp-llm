"""Inference-only Q/K weighted RMS + split-half RoPE and unit V RMS.

The caller owns rollout and the shared BF16 cos/sin tables. Unsupported inputs
return None so it can retain the existing norm/RoPE path. No attention, GEMM,
cache write, input mutation, or phase-dependent state is implemented here.
"""

from __future__ import annotations

import math
from functools import lru_cache
from typing import NamedTuple

import torch
import triton
import triton.language as tl

from rtp_llm.models_py.modules.gemma4.elementwise import _reference_reduce_width


class QKVNormRopeOutput(NamedTuple):
    q: torch.Tensor
    k: torch.Tensor
    v: torch.Tensor


@triton.jit
def _rms_scaled(
    X,
    rows,
    ROWS: tl.constexpr,
    HEADS: tl.constexpr,
    TOKEN_STRIDE: tl.constexpr,
    HEAD_STRIDE: tl.constexpr,
    D: tl.constexpr,
    EPS: tl.constexpr,
    TILE_ROWS: tl.constexpr,
    REDUCE_WIDTH: tl.constexpr,
):
    source = (rows // HEADS) * TOKEN_STRIDE + (rows % HEADS) * HEAD_STRIDE
    active = rows < ROWS
    lane = tl.arange(0, REDUCE_WIDTH)
    a0 = tl.full((TILE_ROWS, REDUCE_WIDTH), 0.0, tl.float32)
    a1 = tl.full((TILE_ROWS, REDUCE_WIDTH), 0.0, tl.float32)
    a2 = tl.full((TILE_ROWS, REDUCE_WIDTH), 0.0, tl.float32)
    a3 = tl.full((TILE_ROWS, REDUCE_WIDTH), 0.0, tl.float32)
    # tl.cdiv returns a scalar tensor; static_range needs a constexpr integer.
    iterations: tl.constexpr = (D + REDUCE_WIDTH * 4 - 1) // (REDUCE_WIDTH * 4)
    for iteration in tl.static_range(iterations):
        base = (lane + iteration * REDUCE_WIDTH) * 4
        p = X + source[:, None] + base[None, :]
        mask = active[:, None] & (base[None, :] < D)
        x0 = tl.load(p, mask=mask, other=0.0).to(tl.float32)
        x1 = tl.load(p + 1, mask=mask, other=0.0).to(tl.float32)
        x2 = tl.load(p + 2, mask=mask, other=0.0).to(tl.float32)
        x3 = tl.load(p + 3, mask=mask, other=0.0).to(tl.float32)
        a0 = a0 + x0 * x0
        a1 = a1 + x1 * x1
        a2 = a2 + x2 * x2
        a3 = a3 + x3 * x3
    total = ((a0 + a1) + a2) + a3
    # Pin the native logical half-tree, independently of the compiler's layout.
    # tl.sum can associate values differently when this RMS is fused with RoPE.
    # D256/512 use widths 32/64/128; only lane zero's final sum is consumed.
    for shift in tl.static_range(6, -1, -1):
        offset = 1 << shift
        if offset < REDUCE_WIDTH:
            paired = tl.broadcast_to(
                ((lane + offset) % REDUCE_WIDTH)[None, :],
                (TILE_ROWS, REDUCE_WIDTH),
            )
            total = total + tl.gather(total, paired, axis=1)
    first_lane = tl.full((TILE_ROWS, 1), 0, tl.int32)
    row_sum = tl.gather(total, first_lane, axis=1).reshape((TILE_ROWS,))
    variance = row_sum * (1.0 / D)
    inverse = tl.rsqrt(variance + EPS)
    columns = tl.arange(0, D)
    value = tl.load(
        X + source[:, None] + columns[None, :],
        mask=active[:, None],
        other=0.0,
    ).to(tl.float32)
    return value * inverse[:, None]


@triton.jit
def _rotate(
    normalized,
    COS,
    SIN,
    rows,
    ROWS: tl.constexpr,
    HEADS: tl.constexpr,
    COS_TOKEN_STRIDE: tl.constexpr,
    COS_DIM_STRIDE: tl.constexpr,
    SIN_TOKEN_STRIDE: tl.constexpr,
    SIN_DIM_STRIDE: tl.constexpr,
    D: tl.constexpr,
    TILE_ROWS: tl.constexpr,
):
    columns = tl.arange(0, D)
    paired = tl.where(columns < D // 2, columns + D // 2, columns - D // 2)
    paired = tl.broadcast_to(paired[None, :], (TILE_ROWS, D))
    partner = tl.gather(normalized, paired, axis=1).to(tl.float32)
    # Triton unary minus lowers to 0 - x, which changes signed zero. CUDA's
    # unary negation flips the sign bit, including for zero.
    sign = tl.where(columns[None, :] < D // 2, 0x80000000, 0).to(tl.uint32)
    partner = (partner.to(tl.uint32, bitcast=True) ^ sign).to(tl.float32, bitcast=True)
    partner = partner.to(tl.bfloat16).to(tl.float32)
    token = rows // HEADS
    mask = (rows < ROWS)[:, None]
    cosine = tl.load(
        COS + token[:, None] * COS_TOKEN_STRIDE + columns[None, :] * COS_DIM_STRIDE,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    sine = tl.load(
        SIN + token[:, None] * SIN_TOKEN_STRIDE + columns[None, :] * SIN_DIM_STRIDE,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    # Keep both multiplication results and the final addition rounded to BF16.
    first = (normalized.to(tl.float32) * cosine).to(tl.bfloat16).to(tl.float32)
    second = (partner * sine).to(tl.bfloat16).to(tl.float32)
    return (first + second).to(tl.bfloat16)


@triton.jit
def _qkv_norm_rope_kernel(
    Q,
    K,
    V,
    QW,
    KW,
    COS,
    SIN,
    Q_OUT,
    K_OUT,
    V_OUT,
    Q_TOKEN_STRIDE: tl.constexpr,
    Q_HEAD_STRIDE: tl.constexpr,
    K_TOKEN_STRIDE: tl.constexpr,
    K_HEAD_STRIDE: tl.constexpr,
    V_TOKEN_STRIDE: tl.constexpr,
    V_HEAD_STRIDE: tl.constexpr,
    COS_TOKEN_STRIDE: tl.constexpr,
    COS_DIM_STRIDE: tl.constexpr,
    SIN_TOKEN_STRIDE: tl.constexpr,
    SIN_DIM_STRIDE: tl.constexpr,
    Q_HEADS: tl.constexpr,
    KV_HEADS: tl.constexpr,
    Q_ROWS: tl.constexpr,
    KV_ROWS: tl.constexpr,
    Q_TILES: tl.constexpr,
    KV_TILES: tl.constexpr,
    D: tl.constexpr,
    EPS: tl.constexpr,
    SHARED_KV: tl.constexpr,
    TILE_ROWS: tl.constexpr,
    Q_REDUCE_WIDTH: tl.constexpr,
    KV_REDUCE_WIDTH: tl.constexpr,
):
    program = tl.program_id(0).to(tl.int64)
    local = tl.arange(0, TILE_ROWS).to(tl.int64)
    columns = tl.arange(0, D)
    if program < Q_TILES:
        rows = program * TILE_ROWS + local
        scaled = _rms_scaled(
            Q,
            rows,
            Q_ROWS,
            Q_HEADS,
            Q_TOKEN_STRIDE,
            Q_HEAD_STRIDE,
            D,
            EPS,
            TILE_ROWS,
            Q_REDUCE_WIDTH,
        )
        weight = tl.load(QW + columns)
        normalized = (scaled * weight[None, :]).to(tl.bfloat16)
        rotated = _rotate(
            normalized,
            COS,
            SIN,
            rows,
            Q_ROWS,
            Q_HEADS,
            COS_TOKEN_STRIDE,
            COS_DIM_STRIDE,
            SIN_TOKEN_STRIDE,
            SIN_DIM_STRIDE,
            D,
            TILE_ROWS,
        )
        tl.store(
            Q_OUT + rows[:, None] * D + columns[None, :],
            rotated,
            mask=(rows < Q_ROWS)[:, None],
        )
    elif program < Q_TILES + KV_TILES:
        rows = (program - Q_TILES) * TILE_ROWS + local
        scaled = _rms_scaled(
            K,
            rows,
            KV_ROWS,
            KV_HEADS,
            K_TOKEN_STRIDE,
            K_HEAD_STRIDE,
            D,
            EPS,
            TILE_ROWS,
            KV_REDUCE_WIDTH,
        )
        weight = tl.load(KW + columns)
        normalized = (scaled * weight[None, :]).to(tl.bfloat16)
        rotated = _rotate(
            normalized,
            COS,
            SIN,
            rows,
            KV_ROWS,
            KV_HEADS,
            COS_TOKEN_STRIDE,
            COS_DIM_STRIDE,
            SIN_TOKEN_STRIDE,
            SIN_DIM_STRIDE,
            D,
            TILE_ROWS,
        )
        target = rows[:, None] * D + columns[None, :]
        tl.store(K_OUT + target, rotated, mask=(rows < KV_ROWS)[:, None])
        if SHARED_KV:
            # Unit V uses the original K statistics, without gamma or RoPE.
            tl.store(
                V_OUT + target, scaled.to(tl.bfloat16), mask=(rows < KV_ROWS)[:, None]
            )
    else:
        rows = (program - Q_TILES - KV_TILES) * TILE_ROWS + local
        scaled = _rms_scaled(
            V,
            rows,
            KV_ROWS,
            KV_HEADS,
            V_TOKEN_STRIDE,
            V_HEAD_STRIDE,
            D,
            EPS,
            TILE_ROWS,
            KV_REDUCE_WIDTH,
        )
        tl.store(
            V_OUT + rows[:, None] * D + columns[None, :],
            scaled.to(tl.bfloat16),
            mask=(rows < KV_ROWS)[:, None],
        )


@lru_cache(maxsize=None)
def _supports_bf16_cuda(device: torch.device) -> bool:
    return torch.cuda.get_device_capability(device)[0] >= 8


def is_supported(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q_weight_float: torch.Tensor,
    k_weight_float: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    eps: float,
    *,
    k_equals_v: bool = False,
) -> bool:
    tensors = (q, k, v, q_weight_float, k_weight_float, cos, sin)
    if not all(isinstance(t, torch.Tensor) for t in tensors):
        return False
    if type(k_equals_v) is not bool:
        return False
    if not q.is_cuda or torch.version.hip is not None or q.ndim != 3:
        return False
    tokens, q_heads, dim = q.shape
    if (
        dim not in (256, 512)
        or q_heads <= 0
        or k.ndim != 3
        or k.shape[0] != tokens
        or k.shape[1] <= 0
        or k.shape[2] != dim
        or v.shape != k.shape
        or v.ndim != 3
        or type(eps) not in (int, float)
        or not math.isfinite(eps)
        or eps <= 0
        or any(t.device != q.device or t.requires_grad for t in tensors)
        or any(
            t.dtype != torch.bfloat16
            or t.stride(-1) != 1
            or any(s < 0 for s in t.stride())
            for t in (q, k, v)
        )
        or any(
            w.dtype != torch.float32 or w.shape != (dim,) or not w.is_contiguous()
            for w in (q_weight_float, k_weight_float)
        )
        or any(
            t.dtype != torch.bfloat16
            or t.shape != (tokens, dim)
            or any(s < 0 for s in t.stride())
            for t in (cos, sin)
        )
    ):
        return False
    if k_equals_v and (v.data_ptr() != k.data_ptr() or v.stride() != k.stride()):
        return False
    return _supports_bf16_cuda(q.device)


def qkv_norm_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q_weight_float: torch.Tensor,
    k_weight_float: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    eps: float,
    *,
    k_equals_v: bool = False,
    rows_per_program: int | None = None,
) -> QKVNormRopeOutput | None:
    if not is_supported(
        q, k, v, q_weight_float, k_weight_float, cos, sin, eps, k_equals_v=k_equals_v
    ):
        return None
    tile = (
        (1 if q.shape[0] < 32 else 4) if rows_per_program is None else rows_per_program
    )
    if type(tile) is not int or tile not in (1, 2, 4, 8):
        return None
    with torch.cuda.device(q.device):
        output = QKVNormRopeOutput(
            torch.empty(q.shape, dtype=q.dtype, device=q.device),
            torch.empty(k.shape, dtype=k.dtype, device=k.device),
            torch.empty(v.shape, dtype=v.dtype, device=v.device),
        )
        if q.shape[0] == 0:
            return output
        q_rows = q.shape[0] * q.shape[1]
        kv_rows = k.shape[0] * k.shape[1]
        q_tiles = triton.cdiv(q_rows, tile)
        kv_tiles = triton.cdiv(kv_rows, tile)
        programs = q_tiles + kv_tiles * (1 if k_equals_v else 2)
        _qkv_norm_rope_kernel[(programs,)](
            q,
            k,
            v,
            q_weight_float,
            k_weight_float,
            cos,
            sin,
            *output,
            q.stride(0),
            q.stride(1),
            k.stride(0),
            k.stride(1),
            v.stride(0),
            v.stride(1),
            cos.stride(0),
            cos.stride(1),
            sin.stride(0),
            sin.stride(1),
            q.shape[1],
            k.shape[1],
            q_rows,
            kv_rows,
            q_tiles,
            kv_tiles,
            q.shape[2],
            eps,
            k_equals_v,
            tile,
            _reference_reduce_width(q.shape[2], q_rows),
            _reference_reduce_width(k.shape[2], kv_rows),
            num_warps=4,
            enable_fp_fusion=False,
        )
    return output
