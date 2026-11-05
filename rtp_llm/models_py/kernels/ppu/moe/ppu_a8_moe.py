"""PPU A8 expert arithmetic with lossless repacking of checkpoint FP4 weights."""

import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice


@triton.jit
def _weight_scales(S, OUT, HI, BAD, N: tl.constexpr, K: tl.constexpr):
    e, nb, kb = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    lane = tl.arange(0, 512)
    exp = tl.load(
        S + (e * N + nb * 128 + lane // 4) * (K // 32) + kb * 4 + lane % 4
    ).to(tl.int32)
    lo, hi = tl.min(exp, 0), tl.max(exp, 0)
    offset = (e * (N // 128) + nb) * (K // 128) + kb
    # E2M1 max=6 maps to E4M3 384. The smallest E2M1 value remains exact
    # through an exponent spread of 14; reject anything outside that proof.
    tl.store(OUT + offset, tl.exp2((hi - 133).to(tl.float32)))
    tl.store(HI + offset, hi)
    tl.store(BAD + offset, (lo <= 0) | (hi >= 255) | (hi < 7) | (hi - lo > 14))


@triton.jit
def _repack_weight(W, S, HI, OUT, N: tl.constexpr, K: tl.constexpr):
    e = tl.program_id(1)
    index = tl.program_id(0) * 512 + tl.arange(0, 512)
    row, col = index // K, index % K
    raw = tl.load(W + e * N * (K // 2) + index // 2, mask=row < N, other=0).to(tl.int32)
    nibble = (raw >> ((col % 2) * 4)) & 15
    mag = nibble & 7
    value = tl.where(
        mag < 4, mag * 0.5, tl.where(mag < 6, mag - 2.0, (mag - 4.0) * 2.0)
    )
    value = tl.where(nibble >= 8, -value, value)
    exponent = tl.load(
        S + (e * N + row) * (K // 32) + col // 32, mask=row < N, other=127
    ).to(tl.int32)
    highest = tl.load(
        HI + (e * (N // 128) + row // 128) * (K // 128) + col // 128,
        mask=row < N,
        other=127,
    )
    value = value * tl.exp2((exponent - highest + 6).to(tl.float32))
    tl.store(OUT + e * N * K + index, value.to(OUT.dtype.element_ty), mask=row < N)


def repack_fp4_to_fp8(weight, scale):
    """Repack once at load time; reject weights that cannot be represented exactly."""
    if weight.ndim != 3 or weight.dtype not in (torch.uint8, torch.int8):
        raise ValueError("Expected packed FP4 [E,N,K/2] weights")
    e, n, packed_k = weight.shape
    k = packed_k * 2
    if (
        n % 128
        or k % 128
        or scale.shape != (e, n, k // 32)
        or scale.dtype != torch.float8_e8m0fnu
        or not weight.is_cuda
        or scale.device != weight.device
        or not weight.is_contiguous()
        or not scale.is_contiguous()
    ):
        raise ValueError(
            "FP4 repack requires contiguous group32 E8M0 scales and aligned PPU weights"
        )
    ws = torch.empty((e, n // 128, k // 128), device=weight.device, dtype=torch.float32)
    bad = torch.empty_like(ws, dtype=torch.int32)
    highest = torch.empty_like(bad)
    _weight_scales[(e, n // 128, k // 128)](
        scale.view(torch.uint8), ws, highest, bad, N=n, K=k
    )
    if bad.sum().item():
        raise ValueError(
            "FP4 weight exponent range cannot be losslessly repacked to FP8"
        )
    out = torch.empty((e, n, k), device=weight.device, dtype=torch.float8_e4m3fn)
    _repack_weight[(triton.cdiv(n * k, 512), e)](
        weight.view(torch.uint8), scale.view(torch.uint8), highest, out, N=n, K=k
    )
    return out, ws


@triton.jit
def _weighted_silu_q8(
    GU,
    COUNTS,
    SRC,
    META,
    EXPERTS,
    Q,
    S,
    M: tl.constexpr,
    I: tl.constexpr,
    CAP: tl.constexpr,
    TOPK: tl.constexpr,
    LIMIT: tl.constexpr,
):
    e = tl.program_id(1)
    g = tl.program_id(2)
    count = tl.load(COUNTS + e)
    expert = tl.load(EXPERTS + e)
    slots = tl.arange(0, TOPK)
    col = g * 512 + tl.arange(0, 512)
    for row in range(tl.program_id(0), count, tl.num_programs(0)):
        source = tl.load(SRC + e * M + row)
        rank = source >> 32
        token = source & 0xFFFFFFFF
        base = ((rank * CAP + token) * TOPK + slots) * 2
        ids = tl.load(META + base).to(tl.int32)
        weights = tl.load(META + base + 1)
        weight = tl.sum(tl.where(ids == expert, weights, 0), 0)
        gate = tl.load(GU + (e * M + row) * 2 * I + col).to(tl.float32)
        up = tl.load(GU + (e * M + row) * 2 * I + I + col).to(tl.float32)
        if LIMIT > 0:
            gate = tl.minimum(gate, LIMIT)
            up = tl.minimum(tl.maximum(up, -LIMIT), LIMIT)
        hidden = (
            (gate / (1.0 + tl.exp(-gate)) * up * weight).to(tl.bfloat16).to(tl.float32)
        )
        hidden = tl.reshape(hidden, (4, 128))
        scale = tl.exp2(
            tl.ceil(tl.log2(tl.maximum(tl.max(tl.abs(hidden), 1), 1.0e-4) / 448.0))
        )
        quant = tl.reshape(
            tl.minimum(tl.maximum(hidden / scale[:, None], -448.0), 448.0).to(
                Q.dtype.element_ty
            ),
            (512,),
        )
        tl.store(Q + (e * M + row) * I + col, quant)
        tl.store(S + (e * (I // 128) + g * 4 + tl.arange(0, 4)) * M + row, scale)


def masked_a8_experts(
    x,
    weight13,
    weight2,
    counts,
    source,
    metadata,
    expert_ids,
    expected_m=16,
    swiglu_limit=10.0,
    out=None,
):
    from deep_gemm.jit_kernels.m_grouped_gemm_fp8_k32 import (
        m_grouped_gemm_fp8_fp8_bf16_nt_masked as grouped,
    )

    e, m, k = x[0].shape
    i = weight13[0].shape[1] // 2
    if i % 512 or expected_m < 1 or counts.shape != (e,) or source.shape != (e, m):
        raise ValueError("Invalid masked A8 expert dimensions/counts")
    if out is None:
        out = torch.empty((e, m, k), dtype=torch.bfloat16, device=x[0].device)
    elif (
        out.shape != (e, m, k)
        or out.dtype != torch.bfloat16
        or out.device != x[0].device
        or not out.is_contiguous()
        or any(
            out.untyped_storage().data_ptr() == t.untyped_storage().data_ptr()
            for t in (*x, *weight13, *weight2, counts, source, metadata, expert_ids)
        )
    ):
        raise ValueError("Masked A8 output must be a distinct contiguous BF16 buffer")
    gu = torch.empty((e, m, 2 * i), dtype=torch.bfloat16, device=x[0].device)
    grouped(x, weight13, gu, counts, expected_m)
    q = torch.empty((e, m, i), dtype=torch.float8_e4m3fn, device=gu.device)
    scales = torch.empty((e, i // 128, m), dtype=torch.float32, device=gu.device)
    _weighted_silu_q8[(min(m, expected_m), e, i // 512)](
        gu,
        counts,
        source,
        metadata,
        expert_ids,
        q,
        scales,
        M=m,
        I=i,
        CAP=metadata.shape[1],
        TOPK=metadata.shape[2],
        LIMIT=swiglu_limit,
        num_warps=4,
    )
    grouped((q, scales.transpose(1, 2)), weight2, out, counts, expected_m)
    return out


@triton.jit
def _quantize_rows(X, Q, S, M: tl.constexpr, K: tl.constexpr):
    row, block = tl.program_id(0), tl.program_id(1)
    col = block * 512 + tl.arange(0, 512)
    value = tl.load(X + row * K + col).to(tl.float32).reshape(4, 128)
    scale = tl.exp2(
        tl.ceil(tl.log2(tl.maximum(tl.max(tl.abs(value), 1), 1.0e-4) / 448.0))
    )
    quant = (value / scale[:, None]).to(Q.dtype.element_ty).reshape(512)
    tl.store(Q + row * K + col, quant)
    tl.store(S + (block * 4 + tl.arange(0, 4)) * M + row, scale)


def quantize_a8(x):
    if (
        x.ndim != 2
        or x.shape[1] % 512
        or x.dtype != torch.bfloat16
        or not x.is_contiguous()
    ):
        raise ValueError("A8 quantization requires contiguous BF16 rows aligned to512")
    m, k = x.shape
    q = torch.empty_like(x, dtype=torch.float8_e4m3fn)
    scale = torch.empty((k // 128, m), device=x.device, dtype=torch.float32)
    if m:
        _quantize_rows[(m, k // 512)](x, q, scale, M=m, K=k, num_warps=4)
    return q, scale.t()


@triton.jit
def _weighted_nopad(
    GU,
    SOURCE,
    EXPERTS,
    IDS,
    WEIGHTS,
    VALID,
    Q,
    S,
    M: tl.constexpr,
    I: tl.constexpr,
    TOPK: tl.constexpr,
    BK: tl.constexpr,
    LIMIT: tl.constexpr,
):
    row, block = tl.program_id(0), tl.program_id(1)
    valid = row < tl.load(VALID)
    token = tl.load(SOURCE + row)
    expert = tl.load(EXPERTS + row)
    slots = tl.arange(0, BK)
    ids = tl.load(IDS + token * TOPK + slots, mask=slots < TOPK, other=-1)
    weights = tl.load(WEIGHTS + token * TOPK + slots, mask=slots < TOPK, other=0.0)
    weight = tl.sum(tl.where(ids == expert, weights, 0.0), 0)
    col = block * 512 + tl.arange(0, 512)
    gate = tl.load(GU + row * 2 * I + col, mask=valid, other=0.0).to(tl.float32)
    up = tl.load(GU + row * 2 * I + I + col, mask=valid, other=0.0).to(tl.float32)
    if LIMIT > 0:
        gate = tl.minimum(gate, LIMIT)
        up = tl.minimum(tl.maximum(up, -LIMIT), LIMIT)
    value = (
        (tl.div_rn(gate, 1.0 + libdevice.exp(-gate)) * up * weight)
        .to(tl.bfloat16)
        .to(tl.float32)
        .reshape(4, 128)
    )
    scale = tl.exp2(
        tl.ceil(tl.log2(tl.maximum(tl.max(tl.abs(value), 1), 1.0e-4) / 448.0))
    )
    quant = (value / scale[:, None]).to(Q.dtype.element_ty).reshape(512)
    tl.store(Q + row * I + col, quant)
    tl.store(S + (block * 4 + tl.arange(0, 4)) * M + row, scale)


def weighted_nopad_a8(gate_up, source, expert_ids, indices, weights, counts, limit):
    m, twice_i = gate_up.shape
    i = twice_i // 2
    if i % 512:
        raise ValueError("PPU A8 expert intermediate size must be aligned to512")
    q = torch.empty((m, i), device=gate_up.device, dtype=torch.float8_e4m3fn)
    scales = torch.empty((i // 128, m), device=gate_up.device, dtype=torch.float32)
    valid = counts.sum()
    _weighted_nopad[(m, i // 512)](
        gate_up,
        source,
        expert_ids,
        indices,
        weights,
        valid,
        q,
        scales,
        M=m,
        I=i,
        TOPK=indices.shape[1],
        BK=triton.next_power_of_2(indices.shape[1]),
        LIMIT=limit,
        num_warps=4,
    )
    return q, scales.t()
