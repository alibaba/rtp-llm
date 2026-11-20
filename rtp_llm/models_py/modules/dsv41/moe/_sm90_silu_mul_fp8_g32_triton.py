"""SM90 SwiGLU and group32 quantization with native MN-major scales."""

from __future__ import annotations

import os
from typing import Tuple

import torch
import triton
import triton.language as tl

from .._sm90_quant_scale_fused import _decode_exponent, mn_major_scale_output


@triton.jit
def _mul_ftz(a, b):
    return tl.inline_asm_elementwise(
        "mul.ftz.f32 $0, $1, $2;",
        "=f,f,f",
        [a, b],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit(do_not_specialize=["scale_stride", "legacy"])
def _silu_mul_fp8_g32_quant_kernel(
    gate_up_ptr,
    q_ptr,
    scales_ptr,
    scale_stride,
    legacy,
    N: tl.constexpr,
    APPLY_CLAMP: tl.constexpr,
    CLAMP_LIMIT: tl.constexpr,
    ROUND_TO_BF16: tl.constexpr,
    BLOCK_N: tl.constexpr,
    FP32_SCALES: tl.constexpr,
):
    GROUPS: tl.constexpr = BLOCK_N // 32
    PACKS: tl.constexpr = GROUPS // 4
    pid_m = tl.program_id(0).to(tl.int64)
    pid_n = tl.program_id(1).to(tl.int64)
    n_off = pid_n * BLOCK_N + tl.arange(0, BLOCK_N).to(tl.int64)
    mask = n_off < N
    row = pid_m * (2 * N)
    g = tl.load(gate_up_ptr + row + n_off, mask=mask, other=0.0).to(tl.float32)
    u = tl.load(gate_up_ptr + row + N + n_off, mask=mask, other=0.0).to(tl.float32)
    if APPLY_CLAMP:
        u = tl.where(u > CLAMP_LIMIT, CLAMP_LIMIT, u)
        u = tl.where(u < -CLAMP_LIMIT, -CLAMP_LIMIT, u)
        g = tl.where(g > CLAMP_LIMIT, CLAMP_LIMIT, g)
    s = g * tl.sigmoid(g)
    y = s * u
    if ROUND_TO_BF16:
        # Legacy callers quantize a materialized BF16 activation.
        y = y.to(tl.bfloat16).to(tl.float32)
    y = y.reshape((GROUPS, 32))
    # Native fmaxf ignores NaNs and seeds each lane with FP32 tiny.
    amax = tl.max(
        tl.maximum(tl.abs(y), 2.0**-126, propagate_nan=tl.PropagateNan.NONE), 1
    )
    if legacy:
        raw = tl.maximum(amax / 448.0, 1.0e-10)
        scale = tl.exp2(tl.ceil(tl.log2(raw)))
        scaled = y / scale[:, None]
        biased = tl.log2(scale).to(tl.int32) + 127
    else:
        # CUDA13 native v2 uses FTZ and fast_log2_ceil/fast_pow2.
        raw = _mul_ftz(amax, 1.0 / 448.0)
        bits = raw.to(tl.int32, bitcast=True)
        biased = ((bits >> 23) & 255) + ((bits & 0x7FFFFF) != 0).to(tl.int32)
        reciprocal = ((254 - biased) << 23).to(tl.float32, bitcast=True)
        scaled = _mul_ftz(y, reciprocal[:, None])
    q = tl.minimum(
        tl.maximum(scaled, -448.0, propagate_nan=tl.PropagateNan.NONE), 448.0
    ).reshape((BLOCK_N,))
    tl.store(q_ptr + pid_m * N + n_off, q.to(tl.float8e4nv), mask=mask)
    group = pid_n * GROUPS + tl.arange(0, GROUPS)
    if FP32_SCALES:
        # Decode the stored UE8M0 byte in-kernel and write the FP32
        # MN-major layout the SM90 DeepGEMM TMA load wants directly.
        byte = tl.where(group < N // 32, biased, 0).to(tl.uint32) & 255
        tl.store(
            scales_ptr + group * scale_stride + pid_m,
            _decode_exponent(byte),
            group < N // 32,
        )
    else:
        biased = tl.where(group < N // 32, biased, 0).to(tl.uint8).to(tl.uint32)
        packed = tl.sum(
            biased.reshape((PACKS, 4)) << (tl.arange(0, 4)[None, :] * 8), 1
        )
        pack = pid_n * PACKS + tl.arange(0, PACKS)
        tl.store(
            scales_ptr + pack * scale_stride + pid_m,
            packed,
            pack < tl.cdiv(N, 128),
        )


def silu_mul_fp8_g32_quant_mn_major(
    gate_up: torch.Tensor, clamp_limit: float = 0.0, *, round_to_bf16: bool = True
) -> Tuple[torch.Tensor, torch.Tensor]:
    """SwiGLU + group32 quant returning native FP32 MN-major scales.

    Arithmetic, BF16 rounding and quantized bytes are identical to
    :func:`silu_mul_fp8_g32_quant`; only the scale store differs. The SM90
    DeepGEMM adapter consumes this layout directly, so the packed UE8M0 words
    and the unpack launch between the two kernels disappear.
    """
    return _silu_mul_fp8_g32_quant(
        gate_up, clamp_limit, fp32_scales=True, round_to_bf16=round_to_bf16
    )


def _silu_mul_fp8_g32_quant(
    gate_up: torch.Tensor, clamp_limit: float, *, fp32_scales: bool, round_to_bf16: bool
) -> Tuple[torch.Tensor, torch.Tensor]:
    m, two_n = gate_up.shape
    n = two_n // 2
    quant = torch.empty((m, n), dtype=torch.float8_e4m3fn, device=gate_up.device)
    if fp32_scales:
        scales = mn_major_scale_output(m, n // 32, gate_up.device)
    else:
        scales = torch.empty(
            (triton.cdiv(n, 128), triton.cdiv(m, 4) * 4),
            dtype=torch.int32,
            device=gate_up.device,
        ).T[:m]
    if m * n == 0:
        return quant, scales
    mode = os.environ.get("DSV4_FP8_QUANT_KERNEL", "auto").strip().lower()
    legacy = mode in ("auto", "legacy")
    apply_clamp = clamp_limit is not None and clamp_limit > 0.0
    _silu_mul_fp8_g32_quant_kernel[(m, triton.cdiv(n, 1024))](
        gate_up,
        quant,
        scales,
        scales.stride(1),
        legacy,
        N=n,
        APPLY_CLAMP=apply_clamp,
        CLAMP_LIMIT=float(clamp_limit) if apply_clamp else 0.0,
        ROUND_TO_BF16=round_to_bf16,
        BLOCK_N=1024,
        FP32_SCALES=fp32_scales,
        num_warps=8,
        enable_fp_fusion=False,
    )
    return quant, scales
