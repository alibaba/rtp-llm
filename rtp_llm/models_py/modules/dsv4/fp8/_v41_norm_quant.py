"""V4.1 attention RMSNorm + group32 MXFP8 fusion.

The materialized BF16 result remains available to global/indexer consumers;
quantization sees that exact BF16 value.
The packed int32 scales are suitable for V41MXFP8Linear's (1, 1, 32) recipe.
"""

from __future__ import annotations

import math
import os

import torch
import triton
import triton.language as tl


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


@triton.jit
def _fma_ftz(a, b, c):
    return tl.inline_asm_elementwise(
        "fma.rn.ftz.f32 $0, $1, $2, $3;",
        "=f,f,f,f",
        [a, b, c],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit(do_not_specialize=["scale_stride", "eps", "legacy"])
def _v41_norm_quant_kernel(x, weight, norm, quant, scales, scale_stride, eps, legacy):
    row = tl.program_id(0).to(tl.int64)
    col = tl.arange(0, 8192)
    value = tl.load(x + row * 5120 + col, col < 5120, 0).to(tl.float32)

    # FlashInfer RMSNormKernel<8, BF16> has 640 threads (20 warps).
    # Each thread accumulates eight squares with FMA, then XOR-reduces
    # within a warp and across 20 warp sums (padded with zeros to 32).
    even, odd = tl.split(value.reshape((1024, 4, 2)))
    v04, v26 = tl.split(even.reshape((1024, 2, 2)))
    v15, v37 = tl.split(odd.reshape((1024, 2, 2)))
    v0, v4 = tl.split(v04)
    v2, v6 = tl.split(v26)
    v1, v5 = tl.split(v15)
    v3, v7 = tl.split(v37)
    partial = _fma_ftz(v0, v0, 0.0)
    partial = _fma_ftz(v1, v1, partial)
    partial = _fma_ftz(v2, v2, partial)
    partial = _fma_ftz(v3, v3, partial)
    partial = _fma_ftz(v4, v4, partial)
    partial = _fma_ftz(v5, v5, partial)
    partial = _fma_ftz(v6, v6, partial)
    partial = _fma_ftz(v7, v7, partial)
    lanes = tl.arange(0, 1024)
    for delta in tl.static_range(5):
        partial = partial + tl.gather(partial, lanes ^ (16 >> delta), 0)
    warp_sums = tl.gather(partial, tl.arange(0, 32) * 32, 0)
    for delta in tl.static_range(5):
        warp_sums = warp_sums + tl.gather(
            warp_sums, tl.arange(0, 32) ^ (16 >> delta), 0
        )
    total = tl.sum(tl.where(tl.arange(0, 32) == 0, warp_sums, 0.0), 0)
    # 5120 is not a power of two: reciprocal multiplication/FMA changes
    # the native BF16 rounding at half-way values. Keep division then add.
    inv = tl.rsqrt(tl.div_rn(total, 5120.0) + eps)
    w = tl.load(weight + col, col < 5120, 0).to(tl.float32)
    # weight_bias=0 in the native norm; preserve signed-zero addition.
    w = tl.inline_asm_elementwise(
        "add.rn.ftz.f32 $0, $1, 0f00000000;",
        "=f,f",
        [w],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )
    # The deployed CUDA13 native build flushes FP32 denormals to zero.
    y = _mul_ftz(_mul_ftz(value, inv), w).to(tl.bfloat16)
    tl.store(norm + row * 5120 + col, y, col < 5120)

    grouped = y.to(tl.float32).reshape((256, 32))
    amax = tl.maximum(tl.max(tl.abs(grouped), 1), 2.0**-126)
    if legacy:
        raw = tl.maximum(amax / 448.0, 1.0e-10)
        exponent = tl.ceil(tl.log2(raw)).to(tl.int32)
        scale = tl.exp2(exponent.to(tl.float32))
        scaled = grouped / scale[:, None]
        biased = exponent + 127
    else:
        # Match native v2 fast_log2_ceil/fast_pow2 with CUDA13 FTZ.
        # A subnormal raw scale becomes zero: byte=0, reciprocal=2**127.
        # Do not substitute log2 or inherit V4's 1e-4 floor.
        raw = _mul_ftz(amax, 1.0 / 448.0)
        bits = raw.to(tl.int32, bitcast=True)
        biased = ((bits >> 23) & 255) + ((bits & 0x7FFFFF) != 0).to(tl.int32)
        reciprocal = ((254 - biased) << 23).to(tl.float32, bitcast=True)
        scaled = _mul_ftz(grouped, reciprocal[:, None])
    q = tl.minimum(tl.maximum(scaled, -448.0), 448.0).reshape((8192,))
    tl.store(quant + row * 5120 + col, q.to(tl.float8e4nv), col < 5120)
    bytes_ = biased.reshape((64, 4)) << (tl.arange(0, 4)[None, :] * 8)
    packed = tl.sum(bytes_, 1)
    pack = tl.arange(0, 64)
    tl.store(scales + pack * scale_stride + row, packed, pack < 40)


def is_supported(x: torch.Tensor, weight: torch.Tensor) -> bool:
    """Metadata gate for contiguous BF16 [M,5120] inference on SM100.

    Finiteness is a caller precondition, not a synchronizing device scan.
    """
    return (
        x.ndim == 2
        and x.shape[1] == 5120
        and x.is_cuda
        and weight.device == x.device
        and x.dtype == weight.dtype == torch.bfloat16
        and weight.shape == (5120,)
        and x.is_contiguous()
        and weight.is_contiguous()
        and x.data_ptr() % 16 == 0
        and weight.data_ptr() % 16 == 0
        and not x.requires_grad
        and not weight.requires_grad
        and torch.cuda.get_device_capability(x.device) == (10, 0)
    )


def rmsnorm_group32_quant(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1.0e-6,
    *,
    quant_kernel: str | None = None,
    out_norm: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
    """Return (BF16 norm, E4M3 payload, packed UE8M0), or None if unsupported.

    ``out_norm=x`` supports the existing attention in-place norm boundary.
    Otherwise out_norm must be disjoint from x and weight. Inputs are finite;
    their FP32 square sum and BF16 normalized output must not overflow.
    Numerical contract is the CUDA13 native FTZ build; no host tensor reads.
    An omitted quant_kernel honors the existing DSV4_FP8_QUANT_KERNEL setting.
    auto uses v2 at M*5120 >= 4Mi elements, exactly as V41MXFP8Linear does.
    """
    if not is_supported(x, weight):
        return None
    if not math.isfinite(eps) or not 2.0**-126 <= eps <= torch.finfo(torch.float32).max:
        raise ValueError("eps must be positive, normal, finite FP32")
    mode = (
        (
            os.environ.get("DSV4_FP8_QUANT_KERNEL", "auto")
            if quant_kernel is None
            else quant_kernel
        )
        .strip()
        .lower()
    )
    if mode not in ("auto", "legacy", "v2"):
        raise ValueError("quant_kernel must be auto, legacy or v2")
    if out_norm is None:
        out_norm = torch.empty_like(x)
    else:
        if (
            out_norm.shape != x.shape
            or out_norm.dtype != x.dtype
            or out_norm.device != x.device
            or not out_norm.is_contiguous()
            or out_norm.data_ptr() % 16 != 0
            or out_norm.requires_grad
        ):
            raise ValueError(
                "out_norm must be aligned contiguous BF16 with x's shape and device"
            )
        if out_norm.numel():
            start = out_norm.data_ptr()
            end = start + out_norm.numel() * 2
            for source in (x, weight):
                left = source.data_ptr()
                right = left + source.numel() * 2
                if start < right and left < end:
                    if source is not x or start != left:
                        raise ValueError(
                            "out_norm may alias x exactly, but must not overlap weight or part of x"
                        )
    m = x.shape[0]
    quant = torch.empty_like(x, dtype=torch.float8_e4m3fn)
    scales = torch.empty(
        (40, triton.cdiv(m, 4) * 4), device=x.device, dtype=torch.int32
    ).T[:m]
    if m:
        legacy = mode == "legacy" or (mode == "auto" and x.numel() < 4 * 1024 * 1024)
        _v41_norm_quant_kernel[(m,)](
            x,
            weight,
            out_norm,
            quant,
            scales,
            scales.stride(1),
            eps,
            legacy,
            num_warps=4,
            enable_fp_fusion=False,
        )
    return out_norm, quant, scales
