"""SM90 query normalization and native FP32 MN-major scale producer."""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from ._sm90_quant_scale_fused import _decode_exponent, mn_major_scale_output


@triton.jit
def _query_norm_quant_kernel(
    X,
    W,
    Y,
    Q,
    SF,
    stride_x,
    stride_sf,
    D: tl.constexpr,
    EPS: tl.constexpr,
    BLOCK: tl.constexpr,
    FP32_SCALES: tl.constexpr,
    PDL: tl.constexpr = False,
):
    row = tl.program_id(0).to(tl.int64)
    cols = tl.arange(0, BLOCK)
    # Immutable weights may be loaded while QA is still producing activations.
    w = tl.load(W + cols, cols < D, 0).to(tl.float32)
    if PDL:
        tl.extra.cuda.gdc_wait()
    x = tl.load(X + row * stride_x + cols, cols < D, 0).to(tl.float32)
    inv = tl.rsqrt(tl.sum(x * x, 0) / D + EPS)
    # Match the existing strided norm's materialized BF16 boundary.
    y = (x * inv * w).to(tl.bfloat16)
    tl.store(Y + row * D + cols, y, cols < D)
    groups = tl.reshape(y.to(tl.float32), (BLOCK // 32, 32))
    amax = tl.max(tl.abs(groups), 1)
    # The legacy CUDA producer clamps the scale itself to 1e-10.
    raw = tl.maximum(tl.div_rn(amax, 448.0), 1.0e-10)
    bits = raw.to(tl.uint32, bitcast=True)
    exponent = (bits >> 23) + ((bits & 0x7FFFFF) != 0).to(tl.uint32)
    scale = (exponent << 23).to(tl.float32, bitcast=True)
    quant = tl.div_rn(groups, scale[:, None])
    quant = tl.minimum(tl.maximum(quant, -448.0), 448.0)
    tl.store(Q + row * D + cols, tl.reshape(quant, (BLOCK,)), cols < D)
    if PDL:
        # QB may initialize during scale packing, but must wait before reads.
        tl.extra.cuda.gdc_launch_dependents()
    group_ids = tl.arange(0, BLOCK // 32)
    exponent = tl.where(group_ids < D // 32, exponent, 0)
    if FP32_SCALES:
        # Decode the stored UE8M0 byte in-kernel and write the FP32 MN-major
        # layout the SM90 DeepGEMM TMA load wants: no packed words, no unpack.
        tl.store(
            SF + group_ids * stride_sf + row,
            _decode_exponent(exponent & 255),
            group_ids < D // 32,
        )
    else:
        packed = tl.sum(
            tl.reshape(exponent, (BLOCK // 128, 4)) << (tl.arange(0, 4)[None, :] * 8),
            1,
        )
        packs = tl.arange(0, BLOCK // 128)
        tl.store(SF + packs * stride_sf + row, packed, packs < triton.cdiv(D, 128))


def query_norm_quant_mn_major(x: torch.Tensor, weight: torch.Tensor, eps: float, *, pdl: bool = False
):
    """Return BF16 Q plus FP8 Q and native FP32 MN-major scales.

    Arithmetic is identical to :func:`query_norm_quant` -- same normalization,
    same UE8M0 exponent, same quantized bytes -- and only the scale store
    differs. The SM90 DeepGEMM adapter consumes this layout directly, so the
    pack/unpack round trip and its launch disappear.
    """
    return _query_norm_quant(x, weight, eps, fp32_scales=True, pdl=pdl)


def _query_norm_quant(
    x: torch.Tensor, weight: torch.Tensor, eps: float, *, fp32_scales: bool, pdl: bool
):
    import deep_gemm

    pdl = pdl or (
        torch.cuda.get_device_capability(x.device)[0] >= 9 and deep_gemm.get_pdl()
    )
    flat = x.view(-1, x.shape[-1])
    rows, width = flat.shape
    if (
        not fp32_scales
        and width == 1280
        and x.is_cuda
        and x.dtype == torch.bfloat16
        and torch.cuda.get_device_capability(x.device) == (10, 0)
    ):
        from rtp_llm.models_py.modules.dsv41._v41_query_quant_cute import (
            try_query_norm_quant,
        )

        result = try_query_norm_quant(x, weight, eps, pdl=pdl)
        if result is not None:
            return result
    normalized = torch.empty(x.shape, device=x.device, dtype=x.dtype)
    quantized = torch.empty(x.shape, device=x.device, dtype=torch.float8_e4m3fn)
    if fp32_scales:
        scales = mn_major_scale_output(rows, width // 32, x.device)
    else:
        scales = torch.empty(
            (triton.cdiv(width // 32, 4), triton.cdiv(rows, 4) * 4),
            device=x.device,
            dtype=torch.int32,
        ).T[:rows]
    if rows:
        _query_norm_quant_kernel[(rows,)](
            flat,
            weight,
            normalized,
            quantized,
            scales,
            flat.stride(0),
            scales.stride(1),
            width,
            eps,
            triton.next_power_of_2(width),
            FP32_SCALES=fp32_scales,
            PDL=pdl,
            launch_pdl=pdl,
            num_warps=8 if width > 1024 else 4,
            enable_fp_fusion=False,
        )
    return normalized, (quantized, scales)
