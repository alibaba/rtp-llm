"""Decode Q normalization with native DeepGEMM group-32 activation scales."""

from __future__ import annotations

import os

import torch
import triton
import triton.language as tl

from rtp_llm.models_py.modules.dsv4._v41_fused_qkv import is_supported
from rtp_llm.models_py.modules.dsv4.utils import V41MXFP8Linear


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
):
    row = tl.program_id(0).to(tl.int64)
    cols = tl.arange(0, BLOCK)
    x = tl.load(X + row * stride_x + cols, cols < D, 0).to(tl.float32)
    w = tl.load(W + cols, cols < D, 0).to(tl.float32)
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
    group_ids = tl.arange(0, BLOCK // 32)
    exponent = tl.where(group_ids < D // 32, exponent, 0)
    packed = tl.sum(
        tl.reshape(exponent, (BLOCK // 128, 4)) << (tl.arange(0, 4)[None, :] * 8),
        1,
    )
    packs = tl.arange(0, BLOCK // 128)
    tl.store(SF + packs * stride_sf + row, packed, packs < triton.cdiv(D, 128))


def query_norm_quant(x: torch.Tensor, weight: torch.Tensor, eps: float):
    """Return BF16 Q plus FP8 Q and packed MN-major UE8M0 scales."""
    if not (
        x.is_cuda
        and x.dtype == torch.bfloat16
        and x.ndim >= 2
        and x.stride(-1) == 1
        and x.shape[-1] % 32 == 0
        and 128 <= x.shape[-1] <= 8192
        and weight.device == x.device
        and weight.dtype == x.dtype
        and weight.shape == (x.shape[-1],)
        and weight.is_contiguous()
    ):
        raise ValueError("Unsupported V4.1 query normalization layout")
    flat = x.view(-1, x.shape[-1])
    rows, width = flat.shape
    normalized = torch.empty(x.shape, device=x.device, dtype=x.dtype)
    quantized = torch.empty(x.shape, device=x.device, dtype=torch.float8_e4m3fn)
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
            num_warps=8 if width > 1024 else 4,
            enable_fp_fusion=False,
        )
    return normalized, (quantized, scales)


def try_project_quantized_qkv(attn, x):
    """Decode-only projection with an explicit, forward-local quantized Q."""
    if os.environ.get("DSV41_FUSED_QUERY_QUANT", "1") != "1":
        return None
    linear = getattr(attn, "wq_a_wkv", None)
    if not isinstance(attn.wq_b, V41MXFP8Linear):
        return None
    if not is_supported(linear, x, attn.q_norm, attn.q_lora_rank):
        return None
    if attn.q_lora_rank % 32 or attn.q_lora_rank < 128:
        return None
    quant_kernel = os.environ.get("DSV4_FP8_QUANT_KERNEL", "auto").strip().lower()
    elements = (x.numel() // x.shape[-1]) * attn.q_lora_rank
    # The v2 producer has different zero/tiny-scale semantics. Let the native
    # consumer select it (and validate unknown settings) on the original path.
    if quant_kernel not in ("auto", "legacy") or (
        quant_kernel == "auto" and elements >= 4 * 1024 * 1024
    ):
        return None
    projected = linear(x)
    qr, quantized = query_norm_quant(
        projected[..., : attn.q_lora_rank], attn.q_norm, attn.eps
    )
    return qr, projected[..., attn.q_lora_rank :], quantized
