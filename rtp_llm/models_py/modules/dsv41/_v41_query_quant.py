"""DeepGEMM producer epilogues with V4.1's original quantization boundaries."""

from __future__ import annotations

import os
import threading

import torch
import triton
import triton.language as tl

from rtp_llm.models_py.modules.dsv41._v41_fused_qkv import is_supported
from rtp_llm.models_py.modules.dsv41.utils import V41MXFP8Linear


class _NativePrepared(threading.local):
    def __init__(self):
        self.keys = set()


_NATIVE_PREPARED = _NativePrepared()


def _native_signature(api, operation, x, rows):
    from rtp_llm.models_py.modules.dsv41.fp8._v41_wo_a_quant import _context

    return (
        operation,
        x.device.index,
        _context(),
        rows,
        api.get_jit_config_fingerprint(),
    )


def try_project_indexer_q(linear, qr, freqs, positions, quantized_input=None):
    """Project directly into RoPE/FP4 query storage; leave other linears unchanged."""
    if (
        not isinstance(linear, V41MXFP8Linear)
        or not qr.is_cuda
        or qr.dtype != torch.bfloat16
        or (linear.N, linear.K) != (4096, 1280)
        or freqs.ndim != 2
        or freqs.shape[1] != 32
        or torch.cuda.get_device_capability(qr.device)[0] != 10
    ):
        return None
    import deep_gemm

    if not callable(getattr(deep_gemm, "fp8_indexer_q_rope_fp4", None)):
        return None
    rows = qr.numel() // 1280
    with torch.cuda.device(qr.device):
        capturing = torch.cuda.is_current_stream_capturing()
        key = _native_signature(deep_gemm, "indexer_q", qr, rows)
        if capturing and key not in _NATIVE_PREPARED.keys:
            return None
        pair = (
            linear._quantize_input(qr) if quantized_input is None else quantized_input
        )
        payload = torch.empty((rows, 32, 64), device=qr.device, dtype=torch.int8)
        scales = torch.empty((rows, 32), device=qr.device, dtype=torch.int32)
        deep_gemm.fp8_indexer_q_rope_fp4(
            (pair[0].reshape(rows, 1280), pair[1]),
            (linear.weight, linear.weight_scales),
            freqs,
            positions.reshape(rows).to(torch.int64).contiguous(),
            payload,
            scales,
        )
        if not capturing:
            _NATIVE_PREPARED.keys.add(key)
    return payload, scales


def try_project_qkv_to_cache(attn, x, freqs, cache, slots, quantized_input=None):
    """Use the native producer when its complete Q/SWA contract is supported.

    Other shapes retain the existing model path. Capture accepts only an
    eagerly prepared configuration on the same host thread and CUDA context.
    """
    linear = getattr(attn, "wq_a_wkv", None)
    if (
        os.environ.get("DSV41_FUSED_QUERY_QUANT", "1") != "1"
        or os.environ.get("DSV41_FUSED_QKV_ROPE_CACHE", "1") != "1"
        or os.environ.get("DSV4_FP8_QUANT_KERNEL", "auto").strip().lower()
        not in ("auto", "legacy")
        or not isinstance(attn.wq_b, V41MXFP8Linear)
        or not isinstance(linear, V41MXFP8Linear)
        or not is_supported(linear, x, attn.q_norm, attn.q_lora_rank)
        or (linear.N, linear.K, attn.q_lora_rank) != (1792, 5120, 1280)
        or not 0 < x.numel() // 5120 <= 512
        or not getattr(attn, "skip_post_q_norm", False)
        or attn.rope_head_dim != 64
        or attn.kv_norm.dtype != torch.bfloat16
        or cache is None
        or slots is None
        or cache.dtype != torch.uint8
        or cache.ndim != 3
        or cache.shape[-1] != 528
        or cache.stride(1) != 528
        or cache.stride(2) != 1
        or cache.stride(0) % 8
        or torch.cuda.get_device_capability(x.device)[0] != 10
    ):
        return None
    import deep_gemm

    if not callable(getattr(deep_gemm, "dsv41_qkv", None)):
        return None
    if deep_gemm.get_num_sms() < 14 * triton.cdiv(x.numel() // 5120, 80):
        return None
    from rtp_llm.models_py.modules.dsv41.fp8._trap_utils import (
        trap_invalid_kv_access_enabled,
    )

    rows = x.numel() // 5120
    with torch.cuda.device(x.device):
        capturing = torch.cuda.is_current_stream_capturing()
        key = _native_signature(deep_gemm, "qkv", x, rows)
        if capturing and key not in _NATIVE_PREPARED.keys:
            return None
        pair = linear._quantize_input(x) if quantized_input is None else quantized_input
        pair = (pair[0].reshape(rows, 5120), pair[1])
        qr = torch.empty((rows, 1280), device=x.device, dtype=torch.bfloat16)
        q8 = torch.empty_like(qr, dtype=torch.float8_e4m3fn)
        sf = torch.empty(
            (10, (rows + 3) // 4 * 4), device=x.device, dtype=torch.int32
        ).T[:rows]
        kv = torch.empty((rows, 512), device=x.device, dtype=torch.bfloat16)
        deep_gemm.dsv41_qkv(
            pair,
            (linear.weight, linear.weight_scales),
            attn.q_norm,
            attn.kv_norm,
            freqs,
            cache,
            slots.to(torch.int64).contiguous(),
            qr,
            q8,
            sf,
            kv,
            eps=attn.eps,
            trap_invalid_slots=trap_invalid_kv_access_enabled(),
        )
        if not capturing:
            _NATIVE_PREPARED.keys.add(key)
    return (
        qr.view(*x.shape[:-1], 1280),
        kv.view(*x.shape[:-1], 512),
        (q8.view(*x.shape[:-1], 1280), sf),
    )


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


def try_project_quantized_qkv(attn, x, quantized_input=None):
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
    # Preserve the existing fusion range; large auto inputs use the original
    # norm path and public quantization entry, which also follows legacy rules.
    # Explicit v2 and unknown settings are handled by the original consumer.
    if quant_kernel not in ("auto", "legacy") or (
        quant_kernel == "auto" and elements >= 4 * 1024 * 1024
    ):
        return None
    projected = (
        linear(x)
        if quantized_input is None
        else linear.forward_quantized(*quantized_input).view(*x.shape[:-1], linear.N)
    )
    qr, quantized = query_norm_quant(
        projected[..., : attn.q_lora_rank], attn.q_norm, attn.eps
    )
    return qr, projected[..., attn.q_lora_rank :], quantized
