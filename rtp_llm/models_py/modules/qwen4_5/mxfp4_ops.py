"""MXFP4 quantization, dequantization, and SM100/SM103 Grouped MoE engines for Qwen 4.5-Flash.

Checkpoint format (OCP MXFP4 / PyTorch ``float4_e2m1fn_x2``):
  - Weight tensor: ``[..., N, K // 2]`` ``uint8``
      Each byte packs two FP4 E2M1 nibbles along K:
        - low nibble  (``byte & 0x0F``)      -> even K index ``2 * j``
        - high nibble (``(byte >> 4) & 0x0F``) -> odd  K index ``2 * j + 1``
      Unsigned magnitude LUT (bits 0..2): ``[0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]``
      Sign bit (bit 3): ``1`` -> negate.
  - Scale tensor: ``[..., N, K // 32]`` ``uint8`` (UE8M0)
      Exponent-only scale: ``2^(scale_u8 - 127)`` in FP32 for ``0..254``; ``255`` is NaN.

Selectable & regime-dispatched backends on SM100 / SM103:
  1. ``triton_splitk``: Optimized Persistent Split-K Triton ``tl.dot_scaled``
     Decode GEMV (zero ``-1`` wasted empty tiles, fused SiTU-GLU epilogue,
     single-pass weighted scatter-reduce for ``M=1..64``).
  2. ``deepgemm_contiguous``: DeepGEMM ``m_grouped_fp8_fp4_gemm_nt_contiguous``
     with pure-Triton block-32 UE8M0 quantization & fused ``situ_glu_fp8_quant_packed``.
  3. ``deepgemm_masked``: DeepGEMM ``m_grouped_fp8_fp4_gemm_nt_masked`` (zero-sync
     CUDA-Graph Decode) with fused ``situ_glu_fp8_quant_masked_packed``.
  4. ``flashinfer`` / ``flashinfer_noswap``: FlashInfer
     ``group_gemm_mxfp4_nt_groupwise`` (``swap_ab=True`` / ``False``) with pure-Triton
     128x4 swizzled UE8M0 quantization.
  5. ``triton``: Baseline grouped ``tl.dot_scaled`` engine.
  6. ``ref``: Exact PyTorch dequant-BF16 reference.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import functools
import os
import threading
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from triton.language.extra import libdevice

from rtp_llm.models_py.modules.qwen4_5.activation import (
    _QUANT_EPS,
    _compute_ue8m0_fp8_block32,
    _cuda_device_ctx,
    per_token_group_quant_fp8_ue8m0,
    per_token_group_quant_fp8_ue8m0_ref,
    situ_and_mul_ref,
    situ_and_mul_triton,
    situ_glu_fp8_quant_masked_packed,
    situ_glu_fp8_quant_packed,
)

MXFP4_BLOCK_SIZE = 32
_DG_CONTIG_ALIGN = 128
_MASKED_WS_MAX_M = 512
_MASKED_WS_CACHE: Dict[Tuple[int, int], torch.Tensor] = {}
_CONTIG_TLS = threading.local()


def _get_contig_sync_state(
    dev: torch.device, need_i32: int
) -> Tuple[torch.Tensor, torch.cuda.Event, torch.Tensor]:
    """Return thread- and stream-local ``(pinned_tok, event, i32_slab)`` for async D2H overlap."""
    cache = getattr(_CONTIG_TLS, "by_stream", None)
    if cache is None:
        cache = {}
        _CONTIG_TLS.by_stream = cache
    dev_idx = int(dev.index if dev.index is not None else torch.cuda.current_device())
    with _cuda_device_ctx(dev):
        stream_id = int(torch.cuda.current_stream(dev).cuda_stream)
        key = (dev_idx, stream_id)
        entry = cache.get(key)
        if entry is None:
            pinned = torch.empty((1,), dtype=torch.int32, pin_memory=True)
            ev = torch.cuda.Event()
            slab = torch.empty((need_i32,), dtype=torch.int32, device=dev)
            entry = [pinned, ev, slab]
            cache[key] = entry
        elif int(entry[2].numel()) < need_i32:
            entry[2] = torch.empty((need_i32,), dtype=torch.int32, device=dev)
    return entry[0], entry[1], entry[2]


def _align256(n: int) -> int:
    return ((n + 255) // 256) * 256


def _get_masked_ws_slab(dev: torch.device, need_bytes: int, *, cacheable: bool) -> torch.Tensor:
    if not cacheable or dev.type != "cuda":
        return torch.empty((need_bytes,), dtype=torch.uint8, device=dev)
    dev_idx = int(dev.index if dev.index is not None else torch.cuda.current_device())
    stream_id = int(torch.cuda.current_stream(dev).cuda_stream)
    key = (dev_idx, stream_id)
    cached = _MASKED_WS_CACHE.get(key)
    if cached is not None and int(cached.numel()) >= need_bytes:
        return cached
    if torch.cuda.is_current_stream_capturing():
        return torch.empty((need_bytes,), dtype=torch.uint8, device=dev)
    alloc_bytes = need_bytes if cached is None else max(need_bytes, int(cached.numel()))
    slab = torch.empty((alloc_bytes,), dtype=torch.uint8, device=dev)
    _MASKED_WS_CACHE[key] = slab
    return slab

_E2M1_LUT_CPU = torch.tensor(
    [
        0.0,
        0.5,
        1.0,
        1.5,
        2.0,
        3.0,
        4.0,
        6.0,
        -0.0,
        -0.5,
        -1.0,
        -1.5,
        -2.0,
        -3.0,
        -4.0,
        -6.0,
    ],
    dtype=torch.float32,
)
_E2M1_LUT_CACHE: dict[torch.device, torch.Tensor] = {}
_SM_COUNT_CACHE: dict[int, int] = {}


def _get_e2m1_lut(device: torch.device) -> torch.Tensor:
    lut = _E2M1_LUT_CACHE.get(device)
    if lut is None:
        lut = _E2M1_LUT_CPU.to(device=device)
        _E2M1_LUT_CACHE[device] = lut
    return lut


def _get_sm_count(device: torch.device) -> int:
    idx = int(device.index or 0)
    cached = _SM_COUNT_CACHE.get(idx)
    if cached is None:
        cached = int(torch.cuda.get_device_properties(device).multi_processor_count)
        _SM_COUNT_CACHE[idx] = cached
    return cached


@functools.lru_cache(maxsize=8)
def _supports_triton_mxfp4_idx(device_index: int) -> bool:
    if not torch.cuda.is_available():
        return False
    major, _ = torch.cuda.get_device_capability(device_index)
    return major >= 9 and hasattr(tl, "dot_scaled")


def supports_triton_mxfp4(device: torch.device) -> bool:
    """Return True if ``tl.dot_scaled`` MXFP4 hardware/Triton execution is supported."""
    if device.type != "cuda":
        return False
    return _supports_triton_mxfp4_idx(int(device.index or 0))


@functools.lru_cache(maxsize=8)
def has_deepgemm_fp8_fp4(device_index: int = 0) -> bool:
    """True iff ``deep_gemm`` ships SM100 FP8xFP4 grouped kernels."""
    if not torch.cuda.is_available():
        return False
    cap = torch.cuda.get_device_capability(device_index)
    if cap[0] != 10:
        return False
    try:
        import deep_gemm
    except Exception:
        return False
    return hasattr(deep_gemm, "m_grouped_fp8_fp4_gemm_nt_contiguous") and hasattr(
        deep_gemm, "m_grouped_fp8_fp4_gemm_nt_masked"
    )


@functools.lru_cache(maxsize=8)
def has_flashinfer_mxfp4(device_index: int = 0) -> bool:
    """True iff ``flashinfer.gemm.group_gemm_mxfp4_nt_groupwise`` is supported."""
    if not torch.cuda.is_available():
        return False
    cap = torch.cuda.get_device_capability(device_index)
    if cap[0] not in (10, 11, 12):
        return False
    try:
        import flashinfer.gemm as fi_gemm
    except Exception:
        return False
    return hasattr(fi_gemm, "group_gemm_mxfp4_nt_groupwise")


# ---------------------------------------------------------------------------
# MXFP4 Quantize & Dequantize (Reference + Triton)
# ---------------------------------------------------------------------------


def quantize_mxfp4(w: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Quantize floating-point tensor ``[..., N, K]`` into MXFP4 ``(w_u8, scale_u8)``."""
    if w.ndim < 2:
        raise ValueError(f"Expected weight tensor with ndim >= 2, got {w.shape}")
    K = w.shape[-1]
    if K % MXFP4_BLOCK_SIZE != 0:
        raise ValueError(
            f"Last dimension K={K} must be divisible by {MXFP4_BLOCK_SIZE}"
        )

    prefix_shape = w.shape[:-1]
    num_groups = K // MXFP4_BLOCK_SIZE

    w_f32 = w.float().reshape(*prefix_shape, num_groups, MXFP4_BLOCK_SIZE)
    amax = w_f32.abs().amax(dim=-1)

    is_zero = amax == 0.0
    safe_amax = torch.where(is_zero, torch.ones_like(amax), amax)
    exp = torch.ceil(torch.log2(safe_amax / 6.0)).clamp(-127.0, 127.0).to(torch.int32)
    exp = torch.where(is_zero, torch.zeros_like(exp), exp)
    scale_u8 = (exp + 127).to(torch.uint8)

    inv_scale = torch.ldexp(torch.ones_like(safe_amax), -exp)
    inv_scale = torch.where(is_zero, torch.zeros_like(inv_scale), inv_scale)
    scaled = w_f32 * inv_scale.unsqueeze(-1)

    abs_v = scaled.abs()
    mag = (
        (abs_v > 0.25).to(torch.uint8)
        + (abs_v >= 0.75).to(torch.uint8)
        + (abs_v > 1.25).to(torch.uint8)
        + (abs_v >= 1.75).to(torch.uint8)
        + (abs_v > 2.5).to(torch.uint8)
        + (abs_v >= 3.5).to(torch.uint8)
        + (abs_v > 5.0).to(torch.uint8)
    )
    sign = (scaled < 0.0).to(torch.uint8) << 3
    nibbles = (sign | mag).reshape(*prefix_shape, K // 2, 2)
    w_u8 = nibbles[..., 0] | (nibbles[..., 1] << 4)
    return w_u8.contiguous(), scale_u8.contiguous()


def dequantize_mxfp4_ref(
    w_u8: torch.Tensor,
    scale_u8: torch.Tensor,
    dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Exact vectorized PyTorch reference MXFP4 dequantizer."""
    if w_u8.dtype != torch.uint8 or scale_u8.dtype != torch.uint8:
        raise TypeError("w_u8 and scale_u8 must be torch.uint8")
    prefix_shape = w_u8.shape[:-1]
    K_half = w_u8.shape[-1]
    K = K_half * 2
    if K % MXFP4_BLOCK_SIZE != 0:
        raise ValueError(f"Unpacked K={K} must be divisible by {MXFP4_BLOCK_SIZE}")
    expected_scale_shape = (*prefix_shape, K // MXFP4_BLOCK_SIZE)
    if tuple(scale_u8.shape) != expected_scale_shape:
        raise ValueError(
            f"Expected scale_u8 shape {expected_scale_shape}, got {tuple(scale_u8.shape)}"
        )

    lut = _get_e2m1_lut(w_u8.device)
    lo = lut[(w_u8 & 0x0F).to(torch.int64)]
    hi = lut[((w_u8 >> 4) & 0x0F).to(torch.int64)]
    vals = torch.stack((lo, hi), dim=-1).reshape(*prefix_shape, K)

    exp_i32 = scale_u8.to(torch.int32) - 127
    scale_f32 = torch.ldexp(
        torch.ones(scale_u8.shape, dtype=torch.float32, device=w_u8.device), exp_i32
    )
    if (scale_u8 == 255).any():
        scale_f32 = torch.where(
            scale_u8 == 255, torch.full_like(scale_f32, float("nan")), scale_f32
        )
    scale_f32 = scale_f32.repeat_interleave(MXFP4_BLOCK_SIZE, dim=-1)
    return (vals * scale_f32).to(dtype)


@triton.jit
def _dequantize_mxfp4_kernel(
    w_ptr,
    sc_ptr,
    out_ptr,
    num_rows,
    K_half,
    stride_w_row: tl.int64,
    stride_sc_row: tl.int64,
    stride_out_row: tl.int64,
    BLOCK_K2: tl.constexpr,
):
    pid_row = tl.program_id(0).to(tl.int64)
    pid_col = tl.program_id(1)
    offs_k2 = pid_col * BLOCK_K2 + tl.arange(0, BLOCK_K2)
    mask_k2 = offs_k2 < K_half
    offs_sc = offs_k2 // 16

    b = tl.load(w_ptr + pid_row * stride_w_row + offs_k2, mask=mask_k2, other=0)
    sc_u8 = tl.load(
        sc_ptr + pid_row * stride_sc_row + offs_sc, mask=mask_k2, other=127
    )

    sc_i32 = sc_u8.to(tl.int32)
    sc_normal = (sc_i32 << 23).to(tl.float32, bitcast=True)
    sc_subnormal = tl.full(sc_i32.shape, 0x00400000, tl.int32).to(
        tl.float32, bitcast=True
    )
    sc_nan = tl.full(sc_i32.shape, 0x7FC00000, tl.int32).to(tl.float32, bitcast=True)
    sc_f32 = tl.where(
        sc_i32 == 0, sc_subnormal, tl.where(sc_i32 == 255, sc_nan, sc_normal)
    )

    lo_n = b & 0x0F
    hi_n = (b >> 4) & 0x0F
    lo_mag = (lo_n & 7).to(tl.int32)
    hi_mag = (hi_n & 7).to(tl.int32)
    lo_sign = tl.where((lo_n & 8) != 0, -1.0, 1.0)
    hi_sign = tl.where((hi_n & 8) != 0, -1.0, 1.0)

    lo_norm = (((lo_mag >> 1) + 126) << 23 | ((lo_mag & 1) << 22)).to(
        tl.float32, bitcast=True
    )
    hi_norm = (((hi_mag >> 1) + 126) << 23 | ((hi_mag & 1) << 22)).to(
        tl.float32, bitcast=True
    )
    lo_val = tl.where(lo_mag < 2, lo_mag.to(tl.float32) * 0.5, lo_norm) * lo_sign * sc_f32
    hi_val = tl.where(hi_mag < 2, hi_mag.to(tl.float32) * 0.5, hi_norm) * hi_sign * sc_f32

    out_base = out_ptr + pid_row * stride_out_row
    out_pair = tl.interleave(lo_val, hi_val).to(out_ptr.dtype.element_ty)
    offs_k = pid_col * (2 * BLOCK_K2) + tl.arange(0, 2 * BLOCK_K2)
    mask_k = offs_k < (2 * K_half)
    tl.store(out_base + offs_k, out_pair, mask=mask_k)


def dequantize_mxfp4(
    w_u8: torch.Tensor,
    scale_u8: torch.Tensor,
    dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Fast Triton MXFP4 dequantizer with automatic CPU fallback to reference."""
    if not w_u8.is_cuda or not scale_u8.is_cuda:
        return dequantize_mxfp4_ref(w_u8, scale_u8, dtype=dtype)
    if w_u8.dtype != torch.uint8 or scale_u8.dtype != torch.uint8:
        raise TypeError("w_u8 and scale_u8 must be torch.uint8")

    prefix_shape = w_u8.shape[:-1]
    K_half = w_u8.shape[-1]
    K = K_half * 2
    if K % MXFP4_BLOCK_SIZE != 0:
        raise ValueError(f"Unpacked K={K} must be divisible by {MXFP4_BLOCK_SIZE}")
    expected_scale_shape = (*prefix_shape, K // MXFP4_BLOCK_SIZE)
    if tuple(scale_u8.shape) != expected_scale_shape:
        raise ValueError(
            f"Expected scale_u8 shape {expected_scale_shape}, got {tuple(scale_u8.shape)}"
        )

    if w_u8.numel() == 0:
        return torch.empty((*prefix_shape, K), dtype=dtype, device=w_u8.device)

    w_2d = w_u8.reshape(-1, K_half)
    sc_2d = scale_u8.reshape(-1, K // MXFP4_BLOCK_SIZE)
    if w_2d.stride(-1) != 1:
        w_2d = w_2d.contiguous()
    if sc_2d.stride(-1) != 1:
        sc_2d = sc_2d.contiguous()

    num_rows = w_2d.shape[0]
    out = torch.empty((num_rows, K), dtype=dtype, device=w_u8.device)
    BLOCK_K2 = 512 if K_half >= 512 else triton.next_power_of_2(max(K_half, 16))
    grid = (num_rows, triton.cdiv(K_half, BLOCK_K2))
    _dequantize_mxfp4_kernel[grid](
        w_2d,
        sc_2d,
        out,
        num_rows,
        K_half,
        w_2d.stride(0),
        sc_2d.stride(0),
        out.stride(0),
        BLOCK_K2=BLOCK_K2,
        num_warps=4,
    )
    return out.reshape(*prefix_shape, K)


# ---------------------------------------------------------------------------
# MXFP8 (E4M3 + UE8M0) Dynamic Activation Quantizer
# ---------------------------------------------------------------------------


@triton.jit
def _quant_mxfp8_e4m3_kernel(
    x_ptr,
    q_ptr,
    sc_ptr,
    num_rows,
    num_groups,
    stride_x_row: tl.int64,
    stride_q_row: tl.int64,
    stride_sc_row: tl.int64,
    GROUPS_PER_CTA: tl.constexpr,
):
    pid_row = tl.program_id(0).to(tl.int64)
    pid_g = tl.program_id(1)
    g_offs = pid_g * GROUPS_PER_CTA + tl.arange(0, GROUPS_PER_CTA)
    g_mask = g_offs < num_groups
    elem_offs = g_offs[:, None] * 32 + tl.arange(0, 32)[None, :]
    x = tl.load(
        x_ptr + pid_row * stride_x_row + elem_offs,
        mask=g_mask[:, None],
        other=0.0,
    ).to(tl.float32)

    amax = tl.max(tl.abs(x), axis=1)
    is_zero = amax == 0.0
    safe_amax = tl.where(is_zero, 1.0, amax)
    ratio = safe_amax * (1.0 / 448.0)
    bits = ratio.to(tl.int32, bitcast=True)
    raw_exp = ((bits >> 23) & 0xFF) - 127
    has_mant = (bits & 0x007FFFFF) != 0
    exp = raw_exp + has_mant.to(tl.int32)
    exp = tl.maximum(tl.minimum(exp, 127), -126)
    exp = tl.where(is_zero, 0, exp)
    sc_u8 = (exp + 127).to(tl.uint8)
    inv_sc = ((127 - exp) << 23).to(tl.float32, bitcast=True)
    inv_sc = tl.where(is_zero, 0.0, inv_sc)

    q = (x * inv_sc[:, None]).to(tl.float8e4nv)
    tl.store(
        q_ptr + pid_row * stride_q_row + elem_offs,
        q,
        mask=g_mask[:, None],
    )
    tl.store(sc_ptr + pid_row * stride_sc_row + g_offs, sc_u8, mask=g_mask)


def quantize_mxfp8_e4m3(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Dynamic per-32-element MXFP8 (E4M3FN + UE8M0) activation quantizer."""
    if x.shape[-1] % MXFP4_BLOCK_SIZE != 0:
        raise ValueError(f"K={x.shape[-1]} must be divisible by 32")
    orig_shape = x.shape[:-1]
    K = x.shape[-1]
    num_groups = K // MXFP4_BLOCK_SIZE
    x_2d = x.reshape(-1, K)
    if x_2d.stride(-1) != 1:
        x_2d = x_2d.contiguous()
    num_rows = x_2d.shape[0]
    q = torch.empty((num_rows, K), dtype=torch.float8_e4m3fn, device=x.device)
    sc = torch.empty((num_rows, num_groups), dtype=torch.uint8, device=x.device)
    if num_rows == 0:
        return q.reshape(*orig_shape, K), sc.reshape(*orig_shape, num_groups)
    GROUPS_PER_CTA = 16
    grid = (num_rows, triton.cdiv(num_groups, GROUPS_PER_CTA))
    _quant_mxfp8_e4m3_kernel[grid](
        x_2d,
        q,
        sc,
        num_rows,
        num_groups,
        x_2d.stride(0),
        q.stride(0),
        sc.stride(0),
        GROUPS_PER_CTA=GROUPS_PER_CTA,
        num_warps=4,
    )
    return q.reshape(*orig_shape, K), sc.reshape(*orig_shape, num_groups)


# ---------------------------------------------------------------------------
# DeepGEMM & FlashInfer Weight Scale Layout Converters (Pure Triton + Fast)
# ---------------------------------------------------------------------------


@triton.jit
def _pack_fp4_scale_deepgemm_kernel(
    in_ptr,
    out_ptr,
    E,
    N,
    num_groups,
    num_packs,
    stride_ie: tl.int64,
    stride_in: tl.int64,
    stride_oe: tl.int64,
    stride_op: tl.int64,
    BLOCK_N: tl.constexpr,
    BLOCK_P: tl.constexpr,
):
    e = tl.program_id(0).to(tl.int64)
    pid_n = tl.program_id(1)
    pid_p = tl.program_id(2)

    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_p = pid_p * BLOCK_P + tl.arange(0, BLOCK_P)
    mask_n = offs_n < N
    mask_p = offs_p < num_packs

    in_base = in_ptr + e * stride_ie + offs_n[:, None].to(tl.int64) * stride_in
    packed = tl.zeros((BLOCK_N, BLOCK_P), dtype=tl.uint32)
    for b_idx in tl.static_range(4):
        g_idx = offs_p * 4 + b_idx
        mask_g = mask_n[:, None] & (g_idx[None, :] < num_groups)
        v = tl.load(in_base + g_idx[None, :], mask=mask_g, other=0).to(tl.uint32)
        packed |= v << (b_idx * 8)

    out_ptrs = (
        out_ptr
        + e * stride_oe
        + offs_p[None, :].to(tl.int64) * stride_op
        + offs_n[:, None]
    )
    tl.store(out_ptrs, packed.to(tl.int32), mask=mask_n[:, None] & mask_p[None, :])


def prepare_fp4_weight_scale_for_deepgemm(
    scale: torch.Tensor,
    mn: int,
    k: int,
    num_groups: Optional[int] = None,
) -> torch.Tensor:
    """Convert raw MXFP4 UE8M0 weight scales to DeepGEMM Section-3 packed ``int32`` layout.

    Accepts:
      - Already-packed ``torch.int32`` (validated & returned zero-copy)
      - Checkpoint ``torch.uint8`` UE8M0 ``[..., mn, ceil_div(k, 32)]``
      - ``torch.float8_e8m0fnu`` (zero-copy ``.view(torch.uint8)``)
      - ``torch.float32`` exact powers of 2 (extracts biased exponent bits)
    """
    if scale.dtype == torch.int32:
        return scale

    if scale.dtype == torch.uint8:
        sc_u8 = scale
    elif hasattr(torch, "float8_e8m0fnu") and scale.dtype == torch.float8_e8m0fnu:
        sc_u8 = scale.view(torch.uint8)
    elif scale.dtype == torch.float32:
        bits = scale.contiguous().view(torch.int32)
        sc_u8 = ((bits >> 23) & 0xFF).to(torch.uint8)
    else:
        raise TypeError(
            f"Expected uint8, float8_e8m0fnu, float32, or int32 scale, got {scale.dtype}"
        )

    num_k_groups = (k + MXFP4_BLOCK_SIZE - 1) // MXFP4_BLOCK_SIZE
    expected_shape = (
        (num_groups, mn, num_k_groups) if num_groups is not None else (mn, num_k_groups)
    )
    if tuple(sc_u8.shape) != expected_shape:
        raise ValueError(f"FP4 scale shape {tuple(sc_u8.shape)} != {expected_shape}")

    num_packs = (num_k_groups + 3) // 4
    mn_align = ((mn + 3) // 4) * 4
    G = 1 if num_groups is None else int(num_groups)
    sc_3d = sc_u8.reshape(G, mn, num_k_groups)
    if sc_3d.stride(-1) != 1:
        sc_3d = sc_3d.contiguous()

    if not sc_3d.is_cuda:
        padded_groups = num_packs * 4
        if padded_groups != num_k_groups:
            pad = torch.zeros(
                (G, mn, padded_groups - num_k_groups),
                dtype=torch.uint8,
                device=sc_3d.device,
            )
            sc_p = torch.cat([sc_3d, pad], dim=-1)
        else:
            sc_p = sc_3d
        sc_i = sc_p.reshape(G, mn, num_packs, 4).to(torch.int64)
        packed = (
            sc_i[..., 0] | (sc_i[..., 1] << 8) | (sc_i[..., 2] << 16) | (sc_i[..., 3] << 24)
        ).to(torch.int32)
        buf = torch.zeros((G, num_packs, mn_align), dtype=torch.int32, device=sc_3d.device)
        buf[:, :, :mn] = packed.transpose(1, 2)
        out = buf.transpose(1, 2)[:, :mn, :]
        return out[0] if num_groups is None else out

    buf = torch.zeros((G, num_packs, mn_align), dtype=torch.int32, device=sc_3d.device)
    BLOCK_N = 64
    BLOCK_P = 32 if num_packs >= 32 else max(4, triton.next_power_of_2(num_packs))
    grid = (G, triton.cdiv(mn, BLOCK_N), triton.cdiv(num_packs, BLOCK_P))
    _pack_fp4_scale_deepgemm_kernel[grid](
        sc_3d,
        buf,
        G,
        mn,
        num_k_groups,
        num_packs,
        sc_3d.stride(0),
        sc_3d.stride(1),
        buf.stride(0),
        buf.stride(1),
        BLOCK_N=BLOCK_N,
        BLOCK_P=BLOCK_P,
        num_warps=4,
    )
    out = buf.transpose(1, 2)[:, :mn, :]
    return out[0] if num_groups is None else out


@triton.jit
def _swizzle_fp4_scale_128x4_kernel(
    in_ptr,
    out_ptr,
    N,
    C,
    n_pad,
    c_pad,
    stride_ie: tl.int64,
    stride_in: tl.int64,
    stride_oe: tl.int64,
    BLOCK_N: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    e = tl.program_id(0).to(tl.int64)
    pid_n = tl.program_id(1)
    pid_c = tl.program_id(2)

    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_c = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
    mask = (offs_n[:, None] < N) & (offs_c[None, :] < C)

    vals = tl.load(
        in_ptr + e * stride_ie + offs_n[:, None].to(tl.int64) * stride_in + offs_c[None, :],
        mask=mask,
        other=0,
    )

    r_blk = offs_n // 128
    r_rem = offs_n % 128
    r4 = r_rem // 32
    r32 = r_rem % 32
    c_blk = offs_c // 4
    c4 = offs_c % 4

    c_blocks = c_pad // 4
    sw_off = (
        ((r_blk[:, None] * c_blocks + c_blk[None, :]) * 32 + r32[:, None]) * 4
        + r4[:, None]
    ) * 4 + c4[None, :]

    out_mask = (offs_n[:, None] < n_pad) & (offs_c[None, :] < c_pad)
    tl.store(
        out_ptr + e * stride_oe + sw_off.to(tl.int64),
        vals,
        mask=out_mask,
    )


def prepare_fp4_weight_scale_for_flashinfer(scale_u8: torch.Tensor) -> torch.Tensor:
    """Swizzle ``[E, N, K // 32] uint8`` scales to CUTLASS 128x4 layout ``[E, N_pad, C_pad] uint8``."""
    if scale_u8.dtype != torch.uint8:
        if hasattr(torch, "float8_e8m0fnu") and scale_u8.dtype == torch.float8_e8m0fnu:
            scale_u8 = scale_u8.view(torch.uint8)
        else:
            raise TypeError(f"Expected uint8 scale, got {scale_u8.dtype}")
    if scale_u8.ndim != 3:
        raise ValueError(f"Expected 3D [E, N, C] scale, got {scale_u8.shape}")
    E, N, C = scale_u8.shape
    n_pad = ((N + 127) // 128) * 128
    c_pad = ((C + 3) // 4) * 4

    sc_c = scale_u8 if scale_u8.stride(-1) == 1 else scale_u8.contiguous()
    if not sc_c.is_cuda:
        buf = torch.zeros((E, n_pad, c_pad), dtype=torch.uint8, device=sc_c.device)
        buf[:, :N, :C] = sc_c
        return (
            buf.reshape(E, n_pad // 128, 4, 32, c_pad // 4, 4)
            .permute(0, 1, 4, 3, 2, 5)
            .reshape(E, n_pad, c_pad)
            .contiguous()
        )

    out = torch.zeros((E, n_pad, c_pad), dtype=torch.uint8, device=sc_c.device)
    BLOCK_N = 64
    BLOCK_C = 32 if c_pad >= 32 else max(4, triton.next_power_of_2(c_pad))
    grid = (E, triton.cdiv(n_pad, BLOCK_N), triton.cdiv(c_pad, BLOCK_C))
    _swizzle_fp4_scale_128x4_kernel[grid](
        sc_c,
        out,
        N,
        C,
        n_pad,
        c_pad,
        sc_c.stride(0),
        sc_c.stride(1),
        out.stride(0),
        BLOCK_N=BLOCK_N,
        BLOCK_C=BLOCK_C,
        num_warps=4,
    )
    return out


# ---------------------------------------------------------------------------
# Single-Matrix MXFP4 Linear (BF16 x E2M1 via tl.dot_scaled)
# ---------------------------------------------------------------------------


@triton.jit
def _mxfp4_linear_bf16_kernel(
    a_ptr,
    b_ptr,
    b_sc_ptr,
    c_ptr,
    M,
    N,
    K,
    stride_am: tl.int64,
    stride_bn,
    stride_bsn,
    stride_cm: tl.int64,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    K_DIVISIBLE: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_m = tl.program_id(1)

    offs_m = pid_m * BM + tl.arange(0, BM)
    offs_n = pid_n * BN + tl.arange(0, BN)
    mask_m = offs_m < M
    mask_n = offs_n < N

    offs_k = tl.arange(0, BK)
    offs_k2 = tl.arange(0, BK // 2)
    offs_ks = tl.arange(0, BK // 32)

    a_ptrs = a_ptr + offs_m[:, None].to(tl.int64) * stride_am + offs_k[None, :]
    b_ptrs = b_ptr + offs_n[:, None].to(tl.int64) * stride_bn + offs_k2[None, :]
    bs_ptrs = b_sc_ptr + offs_n[:, None].to(tl.int64) * stride_bsn + offs_ks[None, :]

    acc = tl.zeros((BM, BN), dtype=tl.float32)
    K_half = K // 2
    K_sc = K // 32
    for k0 in range(0, K, BK):
        if K_DIVISIBLE:
            a = tl.load(a_ptrs, mask=mask_m[:, None], other=0.0)
            b = tl.load(b_ptrs, mask=mask_n[:, None], other=0)
            b_sc = tl.load(bs_ptrs, mask=mask_n[:, None], other=127)
        else:
            mask_k = (k0 + offs_k) < K
            mask_k2 = (k0 // 2 + offs_k2) < K_half
            mask_ks = (k0 // 32 + offs_ks) < K_sc
            a = tl.load(a_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0.0)
            b = tl.load(b_ptrs, mask=mask_n[:, None] & mask_k2[None, :], other=0)
            b_sc = tl.load(bs_ptrs, mask=mask_n[:, None] & mask_ks[None, :], other=127)
        b = tl.trans(b)
        acc = tl.dot_scaled(a, None, "bf16", b, b_sc, "e2m1", acc=acc)
        a_ptrs += BK
        b_ptrs += BK // 2
        bs_ptrs += BK // 32

    c_ptrs = c_ptr + offs_m[:, None].to(tl.int64) * stride_cm + offs_n[None, :]
    tl.store(
        c_ptrs,
        acc.to(c_ptr.dtype.element_ty),
        mask=mask_m[:, None] & mask_n[None, :],
    )


def mxfp4_linear(
    x: torch.Tensor,
    w_u8: torch.Tensor,
    scale_u8: torch.Tensor,
    *,
    use_triton: bool = True,
    out_dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """Single-matrix MXFP4 GEMM: ``x @ dequant(w_u8, scale_u8).T``."""
    target_dtype = out_dtype if out_dtype is not None else x.dtype
    orig_shape = x.shape[:-1]
    K = x.shape[-1]
    N = w_u8.shape[0]
    if w_u8.shape != (N, K // 2) or scale_u8.shape != (N, K // 32):
        raise ValueError(
            f"Weight/scale mismatch for K={K}: w_u8={tuple(w_u8.shape)}, scale_u8={tuple(scale_u8.shape)}"
        )

    if (
        not use_triton
        or not x.is_cuda
        or not supports_triton_mxfp4(x.device)
        or x.dtype != torch.bfloat16
    ):
        w_f = dequantize_mxfp4_ref(w_u8, scale_u8, dtype=torch.float32)
        out = F.linear(x.float(), w_f)
        return out.to(target_dtype)

    x_2d = x.reshape(-1, K)
    if x_2d.stride(-1) != 1:
        x_2d = x_2d.contiguous()
    if w_u8.stride(-1) != 1:
        w_u8 = w_u8.contiguous()
    if scale_u8.stride(-1) != 1:
        scale_u8 = scale_u8.contiguous()

    M = x_2d.shape[0]
    out = torch.empty((M, N), dtype=target_dtype, device=x.device)
    if M == 0:
        return out.reshape(*orig_shape, N)

    BM = 16 if M <= 16 else (32 if M <= 64 else 64)
    BN = 64 if N < 128 else 128
    BK = 64 if (K < 128 or K % 128 != 0) else 128
    grid = (triton.cdiv(N, BN), triton.cdiv(M, BM))
    _mxfp4_linear_bf16_kernel[grid](
        x_2d,
        w_u8,
        scale_u8,
        out,
        M,
        N,
        K,
        x_2d.stride(0),
        w_u8.stride(0),
        scale_u8.stride(0),
        out.stride(0),
        BM=BM,
        BN=BN,
        BK=BK,
        K_DIVISIBLE=(K % BK == 0),
        num_warps=4,
        num_stages=3,
    )
    return out.reshape(*orig_shape, N)


# ---------------------------------------------------------------------------
# Weight Layout Preparation & Lazy Backend Scale Caching
# ---------------------------------------------------------------------------


@dataclass
class MXFP4MoEWeights:
    """Container for packed MXFP4 routed expert weights ready for SM100/SM103 GEMMs."""

    w13_u8: torch.Tensor  # [E, 2*I, L//2] uint8 (or [E, L//2, 2*I] if transposed)
    w13_scale: torch.Tensor  # [E, 2*I, L//32] uint8
    w2_u8: torch.Tensor  # [E, L, I//2] uint8 (or [E, I//2, L] if transposed)
    w2_scale: torch.Tensor  # [E, L, I//32] uint8
    weights_transposed: bool = False
    _dg_w13_s: Optional[torch.Tensor] = field(default=None, repr=False, compare=False)
    _dg_w2_s: Optional[torch.Tensor] = field(default=None, repr=False, compare=False)
    _fi_w13_s: Optional[torch.Tensor] = field(default=None, repr=False, compare=False)
    _fi_w2_s: Optional[torch.Tensor] = field(default=None, repr=False, compare=False)

    @property
    def num_experts(self) -> int:
        return int(self.w13_u8.shape[0])

    @property
    def latent_dim(self) -> int:
        return int(self.w2_scale.shape[1])

    @property
    def intermediate_size(self) -> int:
        return int(self.w13_scale.shape[1] // 2)

    def invalidate_backend_caches(self) -> None:
        self._dg_w13_s = None
        self._dg_w2_s = None
        self._fi_w13_s = None
        self._fi_w2_s = None

    def get_deepgemm_pair(self) -> Tuple[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
        if self.weights_transposed:
            raise ValueError("DeepGEMM FP8xFP4 requires K-major weights (weights_transposed=False)")
        if self._dg_w13_s is None:
            E = self.num_experts
            two_I = int(self.w13_scale.shape[1])
            L = self.latent_dim
            I = two_I // 2
            self._dg_w13_s = prepare_fp4_weight_scale_for_deepgemm(
                self.w13_scale, two_I, L, num_groups=E
            )
            self._dg_w2_s = prepare_fp4_weight_scale_for_deepgemm(
                self.w2_scale, L, I, num_groups=E
            )
        assert self._dg_w2_s is not None
        return (
            (self.w13_u8.view(torch.int8), self._dg_w13_s),
            (self.w2_u8.view(torch.int8), self._dg_w2_s),
        )

    def get_flashinfer_pair(self) -> Tuple[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
        if self.weights_transposed:
            raise ValueError("FlashInfer MXFP4 requires K-major weights (weights_transposed=False)")
        if self._fi_w13_s is None:
            self._fi_w13_s = prepare_fp4_weight_scale_for_flashinfer(self.w13_scale)
            self._fi_w2_s = prepare_fp4_weight_scale_for_flashinfer(self.w2_scale)
        assert self._fi_w2_s is not None
        return (self.w13_u8, self._fi_w13_s), (self.w2_u8, self._fi_w2_s)

    def nbytes(self) -> int:
        return (
            self.w13_u8.numel() * self.w13_u8.element_size()
            + self.w13_scale.numel() * self.w13_scale.element_size()
            + self.w2_u8.numel() * self.w2_u8.element_size()
            + self.w2_scale.numel() * self.w2_scale.element_size()
        )


def prepare_mxfp4_moe_weights(
    *,
    w13_u8: Optional[torch.Tensor] = None,
    w13_scale: Optional[torch.Tensor] = None,
    w_gate_u8: Optional[torch.Tensor] = None,
    w_gate_scale: Optional[torch.Tensor] = None,
    w_up_u8: Optional[torch.Tensor] = None,
    w_up_scale: Optional[torch.Tensor] = None,
    w2_u8: torch.Tensor,
    w2_scale: torch.Tensor,
    transpose_weights: bool = False,
) -> MXFP4MoEWeights:
    """Prepare MXFP4 routed-expert weights from checkpoint tensors."""
    if w13_u8 is None or w13_scale is None:
        if (
            w_gate_u8 is None
            or w_gate_scale is None
            or w_up_u8 is None
            or w_up_scale is None
        ):
            raise ValueError(
                "Provide either (w13_u8, w13_scale) or all of (w_gate_u8, w_gate_scale, w_up_u8, w_up_scale)"
            )
        w13_u8 = torch.cat([w_gate_u8, w_up_u8], dim=1)
        w13_scale = torch.cat([w_gate_scale, w_up_scale], dim=1)

    w13_u8 = w13_u8.contiguous()
    w13_scale = w13_scale.contiguous()
    w2_u8 = w2_u8.contiguous()
    w2_scale = w2_scale.contiguous()

    if transpose_weights:
        w13_u8 = w13_u8.transpose(1, 2).contiguous()
        w2_u8 = w2_u8.transpose(1, 2).contiguous()

    return MXFP4MoEWeights(
        w13_u8=w13_u8,
        w13_scale=w13_scale,
        w2_u8=w2_u8,
        w2_scale=w2_scale,
        weights_transposed=transpose_weights,
    )


# ---------------------------------------------------------------------------
# Zero-Sync GPU MoE Tile Schedule Builder (Standard + Compact Active-Only)
# ---------------------------------------------------------------------------


@dataclass
class MoETileSchedule:
    """Block-aligned per-expert M-tile schedule skipping all 0-token experts."""

    perm: torch.Tensor  # [M_total] int32: sorted slot -> flat (tok*top_k + k)
    sorted_tok_ids: torch.Tensor  # [M_total] int32: sorted slot -> orig token id
    tile_expert_id: torch.Tensor  # [max_num_tiles] int32 (-1 for inactive tail tiles)
    tile_m_start: torch.Tensor  # [max_num_tiles] int32
    tile_m_count: torch.Tensor  # [max_num_tiles] int32
    max_num_tiles: int
    top_k: int
    num_tokens: int
    block_m: int


@dataclass
class CompactMoETileSchedule:
    """Zero-empty-tile active schedule + GPU task count (no CPU-GPU sync)."""

    sorted_tok_ids: torch.Tensor  # [M_total] int32
    sorted_weights: torch.Tensor  # [M_total] float32
    inv_slots: torch.Tensor  # [M, top_k] int32 mapping (tok, k) -> slot (-1 if remote)
    tile_expert_id: torch.Tensor  # [max_num_tiles] int32 (first num_active_tiles valid)
    tile_m_start: torch.Tensor  # [max_num_tiles] int32
    tile_m_count: torch.Tensor  # [max_num_tiles] int32
    num_active_tiles_buf: torch.Tensor  # [1] int32 on device
    num_active_slots_buf: torch.Tensor  # [1] int32 on device
    max_num_tiles: int
    max_slots: int
    top_k: int
    num_tokens: int
    block_m: int


@triton.jit
def _scatter_moe_tile_schedule_kernel(
    counts_ptr,
    m_starts_ptr,
    tile_starts_ptr,
    tile_expert_id_ptr,
    tile_m_start_ptr,
    tile_m_count_ptr,
    BM: tl.constexpr,
):
    e = tl.program_id(0)
    c = tl.load(counts_ptr + e).to(tl.int32)
    if c == 0:
        return
    m0 = tl.load(m_starts_ptr + e).to(tl.int32)
    t0 = tl.load(tile_starts_ptr + e).to(tl.int32)
    nt = (c + BM - 1) // BM
    for i in range(nt):
        t = t0 + i
        rem = c - i * BM
        cnt = tl.minimum(rem, BM)
        tl.store(tile_expert_id_ptr + t, e)
        tl.store(tile_m_start_ptr + t, m0 + i * BM)
        tl.store(tile_m_count_ptr + t, cnt)


@triton.jit
def _scatter_compact_tile_schedule_kernel(
    counts_ptr,
    m_starts_ptr,
    tile_starts_ptr,
    tile_expert_id_ptr,
    tile_m_start_ptr,
    tile_m_count_ptr,
    num_active_tiles_ptr,
    num_active_slots_ptr,
    num_experts,
    BM: tl.constexpr,
):
    e = tl.program_id(0)
    c = tl.load(counts_ptr + e).to(tl.int32)
    m0 = tl.load(m_starts_ptr + e).to(tl.int32)
    t0 = tl.load(tile_starts_ptr + e).to(tl.int32)
    nt = (c + BM - 1) // BM
    if e == num_experts - 1:
        tl.store(num_active_tiles_ptr, t0 + nt)
        tl.store(num_active_slots_ptr, m0 + c)
    if c == 0:
        return
    for i in range(nt):
        t = t0 + i
        rem = c - i * BM
        cnt = tl.minimum(rem, BM)
        tl.store(tile_expert_id_ptr + t, e)
        tl.store(tile_m_start_ptr + t, m0 + i * BM)
        tl.store(tile_m_count_ptr + t, cnt)


@triton.jit
def _build_m1_compact_schedule_kernel(
    topk_ids_ptr,
    topk_weights_ptr,
    sorted_tok_ids_ptr,
    sorted_weights_ptr,
    inv_slots_ptr,
    tile_expert_id_ptr,
    tile_m_start_ptr,
    tile_m_count_ptr,
    num_active_tiles_ptr,
    num_active_slots_ptr,
    ep_start,
    num_experts,
    TOP_K: tl.constexpr,
):
    cnt = 0
    for k in tl.static_range(TOP_K):
        gid = tl.load(topk_ids_ptr + k).to(tl.int32)
        local_e = gid - ep_start
        if local_e >= 0 and local_e < num_experts:
            w = tl.load(topk_weights_ptr + k).to(tl.float32)
            tl.store(tile_expert_id_ptr + cnt, local_e)
            tl.store(tile_m_start_ptr + cnt, cnt)
            tl.store(tile_m_count_ptr + cnt, 1)
            tl.store(sorted_tok_ids_ptr + cnt, 0)
            tl.store(sorted_weights_ptr + cnt, w)
            tl.store(inv_slots_ptr + k, cnt)
            cnt += 1
        else:
            tl.store(inv_slots_ptr + k, -1)
    tl.store(num_active_tiles_ptr, cnt)
    tl.store(num_active_slots_ptr, cnt)


_M1_SCHED_CACHE: dict[tuple[torch.device, int], tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = {}


def _get_m1_static_tensors(
    device: torch.device, top_k: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    key = (device, top_k)
    cached = _M1_SCHED_CACHE.get(key)
    if cached is None:
        perm = torch.arange(top_k, dtype=torch.int32, device=device)
        sorted_tok_ids = torch.zeros((top_k,), dtype=torch.int32, device=device)
        tile_m_start = perm
        tile_m_count = torch.ones((top_k,), dtype=torch.int32, device=device)
        cached = (perm, sorted_tok_ids, tile_m_start, tile_m_count)
        _M1_SCHED_CACHE[key] = cached
    return cached


@triton.jit
def _seg_scan_fwd(k1, c1, k2, c2):
    same = k1 == k2
    return k2, tl.where(same, c1 + c2, c2)


@triton.jit(do_not_specialize=["m_total", "ep_start", "num_experts"])
def _build_small_m_compact_schedule_kernel(
    ids_ptr,
    w_ptr,
    sorted_tok_ids_ptr,
    sorted_weights_ptr,
    inv_slots_ptr,
    tile_e_ptr,
    tile_m0_ptr,
    tile_cnt_ptr,
    n_tiles_buf_ptr,
    n_slots_buf_ptr,
    m_total,
    ep_start,
    num_experts,
    TOP_K: tl.constexpr,
    BM: tl.constexpr,
    M_PAD: tl.constexpr,
):
    """Build complete compact MoE schedule in 1 single CTA without torch.sort."""
    slot = tl.arange(0, M_PAD)
    valid = slot < m_total
    raw_id = tl.load(ids_ptr + slot, mask=valid, other=-1).to(tl.int32)
    le = raw_id - ep_start
    act = valid & (le >= 0) & (le < num_experts)
    eff_e = tl.where(act, le, num_experts)
    key = (eff_e << 11) | slot
    sk = tl.sort(key)
    e_s = sk >> 11
    src = sk & 2047
    act_s = (slot < m_total) & (e_s < num_experts)
    num_act = tl.sum(act_s.to(tl.int32), axis=0)

    seg_key = tl.where(act_s, e_s, -1 - slot)
    ones = act_s.to(tl.int32)
    _, inc_pos = tl.associative_scan((seg_key, ones), axis=0, combine_fn=_seg_scan_fwd)
    rank_in_e = inc_pos - 1
    _, rem_in_e = tl.associative_scan(
        (seg_key, ones), axis=0, combine_fn=_seg_scan_fwd, reverse=True
    )

    valid_s = slot < m_total
    w_v = tl.load(w_ptr + src, mask=valid_s, other=0.0).to(tl.float32)
    tl.store(sorted_tok_ids_ptr + slot, src // TOP_K, mask=valid_s)
    tl.store(sorted_weights_ptr + slot, w_v, mask=valid_s)
    tl.store(inv_slots_ptr + src, tl.where(act_s, slot, -1), mask=(src < m_total))

    is_head = act_s & ((rank_in_e % BM) == 0)
    t_idx = tl.cumsum(is_head.to(tl.int32), axis=0) - 1
    t_cnt = tl.minimum(BM, rem_in_e)
    tl.store(tile_e_ptr + t_idx, e_s, mask=is_head)
    tl.store(tile_m0_ptr + t_idx, slot, mask=is_head)
    tl.store(tile_cnt_ptr + t_idx, t_cnt, mask=is_head)

    n_tiles = tl.sum(is_head.to(tl.int32), axis=0)
    tl.store(n_tiles_buf_ptr, n_tiles)
    tl.store(n_slots_buf_ptr, num_act)


@triton.jit(do_not_specialize=["m_total", "ep_start"])
def _ep_local_and_bincount_kernel(
    ids_ptr,
    flat_local_ptr,
    counts_ptr,
    m_total,
    ep_start,
    num_experts,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < m_total
    gid = tl.load(ids_ptr + offs, mask=mask, other=0).to(tl.int32)
    local_e = gid - ep_start
    valid = (local_e >= 0) & (local_e < num_experts)
    mapped = tl.where(valid, local_e, num_experts)
    tl.store(flat_local_ptr + offs, mapped, mask=mask)
    tl.atomic_add(counts_ptr + mapped, 1, mask=mask & valid)


@triton.jit(do_not_specialize=["m_total", "ep_start"])
def _compact_perm_unpack_kernel(
    perm_ptr,
    flat_ids_ptr,
    flat_weights_ptr,
    sorted_tok_ids_ptr,
    sorted_weights_ptr,
    inv_slots_ptr,
    m_total,
    ep_start,
    num_experts,
    TOP_K: tl.constexpr,
    HAS_EP: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < m_total
    p = tl.load(perm_ptr + offs, mask=mask, other=0).to(tl.int64)
    tok_id = (p // TOP_K).to(tl.int32)
    w = tl.load(flat_weights_ptr + p, mask=mask, other=0.0).to(tl.float32)
    tl.store(sorted_tok_ids_ptr + offs, tok_id, mask=mask)
    tl.store(sorted_weights_ptr + offs, w, mask=mask)
    if HAS_EP:
        gid = tl.load(flat_ids_ptr + p, mask=mask, other=0).to(tl.int32)
        local_e = gid - ep_start
        valid = (local_e >= 0) & (local_e < num_experts)
        slot_val = tl.where(valid, offs.to(tl.int32), -1)
    else:
        slot_val = offs.to(tl.int32)
    tl.store(inv_slots_ptr + p, slot_val, mask=mask)


@triton.jit
def _compact_prefix_scan_kernel(
    counts_ptr,
    m_starts_ptr,
    tile_starts_ptr,
    num_experts,
    BM: tl.constexpr,
    E_PAD: tl.constexpr,
):
    offs = tl.arange(0, E_PAD)
    mask = offs < num_experts
    c = tl.load(counts_ptr + offs, mask=mask, other=0).to(tl.int32)
    nt = tl.where(mask, (c + BM - 1) // BM, 0)
    m_starts = tl.cumsum(c, axis=0) - c
    tile_starts = tl.cumsum(nt, axis=0) - nt
    tl.store(m_starts_ptr + offs, m_starts, mask=mask)
    tl.store(tile_starts_ptr + offs, tile_starts, mask=mask)


@triton.jit
def _moe_expert_bincount_kernel(
    ids_ptr,
    counts_ptr,
    m_total,
    num_experts,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < m_total
    e = tl.load(ids_ptr + offs, mask=mask, other=-1)
    valid = mask & (e >= 0) & (e < num_experts)
    tl.atomic_add(counts_ptr + e, 1, mask=valid)


def _gpu_expert_bincount(flat_local: torch.Tensor, num_experts: int) -> torch.Tensor:
    """CUDA-Graph-safe, zero-sync expert histogram (replaces torch.bincount)."""
    if not flat_local.is_cuda:
        mask = (flat_local >= 0) & (flat_local < num_experts)
        return torch.bincount(flat_local[mask].to(torch.int64), minlength=num_experts)
    counts = torch.zeros((num_experts,), dtype=torch.int32, device=flat_local.device)
    m_total = flat_local.numel()
    if m_total == 0:
        return counts
    flat_c = flat_local.to(torch.int32).contiguous()
    BLOCK = 256
    with _cuda_device_ctx(flat_local.device):
        _moe_expert_bincount_kernel[(triton.cdiv(m_total, BLOCK),)](
            flat_c,
            counts,
            m_total,
            num_experts,
            BLOCK=BLOCK,
            num_warps=4,
        )
    return counts


def build_moe_tile_schedule(
    topk_ids: torch.Tensor,
    num_experts: int,
    block_m: int,
    *,
    unique_per_token: bool = False,
    ep_start: int = 0,
    ep_size: int = 1,
) -> MoETileSchedule:
    """Sort token-expert pairs and build a block-aligned M-tile table on GPU."""
    if topk_ids.ndim != 2:
        raise ValueError(f"Expected topk_ids of shape [M, top_k], got {topk_ids.shape}")
    num_tokens, top_k = topk_ids.shape
    m_total = num_tokens * top_k
    device = topk_ids.device

    if num_tokens == 1 and unique_per_token and top_k > 0:
        perm, sorted_tok_ids, tile_m_start, tile_m_count = _get_m1_static_tensors(
            device, top_k
        )
        if ep_size > 1:
            local_ids = topk_ids[0].to(torch.int32) - int(ep_start)
            valid = (local_ids >= 0) & (local_ids < num_experts)
            tile_expert_id = torch.where(
                valid, local_ids, torch.full_like(local_ids, -1)
            ).contiguous()
        else:
            tile_expert_id = topk_ids[0].to(torch.int32).contiguous()
        return MoETileSchedule(
            perm=perm,
            sorted_tok_ids=sorted_tok_ids,
            tile_expert_id=tile_expert_id,
            tile_m_start=tile_m_start,
            tile_m_count=tile_m_count,
            max_num_tiles=top_k,
            top_k=top_k,
            num_tokens=1,
            block_m=block_m,
        )

    if m_total == 0:
        empty_i32 = torch.empty((0,), dtype=torch.int32, device=device)
        return MoETileSchedule(
            perm=empty_i32,
            sorted_tok_ids=empty_i32,
            tile_expert_id=empty_i32,
            tile_m_start=empty_i32,
            tile_m_count=empty_i32,
            max_num_tiles=0,
            top_k=top_k,
            num_tokens=num_tokens,
            block_m=block_m,
        )

    flat_ids = topk_ids.reshape(-1)
    if ep_size > 1:
        ep_end = int(ep_start) + num_experts
        in_range = (flat_ids >= ep_start) & (flat_ids < ep_end)
        flat_local = torch.where(
            in_range,
            flat_ids - int(ep_start),
            torch.full_like(flat_ids, num_experts),
        )
        _, perm_i64 = torch.sort(flat_local)
        perm = perm_i64.to(torch.int32)
        sorted_tok_ids = (perm_i64 // top_k).to(torch.int32)
        counts = _gpu_expert_bincount(flat_local, num_experts)
    else:
        _, perm_i64 = torch.sort(flat_ids)
        perm = perm_i64.to(torch.int32)
        sorted_tok_ids = (perm_i64 // top_k).to(torch.int32)
        counts = _gpu_expert_bincount(flat_ids, num_experts)
    m_starts = torch.cumsum(counts, dim=0) - counts
    num_tiles_per_e = (counts + block_m - 1) // block_m
    tile_starts = torch.cumsum(num_tiles_per_e, dim=0) - num_tiles_per_e

    a_max = min(num_experts, m_total)
    max_num_tiles = a_max + (m_total - a_max) // block_m

    tile_expert_id = torch.full(
        (max_num_tiles,), -1, dtype=torch.int32, device=device
    )
    tile_m_start = torch.empty((max_num_tiles,), dtype=torch.int32, device=device)
    tile_m_count = torch.empty((max_num_tiles,), dtype=torch.int32, device=device)

    if topk_ids.is_cuda:
        with _cuda_device_ctx(device):
            _scatter_moe_tile_schedule_kernel[(num_experts,)](
                counts,
                m_starts,
                tile_starts,
                tile_expert_id,
                tile_m_start,
                tile_m_count,
                BM=block_m,
                num_warps=1,
            )
    else:
        for e in range(num_experts):
            c = int(counts[e].item())
            m0 = int(m_starts[e].item())
            t0 = int(tile_starts[e].item())
            nt = (c + block_m - 1) // block_m
            for i in range(nt):
                tile_expert_id[t0 + i] = e
                tile_m_start[t0 + i] = m0 + i * block_m
                tile_m_count[t0 + i] = min(block_m, c - i * block_m)

    return MoETileSchedule(
        perm=perm,
        sorted_tok_ids=sorted_tok_ids,
        tile_expert_id=tile_expert_id,
        tile_m_start=tile_m_start,
        tile_m_count=tile_m_count,
        max_num_tiles=max_num_tiles,
        top_k=top_k,
        num_tokens=num_tokens,
        block_m=block_m,
    )


def _build_compact_decode_schedule(
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    num_experts: int,
    block_m: int,
    *,
    ep_start: int = 0,
    ep_size: int = 1,
) -> CompactMoETileSchedule:
    """Build zero-empty-tile compact schedule with zero CPU-GPU sync."""
    num_tokens, top_k = topk_ids.shape
    m_total = num_tokens * top_k
    device = topk_ids.device

    if m_total == 0:
        empty_i32 = torch.empty((0,), dtype=torch.int32, device=device)
        empty_f32 = torch.empty((0,), dtype=torch.float32, device=device)
        return CompactMoETileSchedule(
            sorted_tok_ids=empty_i32,
            sorted_weights=empty_f32,
            inv_slots=torch.empty((num_tokens, top_k), dtype=torch.int32, device=device),
            tile_expert_id=empty_i32,
            tile_m_start=empty_i32,
            tile_m_count=empty_i32,
            num_active_tiles_buf=torch.zeros((1,), dtype=torch.int32, device=device),
            num_active_slots_buf=torch.zeros((1,), dtype=torch.int32, device=device),
            max_num_tiles=0,
            max_slots=0,
            top_k=top_k,
            num_tokens=num_tokens,
            block_m=block_m,
        )

    if topk_ids.is_cuda and num_tokens == 1 and top_k <= 32:
        sorted_tok_ids = torch.empty((top_k,), dtype=torch.int32, device=device)
        sorted_weights = torch.empty((top_k,), dtype=torch.float32, device=device)
        inv_slots = torch.empty((1, top_k), dtype=torch.int32, device=device)
        tile_expert_id = torch.empty((top_k,), dtype=torch.int32, device=device)
        tile_m_start = torch.empty((top_k,), dtype=torch.int32, device=device)
        tile_m_count = torch.empty((top_k,), dtype=torch.int32, device=device)
        num_active_tiles_buf = torch.empty((1,), dtype=torch.int32, device=device)
        num_active_slots_buf = torch.empty((1,), dtype=torch.int32, device=device)
        ids_0 = topk_ids[0] if topk_ids[0].is_contiguous() else topk_ids[0].contiguous()
        w_0 = topk_weights[0] if topk_weights[0].is_contiguous() else topk_weights[0].contiguous()
        with _cuda_device_ctx(device):
            _build_m1_compact_schedule_kernel[(1,)](
                ids_0,
                w_0,
                sorted_tok_ids,
                sorted_weights,
                inv_slots.view(-1),
                tile_expert_id,
                tile_m_start,
                tile_m_count,
                num_active_tiles_buf,
                num_active_slots_buf,
                int(ep_start) if ep_size > 1 else 0,
                num_experts,
                TOP_K=top_k,
                num_warps=1,
            )
        return CompactMoETileSchedule(
            sorted_tok_ids=sorted_tok_ids,
            sorted_weights=sorted_weights,
            inv_slots=inv_slots,
            tile_expert_id=tile_expert_id,
            tile_m_start=tile_m_start,
            tile_m_count=tile_m_count,
            num_active_tiles_buf=num_active_tiles_buf,
            num_active_slots_buf=num_active_slots_buf,
            max_num_tiles=top_k,
            max_slots=top_k,
            top_k=top_k,
            num_tokens=1,
            block_m=block_m,
        )

    flat_ids = topk_ids.reshape(-1)
    if not flat_ids.is_contiguous():
        flat_ids = flat_ids.contiguous()
    flat_weights = topk_weights.reshape(-1)
    if not flat_weights.is_contiguous():
        flat_weights = flat_weights.contiguous()
    a_max = min(num_experts, m_total)
    max_num_tiles = max(1, a_max + (m_total - a_max) // block_m)

    if flat_ids.is_cuda and m_total <= 1024 and num_experts <= 2048:
        sorted_tok_ids = torch.empty((m_total,), dtype=torch.int32, device=device)
        sorted_weights = torch.empty((m_total,), dtype=torch.float32, device=device)
        inv_slots = torch.empty((num_tokens, top_k), dtype=torch.int32, device=device)
        tile_expert_id = torch.empty((max_num_tiles,), dtype=torch.int32, device=device)
        tile_m_start = torch.empty((max_num_tiles,), dtype=torch.int32, device=device)
        tile_m_count = torch.empty((max_num_tiles,), dtype=torch.int32, device=device)
        num_active_tiles_buf = torch.empty((1,), dtype=torch.int32, device=device)
        num_active_slots_buf = torch.empty((1,), dtype=torch.int32, device=device)
        m_pad = max(16, triton.next_power_of_2(m_total))
        with _cuda_device_ctx(device):
            _build_small_m_compact_schedule_kernel[(1,)](
                flat_ids,
                flat_weights,
                sorted_tok_ids,
                sorted_weights,
                inv_slots.view(-1),
                tile_expert_id,
                tile_m_start,
                tile_m_count,
                num_active_tiles_buf,
                num_active_slots_buf,
                m_total,
                int(ep_start) if ep_size > 1 else 0,
                num_experts,
                TOP_K=top_k,
                BM=block_m,
                M_PAD=m_pad,
                num_warps=4,
            )
        return CompactMoETileSchedule(
            sorted_tok_ids=sorted_tok_ids,
            sorted_weights=sorted_weights,
            inv_slots=inv_slots,
            tile_expert_id=tile_expert_id,
            tile_m_start=tile_m_start,
            tile_m_count=tile_m_count,
            num_active_tiles_buf=num_active_tiles_buf,
            num_active_slots_buf=num_active_slots_buf,
            max_num_tiles=max_num_tiles,
            max_slots=m_total,
            top_k=top_k,
            num_tokens=num_tokens,
            block_m=block_m,
        )

    if flat_ids.is_cuda:
        with _cuda_device_ctx(device):
            BLOCK = 256
            grid_m = (triton.cdiv(m_total, BLOCK),)
            if ep_size > 1:
                flat_local = torch.empty((m_total,), dtype=torch.int32, device=device)
                counts = torch.zeros((num_experts,), dtype=torch.int32, device=device)
                _ep_local_and_bincount_kernel[grid_m](
                    flat_ids,
                    flat_local,
                    counts,
                    m_total,
                    int(ep_start),
                    num_experts,
                    BLOCK=BLOCK,
                    num_warps=4,
                )
                _, perm_i64 = torch.sort(flat_local, stable=True)
            else:
                counts = _gpu_expert_bincount(flat_ids, num_experts)
                _, perm_i64 = torch.sort(flat_ids, stable=True)

            sorted_tok_ids = torch.empty((m_total,), dtype=torch.int32, device=device)
            sorted_weights = torch.empty((m_total,), dtype=torch.float32, device=device)
            inv_slots_flat = torch.empty((m_total,), dtype=torch.int32, device=device)
            _compact_perm_unpack_kernel[grid_m](
                perm_i64,
                flat_ids,
                flat_weights,
                sorted_tok_ids,
                sorted_weights,
                inv_slots_flat,
                m_total,
                int(ep_start) if ep_size > 1 else 0,
                num_experts,
                TOP_K=top_k,
                HAS_EP=(ep_size > 1),
                BLOCK=BLOCK,
                num_warps=4,
            )
            inv_slots = inv_slots_flat.reshape(num_tokens, top_k)

            if num_experts <= 2048:
                m_starts = torch.empty((num_experts,), dtype=torch.int32, device=device)
                tile_starts = torch.empty((num_experts,), dtype=torch.int32, device=device)
                e_pad = max(16, triton.next_power_of_2(num_experts))
                _compact_prefix_scan_kernel[(1,)](
                    counts,
                    m_starts,
                    tile_starts,
                    num_experts,
                    BM=block_m,
                    E_PAD=e_pad,
                    num_warps=4,
                )
            else:
                m_starts = torch.cumsum(counts, dim=0) - counts
                num_tiles_per_e = (counts + block_m - 1) // block_m
                tile_starts = torch.cumsum(num_tiles_per_e, dim=0) - num_tiles_per_e

            tile_expert_id = torch.empty((max_num_tiles,), dtype=torch.int32, device=device)
            tile_m_start = torch.empty((max_num_tiles,), dtype=torch.int32, device=device)
            tile_m_count = torch.empty((max_num_tiles,), dtype=torch.int32, device=device)
            num_active_tiles_buf = torch.empty((1,), dtype=torch.int32, device=device)
            num_active_slots_buf = torch.empty((1,), dtype=torch.int32, device=device)
            _scatter_compact_tile_schedule_kernel[(num_experts,)](
                counts,
                m_starts,
                tile_starts,
                tile_expert_id,
                tile_m_start,
                tile_m_count,
                num_active_tiles_buf,
                num_active_slots_buf,
                num_experts,
                BM=block_m,
                num_warps=1,
            )
    else:
        flat_weights_f32 = flat_weights.float()
        slot_idx = torch.arange(m_total, dtype=torch.int32, device=device)
        if ep_size > 1:
            ep_end = int(ep_start) + num_experts
            in_range = (flat_ids >= ep_start) & (flat_ids < ep_end)
            flat_local = torch.where(
                in_range,
                flat_ids - int(ep_start),
                torch.full_like(flat_ids, num_experts),
            )
            _, perm_i64 = torch.sort(flat_local, stable=True)
            sorted_tok_ids = (perm_i64 // top_k).to(torch.int32)
            sorted_weights = flat_weights_f32.index_select(0, perm_i64)
            slot_vals = torch.where(
                in_range.index_select(0, perm_i64),
                slot_idx,
                torch.full_like(slot_idx, -1),
            )
            counts = _gpu_expert_bincount(flat_local, num_experts).to(torch.int32)
        else:
            _, perm_i64 = torch.sort(flat_ids, stable=True)
            sorted_tok_ids = (perm_i64 // top_k).to(torch.int32)
            sorted_weights = flat_weights_f32.index_select(0, perm_i64)
            slot_vals = slot_idx
            counts = _gpu_expert_bincount(flat_ids, num_experts).to(torch.int32)
        inv_slots_flat = torch.empty((m_total,), dtype=torch.int32, device=device)
        inv_slots_flat.index_copy_(0, perm_i64, slot_vals)
        inv_slots = inv_slots_flat.reshape(num_tokens, top_k)

        m_starts = torch.cumsum(counts, dim=0) - counts
        num_tiles_per_e = (counts + block_m - 1) // block_m
        tile_starts = torch.cumsum(num_tiles_per_e, dim=0) - num_tiles_per_e
        tile_expert_id = torch.empty((max_num_tiles,), dtype=torch.int32, device=device)
        tile_m_start = torch.empty((max_num_tiles,), dtype=torch.int32, device=device)
        tile_m_count = torch.empty((max_num_tiles,), dtype=torch.int32, device=device)
        num_active_tiles = 0
        for e in range(num_experts):
            c = int(counts[e].item())
            m0 = int(m_starts[e].item())
            nt = (c + block_m - 1) // block_m
            for i in range(nt):
                tile_expert_id[num_active_tiles] = e
                tile_m_start[num_active_tiles] = m0 + i * block_m
                tile_m_count[num_active_tiles] = min(block_m, c - i * block_m)
                num_active_tiles += 1
        num_active_tiles_buf = torch.tensor([num_active_tiles], dtype=torch.int32, device=device)
        num_active_slots_buf = torch.tensor(
            [int((m_starts[-1] + counts[-1]).item())], dtype=torch.int32, device=device
        )

    return CompactMoETileSchedule(
        sorted_tok_ids=sorted_tok_ids,
        sorted_weights=sorted_weights,
        inv_slots=inv_slots,
        tile_expert_id=tile_expert_id,
        tile_m_start=tile_m_start,
        tile_m_count=tile_m_count,
        num_active_tiles_buf=num_active_tiles_buf,
        num_active_slots_buf=num_active_slots_buf,
        max_num_tiles=max_num_tiles,
        max_slots=m_total,
        top_k=top_k,
        num_tokens=num_tokens,
        block_m=block_m,
    )


# ---------------------------------------------------------------------------
# Baseline Triton tl.dot_scaled Grouped MXFP4 GEMM Kernels
# ---------------------------------------------------------------------------


@triton.jit
def _mxfp4_grouped_gemm_bf16_kernel(
    a_ptr,
    b_ptr,
    b_sc_ptr,
    c_ptr,
    sorted_tok_ids_ptr,
    perm_ptr,
    tile_expert_id_ptr,
    tile_m_start_ptr,
    tile_m_count_ptr,
    N,
    K,
    stride_am,
    stride_be: tl.int64,
    stride_b0,
    stride_bse: tl.int64,
    stride_bsn,
    stride_cm,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    TRANS_B: tl.constexpr,
    GATHER_INPUT: tl.constexpr,
    SCATTER_OUTPUT: tl.constexpr,
    K_DIVISIBLE: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_tile = tl.program_id(1)

    expert_id = tl.load(tile_expert_id_ptr + pid_tile)
    if expert_id < 0:
        return

    m_start = tl.load(tile_m_start_ptr + pid_tile)
    m_count = tl.load(tile_m_count_ptr + pid_tile)

    offs_local = tl.arange(0, BM)
    mask_m = offs_local < m_count
    slot_ids = m_start + offs_local

    if GATHER_INPUT:
        row_ids_a = tl.load(sorted_tok_ids_ptr + slot_ids, mask=mask_m, other=0)
    else:
        row_ids_a = slot_ids

    offs_n = pid_n * BN + tl.arange(0, BN)
    mask_n = offs_n < N
    offs_k = tl.arange(0, BK)
    offs_k2 = tl.arange(0, BK // 2)
    offs_ks = tl.arange(0, BK // 32)

    expert_id_64 = expert_id.to(tl.int64)
    b_base = b_ptr + expert_id_64 * stride_be
    bs_base = b_sc_ptr + expert_id_64 * stride_bse

    a_ptrs = a_ptr + row_ids_a[:, None].to(tl.int64) * stride_am + offs_k[None, :]
    if TRANS_B:
        b_ptrs = b_base + offs_n[:, None] * stride_b0 + offs_k2[None, :]
        b_step = BK // 2
    else:
        b_ptrs = b_base + offs_k2[:, None] * stride_b0 + offs_n[None, :]
        b_step = (BK // 2) * stride_b0
    bs_ptrs = bs_base + offs_n[:, None] * stride_bsn + offs_ks[None, :]

    acc = tl.zeros((BM, BN), dtype=tl.float32)
    K_half = K // 2
    K_sc = K // 32
    for k0 in range(0, K, BK):
        if K_DIVISIBLE:
            a = tl.load(a_ptrs, mask=mask_m[:, None], other=0.0)
            if TRANS_B:
                b = tl.load(
                    b_ptrs,
                    mask=mask_n[:, None],
                    other=0,
                    eviction_policy="evict_first",
                )
                b = tl.trans(b)
            else:
                b = tl.load(
                    b_ptrs,
                    mask=mask_n[None, :],
                    other=0,
                    eviction_policy="evict_first",
                )
            b_sc = tl.load(
                bs_ptrs,
                mask=mask_n[:, None],
                other=127,
                eviction_policy="evict_first",
            )
        else:
            mask_k = (k0 + offs_k) < K
            mask_k2 = (k0 // 2 + offs_k2) < K_half
            mask_ks = (k0 // 32 + offs_ks) < K_sc
            a = tl.load(a_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0.0)
            if TRANS_B:
                b = tl.load(
                    b_ptrs,
                    mask=mask_n[:, None] & mask_k2[None, :],
                    other=0,
                    eviction_policy="evict_first",
                )
                b = tl.trans(b)
            else:
                b = tl.load(
                    b_ptrs,
                    mask=mask_k2[:, None] & mask_n[None, :],
                    other=0,
                    eviction_policy="evict_first",
                )
            b_sc = tl.load(
                bs_ptrs,
                mask=mask_n[:, None] & mask_ks[None, :],
                other=127,
                eviction_policy="evict_first",
            )
        acc = tl.dot_scaled(a, None, "bf16", b, b_sc, "e2m1", acc=acc)
        a_ptrs += BK
        b_ptrs += b_step
        bs_ptrs += BK // 32

    if SCATTER_OUTPUT:
        row_ids_c = tl.load(perm_ptr + slot_ids, mask=mask_m, other=0)
    else:
        row_ids_c = slot_ids

    c_ptrs = c_ptr + row_ids_c[:, None].to(tl.int64) * stride_cm + offs_n[None, :]
    tl.store(
        c_ptrs,
        acc.to(c_ptr.dtype.element_ty),
        mask=mask_m[:, None] & mask_n[None, :],
    )


@triton.jit
def _mxfp4_grouped_gemm_e4m3_kernel(
    a_ptr,
    a_sc_ptr,
    b_ptr,
    b_sc_ptr,
    c_ptr,
    sorted_tok_ids_ptr,
    perm_ptr,
    tile_expert_id_ptr,
    tile_m_start_ptr,
    tile_m_count_ptr,
    N,
    K,
    stride_am,
    stride_asm,
    stride_be: tl.int64,
    stride_b0,
    stride_bse: tl.int64,
    stride_bsn,
    stride_cm,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    TRANS_B: tl.constexpr,
    GATHER_INPUT: tl.constexpr,
    SCATTER_OUTPUT: tl.constexpr,
    K_DIVISIBLE: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_tile = tl.program_id(1)

    expert_id = tl.load(tile_expert_id_ptr + pid_tile)
    if expert_id < 0:
        return

    m_start = tl.load(tile_m_start_ptr + pid_tile)
    m_count = tl.load(tile_m_count_ptr + pid_tile)

    offs_local = tl.arange(0, BM)
    mask_m = offs_local < m_count
    slot_ids = m_start + offs_local

    if GATHER_INPUT:
        row_ids_a = tl.load(sorted_tok_ids_ptr + slot_ids, mask=mask_m, other=0)
    else:
        row_ids_a = slot_ids

    offs_n = pid_n * BN + tl.arange(0, BN)
    mask_n = offs_n < N
    offs_k = tl.arange(0, BK)
    offs_k2 = tl.arange(0, BK // 2)
    offs_ks = tl.arange(0, BK // 32)

    expert_id_64 = expert_id.to(tl.int64)
    b_base = b_ptr + expert_id_64 * stride_be
    bs_base = b_sc_ptr + expert_id_64 * stride_bse

    a_ptrs = a_ptr + row_ids_a[:, None].to(tl.int64) * stride_am + offs_k[None, :]
    as_ptrs = a_sc_ptr + row_ids_a[:, None].to(tl.int64) * stride_asm + offs_ks[None, :]
    if TRANS_B:
        b_ptrs = b_base + offs_n[:, None] * stride_b0 + offs_k2[None, :]
        b_step = BK // 2
    else:
        b_ptrs = b_base + offs_k2[:, None] * stride_b0 + offs_n[None, :]
        b_step = (BK // 2) * stride_b0
    bs_ptrs = bs_base + offs_n[:, None] * stride_bsn + offs_ks[None, :]

    acc = tl.zeros((BM, BN), dtype=tl.float32)
    K_half = K // 2
    K_sc = K // 32
    for k0 in range(0, K, BK):
        if K_DIVISIBLE:
            a = tl.load(a_ptrs, mask=mask_m[:, None], other=0.0)
            a_sc = tl.load(as_ptrs, mask=mask_m[:, None], other=127)
            if TRANS_B:
                b = tl.load(
                    b_ptrs,
                    mask=mask_n[:, None],
                    other=0,
                    eviction_policy="evict_first",
                )
                b = tl.trans(b)
            else:
                b = tl.load(
                    b_ptrs,
                    mask=mask_n[None, :],
                    other=0,
                    eviction_policy="evict_first",
                )
            b_sc = tl.load(
                bs_ptrs,
                mask=mask_n[:, None],
                other=127,
                eviction_policy="evict_first",
            )
        else:
            mask_k = (k0 + offs_k) < K
            mask_k2 = (k0 // 2 + offs_k2) < K_half
            mask_ks = (k0 // 32 + offs_ks) < K_sc
            a = tl.load(a_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0.0)
            a_sc = tl.load(as_ptrs, mask=mask_m[:, None] & mask_ks[None, :], other=127)
            if TRANS_B:
                b = tl.load(
                    b_ptrs,
                    mask=mask_n[:, None] & mask_k2[None, :],
                    other=0,
                    eviction_policy="evict_first",
                )
                b = tl.trans(b)
            else:
                b = tl.load(
                    b_ptrs,
                    mask=mask_k2[:, None] & mask_n[None, :],
                    other=0,
                    eviction_policy="evict_first",
                )
            b_sc = tl.load(
                bs_ptrs,
                mask=mask_n[:, None] & mask_ks[None, :],
                other=127,
                eviction_policy="evict_first",
            )
        acc = tl.dot_scaled(a, a_sc, "e4m3", b, b_sc, "e2m1", acc=acc)
        a_ptrs += BK
        as_ptrs += BK // 32
        b_ptrs += b_step
        bs_ptrs += BK // 32

    if SCATTER_OUTPUT:
        row_ids_c = tl.load(perm_ptr + slot_ids, mask=mask_m, other=0)
    else:
        row_ids_c = slot_ids

    c_ptrs = c_ptr + row_ids_c[:, None].to(tl.int64) * stride_cm + offs_n[None, :]
    tl.store(
        c_ptrs,
        acc.to(c_ptr.dtype.element_ty),
        mask=mask_m[:, None] & mask_n[None, :],
    )


@triton.jit
def _moe_weighted_reduce_kernel(
    in_ptr,
    weights_ptr,
    out_ptr,
    M,
    K,
    stride_im: tl.int64,
    stride_ik,
    stride_wm,
    stride_om: tl.int64,
    TOP_K: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0).to(tl.int64)
    pid_k = tl.program_id(1)
    offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    mask_k = offs_k < K

    in_row_base = in_ptr + pid_m * stride_im
    w_row_base = weights_ptr + pid_m * stride_wm
    acc = tl.zeros((BLOCK_K,), dtype=tl.float32)
    for k_idx in tl.static_range(TOP_K):
        w = tl.load(w_row_base + k_idx).to(tl.float32)
        if w != 0.0:
            v = tl.load(
                in_row_base + k_idx * stride_ik + offs_k, mask=mask_k, other=0.0
            ).to(tl.float32)
            acc += v * w

    tl.store(
        out_ptr + pid_m * stride_om + offs_k,
        acc.to(out_ptr.dtype.element_ty),
        mask=mask_k,
    )


def moe_weighted_reduce_triton(
    expert_outputs: torch.Tensor, topk_weights: torch.Tensor
) -> torch.Tensor:
    """Deterministic FP32 weighted sum over ``top_k`` expert outputs: ``[M, top_k, K] -> [M, K]``."""
    M, top_k, K = expert_outputs.shape
    if not expert_outputs.is_cuda:
        return (
            (expert_outputs.float() * topk_weights.float().unsqueeze(-1))
            .sum(dim=1)
            .to(expert_outputs.dtype)
        )
    out = torch.empty((M, K), dtype=expert_outputs.dtype, device=expert_outputs.device)
    if M == 0:
        return out
    w = topk_weights.float().contiguous()
    if expert_outputs.stride(-1) != 1:
        expert_outputs = expert_outputs.contiguous()
    BLOCK_K = 512 if K >= 512 else triton.next_power_of_2(max(K, 64))
    grid = (M, triton.cdiv(K, BLOCK_K))
    _moe_weighted_reduce_kernel[grid](
        expert_outputs,
        w,
        out,
        M,
        K,
        expert_outputs.stride(0),
        expert_outputs.stride(1),
        w.stride(0),
        out.stride(0),
        TOP_K=top_k,
        BLOCK_K=BLOCK_K,
        num_warps=4,
    )
    return out


def _select_grouped_gemm_config(
    num_tokens: int,
    top_k: int,
    num_experts: int,
    N: int,
    K: int,
    block_m: Optional[int] = None,
) -> Tuple[int, int, int, int, int]:
    """Select tuned SM100/SM103 ``(BM, BN, BK, num_warps, num_stages)``."""
    m_total = num_tokens * top_k
    est_active = max(1, min(num_experts, m_total))
    avg_m_per_e = m_total / est_active

    if block_m is None:
        if avg_m_per_e <= 12.0:
            BM = 16
        elif avg_m_per_e <= 24.0:
            BM = 32
        else:
            BM = 64
    else:
        BM = block_m

    if N < 128:
        BN = 64 if N >= 64 else max(16, triton.next_power_of_2(N))
    elif num_tokens <= 4 and N <= 2560:
        BN = 64
    else:
        BN = 128

    if K < 128 or (K % 128 != 0):
        BK = 64
    else:
        BK = 128

    num_warps = 4
    num_stages = 4 if (BM == 16 and BK == 128) else 3
    return BM, BN, BK, num_warps, num_stages


def mxfp4_grouped_moe_gemm(
    x: torch.Tensor,
    w_u8: torch.Tensor,
    scale_u8: torch.Tensor,
    *,
    schedule: Optional[MoETileSchedule] = None,
    topk_ids: Optional[torch.Tensor] = None,
    gather_input: bool = True,
    scatter_output: bool = False,
    weights_transposed: bool = False,
    compute_mode: str = "bf16",
    use_triton: bool = True,
) -> torch.Tensor:
    """Execute one grouped MXFP4 GEMM over active MoE experts."""
    num_experts = int(scale_u8.shape[0])
    N = int(scale_u8.shape[1])
    K = int(x.shape[-1])
    if schedule is None:
        if topk_ids is None:
            raise ValueError("Provide either schedule or topk_ids")
        BM_init, _, _, _, _ = _select_grouped_gemm_config(
            topk_ids.shape[0], topk_ids.shape[1], num_experts, N, K
        )
        schedule = build_moe_tile_schedule(topk_ids, num_experts, BM_init)

    m_total = schedule.num_tokens * schedule.top_k
    out = torch.empty((m_total, N), dtype=x.dtype, device=x.device)
    if m_total == 0:
        return out

    if (
        not use_triton
        or not x.is_cuda
        or not supports_triton_mxfp4(x.device)
        or x.dtype != torch.bfloat16
    ):
        w_nk = w_u8.transpose(1, 2).contiguous() if weights_transposed else w_u8
        tile_e = schedule.tile_expert_id.tolist()
        tile_s = schedule.tile_m_start.tolist()
        tile_c = schedule.tile_m_count.tolist()
        deq_cache: dict[int, torch.Tensor] = {}
        for e, m0, cnt in zip(tile_e, tile_s, tile_c):
            if e < 0:
                continue
            w_f = deq_cache.get(e)
            if w_f is None:
                w_f = dequantize_mxfp4_ref(w_nk[e], scale_u8[e], dtype=x.dtype)
                deq_cache[e] = w_f
            slots = slice(m0, m0 + cnt)
            if gather_input:
                a_sub = x.index_select(0, schedule.sorted_tok_ids[slots].to(torch.int64))
            else:
                a_sub = x[slots]
            y_sub = F.linear(a_sub, w_f)
            if scatter_output:
                out.index_copy_(0, schedule.perm[slots].to(torch.int64), y_sub)
            else:
                out[slots] = y_sub
        return out

    if x.stride(-1) != 1:
        x = x.contiguous()
    if w_u8.stride(-1) != 1:
        w_u8 = w_u8.contiguous()
    if scale_u8.stride(-1) != 1:
        scale_u8 = scale_u8.contiguous()

    BM, BN, BK, num_warps, num_stages = _select_grouped_gemm_config(
        schedule.num_tokens,
        schedule.top_k,
        num_experts,
        N,
        K,
        block_m=schedule.block_m,
    )
    grid = (triton.cdiv(N, BN), schedule.max_num_tiles)
    k_divisible = (K % BK) == 0

    mode_norm = "bf16" if str(compute_mode).lower() in ("bf16", "bf16_strict", "strict_bf16") else str(compute_mode).lower()
    if mode_norm == "bf16":
        _mxfp4_grouped_gemm_bf16_kernel[grid](
            x,
            w_u8,
            scale_u8,
            out,
            schedule.sorted_tok_ids,
            schedule.perm,
            schedule.tile_expert_id,
            schedule.tile_m_start,
            schedule.tile_m_count,
            N,
            K,
            x.stride(0),
            w_u8.stride(0),
            w_u8.stride(1),
            scale_u8.stride(0),
            scale_u8.stride(1),
            out.stride(0),
            BM=BM,
            BN=BN,
            BK=BK,
            TRANS_B=not weights_transposed,
            GATHER_INPUT=gather_input,
            SCATTER_OUTPUT=scatter_output,
            K_DIVISIBLE=k_divisible,
            num_warps=num_warps,
            num_stages=num_stages,
        )
    elif mode_norm == "e4m3":
        x_q, x_sc = quantize_mxfp8_e4m3(x)
        _mxfp4_grouped_gemm_e4m3_kernel[grid](
            x_q,
            x_sc,
            w_u8,
            scale_u8,
            out,
            schedule.sorted_tok_ids,
            schedule.perm,
            schedule.tile_expert_id,
            schedule.tile_m_start,
            schedule.tile_m_count,
            N,
            K,
            x_q.stride(0),
            x_sc.stride(0),
            w_u8.stride(0),
            w_u8.stride(1),
            scale_u8.stride(0),
            scale_u8.stride(1),
            out.stride(0),
            BM=BM,
            BN=BN,
            BK=BK,
            TRANS_B=not weights_transposed,
            GATHER_INPUT=gather_input,
            SCATTER_OUTPUT=scatter_output,
            K_DIVISIBLE=k_divisible,
            num_warps=num_warps,
            num_stages=num_stages,
        )
    else:
        raise ValueError(f"Unsupported compute_mode={compute_mode!r}; expected bf16, bf16_strict, or e4m3")

    return out


# ---------------------------------------------------------------------------
# Optimized Persistent Split-K Triton Decode GEMV (M = 1..64)
#   - Zero -1 wasted empty tiles (compact active-task queue on GPU)
#   - Fused SiTU-GLU epilogue (inline at SPLIT_K=1, fused reduce+SiTU at SPLIT_K>1)
#   - Single-pass weighted scatter-reduce in Stage 2 (atomic_add into FP32)
# ---------------------------------------------------------------------------


@triton.jit
def _mxfp4_decode_stage1_persistent_kernel(
    a_ptr,
    b_ptr,
    b_sc_ptr,
    out_g_ptr,  # if SPLIT_K==1: act_ptr [M_slots, I] BF16; else partial_g [SPLIT_K, M_slots, I] FP32
    out_u_ptr,  # if SPLIT_K==1: unused; else partial_u [SPLIT_K, M_slots, I] FP32
    sorted_tok_ids_ptr,
    tile_expert_id_ptr,
    tile_m_start_ptr,
    tile_m_count_ptr,
    num_active_tiles_ptr,
    I,
    K,
    stride_am,
    stride_be: tl.int64,
    stride_b0,
    stride_bse: tl.int64,
    stride_bsn,
    stride_os: tl.int64,
    stride_om,
    gate_beta: tl.constexpr,
    up_beta: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    SPLIT_K: tl.constexpr,
    N_BLOCKS: tl.constexpr,
    NUM_CTAS: tl.constexpr,
    TRANS_B: tl.constexpr,
    K_DIVISIBLE: tl.constexpr,
):
    pid = tl.program_id(0)
    num_active = tl.load(num_active_tiles_ptr).to(tl.int32)
    total_tasks = num_active * N_BLOCKS * SPLIT_K
    if total_tasks == 0:
        return

    K_split = K // SPLIT_K
    offs_local = tl.arange(0, BM)
    offs_k_base = tl.arange(0, BK)
    offs_k2_base = tl.arange(0, BK // 2)
    offs_ks_base = tl.arange(0, BK // 32)
    K_half = K // 2
    K_sc = K // 32

    for task_id in range(pid, total_tasks, NUM_CTAS):
        split_id = task_id % SPLIT_K
        tmp = task_id // SPLIT_K
        n_blk = tmp % N_BLOCKS
        tile_id = tmp // N_BLOCKS

        expert_id = tl.load(tile_expert_id_ptr + tile_id).to(tl.int64)
        m_start = tl.load(tile_m_start_ptr + tile_id)
        m_count = tl.load(tile_m_count_ptr + tile_id)

        mask_m = offs_local < m_count
        slot_ids = m_start + offs_local
        row_ids_a = tl.load(sorted_tok_ids_ptr + slot_ids, mask=mask_m, other=0)

        offs_ng = n_blk * BN + tl.arange(0, BN)
        offs_nu = I + offs_ng
        mask_n = offs_ng < I

        k_start = split_id * K_split
        k_end = K if split_id == SPLIT_K - 1 else k_start + K_split

        b_base = b_ptr + expert_id * stride_be
        bs_base = b_sc_ptr + expert_id * stride_bse

        a_ptrs = (
            a_ptr
            + row_ids_a[:, None].to(tl.int64) * stride_am
            + (k_start + offs_k_base)[None, :]
        )
        if TRANS_B:
            bg_ptrs = b_base + offs_ng[:, None] * stride_b0 + (k_start // 2 + offs_k2_base)[None, :]
            bu_ptrs = b_base + offs_nu[:, None] * stride_b0 + (k_start // 2 + offs_k2_base)[None, :]
            b_step = BK // 2
        else:
            bg_ptrs = b_base + (k_start // 2 + offs_k2_base)[:, None] * stride_b0 + offs_ng[None, :]
            bu_ptrs = b_base + (k_start // 2 + offs_k2_base)[:, None] * stride_b0 + offs_nu[None, :]
            b_step = (BK // 2) * stride_b0

        bsg_ptrs = bs_base + offs_ng[:, None] * stride_bsn + (k_start // 32 + offs_ks_base)[None, :]
        bsu_ptrs = bs_base + offs_nu[:, None] * stride_bsn + (k_start // 32 + offs_ks_base)[None, :]

        acc_g = tl.zeros((BM, BN), dtype=tl.float32)
        acc_u = tl.zeros((BM, BN), dtype=tl.float32)

        for k0 in range(k_start, k_end, BK):
            if K_DIVISIBLE:
                a = tl.load(a_ptrs, mask=mask_m[:, None], other=0.0)
                if TRANS_B:
                    bg = tl.load(bg_ptrs, mask=mask_n[:, None], other=0, eviction_policy="evict_first")
                    bu = tl.load(bu_ptrs, mask=mask_n[:, None], other=0, eviction_policy="evict_first")
                    bg = tl.trans(bg)
                    bu = tl.trans(bu)
                else:
                    bg = tl.load(bg_ptrs, mask=mask_n[None, :], other=0, eviction_policy="evict_first")
                    bu = tl.load(bu_ptrs, mask=mask_n[None, :], other=0, eviction_policy="evict_first")
                bsg = tl.load(bsg_ptrs, mask=mask_n[:, None], other=127, eviction_policy="evict_first")
                bsu = tl.load(bsu_ptrs, mask=mask_n[:, None], other=127, eviction_policy="evict_first")
            else:
                mask_k = (k0 + offs_k_base) < k_end
                mask_k2 = (k0 // 2 + offs_k2_base) < K_half
                mask_ks = (k0 // 32 + offs_ks_base) < K_sc
                a = tl.load(a_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0.0)
                if TRANS_B:
                    bg = tl.load(bg_ptrs, mask=mask_n[:, None] & mask_k2[None, :], other=0, eviction_policy="evict_first")
                    bu = tl.load(bu_ptrs, mask=mask_n[:, None] & mask_k2[None, :], other=0, eviction_policy="evict_first")
                    bg = tl.trans(bg)
                    bu = tl.trans(bu)
                else:
                    bg = tl.load(bg_ptrs, mask=mask_k2[:, None] & mask_n[None, :], other=0, eviction_policy="evict_first")
                    bu = tl.load(bu_ptrs, mask=mask_k2[:, None] & mask_n[None, :], other=0, eviction_policy="evict_first")
                bsg = tl.load(bsg_ptrs, mask=mask_n[:, None] & mask_ks[None, :], other=127, eviction_policy="evict_first")
                bsu = tl.load(bsu_ptrs, mask=mask_n[:, None] & mask_ks[None, :], other=127, eviction_policy="evict_first")

            acc_g = tl.dot_scaled(a, None, "bf16", bg, bsg, "e2m1", acc=acc_g)
            acc_u = tl.dot_scaled(a, None, "bf16", bu, bsu, "e2m1", acc=acc_u)
            a_ptrs += BK
            bg_ptrs += b_step
            bu_ptrs += b_step
            bsg_ptrs += BK // 32
            bsu_ptrs += BK // 32

        if SPLIT_K == 1:
            # Round gate/up to BF16 to match unfused Stage1 -> SiTU-GLU parity
            gf = acc_g.to(tl.bfloat16).to(tl.float32)
            uf = acc_u.to(tl.bfloat16).to(tl.float32)
            sig_g = tl.div_rn(1.0, 1.0 + libdevice.exp(-gf))
            g_val = gate_beta * libdevice.tanh(tl.div_rn(gf, gate_beta)) * sig_g
            u_val = up_beta * libdevice.tanh(tl.div_rn(uf, up_beta))
            act = (g_val * u_val).to(out_g_ptr.dtype.element_ty)
            c_ptrs = out_g_ptr + slot_ids[:, None].to(tl.int64) * stride_om + offs_ng[None, :]
            tl.store(c_ptrs, act, mask=mask_m[:, None] & mask_n[None, :])
        else:
            s_off = split_id.to(tl.int64) * stride_os
            og_ptrs = out_g_ptr + s_off + slot_ids[:, None].to(tl.int64) * stride_om + offs_ng[None, :]
            ou_ptrs = out_u_ptr + s_off + slot_ids[:, None].to(tl.int64) * stride_om + offs_ng[None, :]
            tl.store(og_ptrs, acc_g, mask=mask_m[:, None] & mask_n[None, :])
            tl.store(ou_ptrs, acc_u, mask=mask_m[:, None] & mask_n[None, :])


@triton.jit
def _splitk_reduce_situ_glu_kernel(
    pg_ptr,
    pu_ptr,
    act_ptr,
    num_active_slots_ptr,
    I,
    stride_s: tl.int64,
    stride_m,
    gate_beta: tl.constexpr,
    up_beta: tl.constexpr,
    SPLIT_K: tl.constexpr,
    BLOCK_I: tl.constexpr,
):
    slot = tl.program_id(0).to(tl.int64)
    n_active = tl.load(num_active_slots_ptr).to(tl.int64)
    if slot >= n_active:
        return
    pid_i = tl.program_id(1)
    offs_i = pid_i * BLOCK_I + tl.arange(0, BLOCK_I)
    mask_i = offs_i < I

    acc_g = tl.zeros((BLOCK_I,), dtype=tl.float32)
    acc_u = tl.zeros((BLOCK_I,), dtype=tl.float32)
    row_off = slot * stride_m + offs_i
    for s in tl.static_range(SPLIT_K):
        acc_g += tl.load(pg_ptr + s * stride_s + row_off, mask=mask_i, other=0.0)
        acc_u += tl.load(pu_ptr + s * stride_s + row_off, mask=mask_i, other=0.0)

    gf = acc_g.to(tl.bfloat16).to(tl.float32)
    uf = acc_u.to(tl.bfloat16).to(tl.float32)
    sig_g = tl.div_rn(1.0, 1.0 + libdevice.exp(-gf))
    g_val = gate_beta * libdevice.tanh(tl.div_rn(gf, gate_beta)) * sig_g
    u_val = up_beta * libdevice.tanh(tl.div_rn(uf, up_beta))
    out = (g_val * u_val).to(act_ptr.dtype.element_ty)
    tl.store(act_ptr + row_off, out, mask=mask_i)


@triton.jit
def _mxfp4_decode_stage2_persistent_kernel(
    act_ptr,
    b_ptr,
    b_sc_ptr,
    partial_ptr,
    tile_expert_id_ptr,
    tile_m_start_ptr,
    tile_m_count_ptr,
    num_active_tiles_ptr,
    N,
    K,
    stride_am,
    stride_be: tl.int64,
    stride_b0,
    stride_bse: tl.int64,
    stride_bsn,
    stride_ps: tl.int64,
    stride_pm: tl.int64,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    SPLIT_K: tl.constexpr,
    N_BLOCKS: tl.constexpr,
    NUM_CTAS: tl.constexpr,
    TRANS_B: tl.constexpr,
    K_DIVISIBLE: tl.constexpr,
):
    pid = tl.program_id(0)
    num_active = tl.load(num_active_tiles_ptr).to(tl.int32)
    total_tasks = num_active * N_BLOCKS * SPLIT_K
    if total_tasks == 0:
        return

    K_split = K // SPLIT_K
    offs_local = tl.arange(0, BM)
    offs_k_base = tl.arange(0, BK)
    offs_k2_base = tl.arange(0, BK // 2)
    offs_ks_base = tl.arange(0, BK // 32)
    K_half = K // 2
    K_sc = K // 32

    for task_id in range(pid, total_tasks, NUM_CTAS):
        split_id = task_id % SPLIT_K
        tmp = task_id // SPLIT_K
        n_blk = tmp % N_BLOCKS
        tile_id = tmp // N_BLOCKS

        expert_id = tl.load(tile_expert_id_ptr + tile_id).to(tl.int64)
        m_start = tl.load(tile_m_start_ptr + tile_id)
        m_count = tl.load(tile_m_count_ptr + tile_id)

        mask_m = offs_local < m_count
        slot_ids = m_start + offs_local
        offs_n = n_blk * BN + tl.arange(0, BN)
        mask_n = offs_n < N

        k_start = split_id * K_split
        k_end = K if split_id == SPLIT_K - 1 else k_start + K_split

        b_base = b_ptr + expert_id * stride_be
        bs_base = b_sc_ptr + expert_id * stride_bse

        a_ptrs = (
            act_ptr
            + slot_ids[:, None].to(tl.int64) * stride_am
            + (k_start + offs_k_base)[None, :]
        )
        if TRANS_B:
            b_ptrs = b_base + offs_n[:, None] * stride_b0 + (k_start // 2 + offs_k2_base)[None, :]
            b_step = BK // 2
        else:
            b_ptrs = b_base + (k_start // 2 + offs_k2_base)[:, None] * stride_b0 + offs_n[None, :]
            b_step = (BK // 2) * stride_b0
        bs_ptrs = bs_base + offs_n[:, None] * stride_bsn + (k_start // 32 + offs_ks_base)[None, :]

        acc = tl.zeros((BM, BN), dtype=tl.float32)
        for k0 in range(k_start, k_end, BK):
            if K_DIVISIBLE:
                a = tl.load(a_ptrs, mask=mask_m[:, None], other=0.0)
                if TRANS_B:
                    b = tl.load(b_ptrs, mask=mask_n[:, None], other=0, eviction_policy="evict_first")
                    b = tl.trans(b)
                else:
                    b = tl.load(b_ptrs, mask=mask_n[None, :], other=0, eviction_policy="evict_first")
                b_sc = tl.load(bs_ptrs, mask=mask_n[:, None], other=127, eviction_policy="evict_first")
            else:
                mask_k = (k0 + offs_k_base) < k_end
                mask_k2 = (k0 // 2 + offs_k2_base) < K_half
                mask_ks = (k0 // 32 + offs_ks_base) < K_sc
                a = tl.load(a_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0.0)
                if TRANS_B:
                    b = tl.load(b_ptrs, mask=mask_n[:, None] & mask_k2[None, :], other=0, eviction_policy="evict_first")
                    b = tl.trans(b)
                else:
                    b = tl.load(b_ptrs, mask=mask_k2[:, None] & mask_n[None, :], other=0, eviction_policy="evict_first")
                b_sc = tl.load(bs_ptrs, mask=mask_n[:, None] & mask_ks[None, :], other=127, eviction_policy="evict_first")

            acc = tl.dot_scaled(a, None, "bf16", b, b_sc, "e2m1", acc=acc)
            a_ptrs += BK
            b_ptrs += b_step
            bs_ptrs += BK // 32

        out_ptrs = (
            partial_ptr
            + split_id.to(tl.int64) * stride_ps
            + slot_ids[:, None].to(tl.int64) * stride_pm
            + offs_n[None, :]
        )
        tl.store(out_ptrs, acc, mask=mask_m[:, None] & mask_n[None, :])


@triton.jit
def _decode_stage2_deterministic_reduce_kernel(
    partial_ptr,
    inv_slots_ptr,
    topk_weights_ptr,
    out_ptr,
    L,
    stride_ps: tl.int64,
    stride_pm: tl.int64,
    stride_wm,
    stride_om: tl.int64,
    SPLIT_K: tl.constexpr,
    TOP_K: tl.constexpr,
    BLOCK_L: tl.constexpr,
):
    tok = tl.program_id(0).to(tl.int64)
    pid_l = tl.program_id(1)
    offs_l = pid_l * BLOCK_L + tl.arange(0, BLOCK_L)
    mask_l = offs_l < L

    acc = tl.zeros((BLOCK_L,), dtype=tl.float32)
    slot_base = inv_slots_ptr + tok * TOP_K
    w_base = topk_weights_ptr + tok * stride_wm
    for k_idx in tl.static_range(TOP_K):
        slot = tl.load(slot_base + k_idx).to(tl.int64)
        if slot >= 0:
            w = tl.load(w_base + k_idx).to(tl.float32)
            if w != 0.0:
                val = tl.zeros((BLOCK_L,), dtype=tl.float32)
                for s in tl.static_range(SPLIT_K):
                    val += tl.load(
                        partial_ptr + s * stride_ps + slot * stride_pm + offs_l,
                        mask=mask_l,
                        other=0.0,
                    )
                acc += val * w

    tl.store(
        out_ptr + tok * stride_om + offs_l,
        acc.to(out_ptr.dtype.element_ty),
        mask=mask_l,
    )


def mxfp4_moe_triton_splitk(
    x: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    weights: MXFP4MoEWeights,
    *,
    gate_beta: float = 4.0,
    up_beta: float = 25.0,
    ep_start: int = 0,
    ep_size: int = 1,
) -> torch.Tensor:
    """Persistent Split-K Triton ``tl.dot_scaled`` Decode GEMV (M=1..64).

    Eliminates all ``-1`` wasted empty tiles via a compact GPU task queue,
    fuses ``SiTU-GLU`` into Stage 1, and performs a single-pass FP32 weighted
    scatter-reduce directly from Stage 2. 100% CUDA Graph capture/replay safe.
    """
    M, L = x.shape
    if M == 0:
        return torch.empty_like(x)
    if not x.is_cuda or x.dtype != torch.bfloat16 or not supports_triton_mxfp4(x.device):
        return mxfp4_moe_forward_ref(
            x, topk_ids, topk_weights, weights,
            gate_beta=gate_beta, up_beta=up_beta, ep_start=ep_start, ep_size=ep_size,
        )

    x_c = x if x.stride(-1) == 1 else x.contiguous()
    w13 = weights.w13_u8 if weights.w13_u8.stride(-1) == 1 else weights.w13_u8.contiguous()
    w13_sc = weights.w13_scale if weights.w13_scale.stride(-1) == 1 else weights.w13_scale.contiguous()
    w2 = weights.w2_u8 if weights.w2_u8.stride(-1) == 1 else weights.w2_u8.contiguous()
    w2_sc = weights.w2_scale if weights.w2_scale.stride(-1) == 1 else weights.w2_scale.contiguous()

    E_local = weights.num_experts
    I = weights.intermediate_size
    top_k = topk_ids.shape[1]

    BM = 16
    BN1 = 64 if I >= 64 else max(16, triton.next_power_of_2(I))
    BK1 = 128 if (L >= 128 and L % 128 == 0) else 64
    if M <= 4 and L % (4 * BK1) == 0:
        SPLIT_K1 = 4
    elif M <= 16 and L % (2 * BK1) == 0:
        SPLIT_K1 = 2
    else:
        SPLIT_K1 = 1

    BN2 = 64 if L >= 64 else max(16, triton.next_power_of_2(L))
    BK2 = 128 if (I >= 128 and I % 128 == 0) else 64
    if M <= 4 and I % (4 * BK2) == 0:
        SPLIT_K2 = 4
    elif M <= 16 and I % (2 * BK2) == 0:
        SPLIT_K2 = 2
    else:
        SPLIT_K2 = 1

    sched = _build_compact_decode_schedule(
        topk_ids, topk_weights, E_local, BM, ep_start=ep_start, ep_size=ep_size
    )

    sm_count = _get_sm_count(x.device)
    N1_BLOCKS = triton.cdiv(I, BN1)
    N2_BLOCKS = triton.cdiv(L, BN2)
    num_ctas1 = max(1, min(sm_count * 2, sched.max_num_tiles * N1_BLOCKS * SPLIT_K1))
    num_ctas2 = max(1, min(sm_count * 2, sched.max_num_tiles * N2_BLOCKS * SPLIT_K2))

    act = torch.empty((sched.max_slots, I), dtype=x.dtype, device=x.device)
    if SPLIT_K1 == 1:
        _mxfp4_decode_stage1_persistent_kernel[(num_ctas1,)](
            x_c,
            w13,
            w13_sc,
            act,
            act,
            sched.sorted_tok_ids,
            sched.tile_expert_id,
            sched.tile_m_start,
            sched.tile_m_count,
            sched.num_active_tiles_buf,
            I,
            L,
            x_c.stride(0),
            w13.stride(0),
            w13.stride(1),
            w13_sc.stride(0),
            w13_sc.stride(1),
            0,
            act.stride(0),
            gate_beta=float(gate_beta),
            up_beta=float(up_beta),
            BM=BM,
            BN=BN1,
            BK=BK1,
            SPLIT_K=1,
            N_BLOCKS=N1_BLOCKS,
            NUM_CTAS=num_ctas1,
            TRANS_B=not weights.weights_transposed,
            K_DIVISIBLE=(L % BK1 == 0),
            num_warps=4,
            num_stages=3,
        )
    else:
        pg = torch.empty((SPLIT_K1, sched.max_slots, I), dtype=torch.float32, device=x.device)
        pu = torch.empty((SPLIT_K1, sched.max_slots, I), dtype=torch.float32, device=x.device)
        _mxfp4_decode_stage1_persistent_kernel[(num_ctas1,)](
            x_c,
            w13,
            w13_sc,
            pg,
            pu,
            sched.sorted_tok_ids,
            sched.tile_expert_id,
            sched.tile_m_start,
            sched.tile_m_count,
            sched.num_active_tiles_buf,
            I,
            L,
            x_c.stride(0),
            w13.stride(0),
            w13.stride(1),
            w13_sc.stride(0),
            w13_sc.stride(1),
            pg.stride(0),
            pg.stride(1),
            gate_beta=float(gate_beta),
            up_beta=float(up_beta),
            BM=BM,
            BN=BN1,
            BK=BK1,
            SPLIT_K=SPLIT_K1,
            N_BLOCKS=N1_BLOCKS,
            NUM_CTAS=num_ctas1,
            TRANS_B=not weights.weights_transposed,
            K_DIVISIBLE=(L % BK1 == 0),
            num_warps=4,
            num_stages=3,
        )
        BLOCK_I = 512 if I >= 512 else triton.next_power_of_2(max(I, 64))
        _splitk_reduce_situ_glu_kernel[(sched.max_slots, triton.cdiv(I, BLOCK_I))](
            pg,
            pu,
            act,
            sched.num_active_slots_buf,
            I,
            pg.stride(0),
            pg.stride(1),
            gate_beta=float(gate_beta),
            up_beta=float(up_beta),
            SPLIT_K=SPLIT_K1,
            BLOCK_I=BLOCK_I,
            num_warps=4,
        )

    partial_out = torch.empty((SPLIT_K2, sched.max_slots, L), dtype=torch.float32, device=x.device)
    _mxfp4_decode_stage2_persistent_kernel[(num_ctas2,)](
        act,
        w2,
        w2_sc,
        partial_out,
        sched.tile_expert_id,
        sched.tile_m_start,
        sched.tile_m_count,
        sched.num_active_tiles_buf,
        L,
        I,
        act.stride(0),
        w2.stride(0),
        w2.stride(1),
        w2_sc.stride(0),
        w2_sc.stride(1),
        partial_out.stride(0),
        partial_out.stride(1),
        BM=BM,
        BN=BN2,
        BK=BK2,
        SPLIT_K=SPLIT_K2,
        N_BLOCKS=N2_BLOCKS,
        NUM_CTAS=num_ctas2,
        TRANS_B=not weights.weights_transposed,
        K_DIVISIBLE=(I % BK2 == 0),
        num_warps=4,
        num_stages=3,
    )
    out = torch.empty((M, L), dtype=x.dtype, device=x.device)
    BLOCK_L = 512 if L >= 512 else triton.next_power_of_2(max(L, 64))
    w_f = (
        topk_weights
        if (topk_weights.dtype == torch.float32 and topk_weights.is_contiguous())
        else topk_weights.float().contiguous()
    )
    _decode_stage2_deterministic_reduce_kernel[(M, triton.cdiv(L, BLOCK_L))](
        partial_out,
        sched.inv_slots,
        w_f,
        out,
        L,
        partial_out.stride(0),
        partial_out.stride(1),
        w_f.stride(0),
        out.stride(0),
        SPLIT_K=SPLIT_K2,
        TOP_K=top_k,
        BLOCK_L=BLOCK_L,
        num_warps=4,
    )
    return out


# ---------------------------------------------------------------------------
# DeepGEMM SM100 FP8xMXFP4 Backends (Contiguous Prefill + Masked CG Decode)
#   + FlashInfer group_gemm_mxfp4_nt_groupwise
# ---------------------------------------------------------------------------


@triton.jit(do_not_specialize=["m_total", "ep_start"])
def _contig_ep_bincount_only_kernel(
    ids_ptr,
    counts_ptr,
    m_total,
    ep_start,
    num_experts: tl.constexpr,
    HAS_EP: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < m_total
    gid = tl.load(ids_ptr + offs, mask=mask, other=-1).to(tl.int32)
    le = gid - ep_start if HAS_EP else gid
    valid = mask & (le >= 0) & (le < num_experts)
    tl.atomic_add(counts_ptr + le, 1, mask=valid)


@triton.jit
def _contig_prefix_scan_align_kernel(
    counts_ptr,
    aligned_counts_ptr,
    offsets_ptr,
    all_tokens_buf_ptr,
    num_experts: tl.constexpr,
    ALIGNMENT: tl.constexpr,
    E_PAD: tl.constexpr,
):
    offs = tl.arange(0, E_PAD)
    mask = offs < num_experts
    c = tl.load(counts_ptr + offs, mask=mask, other=0).to(tl.int32)
    ac = tl.where(mask, ((c + ALIGNMENT - 1) // ALIGNMENT) * ALIGNMENT, 0)
    off = tl.cumsum(ac, axis=0) - ac
    tl.store(aligned_counts_ptr + offs, ac, mask=mask)
    tl.store(offsets_ptr + offs, off, mask=mask)
    tl.store(all_tokens_buf_ptr, tl.sum(ac, axis=0))


@triton.jit
def _init_m_indices_kernel(
    counts_ptr,
    aligned_counts_ptr,
    offsets_ptr,
    m_indices_ptr,
    write_cnt_ptr,
    active_rows_ptr,
    cap_tokens,
    num_experts: tl.constexpr,
    FILL_TAIL: tl.constexpr,
    BLOCK: tl.constexpr,
):
    e = tl.program_id(0)
    if e < num_experts:
        tl.store(write_cnt_ptr + e, 0)
        c = tl.load(counts_ptr + e)
        if c > 0:
            ac = tl.load(aligned_counts_ptr + e).to(tl.int32)
            start = tl.load(offsets_ptr + e).to(tl.int32)
            for base in range(0, ac, BLOCK):
                offs = base + tl.arange(0, BLOCK)
                tl.store(m_indices_ptr + start + offs, e, mask=(offs < ac))
    elif FILL_TAIL:
        total_aligned = (
            tl.load(offsets_ptr + num_experts - 1) + tl.load(aligned_counts_ptr + num_experts - 1)
        ).to(tl.int32)
        if e == num_experts:
            tl.store(active_rows_ptr, total_aligned)
        pid_tail = e - num_experts
        n_tail_ctas = tl.num_programs(0) - num_experts
        for base in range(total_aligned + pid_tail * BLOCK, cap_tokens, n_tail_ctas * BLOCK):
            offs = base + tl.arange(0, BLOCK)
            tl.store(m_indices_ptr + offs, -1, mask=(offs < cap_tokens))


@triton.jit(do_not_specialize=["stride_q_row", "stride_sc_row", "stride_out_q_row", "stride_out_sc_k"])
def _scatter_fp8_to_contig_kernel(
    x_q_ptr,
    x_sc_ptr,
    topk_ids_ptr,
    offsets_ptr,
    write_cnt_ptr,
    out_q_ptr,
    out_sc_ptr,
    dst_slots_ptr,
    M_total,
    top_k,
    K,
    num_groups,
    num_packs,
    ep_start,
    num_experts,
    stride_q_row: tl.int64,
    stride_sc_row: tl.int64,
    stride_out_q_row: tl.int64,
    stride_out_sc_k: tl.int64,
    BLOCK_K: tl.constexpr,
):
    idx = tl.program_id(0).to(tl.int64)
    if idx >= M_total:
        return
    tok = idx // top_k
    gid = tl.load(topk_ids_ptr + idx).to(tl.int32)
    local_e = gid - ep_start
    if local_e < 0 or local_e >= num_experts:
        tl.store(dst_slots_ptr + idx, -1)
        return

    pos = tl.atomic_add(write_cnt_ptr + local_e, 1)
    slot = (tl.load(offsets_ptr + local_e) + pos).to(tl.int64)
    tl.store(dst_slots_ptr + idx, slot.to(tl.int32))

    for k0 in range(0, K, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        mk = offs_k < K
        vals = tl.load(x_q_ptr + tok * stride_q_row + offs_k, mask=mk, other=0)
        tl.store(out_q_ptr + slot * stride_out_q_row + offs_k, vals, mask=mk)

    PACKS_BLK: tl.constexpr = BLOCK_K // 128
    for p0 in range(0, num_packs, PACKS_BLK):
        offs_p = p0 + tl.arange(0, PACKS_BLK)
        mp = offs_p < num_packs
        packed = tl.zeros((PACKS_BLK,), dtype=tl.uint32)
        sc_base = x_sc_ptr + tok * stride_sc_row
        for b_idx in tl.static_range(4):
            g = offs_p * 4 + b_idx
            v = tl.load(sc_base + g, mask=mp & (g < num_groups), other=0).to(tl.uint32)
            packed |= v << (b_idx * 8)
        tl.store(
            out_sc_ptr + offs_p.to(tl.int64) * stride_out_sc_k + slot,
            packed.to(tl.int32),
            mask=mp,
        )


@triton.jit
def _gather_slot_weighted_reduce_kernel(
    down_ptr,
    dst_slots_ptr,
    topk_weights_ptr,
    out_ptr,
    K,
    stride_dm: tl.int64,
    stride_wm,
    stride_om: tl.int64,
    TOP_K: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    tok = tl.program_id(0).to(tl.int64)
    pid_k = tl.program_id(1)
    offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    mask_k = offs_k < K

    acc = tl.zeros((BLOCK_K,), dtype=tl.float32)
    slot_base = dst_slots_ptr + tok * TOP_K
    w_base = topk_weights_ptr + tok * stride_wm
    for k_idx in tl.static_range(TOP_K):
        slot = tl.load(slot_base + k_idx).to(tl.int64)
        if slot >= 0:
            w = tl.load(w_base + k_idx).to(tl.float32)
            if w != 0.0:
                v = tl.load(down_ptr + slot * stride_dm + offs_k, mask=mask_k, other=0.0).to(
                    tl.float32
                )
                acc += v * w

    tl.store(
        out_ptr + tok * stride_om + offs_k,
        acc.to(out_ptr.dtype.element_ty),
        mask=mask_k,
    )


def mxfp4_moe_deepgemm_contiguous(
    x: Optional[torch.Tensor],
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    weights: MXFP4MoEWeights,
    *,
    gate_beta: float = 4.0,
    up_beta: float = 25.0,
    ep_start: int = 0,
    ep_size: int = 1,
    prequant_fp8: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    out_dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """DeepGEMM SM100 ``m_grouped_fp8_fp4_gemm_nt_contiguous`` (Prefill / large-M)."""
    import deep_gemm

    x_ref = prequant_fp8[0] if prequant_fp8 is not None else x
    assert x_ref is not None
    M, L = x_ref.shape
    dev = x_ref.device
    res_dtype = (
        out_dtype
        if out_dtype is not None
        else (x.dtype if x is not None else torch.bfloat16)
    )
    if M == 0:
        return torch.empty((0, L), dtype=res_dtype, device=dev)
    top_k = topk_ids.shape[1]
    E_local = weights.num_experts
    I = weights.intermediate_size
    if L % 128 != 0 or I % 128 != 0:
        raise ValueError(f"DeepGEMM FP8xFP4 requires L={L}, I={I} divisible by 128")

    alignment = int(deep_gemm.get_mk_alignment_for_contiguous_layout())
    ids_c = topk_ids if topk_ids.is_contiguous() else topk_ids.contiguous()
    flat_ids = ids_c.view(-1)
    e_lo = int(ep_start) if ep_size > 1 else 0
    m_total_slots = M * top_k
    nosync_env = os.environ.get("QWEN4_5_MOE_CONTIG_NOSYNC")
    explicit_ep_nosync = (
        nosync_env is not None
        and nosync_env.strip().lower() in ("1", "true", "yes")
    )
    capturing_cg = (
        torch.cuda.is_current_stream_capturing()
        or os.environ.get("RTP_LLM_CUDA_GRAPH_WARMUP_FORWARD") == "1"
    )
    # Keep EP>1 non-CG prefill exact via async D2H + event overlap unless explicitly opted in;
    # otherwise the M*top_k + E_local*127 upper bound bloats all_tokens >3x on sparse EP=4 ranks.
    use_cap_nosync = (
        (ep_size <= 1)
        or capturing_cg
        or (
            explicit_ep_nosync
            and E_local <= 2048
            and ids_c.dtype == torch.int32
            and m_total_slots <= 524288
        )
    )
    sync_state: Optional[Tuple[torch.Tensor, torch.cuda.Event]] = None
    slab_write_cnt: Optional[torch.Tensor] = None
    if E_local <= 2048 and ids_c.dtype == torch.int32:
        if not use_cap_nosync:
            pinned_tok, ev_tok, i32_slab = _get_contig_sync_state(dev, E_local * 4 + 1)
            counts = i32_slab[:E_local]
            counts.zero_()
            aligned_counts = i32_slab[E_local : 2 * E_local]
            offsets = i32_slab[2 * E_local : 3 * E_local]
            slab_write_cnt = i32_slab[3 * E_local : 4 * E_local]
            all_tokens_buf = i32_slab[4 * E_local : 4 * E_local + 1]
            sync_state = (pinned_tok, ev_tok)
        else:
            counts = torch.zeros((E_local,), dtype=torch.int32, device=dev)
            aligned_counts = torch.empty((E_local,), dtype=torch.int32, device=dev)
            offsets = torch.empty((E_local,), dtype=torch.int32, device=dev)
            all_tokens_buf = torch.empty((1,), dtype=torch.int32, device=dev)
        _contig_ep_bincount_only_kernel[(triton.cdiv(m_total_slots, 256),)](
            flat_ids,
            counts,
            m_total_slots,
            e_lo,
            num_experts=E_local,
            HAS_EP=(ep_size > 1),
            BLOCK=256,
            num_warps=4,
        )
        e_pad = max(16, triton.next_power_of_2(E_local))
        _contig_prefix_scan_align_kernel[(1,)](
            counts,
            aligned_counts,
            offsets,
            all_tokens_buf,
            num_experts=E_local,
            ALIGNMENT=alignment,
            E_PAD=e_pad,
            num_warps=4,
        )
        if sync_state is not None:
            with _cuda_device_ctx(dev):
                sync_state[0].copy_(all_tokens_buf, non_blocking=True)
                sync_state[1].record(torch.cuda.current_stream(dev))
    else:
        if ep_size > 1:
            in_range = (flat_ids >= e_lo) & (flat_ids < e_lo + E_local)
            flat_local = torch.where(
                in_range, flat_ids - e_lo, torch.full_like(flat_ids, E_local)
            )
        else:
            flat_local = flat_ids
        counts = _gpu_expert_bincount(flat_local, E_local)
        aligned_counts = ((counts + alignment - 1) // alignment) * alignment
        offsets = torch.cumsum(aligned_counts, dim=0) - aligned_counts
        all_tokens_buf = None

    # Overlap independent weight lookup, output slot alloc, and FP8 quant with D2H before waiting.
    (w13_i8, w13_s), (w2_i8, w2_s) = weights.get_deepgemm_pair()
    if prequant_fp8 is not None:
        x_q, x_sc_raw = prequant_fp8
    else:
        assert x is not None
        x_q, x_sc_raw = per_token_group_quant_fp8_ue8m0(x, scale_format="raw")
    num_groups = L // 32
    num_packs = num_groups // 4
    M_total = M * top_k
    dst_slots = torch.empty((M, top_k), dtype=torch.int32, device=dev)

    if use_cap_nosync:
        all_tokens = (
            (m_total_slots + E_local * (alignment - 1) + alignment - 1) // alignment
        ) * alignment
        active_rows_t: Optional[torch.Tensor] = torch.empty((1,), dtype=torch.int32, device=dev)
        grid_init = (E_local + 32,)
    else:
        if sync_state is not None:
            with _cuda_device_ctx(dev):
                sync_state[1].synchronize()
            all_tokens = int(sync_state[0][0])
        else:
            all_tokens = int(
                all_tokens_buf.item() if all_tokens_buf is not None else aligned_counts.sum().item()
            )
        if all_tokens == 0:
            return torch.zeros((M, L), dtype=res_dtype, device=dev)
        active_rows_t = None
        grid_init = (E_local,)

    a_fp8 = torch.empty((all_tokens, L), dtype=torch.float8_e4m3fn, device=dev)
    tma_m = ((all_tokens + 3) // 4) * 4
    a_scale = torch.empty((num_packs, tma_m), dtype=torch.int32, device=dev).T[:all_tokens]

    m_indices = torch.empty((all_tokens,), dtype=torch.int32, device=dev)
    write_cnt = (
        slab_write_cnt
        if slab_write_cnt is not None
        else torch.empty((E_local,), dtype=torch.int32, device=dev)
    )
    _init_m_indices_kernel[grid_init](
        counts,
        aligned_counts,
        offsets,
        m_indices,
        write_cnt,
        active_rows_t if active_rows_t is not None else write_cnt,
        all_tokens,
        num_experts=E_local,
        FILL_TAIL=use_cap_nosync,
        BLOCK=256,
        num_warps=4,
    )

    _scatter_fp8_to_contig_kernel[(M_total,)](
        x_q.view(torch.uint8),
        x_sc_raw,
        ids_c,
        offsets,
        write_cnt,
        a_fp8.view(torch.uint8),
        a_scale,
        dst_slots.view(-1),
        M_total,
        top_k,
        L,
        num_groups,
        num_packs,
        e_lo,
        E_local,
        x_q.stride(0),
        x_sc_raw.stride(0),
        a_fp8.stride(0),
        a_scale.stride(1),
        BLOCK_K=512,
        num_warps=4,
    )

    gate_up = torch.empty((all_tokens, 2 * I), dtype=torch.bfloat16, device=dev)
    deep_gemm.m_grouped_fp8_fp4_gemm_nt_contiguous(
        (a_fp8, a_scale),
        (w13_i8, w13_s),
        gate_up,
        m_indices,
        recipe_a=(1, 32),
        recipe_b=(1, 32),
    )
    del a_fp8, a_scale

    h_fp8, h_scale = situ_glu_fp8_quant_packed(
        gate_up,
        gate_beta=gate_beta,
        up_beta=up_beta,
        scale_format="deepgemm",
        zero_unused_scales=False,
        max_active_rows=active_rows_t,
    )
    del gate_up

    down_out = torch.empty((all_tokens, L), dtype=torch.bfloat16, device=dev)
    deep_gemm.m_grouped_fp8_fp4_gemm_nt_contiguous(
        (h_fp8, h_scale),
        (w2_i8, w2_s),
        down_out,
        m_indices,
        recipe_a=(1, 32),
        recipe_b=(1, 32),
    )
    del h_fp8, h_scale

    out = torch.empty((M, L), dtype=res_dtype, device=dev)
    BLOCK_K = 512 if L >= 512 else triton.next_power_of_2(max(L, 64))
    w_f = (
        topk_weights
        if (topk_weights.dtype == torch.float32 and topk_weights.is_contiguous())
        else topk_weights.float().contiguous()
    )
    _gather_slot_weighted_reduce_kernel[(M, triton.cdiv(L, BLOCK_K))](
        down_out,
        dst_slots,
        w_f,
        out,
        L,
        down_out.stride(0),
        w_f.stride(0),
        out.stride(0),
        TOP_K=top_k,
        BLOCK_K=BLOCK_K,
        num_warps=4,
    )
    return out


@triton.jit(
    do_not_specialize=[
        "M",
        "e_lo",
        "E_local",
        "stride_xm",
        "stride_aqe",
        "stride_aqm",
        "stride_asce",
        "stride_asck",
    ]
)
def _quant_and_scatter_fp8_to_masked_kernel(
    x_ptr,
    ids_ptr,
    masked_m_ptr,
    a_q_ptr,
    a_sc_ptr,
    dst_pos_ptr,
    M,
    TOP_K: tl.constexpr,
    TOP_K_PAD: tl.constexpr,
    L: tl.constexpr,
    e_lo,
    E_local,
    stride_xm: tl.int64,
    stride_aqe: tl.int64,
    stride_aqm: tl.int64,
    stride_asce: tl.int64,
    stride_asck: tl.int64,
    eps: tl.constexpr,
    BLOCK_L: tl.constexpr,
):
    """Fused block-32 UE8M0 FP8 quant + scatter to DeepGEMM masked expert layout."""
    m = tl.program_id(0).to(tl.int64)
    if m >= M:
        return
    slots_k = tl.arange(0, TOP_K_PAD)
    mk = slots_k < TOP_K
    e_raw = tl.load(ids_ptr + m * TOP_K + slots_k, mask=mk, other=-1).to(tl.int32)
    le = e_raw - e_lo
    valid_e = mk & (le >= 0) & (le < E_local)
    pos = tl.where(valid_e, tl.atomic_add(masked_m_ptr + tl.where(valid_e, le, 0), 1, mask=valid_e), -1)
    tl.store(dst_pos_ptr + m * TOP_K + slots_k, pos, mask=mk)
    # Sync CTA so Warp 0 global writes to dst_pos are visible to all warps before loading pk_slot.
    tl.debug_barrier()
    if tl.sum((pos >= 0).to(tl.int32), axis=0) == 0:
        return

    TILES: tl.constexpr = L // BLOCK_L
    GROUPS: tl.constexpr = BLOCK_L // 32
    PACKS: tl.constexpr = GROUPS // 4
    shifts = (tl.arange(0, 4)[None, :] * 8).to(tl.uint32)
    xr = x_ptr + m * stride_xm
    for t in tl.static_range(TILES):
        ol = t * BLOCK_L + tl.arange(0, BLOCK_L)
        xv = tl.load(xr + ol).to(tl.float32)
        q, biased = _compute_ue8m0_fp8_block32(xv.reshape((GROUPS, 32)), eps=eps)
        qf_u8 = q.reshape((BLOCK_L,)).to(tl.uint8, bitcast=True)
        pk = tl.sum(
            biased.to(tl.uint8).to(tl.uint32).reshape((PACKS, 4)) << shifts,
            axis=1,
        ).to(tl.int32)
        po = t * PACKS + tl.arange(0, PACKS)
        for k in range(TOP_K):
            ek = tl.load(ids_ptr + m * TOP_K + k).to(tl.int32) - e_lo
            pk_slot = tl.load(dst_pos_ptr + m * TOP_K + k)
            ok = (
                (pk_slot >= 0)
                & (pk_slot.to(tl.int64) * stride_aqm < stride_aqe)
                & (ek >= 0)
                & (ek < E_local)
            )
            ek_s = tl.where(ok, ek, 0).to(tl.int64)
            pk_s = tl.where(ok, pk_slot, 0).to(tl.int64)
            tl.store(
                a_q_ptr + ek_s * stride_aqe + pk_s * stride_aqm + ol,
                qf_u8,
                mask=ok,
            )
            tl.store(
                a_sc_ptr + ek_s * stride_asce + po.to(tl.int64) * stride_asck + pk_s,
                pk,
                mask=ok,
            )


@triton.jit(do_not_specialize=["stride_q_row", "stride_sc_row", "stride_aqe", "stride_aqm", "stride_asce", "stride_asck"])
def _scatter_fp8_to_masked_kernel(
    x_q_ptr,
    x_sc_ptr,
    topk_ids_ptr,
    masked_m_ptr,
    a_q_ptr,
    a_sc_ptr,
    dst_pos_ptr,
    M_total,
    top_k,
    K,
    num_groups,
    num_packs,
    ep_start,
    num_experts,
    stride_q_row: tl.int64,
    stride_sc_row: tl.int64,
    stride_aqe: tl.int64,
    stride_aqm: tl.int64,
    stride_asce: tl.int64,
    stride_asck: tl.int64,
    BLOCK_K: tl.constexpr,
):
    idx = tl.program_id(0).to(tl.int64)
    if idx >= M_total:
        return
    tok = idx // top_k
    gid = tl.load(topk_ids_ptr + idx).to(tl.int32)
    local_e = gid - ep_start
    if local_e < 0 or local_e >= num_experts:
        tl.store(dst_pos_ptr + idx, -1)
        return

    pos = tl.atomic_add(masked_m_ptr + local_e, 1).to(tl.int64)
    tl.store(dst_pos_ptr + idx, pos.to(tl.int32))
    e64 = local_e.to(tl.int64)

    dst_q_base = a_q_ptr + e64 * stride_aqe + pos * stride_aqm
    for k0 in range(0, K, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        mk = offs_k < K
        vals = tl.load(x_q_ptr + tok * stride_q_row + offs_k, mask=mk, other=0)
        tl.store(dst_q_base + offs_k, vals, mask=mk)

    PACKS_BLK: tl.constexpr = BLOCK_K // 128
    dst_sc_base = a_sc_ptr + e64 * stride_asce + pos
    src_sc_base = x_sc_ptr + tok * stride_sc_row
    for p0 in range(0, num_packs, PACKS_BLK):
        offs_p = p0 + tl.arange(0, PACKS_BLK)
        mp = offs_p < num_packs
        packed = tl.zeros((PACKS_BLK,), dtype=tl.uint32)
        for b_idx in tl.static_range(4):
            g = offs_p * 4 + b_idx
            v = tl.load(src_sc_base + g, mask=mp & (g < num_groups), other=0).to(tl.uint32)
            packed |= v << (b_idx * 8)
        tl.store(
            dst_sc_base + offs_p.to(tl.int64) * stride_asck,
            packed.to(tl.int32),
            mask=mp,
        )


@triton.jit(
    do_not_specialize=[
        "M",
        "e_lo",
        "E_local",
        "sc_off",
        "meta_off",
        "row_bytes",
        "stride_aqe",
        "stride_aqm",
        "stride_asce",
        "stride_asck",
    ]
)
def _scatter_packed_fp8_to_masked_kernel(
    all_buf_ptr,
    masked_m_ptr,
    a_q_ptr,
    a_sc_ptr,
    dst_pos_ptr,
    M,
    sc_off: tl.int64,
    meta_off: tl.int64,
    row_bytes: tl.int64,
    e_lo,
    E_local,
    stride_aqe: tl.int64,
    stride_aqm: tl.int64,
    stride_asce: tl.int64,
    stride_asck: tl.int64,
    TOP_K: tl.constexpr,
    TOP_K_PAD: tl.constexpr,
    L: tl.constexpr,
    BLOCK_L: tl.constexpr,
):
    """Fused AllToAll unpack + DeepGEMM masked expert scatter (single read of all_buf)."""
    m = tl.program_id(0).to(tl.int64)
    if m >= M:
        return
    in_row = all_buf_ptr + m * row_bytes
    meta_i32 = (in_row + meta_off).to(tl.pointer_type(tl.int32))
    slots_k = tl.arange(0, TOP_K_PAD)
    mk = slots_k < TOP_K
    e_raw = tl.load(meta_i32 + slots_k, mask=mk, other=-1).to(tl.int32)
    le = e_raw - e_lo
    valid_e = mk & (le >= 0) & (le < E_local)
    pos = tl.where(valid_e, tl.atomic_add(masked_m_ptr + tl.where(valid_e, le, 0), 1, mask=valid_e), -1)
    tl.store(dst_pos_ptr + m * TOP_K + slots_k, pos, mask=mk)
    # Sync CTA so Warp 0 global writes to dst_pos are visible to all warps before loading pk_slot.
    tl.debug_barrier()
    if tl.sum((pos >= 0).to(tl.int32), axis=0) == 0:
        return

    TILES: tl.constexpr = L // BLOCK_L
    GROUPS: tl.constexpr = BLOCK_L // 32
    PACKS: tl.constexpr = GROUPS // 4
    shifts = (tl.arange(0, 4)[None, :] * 8).to(tl.uint32)
    for t in tl.static_range(TILES):
        ol = t * BLOCK_L + tl.arange(0, BLOCK_L)
        og = t * GROUPS + tl.arange(0, GROUPS)
        qf_u8 = tl.load(in_row + ol)
        sc_u8 = tl.load(in_row + sc_off + og).to(tl.uint32)
        pk = tl.sum(sc_u8.reshape((PACKS, 4)) << shifts, axis=1).to(tl.int32)
        po = t * PACKS + tl.arange(0, PACKS)
        for k in range(TOP_K):
            ek = tl.load(meta_i32 + k).to(tl.int32) - e_lo
            pk_slot = tl.load(dst_pos_ptr + m * TOP_K + k)
            ok = (
                (pk_slot >= 0)
                & (pk_slot.to(tl.int64) * stride_aqm < stride_aqe)
                & (ek >= 0)
                & (ek < E_local)
            )
            ek_s = tl.where(ok, ek, 0).to(tl.int64)
            pk_s = tl.where(ok, pk_slot, 0).to(tl.int64)
            tl.store(
                a_q_ptr + ek_s * stride_aqe + pk_s * stride_aqm + ol,
                qf_u8,
                mask=ok,
            )
            tl.store(
                a_sc_ptr + ek_s * stride_asce + po.to(tl.int64) * stride_asck + pk_s,
                pk,
                mask=ok,
            )


@triton.jit
def _gather_masked_weighted_reduce_kernel(
    down_ptr,
    topk_ids_ptr,
    dst_pos_ptr,
    topk_weights_ptr,
    all_buf_ptr,
    scale_gate_ptr,
    out_ptr,
    K,
    ep_start,
    num_experts,
    meta_off: tl.int64,
    row_bytes: tl.int64,
    stride_de: tl.int64,
    stride_dm: tl.int64,
    stride_wm,
    stride_sg: tl.int64,
    stride_om: tl.int64,
    scale_gate_col: tl.constexpr,
    PACKED_META: tl.constexpr,
    HAS_SCALE_GATE: tl.constexpr,
    TOP_K: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    tok = tl.program_id(0).to(tl.int64)
    pid_k = tl.program_id(1)
    offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    mask_k = offs_k < K

    acc = tl.zeros((BLOCK_K,), dtype=tl.float32)
    pos_base = dst_pos_ptr + tok * TOP_K
    if PACKED_META:
        meta_ptr = all_buf_ptr + tok * row_bytes + meta_off
        id_base = meta_ptr.to(tl.pointer_type(tl.int32))
        w_base = (meta_ptr + TOP_K * 4).to(tl.pointer_type(tl.float32))
    else:
        id_base = topk_ids_ptr + tok * TOP_K
        w_base = topk_weights_ptr + tok * stride_wm
    for k_idx in tl.static_range(TOP_K):
        pos = tl.load(pos_base + k_idx).to(tl.int64)
        if pos >= 0:
            local_e = (tl.load(id_base + k_idx).to(tl.int32) - ep_start).to(tl.int64)
            w = tl.load(w_base + k_idx).to(tl.float32)
            if local_e >= 0 and local_e < num_experts and w != 0.0:
                v = tl.load(
                    down_ptr + local_e * stride_de + pos * stride_dm + offs_k,
                    mask=mask_k,
                    other=0.0,
                ).to(tl.float32)
                acc += v * w

    out_val = acc.to(out_ptr.dtype.element_ty)
    if HAS_SCALE_GATE:
        sg_raw = tl.load(scale_gate_ptr + tok * stride_sg + scale_gate_col).to(tl.float32)
        sig_sg = tl.div_rn(1.0, 1.0 + libdevice.exp(-sg_raw)).to(out_ptr.dtype.element_ty)
        out_val = (out_val.to(tl.float32) * sig_sg.to(tl.float32)).to(out_ptr.dtype.element_ty)

    tl.store(
        out_ptr + tok * stride_om + offs_k,
        out_val,
        mask=mask_k,
    )


def _unpack_packed_fp8_payload(
    packed: Tuple[torch.Tensor, int, int, int, int, int],
) -> Tuple[Tuple[torch.Tensor, torch.Tensor], torch.Tensor, torch.Tensor]:
    all_buf, L, sc_off, meta_off, row_bytes, top_k = packed
    M = int(all_buf.shape[0])
    num_groups = L // 32
    dev = all_buf.device
    xq = all_buf[:, :L].contiguous().view(torch.float8_e4m3fn)
    xsc = all_buf[:, sc_off : sc_off + num_groups].contiguous()
    meta_u8 = all_buf[:, meta_off : meta_off + top_k * 8].contiguous()
    ids = meta_u8[:, : top_k * 4].view(torch.int32).reshape(M, top_k)
    w = meta_u8[:, top_k * 4 :].view(torch.float32).reshape(M, top_k)
    return (xq, xsc), ids, w


def mxfp4_moe_deepgemm_masked(
    x: Optional[torch.Tensor],
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    weights: MXFP4MoEWeights,
    *,
    gate_beta: float = 4.0,
    up_beta: float = 25.0,
    ep_start: int = 0,
    ep_size: int = 1,
    prequant_fp8: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    prequant_packed: Optional[Tuple[torch.Tensor, int, int, int, int, int]] = None,
    scale_gate: Optional[torch.Tensor] = None,
    scale_gate_col: int = 0,
    out_dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """DeepGEMM SM100 ``m_grouped_fp8_fp4_gemm_nt_masked`` (100% CUDA-Graph Decode)."""
    import deep_gemm

    if prequant_packed is not None:
        all_buf, L, sc_off, meta_off, row_bytes, top_k = prequant_packed
        M = int(all_buf.shape[0])
        dev = all_buf.device
    else:
        x_ref = prequant_fp8[0] if prequant_fp8 is not None else x
        assert x_ref is not None
        M, L = x_ref.shape
        dev = x_ref.device
        top_k = int(topk_ids.shape[1])
        all_buf = None
        sc_off = 0
        meta_off = 0
        row_bytes = 0
    res_dtype = (
        out_dtype
        if out_dtype is not None
        else (x.dtype if x is not None else torch.bfloat16)
    )
    if M == 0:
        return torch.empty((0, L), dtype=res_dtype, device=dev)
    E_local = weights.num_experts
    I = weights.intermediate_size
    if L % 128 != 0 or I % 128 != 0:
        raise ValueError(f"DeepGEMM FP8xFP4 requires L={L}, I={I} divisible by 128")

    # DeepGEMM SM100 MGroupedMasked uses BLOCK_M=128 (alignment=128) TMA descriptors along M;
    # max_m must be aligned to get_mk_alignment_for_contiguous_layout() for exact M < 128 parity.
    align = int(deep_gemm.get_mk_alignment_for_contiguous_layout())
    max_m = ((M + align - 1) // align) * align
    total_e = E_local * max(int(ep_size), 1)
    expected_m = max(1, (M * top_k + total_e - 1) // total_e)
    e_lo = int(ep_start) if ep_size > 1 else 0

    (w13_i8, w13_s), (w2_i8, w2_s) = weights.get_deepgemm_pair()

    num_groups = L // 32
    num_packs = num_groups // 4
    num_groups_i = I // 32
    num_packs_i = (num_groups_i + 3) // 4
    tma_m_h = ((max_m + 3) // 4) * 4

    # Single reusable slab eliminates per-layer 100+ MiB allocator churn across all MoE layers.
    sz_mm = _align256(E_local * 4)
    sz_dp = _align256(M * top_k * 4)
    sz_a_q = _align256(E_local * max_m * L)
    sz_a_sc = _align256(E_local * num_packs * max_m * 4)
    sz_h_q = _align256(E_local * max_m * I)
    sz_h_sc = _align256(E_local * num_packs_i * tma_m_h * 4)
    sz_reg2 = max(sz_a_q + sz_a_sc, sz_h_q + sz_h_sc)
    sz_gu = _align256(E_local * max_m * 2 * I * 2)
    sz_dn = _align256(E_local * max_m * L * 2)
    sz_reg1 = max(sz_gu, sz_dn)
    off_dp = sz_mm
    off_reg2 = off_dp + sz_dp
    off_reg1 = off_reg2 + sz_reg2
    total_ws = off_reg1 + sz_reg1

    slab = _get_masked_ws_slab(dev, total_ws, cacheable=(max_m <= _MASKED_WS_MAX_M))
    masked_m = slab[: E_local * 4].view(torch.int32)
    masked_m.zero_()
    dst_pos = slab[off_dp : off_dp + M * top_k * 4].view(torch.int32).view(M, top_k)

    a_fp8 = (
        slab[off_reg2 : off_reg2 + E_local * max_m * L]
        .view(torch.float8_e4m3fn)
        .view(E_local, max_m, L)
    )
    a_scale = (
        slab[off_reg2 + sz_a_q : off_reg2 + sz_a_q + E_local * num_packs * max_m * 4]
        .view(torch.int32)
        .view(E_local, num_packs, max_m)
        .transpose(1, 2)
    )
    gate_up = (
        slab[off_reg1 : off_reg1 + E_local * max_m * 2 * I * 2]
        .view(torch.bfloat16)
        .view(E_local, max_m, 2 * I)
    )
    h_fp8 = (
        slab[off_reg2 : off_reg2 + E_local * max_m * I]
        .view(torch.float8_e4m3fn)
        .view(E_local, max_m, I)
    )
    h_scale = (
        slab[off_reg2 + sz_h_q : off_reg2 + sz_h_q + E_local * num_packs_i * tma_m_h * 4]
        .view(torch.int32)
        .view(E_local, num_packs_i, tma_m_h)
        .transpose(1, 2)[:, :max_m, :]
    )
    down_out = (
        slab[off_reg1 : off_reg1 + E_local * max_m * L * 2]
        .view(torch.bfloat16)
        .view(E_local, max_m, L)
    )

    if prequant_packed is not None:
        assert all_buf is not None
        block_l = 512 if (L % 512 == 0) else (256 if (L % 256 == 0) else 128)
        _scatter_packed_fp8_to_masked_kernel[(M,)](
            all_buf,
            masked_m,
            a_fp8.view(torch.uint8),
            a_scale,
            dst_pos.view(-1),
            M,
            sc_off,
            meta_off,
            row_bytes,
            e_lo,
            E_local,
            a_fp8.stride(0),
            a_fp8.stride(1),
            a_scale.stride(0),
            a_scale.stride(2),
            TOP_K=top_k,
            TOP_K_PAD=max(16, triton.next_power_of_2(top_k)),
            L=L,
            BLOCK_L=block_l,
            num_warps=4,
        )
        ids_c = all_buf
    elif prequant_fp8 is None and x is not None and M <= 2048:
        ids_c = topk_ids if topk_ids.is_contiguous() else topk_ids.contiguous()
        x_c = x if x.stride(-1) == 1 else x.contiguous()
        block_l = 512 if (L % 512 == 0) else (256 if (L % 256 == 0) else 128)
        _quant_and_scatter_fp8_to_masked_kernel[(M,)](
            x_c,
            ids_c,
            masked_m,
            a_fp8.view(torch.uint8),
            a_scale,
            dst_pos.view(-1),
            M,
            TOP_K=top_k,
            TOP_K_PAD=max(16, triton.next_power_of_2(top_k)),
            L=L,
            e_lo=e_lo,
            E_local=E_local,
            stride_xm=x_c.stride(0),
            stride_aqe=a_fp8.stride(0),
            stride_aqm=a_fp8.stride(1),
            stride_asce=a_scale.stride(0),
            stride_asck=a_scale.stride(2),
            eps=float(_QUANT_EPS),
            BLOCK_L=block_l,
            num_warps=4,
        )
    else:
        ids_c = topk_ids if topk_ids.is_contiguous() else topk_ids.contiguous()
        if prequant_fp8 is not None:
            x_q, x_sc_raw = prequant_fp8
        else:
            assert x is not None
            x_q, x_sc_raw = per_token_group_quant_fp8_ue8m0(x, scale_format="raw")
        M_total = M * top_k
        _scatter_fp8_to_masked_kernel[(M_total,)](
            x_q.view(torch.uint8),
            x_sc_raw,
            ids_c,
            masked_m,
            a_fp8.view(torch.uint8),
            a_scale,
            dst_pos.view(-1),
            M_total,
            top_k,
            L,
            num_groups,
            num_packs,
            e_lo,
            E_local,
            x_q.stride(0),
            x_sc_raw.stride(0),
            a_fp8.stride(0),
            a_fp8.stride(1),
            a_scale.stride(0),
            a_scale.stride(2),
            BLOCK_K=512,
            num_warps=4,
        )
        del x_q, x_sc_raw

    deep_gemm.m_grouped_fp8_fp4_gemm_nt_masked(
        (a_fp8, a_scale),
        (w13_i8, w13_s),
        gate_up,
        masked_m,
        expected_m,
        recipe_a=(1, 32),
        recipe_b=(1, 32),
    )

    situ_glu_fp8_quant_masked_packed(
        gate_up,
        masked_m,
        gate_beta=gate_beta,
        up_beta=up_beta,
        zero_unused_scales=False,
        out_q=h_fp8,
        out_sc=h_scale,
    )

    deep_gemm.m_grouped_fp8_fp4_gemm_nt_masked(
        (h_fp8, h_scale),
        (w2_i8, w2_s),
        down_out,
        masked_m,
        expected_m,
        recipe_a=(1, 32),
        recipe_b=(1, 32),
    )

    out = torch.empty((M, L), dtype=res_dtype, device=dev)
    BLOCK_K = 512 if L >= 512 else triton.next_power_of_2(max(L, 64))
    packed_meta = prequant_packed is not None
    if packed_meta:
        assert all_buf is not None
        ids_arg = all_buf
        w_arg = all_buf
        all_arg = all_buf
        w_stride = 0
    else:
        w_f = (
            topk_weights
            if (topk_weights.dtype == torch.float32 and topk_weights.is_contiguous())
            else topk_weights.float().contiguous()
        )
        ids_arg = ids_c
        w_arg = w_f
        all_arg = ids_c
        w_stride = int(w_f.stride(0))
    has_sg = scale_gate is not None
    if has_sg:
        assert scale_gate is not None
        sg_2d = scale_gate.reshape(M, -1) if scale_gate.ndim >= 2 else scale_gate.reshape(M, 1)
        if sg_2d.stride(-1) != 1:
            sg_2d = sg_2d.contiguous()
        sg_ptr: torch.Tensor = sg_2d
        sg_stride = int(sg_2d.stride(0))
        sg_col = int(scale_gate_col)
    else:
        sg_ptr = out
        sg_stride = 0
        sg_col = 0
    _gather_masked_weighted_reduce_kernel[(M, triton.cdiv(L, BLOCK_K))](
        down_out,
        ids_arg,
        dst_pos,
        w_arg,
        all_arg,
        sg_ptr,
        out,
        L,
        e_lo,
        E_local,
        meta_off,
        row_bytes,
        down_out.stride(0),
        down_out.stride(1),
        w_stride,
        sg_stride,
        out.stride(0),
        scale_gate_col=sg_col,
        PACKED_META=packed_meta,
        HAS_SCALE_GATE=has_sg,
        TOP_K=top_k,
        BLOCK_K=BLOCK_K,
        num_warps=4,
    )
    return out


@triton.jit(do_not_specialize=["stride_q_row", "stride_sc_row", "stride_out_q_row"])
def _scatter_fp8_to_flashinfer_kernel(
    x_q_ptr,
    x_sc_ptr,
    topk_ids_ptr,
    m_indptr_ptr,
    write_cnt_ptr,
    out_q_ptr,
    out_sc_ptr,
    dst_slots_ptr,
    M_total,
    top_k,
    K,
    num_groups,
    sf_cols_pad,
    ep_start,
    num_experts,
    stride_q_row: tl.int64,
    stride_sc_row: tl.int64,
    stride_out_q_row: tl.int64,
    BLOCK_K: tl.constexpr,
):
    idx = tl.program_id(0).to(tl.int64)
    if idx >= M_total:
        return
    tok = idx // top_k
    gid = tl.load(topk_ids_ptr + idx).to(tl.int32)
    local_e = gid - ep_start
    if local_e < 0 or local_e >= num_experts:
        tl.store(dst_slots_ptr + idx, -1)
        return

    pos = tl.atomic_add(write_cnt_ptr + local_e, 1).to(tl.int64)
    m_off = tl.load(m_indptr_ptr + local_e).to(tl.int64)
    slot = m_off + pos
    tl.store(dst_slots_ptr + idx, slot.to(tl.int32))

    for k0 in range(0, K, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        mk = offs_k < K
        vals = tl.load(x_q_ptr + tok * stride_q_row + offs_k, mask=mk, other=0)
        tl.store(out_q_ptr + slot * stride_out_q_row + offs_k, vals, mask=mk)

    sf_m_off = ((m_off + local_e * 127) // 128) * 128
    r_blk = pos // 128
    r_rem = pos % 128
    r4 = r_rem // 32
    r32 = r_rem % 32
    c_blocks = sf_cols_pad // 4

    for g0 in range(0, num_groups, 128):
        offs_g = g0 + tl.arange(0, 128)
        mg = offs_g < num_groups
        vals_sc = tl.load(x_sc_ptr + tok * stride_sc_row + offs_g, mask=mg, other=0)
        c_blk = offs_g // 4
        c4 = offs_g % 4
        sw_off = (
            (sf_m_off + r_blk * 128) * sf_cols_pad
            + ((c_blk * 32 + r32) * 4 + r4) * 4
            + c4
        )
        tl.store(out_sc_ptr + sw_off, vals_sc, mask=mg)


def mxfp4_moe_flashinfer(
    x: Optional[torch.Tensor],
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    weights: MXFP4MoEWeights,
    *,
    gate_beta: float = 4.0,
    up_beta: float = 25.0,
    ep_start: int = 0,
    ep_size: int = 1,
    swap_ab: bool = True,
    prequant_fp8: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    out_dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """FlashInfer ``group_gemm_mxfp4_nt_groupwise`` SM100/SM103 MXFP4 MoE backend."""
    import flashinfer.gemm as fi_gemm

    x_ref = prequant_fp8[0] if prequant_fp8 is not None else x
    assert x_ref is not None
    M, L = x_ref.shape
    dev = x_ref.device
    res_dtype = (
        out_dtype
        if out_dtype is not None
        else (x.dtype if x is not None else torch.bfloat16)
    )
    if M == 0:
        return torch.empty((0, L), dtype=res_dtype, device=dev)
    top_k = topk_ids.shape[1]
    E_local = weights.num_experts
    I = weights.intermediate_size
    if L % 128 != 0 or I % 128 != 0:
        raise ValueError(f"FlashInfer group_gemm_mxfp4 requires L={L}, I={I} divisible by 128")

    e_lo = int(ep_start) if ep_size > 1 else 0
    flat_ids = topk_ids.reshape(-1)
    if ep_size > 1:
        in_range = (flat_ids >= e_lo) & (flat_ids < e_lo + E_local)
        flat_local = torch.where(
            in_range, flat_ids - e_lo, torch.full_like(flat_ids, E_local)
        )
    else:
        flat_local = flat_ids
    counts = _gpu_expert_bincount(flat_local, E_local)
    aligned_counts = ((counts + 3) // 4) * 4
    m_indptr = torch.zeros((E_local + 1,), dtype=torch.int32, device=dev)
    torch.cumsum(aligned_counts, dim=0, out=m_indptr[1:])
    cum_m = int(m_indptr[-1].item())
    if cum_m == 0:
        return torch.zeros((M, L), dtype=res_dtype, device=dev)

    (w13_u8, w13_fi_s), (w2_u8, w2_fi_s) = weights.get_flashinfer_pair()

    if prequant_fp8 is not None:
        x_q, x_sc_raw = prequant_fp8
    else:
        assert x is not None
        x_q, x_sc_raw = per_token_group_quant_fp8_ue8m0(x, scale_format="raw")
    a_fp8 = torch.empty((cum_m, L), dtype=torch.float8_e4m3fn, device=dev)
    num_groups = L // 32
    sf_cols_pad = ((num_groups + 3) // 4) * 4
    total_sf_rows = ((cum_m + max(E_local - 1, 0) * 127 + 127) // 128) * 128
    a_scale = torch.empty((total_sf_rows * sf_cols_pad,), dtype=torch.uint8, device=dev)

    M_total = M * top_k
    write_cnt = torch.zeros((E_local,), dtype=torch.int32, device=dev)
    dst_slots = torch.empty((M, top_k), dtype=torch.int32, device=dev)
    _scatter_fp8_to_flashinfer_kernel[(M_total,)](
        x_q.view(torch.uint8),
        x_sc_raw,
        topk_ids.contiguous(),
        m_indptr,
        write_cnt,
        a_fp8.view(torch.uint8),
        a_scale,
        dst_slots.view(-1),
        M_total,
        top_k,
        L,
        num_groups,
        sf_cols_pad,
        e_lo,
        E_local,
        x_q.stride(0),
        x_sc_raw.stride(0),
        a_fp8.stride(0),
        BLOCK_K=512,
        num_warps=4,
    )

    gate_up = torch.empty((cum_m, 2 * I), dtype=torch.bfloat16, device=dev)
    fi_gemm.group_gemm_mxfp4_nt_groupwise(
        a=a_fp8,
        b=w13_u8,
        a_scale=a_scale,
        b_scale=w13_fi_s,
        m_indptr=m_indptr,
        swap_ab=swap_ab,
        out=gate_up,
        out_dtype=torch.bfloat16,
    )

    h_fp8, h_scale = situ_glu_fp8_quant_packed(
        gate_up,
        gate_beta=gate_beta,
        up_beta=up_beta,
        scale_format="flashinfer",
        m_indptr=m_indptr,
        zero_unused_scales=False,
    )

    down_out = torch.empty((cum_m, L), dtype=torch.bfloat16, device=dev)
    fi_gemm.group_gemm_mxfp4_nt_groupwise(
        a=h_fp8,
        b=w2_u8,
        a_scale=h_scale,
        b_scale=w2_fi_s,
        m_indptr=m_indptr,
        swap_ab=swap_ab,
        out=down_out,
        out_dtype=torch.bfloat16,
    )

    out = torch.empty((M, L), dtype=res_dtype, device=dev)
    BLOCK_K = 512 if L >= 512 else triton.next_power_of_2(max(L, 64))
    w_f = (
        topk_weights
        if (topk_weights.dtype == torch.float32 and topk_weights.is_contiguous())
        else topk_weights.float().contiguous()
    )
    _gather_slot_weighted_reduce_kernel[(M, triton.cdiv(L, BLOCK_K))](
        down_out,
        dst_slots,
        w_f,
        out,
        L,
        down_out.stride(0),
        w_f.stride(0),
        out.stride(0),
        TOP_K=top_k,
        BLOCK_K=BLOCK_K,
        num_warps=4,
    )
    return out


# ---------------------------------------------------------------------------
# Full 2-Stage Routed MXFP4 Experts Forward (Regime-Dispatched + Reference)
# ---------------------------------------------------------------------------


def resolve_moe_backend_for_shape(
    M: int,
    L: int,
    device: torch.device,
    weights: MXFP4MoEWeights,
    backend: str,
    use_triton: bool,
    compute_mode: str = "bf16",
) -> str:
    """Resolve active MXFP4 MoE backend without needing a live activation tensor."""
    if backend == "ref" or not use_triton or device.type != "cuda" or not supports_triton_mxfp4(device):
        return "ref"
    req = backend if backend != "auto" else os.environ.get("QWEN4_5_MXFP4_BACKEND", "auto")
    capturing = (
        torch.cuda.is_current_stream_capturing()
        or os.environ.get("RTP_LLM_CUDA_GRAPH_WARMUP_FORWARD") == "1"
    )
    I = weights.intermediate_size
    dev_idx = int(device.index or 0)
    dg_masked_ok = (
        not weights.weights_transposed
        and L % 128 == 0
        and I % 128 == 0
        and has_deepgemm_fp8_fp4(dev_idx)
    )

    if req != "auto":
        if capturing and req in ("deepgemm_contiguous", "flashinfer", "flashinfer_noswap"):
            # Contiguous / FlashInfer scatter allocates dynamic shapes via .item().
            return "deepgemm_masked" if dg_masked_ok else ("triton_splitk" if M <= 64 else "triton")
        return req

    cm_l = str(compute_mode).lower()
    if (
        cm_l in ("bf16_strict", "strict_bf16")
        or os.environ.get("QWEN4_5_MXFP4_STRICT_BF16", "0") == "1"
    ):
        return "triton_splitk" if M <= 64 else "triton"

    if M <= 64:
        dec_env = os.environ.get("QWEN4_5_MXFP4_DECODE_BACKEND", "auto")
        if dec_env == "auto":
            default_splitk_max = "32" if (L >= 1024 and I >= 1024) else "64"
            splitk_max = int(os.environ.get("QWEN4_5_MXFP4_SPLITK_MAX_M", default_splitk_max))
            return "deepgemm_masked" if (M > splitk_max and dg_masked_ok) else "triton_splitk"
        if dec_env == "sm100_winner":
            return "deepgemm_masked" if (M >= 2 and dg_masked_ok) else "triton_splitk"
        if capturing and dec_env in ("deepgemm_contiguous", "flashinfer", "flashinfer_noswap"):
            return "deepgemm_masked" if dg_masked_ok else "triton_splitk"
        if dec_env == "deepgemm_masked" and not dg_masked_ok:
            return "triton_splitk"
        if dec_env in ("flashinfer", "flashinfer_noswap"):
            fi_ok = (
                not weights.weights_transposed
                and L % 128 == 0
                and I % 128 == 0
                and has_flashinfer_mxfp4(dev_idx)
            )
            if not fi_ok:
                return "triton_splitk"
        return dec_env

    pref_env = os.environ.get("QWEN4_5_MXFP4_PREFILL_BACKEND", "auto")
    if pref_env == "auto" and capturing:
        return "deepgemm_masked" if dg_masked_ok else "triton"
    fi_ok = (
        not weights.weights_transposed
        and L % 128 == 0
        and I % 128 == 0
        and has_flashinfer_mxfp4(dev_idx)
    )
    if pref_env != "auto":
        if capturing and pref_env in ("deepgemm_contiguous", "flashinfer", "flashinfer_noswap"):
            return "deepgemm_masked" if dg_masked_ok else "triton"
        return pref_env
    if capturing:
        return "deepgemm_masked" if dg_masked_ok else "triton"
    # Empirical SM103 Pareto:
    #   - EP>=4 (E_local<=256): deepgemm_masked beats flashinfer up to M<=2048 (265 vs 457us),
    #     and FillFunctor-free deepgemm_contiguous beats flashinfer for M>2048 (731 vs 768us @ 8192).
    #   - EP=1..2 (E_local>256): deepgemm_masked for M<=1024, flashinfer for 1024<M<16384, contiguous for M>=16384.
    e_local = weights.num_experts
    default_masked_max = "2048" if e_local <= 256 else "1024"
    default_contig_min = "2049" if e_local <= 256 else "16384"
    masked_max_m = int(os.environ.get("QWEN4_5_MXFP4_MASKED_MAX_M", default_masked_max))
    contig_min_m = int(os.environ.get("QWEN4_5_MXFP4_CONTIG_MIN_M", default_contig_min))
    if M <= masked_max_m and dg_masked_ok:
        return "deepgemm_masked"
    if M >= contig_min_m and dg_masked_ok:
        return "deepgemm_contiguous"
    if fi_ok:
        return "flashinfer"
    if dg_masked_ok:
        return "deepgemm_contiguous"
    return "triton"


def _resolve_moe_backend(
    x: torch.Tensor,
    weights: MXFP4MoEWeights,
    backend: str,
    use_triton: bool,
    compute_mode: str = "bf16",
) -> str:
    return resolve_moe_backend_for_shape(
        int(x.shape[0]),
        int(x.shape[1]),
        x.device,
        weights,
        backend,
        use_triton,
        compute_mode=compute_mode,
    )


def mxfp4_moe_forward(
    x: Optional[torch.Tensor],
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    weights: MXFP4MoEWeights,
    *,
    gate_beta: float = 4.0,
    up_beta: float = 25.0,
    compute_mode: str = "bf16",
    use_triton: bool = True,
    ep_start: int = 0,
    ep_size: int = 1,
    backend: str = "auto",
    prequant_fp8: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    prequant_packed: Optional[Tuple[torch.Tensor, int, int, int, int, int]] = None,
    scale_gate: Optional[torch.Tensor] = None,
    scale_gate_col: int = 0,
    out_dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """Full 2-stage MXFP4 Routed Experts execution with regime dispatch.

    Supported ``backend`` values:
      - ``"auto"``: Regime-dispatched winner (``triton_splitk`` for Decode ``M <= 64``,
        or ``sm100_winner`` via ``QWEN4_5_MXFP4_DECODE_BACKEND=sm100_winner`` for
        ``triton_splitk`` at ``M<=2`` + ``deepgemm_masked`` at ``M>=3``; ``deepgemm_masked``
        for ``64 < M <= 256`` / CG capture; ``flashinfer`` -> ``deepgemm_contiguous`` ->
        ``triton`` for Prefill ``M > 256``).
      - ``"triton_splitk"``: Persistent Split-K Triton ``tl.dot_scaled`` Decode GEMV.
      - ``"deepgemm_contiguous"``: DeepGEMM ``m_grouped_fp8_fp4_gemm_nt_contiguous``.
      - ``"deepgemm_masked"``: DeepGEMM ``m_grouped_fp8_fp4_gemm_nt_masked``.
      - ``"flashinfer"`` / ``"flashinfer_noswap"``: FlashInfer ``group_gemm_mxfp4_nt_groupwise``.
      - ``"triton"``: Baseline grouped Triton ``tl.dot_scaled``.
      - ``"ref"``: Exact dequant-BF16 reference.
    """
    if prequant_packed is not None and x is None and prequant_fp8 is None:
        M = int(prequant_packed[0].shape[0])
        L = int(prequant_packed[1])
        dev = prequant_packed[0].device
    else:
        x_ref = prequant_fp8[0] if (x is None and prequant_fp8 is not None) else x
        assert x_ref is not None
        M, L = x_ref.shape
        dev = x_ref.device
    res_dtype = (
        out_dtype
        if out_dtype is not None
        else (x.dtype if x is not None else torch.bfloat16)
    )
    if M == 0:
        return torch.empty((0, L), dtype=res_dtype, device=dev)

    chosen = resolve_moe_backend_for_shape(
        M, L, dev, weights, backend, use_triton, compute_mode=compute_mode
    )
    if prequant_packed is not None and chosen != "deepgemm_masked":
        prequant_fp8, topk_ids, topk_weights = _unpack_packed_fp8_payload(prequant_packed)
        prequant_packed = None

    with _cuda_device_ctx(dev):
        out = _dispatch_mxfp4_moe_forward(
            chosen,
            x,
            topk_ids,
            topk_weights,
            weights,
            gate_beta=gate_beta,
            up_beta=up_beta,
            ep_start=ep_start,
            ep_size=ep_size,
            prequant_fp8=prequant_fp8,
            prequant_packed=prequant_packed,
            scale_gate=scale_gate if chosen == "deepgemm_masked" else None,
            scale_gate_col=scale_gate_col,
            res_dtype=res_dtype,
            M=M,
            L=L,
            compute_mode=compute_mode,
        )
        if scale_gate is not None and chosen != "deepgemm_masked" and out.shape[0] > 0:
            sg = scale_gate.reshape(M, -1)[:, scale_gate_col : scale_gate_col + 1]
            sig = torch.sigmoid(sg.float()).to(out.dtype)
            out = (out.float() * sig.float()).to(out.dtype)
        return out


def _dispatch_mxfp4_moe_forward(
    chosen: str,
    x: Optional[torch.Tensor],
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    weights: MXFP4MoEWeights,
    *,
    gate_beta: float,
    up_beta: float,
    ep_start: int,
    ep_size: int,
    prequant_fp8: Optional[Tuple[torch.Tensor, torch.Tensor]],
    prequant_packed: Optional[Tuple[torch.Tensor, int, int, int, int, int]] = None,
    scale_gate: Optional[torch.Tensor] = None,
    scale_gate_col: int = 0,
    res_dtype: torch.dtype = torch.bfloat16,
    M: int = 0,
    L: int = 0,
    compute_mode: str = "bf16",
) -> torch.Tensor:
    if chosen == "deepgemm_contiguous":
        return mxfp4_moe_deepgemm_contiguous(
            x,
            topk_ids,
            topk_weights,
            weights,
            gate_beta=gate_beta,
            up_beta=up_beta,
            ep_start=ep_start,
            ep_size=ep_size,
            prequant_fp8=prequant_fp8,
            out_dtype=res_dtype,
        )
    if chosen == "deepgemm_masked":
        return mxfp4_moe_deepgemm_masked(
            x,
            topk_ids,
            topk_weights,
            weights,
            gate_beta=gate_beta,
            up_beta=up_beta,
            ep_start=ep_start,
            ep_size=ep_size,
            prequant_fp8=prequant_fp8,
            prequant_packed=prequant_packed,
            scale_gate=scale_gate,
            scale_gate_col=scale_gate_col,
            out_dtype=res_dtype,
        )
    if chosen == "flashinfer":
        return mxfp4_moe_flashinfer(
            x,
            topk_ids,
            topk_weights,
            weights,
            gate_beta=gate_beta,
            up_beta=up_beta,
            ep_start=ep_start,
            ep_size=ep_size,
            swap_ab=True,
            prequant_fp8=prequant_fp8,
            out_dtype=res_dtype,
        )
    if chosen == "flashinfer_noswap":
        return mxfp4_moe_flashinfer(
            x,
            topk_ids,
            topk_weights,
            weights,
            gate_beta=gate_beta,
            up_beta=up_beta,
            ep_start=ep_start,
            ep_size=ep_size,
            swap_ab=False,
            prequant_fp8=prequant_fp8,
            out_dtype=res_dtype,
        )

    if x is None:
        assert prequant_fp8 is not None
        x = _dequant_fp8_ue8m0_ref(prequant_fp8[0], prequant_fp8[1], res_dtype)

    if chosen == "ref":
        return mxfp4_moe_forward_ref(
            x,
            topk_ids,
            topk_weights,
            weights,
            gate_beta=gate_beta,
            up_beta=up_beta,
            ep_start=ep_start,
            ep_size=ep_size,
        )
    if chosen == "triton_splitk":
        return mxfp4_moe_triton_splitk(
            x,
            topk_ids,
            topk_weights,
            weights,
            gate_beta=gate_beta,
            up_beta=up_beta,
            ep_start=ep_start,
            ep_size=ep_size,
        )
    if chosen != "triton":
        raise ValueError(f"Unknown MXFP4 MoE backend={chosen!r}")

    top_k = topk_ids.shape[1]
    num_experts = weights.num_experts
    two_I = weights.w13_scale.shape[1]
    BM, _, _, _, _ = _select_grouped_gemm_config(M, top_k, num_experts, two_I, L)
    schedule = build_moe_tile_schedule(
        topk_ids,
        num_experts,
        BM,
        unique_per_token=True,
        ep_start=ep_start,
        ep_size=ep_size,
    )

    gate_up = mxfp4_grouped_moe_gemm(
        x,
        weights.w13_u8,
        weights.w13_scale,
        schedule=schedule,
        gather_input=True,
        scatter_output=False,
        weights_transposed=weights.weights_transposed,
        compute_mode=compute_mode,
        use_triton=True,
    )
    act = situ_and_mul_triton(gate_up, gate_beta=gate_beta, up_beta=up_beta)
    down_out = mxfp4_grouped_moe_gemm(
        act,
        weights.w2_u8,
        weights.w2_scale,
        schedule=schedule,
        gather_input=False,
        scatter_output=True,
        weights_transposed=weights.weights_transposed,
        compute_mode=compute_mode,
        use_triton=True,
    )

    if ep_size > 1:
        ep_end = int(ep_start) + num_experts
        in_range = (topk_ids >= ep_start) & (topk_ids < ep_end)
        reduce_weights = torch.where(
            in_range, topk_weights, torch.zeros_like(topk_weights)
        )
    else:
        reduce_weights = topk_weights

    return moe_weighted_reduce_triton(down_out.view(M, top_k, L), reduce_weights)


def _dequant_fp8_ue8m0_ref(q: torch.Tensor, sc_u8: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    scale = torch.pow(2.0, sc_u8.to(torch.float32) - 127.0)
    return (q.float().reshape(q.shape[0], -1, 32) * scale.unsqueeze(-1)).reshape(q.shape).to(dtype)


def mxfp4_moe_forward_ref(
    x: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    weights: MXFP4MoEWeights,
    *,
    gate_beta: float = 4.0,
    up_beta: float = 25.0,
    ep_start: int = 0,
    ep_size: int = 1,
    fp8_activations: bool = False,
) -> torch.Tensor:
    """Exact dequant reference for MXFP4 Routed Experts (BF16 or FP8+UE8M0 activations)."""
    M, L = x.shape
    if M == 0:
        return torch.empty_like(x)
    x_eff = x
    if fp8_activations:
        x_q, x_s = per_token_group_quant_fp8_ue8m0_ref(x, scale_format="raw")
        x_eff = _dequant_fp8_ue8m0_ref(x_q, x_s, x.dtype)

    out_fp32 = torch.zeros((M, L), dtype=torch.float32, device=x.device)
    w13_nk = (
        weights.w13_u8.transpose(1, 2).contiguous()
        if weights.weights_transposed
        else weights.w13_u8
    )
    w2_nk = (
        weights.w2_u8.transpose(1, 2).contiguous()
        if weights.weights_transposed
        else weights.w2_u8
    )
    local_e_count = weights.num_experts
    ep_end = int(ep_start) + local_e_count if ep_size > 1 else local_e_count
    e_lo = int(ep_start) if ep_size > 1 else 0

    unique_experts = torch.unique(topk_ids).tolist()
    for e in unique_experts:
        if e < e_lo or e >= ep_end:
            continue
        local_e = e - e_lo
        tok_idx, k_idx = torch.where(topk_ids == e)
        if tok_idx.numel() == 0:
            continue
        w13_bf16 = dequantize_mxfp4_ref(
            w13_nk[local_e], weights.w13_scale[local_e], dtype=x.dtype
        )
        w2_bf16 = dequantize_mxfp4_ref(
            w2_nk[local_e], weights.w2_scale[local_e], dtype=x.dtype
        )

        x_e = x_eff.index_select(0, tok_idx)
        gate_up = F.linear(x_e, w13_bf16)
        act = situ_and_mul_ref(gate_up, gate_beta=gate_beta, up_beta=up_beta)
        if fp8_activations:
            a_q, a_s = per_token_group_quant_fp8_ue8m0_ref(act, scale_format="raw")
            act = _dequant_fp8_ue8m0_ref(a_q, a_s, x.dtype)
        down = F.linear(act, w2_bf16)

        w_e = topk_weights[tok_idx, k_idx].float().unsqueeze(-1)
        out_fp32.index_add_(0, tok_idx, down.float() * w_e)

    return out_fp32.to(x.dtype)


# ---------------------------------------------------------------------------
# Standalone SM100 Shootout Benchmark (for @infra-bench on P-001)
# ---------------------------------------------------------------------------


def bench_sm100_shootout(
    *,
    ep_size: int = 4,
    num_experts: int = 640,
    top_k: int = 10,
    latent_dim: int = 2560,
    intermediate_dim: int = 2560,
    m_values: Sequence[int] = (1, 4, 8, 16, 32, 64, 512, 2048),
    warmup: int = 8,
    iters: int = 30,
    device: Optional[torch.device] = None,
    group: Any = None,
) -> Dict[str, Any]:
    """Benchmark all available SM100 MXFP4 MoE backends (and AllReduce if distributed)
    across Decode & Prefill regimes to pick empirical winners per (M, K, N).
    """
    dev = device or torch.device("cuda", torch.cuda.current_device())
    cap = torch.cuda.get_device_capability(dev)
    dev_idx = int(dev.index or 0)
    local_e = num_experts // ep_size

    g = torch.Generator(device=dev)
    g.manual_seed(2026)
    w13_u8 = torch.randint(0, 256, (local_e, 2 * intermediate_dim, latent_dim // 2), dtype=torch.uint8, device=dev, generator=g)
    w13_sc = torch.randint(120, 124, (local_e, 2 * intermediate_dim, latent_dim // 32), dtype=torch.uint8, device=dev, generator=g)
    w2_u8 = torch.randint(0, 256, (local_e, latent_dim, intermediate_dim // 2), dtype=torch.uint8, device=dev, generator=g)
    w2_sc = torch.randint(120, 124, (local_e, latent_dim, intermediate_dim // 32), dtype=torch.uint8, device=dev, generator=g)
    weights = MXFP4MoEWeights(w13_u8=w13_u8, w13_scale=w13_sc, w2_u8=w2_u8, w2_scale=w2_sc)

    backends: List[str] = ["triton", "triton_splitk"]
    if has_deepgemm_fp8_fp4(dev_idx):
        weights.get_deepgemm_pair()
        backends.extend(["deepgemm_contiguous", "deepgemm_masked"])
    if has_flashinfer_mxfp4(dev_idx):
        weights.get_flashinfer_pair()
        backends.extend(["flashinfer", "flashinfer_noswap"])

    def _time_ms(fn) -> float:
        for _ in range(warmup):
            fn()
        torch.cuda.synchronize(dev)
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        for _ in range(iters):
            fn()
        e.record()
        torch.cuda.synchronize(dev)
        return float(s.elapsed_time(e) / iters)

    print(
        f"\n=== SM100 MXFP4 MoE Shootout ({torch.cuda.get_device_name(dev)}, SM{cap}, "
        f"E={num_experts}, EP={ep_size}, local_E={local_e}, L={latent_dim}, I={intermediate_dim}, top_k={top_k}) ==="
    )
    results: Dict[str, Any] = {"device": torch.cuda.get_device_name(dev), "sm": cap, "moe": {}, "allreduce": {}}

    for M in m_values:
        x = torch.randn((M, latent_dim), dtype=torch.bfloat16, device=dev, generator=g) * 0.5
        logits = torch.randn((M, num_experts), dtype=torch.float32, device=dev, generator=g)
        topk_ids = torch.topk(logits, k=top_k, dim=-1).indices.to(torch.int32)
        topk_weights = torch.softmax(logits.gather(-1, topk_ids.long()), dim=-1).to(torch.float32)

        if ep_size > 1 and int((topk_ids < local_e).sum().item()) == 0:
            topk_ids[0, 0] = 0
        y_bf16_ref = mxfp4_moe_forward_ref(x, topk_ids, topk_weights, weights, ep_start=0, ep_size=ep_size)
        y_fp8_ref = mxfp4_moe_forward_ref(x, topk_ids, topk_weights, weights, ep_start=0, ep_size=ep_size, fp8_activations=True)
        row: Dict[str, Any] = {}
        best_b, best_ms = None, 1e9
        for b in backends:
            if b == "deepgemm_masked" and M > 1024:
                continue
            if b == "triton_splitk" and M > 256:
                continue
            try:
                y_b = mxfp4_moe_forward(x, topk_ids, topk_weights, weights, ep_start=0, ep_size=ep_size, backend=b)
                cos_bf16 = F.cosine_similarity(y_b.float().reshape(-1), y_bf16_ref.float().reshape(-1), dim=0).item()
                cos_fp8 = F.cosine_similarity(y_b.float().reshape(-1), y_fp8_ref.float().reshape(-1), dim=0).item()
                ms = _time_ms(lambda b=b: mxfp4_moe_forward(x, topk_ids, topk_weights, weights, ep_start=0, ep_size=ep_size, backend=b))
                is_fp8 = b.startswith("deepgemm") or b.startswith("flashinfer")
                gate_cos = cos_fp8 if is_fp8 else cos_bf16
                row[b] = {"ms": round(ms, 4), "cos_bf16": round(cos_bf16, 6), "cos_fp8": round(cos_fp8, 6)}
                if gate_cos >= 0.999 and ms < best_ms:
                    best_ms = ms
                    best_b = b
            except Exception as exc:
                row[b] = {"error": str(exc)}
        row["winner"] = best_b
        results["moe"][int(M)] = row
        summary = " | ".join(
            f"{k}={v['ms']:.3f}ms(bf16={v['cos_bf16']:.4f},fp8={v['cos_fp8']:.5f})" if "ms" in v else f"{k}=ERR"
            for k, v in row.items() if k != "winner"
        )
        print(f"  M={M:4d}: {summary}  --> WINNER={best_b}")

    if group is not None:
        import torch.distributed as dist
        from rtp_llm.models_py.modules.qwen4_5.fast_comm import Qwen4_5FastAllReduce

        if dist.is_initialized() and dist.get_world_size(group) > 1:
            is_r0 = dist.get_rank(group) == 0
            if is_r0:
                print("\n--- AllReduce Shootout (Eager & CUDA Graph) ---")
            ar_backends = {}
            for ab, backend_arg, oneshot_arg in (
                ("nccl", "nccl", None),
                ("symm_mem", "symm_mem", None),
                ("flashinfer_oneshot", "flashinfer", True),
                ("flashinfer_twoshot", "flashinfer", False),
            ):
                inst = Qwen4_5FastAllReduce(
                    group, dev, backend=backend_arg, max_tokens=1024,
                    hidden_dims=(5120, 2560), use_oneshot=oneshot_arg,
                )
                if inst.active_backend == backend_arg:
                    ar_backends[ab] = inst
                else:
                    inst.destroy()
            ws_ar = dist.get_world_size(group)
            for H in (2560, 5120):
                for M in (1, 4, 8, 16, 32, 64, 128, 256, 512, 1024):
                    x_ar = torch.randn((M, H), dtype=torch.bfloat16, device=dev)
                    ar_row: Dict[str, Any] = {}
                    best_ab, best_ar_ms = None, 1e9
                    for ab, inst in ar_backends.items():
                        if ab == "flashinfer_twoshot" and M <= ws_ar:
                            continue
                        ms_e = _time_ms(lambda inst=inst: inst(x_ar))
                        for _ in range(3):
                            inst(x_ar)
                        torch.cuda.synchronize(dev)
                        g_ar = torch.cuda.CUDAGraph()
                        with torch.cuda.graph(g_ar):
                            inst(x_ar)
                        ms_cg = _time_ms(lambda g_ar=g_ar: g_ar.replay())
                        del g_ar
                        ar_row[ab] = {"eager_ms": round(ms_e, 4), "cg_ms": round(ms_cg, 4)}
                        if ms_cg < best_ar_ms:
                            best_ar_ms = ms_cg
                            best_ab = ab
                    ar_row["winner"] = best_ab
                    results["allreduce"][f"M{M}_H{H}"] = ar_row
                    if is_r0:
                        print(f"  H={H} M={M:4d}: {ar_row}")
            for inst in ar_backends.values():
                inst.destroy()

    return results


__all__ = [
    "CompactMoETileSchedule",
    "MXFP4MoEWeights",
    "MXFP4_BLOCK_SIZE",
    "MoETileSchedule",
    "bench_sm100_shootout",
    "build_moe_tile_schedule",
    "dequantize_mxfp4",
    "dequantize_mxfp4_ref",
    "has_deepgemm_fp8_fp4",
    "has_flashinfer_mxfp4",
    "moe_weighted_reduce_triton",
    "mxfp4_grouped_moe_gemm",
    "mxfp4_linear",
    "mxfp4_moe_deepgemm_contiguous",
    "mxfp4_moe_deepgemm_masked",
    "mxfp4_moe_flashinfer",
    "mxfp4_moe_forward",
    "mxfp4_moe_forward_ref",
    "mxfp4_moe_triton_splitk",
    "per_token_group_quant_fp8_ue8m0",
    "per_token_group_quant_fp8_ue8m0_ref",
    "prepare_fp4_weight_scale_for_deepgemm",
    "prepare_fp4_weight_scale_for_flashinfer",
    "prepare_mxfp4_moe_weights",
    "quantize_mxfp4",
    "quantize_mxfp8_e4m3",
    "supports_triton_mxfp4",
]
