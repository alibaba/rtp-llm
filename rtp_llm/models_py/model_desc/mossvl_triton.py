"""MOSS-VL cross-attention Triton fused kernels.

Eliminates native torch elementwise kernel launches from the cross-attn path
(timeline shows 216 vectorized_elementwise calls ≈ 4% GPU time in decode):

1. fused_gated_residual_add: `residual + gate * x` in one launch
   (replaces multiply + add = 2 native kernels per call)
2. fused_mrope_rope: interleaved MRoPE rotation in one launch
   (replaces mul + cat/neg + mul + add ≈ 5 native kernels per call)
3. fused_qk_rmsnorm: per-head RMSNorm in one launch
   (replaces pow + mean + rsqrt + mul ≈ 4 native kernels per call)

Each kernel has a `_supported()` gate and a pure-torch fallback; the fused
path activates only on CUDA + contiguous + bf16/fp16 inputs.
"""

from __future__ import annotations

import os
from typing import Optional, Tuple

import torch
import triton
import triton.language as tl


def _triton_enabled() -> bool:
    return os.environ.get("RTP_MOSSVL_TRITON", "1") == "1"


def _supported(x: torch.Tensor) -> bool:
    return (
        _triton_enabled()
        and x.is_cuda
        and x.is_contiguous()
        and x.dtype in (torch.float16, torch.bfloat16)
    )


# ---------------------------------------------------------------------------
#  1. Fused gated residual add:  out = residual + gate * x
# ---------------------------------------------------------------------------


@triton.jit
def _gated_residual_add_kernel(
    residual_ptr,
    x_ptr,
    gate_ptr,
    out_ptr,
    n_elements,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    residual = tl.load(residual_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    gate = tl.load(gate_ptr).to(tl.float32)
    out = residual + gate * x
    tl.store(out_ptr + offsets, out, mask=mask)


def fused_gated_residual_add(
    residual: torch.Tensor, x: torch.Tensor, gate: torch.Tensor
) -> Optional[torch.Tensor]:
    """out = residual + gate * x (fused single kernel)."""
    if not (_supported(residual) and _supported(x)):
        return None
    if residual.shape != x.shape:
        return None
    out = torch.empty_like(x)
    n = x.numel()
    BLOCK = 1024
    grid = (triton.cdiv(n, BLOCK),)
    _gated_residual_add_kernel[grid](residual, x, gate, out, n, BLOCK)
    return out


# ---------------------------------------------------------------------------
#  2. Fused interleaved MRoPE rope:  out = x*cos + rotate_half(x)*sin
#     x: [..., head_dim]; cos/sin: [..., head_dim]
#     rotate_half(x) = cat(-x[half:], x[:half])
# ---------------------------------------------------------------------------


@triton.jit
def _mrope_rope_kernel(
    x_ptr,
    cos_ptr,
    sin_ptr,
    out_ptr,
    n_heads_total,  # total (seq*heads) rows
    head_dim: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
):
    pid = tl.program_id(0)
    row_start = pid * BLOCK_ROWS
    row_offs = row_start + tl.arange(0, BLOCK_ROWS)
    row_mask = row_offs < n_heads_total
    half = head_dim // 2

    # process first half: out[i] = x[i]*cos[i] + (-x[half+i])*sin[i]
    for d in range(0, half, 64):
        d_offs = d + tl.arange(0, 64)
        d_mask = d_offs < half
        base = row_offs[:, None] * head_dim + d_offs[None, :]
        m = row_mask[:, None] & d_mask[None, :]
        x_first = tl.load(x_ptr + base, mask=m, other=0.0).to(tl.float32)
        x_second = tl.load(x_ptr + base + half, mask=m, other=0.0).to(tl.float32)
        cos_v = tl.load(cos_ptr + base, mask=m, other=0.0).to(tl.float32)
        sin_v = tl.load(sin_ptr + base, mask=m, other=0.0).to(tl.float32)
        out_first = x_first * cos_v - x_second * sin_v
        tl.store(out_ptr + base, out_first, mask=m)

    # process second half: out[half+i] = x[half+i]*cos[half+i] + x[i]*sin[half+i]
    for d in range(0, half, 64):
        d_offs = d + tl.arange(0, 64)
        d_mask = d_offs < half
        base = row_offs[:, None] * head_dim + d_offs[None, :]
        m = row_mask[:, None] & d_mask[None, :]
        x_first = tl.load(x_ptr + base, mask=m, other=0.0).to(tl.float32)
        x_second = tl.load(x_ptr + base + half, mask=m, other=0.0).to(tl.float32)
        cos_v = tl.load(cos_ptr + base + half, mask=m, other=0.0).to(tl.float32)
        sin_v = tl.load(sin_ptr + base + half, mask=m, other=0.0).to(tl.float32)
        out_second = x_second * cos_v + x_first * sin_v
        tl.store(out_ptr + base + half, out_second, mask=m)


def fused_mrope_rope(
    x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> Optional[torch.Tensor]:
    """Apply interleaved MRoPE rotation: x*cos + rotate_half(x)*sin."""
    if not _supported(x):
        return None
    if x.dim() < 2 or x.shape[-1] < 2:
        return None
    if not (cos.is_cuda and sin.is_cuda):
        return None
    head_dim = x.shape[-1]
    n_rows = x.numel() // head_dim
    out = torch.empty_like(x)
    BLOCK_ROWS = 4
    grid = (triton.cdiv(n_rows, BLOCK_ROWS),)
    # flatten to [n_rows, head_dim] view
    x_flat = x.reshape(n_rows, head_dim)
    cos_flat = cos.reshape(n_rows, head_dim) if cos.numel() == x.numel() else cos
    sin_flat = sin.reshape(n_rows, head_dim) if sin.numel() == x.numel() else sin
    if cos_flat.shape[0] != n_rows or sin_flat.shape[0] != n_rows:
        return None
    _mrope_rope_kernel[grid](
        x_flat, cos_flat, sin_flat, out.reshape(n_rows, head_dim),
        n_rows, head_dim, BLOCK_ROWS,
    )
    return out


# ---------------------------------------------------------------------------
#  3. Fused QK RMSNorm: per-head RMS normalization
#     x: [..., head_dim]; weight: [head_dim]
#     out = weight * x * rsqrt(mean(x^2) + eps)
# ---------------------------------------------------------------------------


@triton.jit
def _qk_rmsnorm_kernel(
    x_ptr,
    weight_ptr,
    out_ptr,
    n_rows,
    head_dim: tl.constexpr,
    eps,
    BLOCK_ROWS: tl.constexpr,
    BLOCK_DIM: tl.constexpr,
):
    pid = tl.program_id(0)
    row_start = pid * BLOCK_ROWS
    row_offs = row_start + tl.arange(0, BLOCK_ROWS)
    row_mask = row_offs < n_rows
    d_offs = tl.arange(0, BLOCK_DIM)
    d_mask = d_offs < head_dim

    base = row_offs[:, None] * head_dim + d_offs[None, :]
    m = row_mask[:, None] & d_mask[None, :]

    x = tl.load(x_ptr + base, mask=m, other=0.0).to(tl.float32)
    variance = tl.sum(x * x, axis=1) / head_dim
    rstd = 1.0 / tl.sqrt(variance + eps)
    weight = tl.load(weight_ptr + d_offs, mask=d_mask, other=0.0).to(tl.float32)
    out = x * rstd[:, None] * weight[None, :]
    tl.store(out_ptr + base, out, mask=m)


def fused_qk_rmsnorm(
    x: torch.Tensor, weight: torch.Tensor, eps: float
) -> Optional[torch.Tensor]:
    """Per-head RMSNorm: weight * x * rsqrt(mean(x^2) + eps)."""
    if not _supported(x):
        return None
    if x.shape[-1] > 4096:
        return None  # Triton block limit
    head_dim = x.shape[-1]
    n_rows = x.numel() // head_dim
    out = torch.empty_like(x)
    BLOCK_ROWS = 4
    BLOCK_DIM = triton.next_power_of_2(head_dim)
    grid = (triton.cdiv(n_rows, BLOCK_ROWS),)
    x_flat = x.reshape(n_rows, head_dim)
    _qk_rmsnorm_kernel[grid](
        x_flat, weight, out.reshape(n_rows, head_dim),
        n_rows, head_dim, eps, BLOCK_ROWS, BLOCK_DIM,
    )
    return out
