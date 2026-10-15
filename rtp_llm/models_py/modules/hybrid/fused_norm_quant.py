"""Fused RMSNorm + FP8 UE8M0 per-token-group quantization (shared wrapper).

Reuses the DSV4 Triton kernel whose packed-scale output layout
(``[M, K/128/4]`` int32, column-major TMA-aligned) is exactly what
``CudaFp8DeepGEMMLinear.quantize_input`` produces with
``sgl_per_token_group_quant_fp8(..., column_major_scales=True,
scale_tma_aligned=True, scale_ue8m0=True)`` on SM100/SM120, so a
``forward_quantized`` consumer cannot tell the fused result from the
two-kernel chain.

This wrapper adds what the DSV4 entry point deliberately rejects: 2D inputs
whose rows are strided (``stride(1) == 1``, ``stride(0) >= N``). MlaAttention
feeds ``q_a_layernorm`` the strided split view of the fused QKV projection;
the kernel reads rows through ``x_stride_m``, so the strided view goes in
without a ``.contiguous()`` copy and the norm output still lands in a fresh
contiguous tensor (matching flashinfer's strided-input behavior).
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import triton

from rtp_llm.models_py.modules.dsv4._fused_rmsnorm_fp8_quant_triton import (
    _make_ue8m0_scale_like,
    _rmsnorm_fp8_quant_kernel,
)


def fused_rmsnorm_fp8_quant_ue8m0(
    x: torch.Tensor,
    weight: torch.Tensor,
    *,
    eps: float,
    group_size: int = 128,
    clamp_eps: float = 1.0e-4,
    out_norm: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return ``(norm_bf16, fp8, packed_ue8m0_scale)`` for a 2D BF16 input.

    ``x`` may be row-strided (``stride(1) == 1``); ``out_norm`` (when given)
    must be contiguous and is written in place — pass ``out_norm=x`` only when
    ``x`` itself is contiguous and dead afterwards.
    """
    if x.dim() != 2:
        raise ValueError(f"x must be [M,N], got {tuple(x.shape)}")
    if x.dtype != torch.bfloat16:
        raise ValueError(f"x must be bf16, got {x.dtype}")
    if weight.dtype != torch.bfloat16:
        raise ValueError(f"weight must be bf16, got {weight.dtype}")
    if not x.is_cuda or not weight.is_cuda:
        raise RuntimeError("fused_rmsnorm_fp8_quant_ue8m0 requires CUDA tensors")
    if x.stride(1) != 1 or x.stride(0) < x.shape[1]:
        raise ValueError(
            "x must have contiguous rows (stride(1)==1, stride(0)>=N), got "
            f"stride={x.stride()}"
        )
    m, n = x.shape
    if weight.shape != (n,) or not weight.is_contiguous():
        raise ValueError(f"weight must be contiguous [{n}], got {tuple(weight.shape)}")
    if n % group_size != 0:
        raise ValueError(f"N={n} must be divisible by group_size={group_size}")

    if out_norm is None:
        out_norm = torch.empty((m, n), device=x.device, dtype=torch.bfloat16)
    else:
        if out_norm.shape != x.shape:
            raise ValueError(
                f"out_norm must have shape {tuple(x.shape)}, got {tuple(out_norm.shape)}"
            )
        if out_norm.dtype != torch.bfloat16:
            raise ValueError(f"out_norm must be bf16, got {out_norm.dtype}")
        if out_norm.device != x.device:
            raise ValueError("out_norm must be on the same device as x")
        if not out_norm.is_contiguous():
            raise ValueError("out_norm must be contiguous")

    out_q = torch.empty((m, n), device=x.device, dtype=torch.float8_e4m3fn)
    out_scale = _make_ue8m0_scale_like((m, n), device=x.device, group_size=group_size)
    if m == 0:
        return out_norm, out_q, out_scale

    finfo = torch.finfo(torch.float8_e4m3fn)
    scale_packs = (n // group_size + 3) // 4
    block_n = triton.next_power_of_2(n)
    _rmsnorm_fp8_quant_kernel[(m,)](
        x,
        weight,
        out_norm,
        out_q,
        out_scale,
        m,
        x.stride(0),
        out_norm.stride(0),
        out_q.stride(0),
        out_scale.stride(1),
        EPS=eps,
        CLAMP_EPS=clamp_eps,
        FP8_MIN=finfo.min,
        FP8_MAX=finfo.max,
        N=n,
        BLOCK_N=block_n,
        GROUP_SIZE=group_size,
        SCALE_PACKS=scale_packs,
        num_warps=8 if block_n >= 4096 else 4,
        num_stages=2,
    )
    return out_norm, out_q, out_scale


def _deepgemm_backend(linear: torch.nn.Module):
    """Return the DeepGEMM FP8 inner linear, whether bare or wrapped.

    ``CudaFp8GEMMLinear`` wraps a DeepGEMM inner (and optionally a small-M
    FlashInfer backend) without forwarding ``quantize_input`` /
    ``forward_quantized``; unwrap it before the quantized-input contract.
    """
    inner = getattr(linear, "_deepgemm_linear", None)
    if inner is not None:
        return inner
    if hasattr(linear, "quantize_input") and hasattr(linear, "forward_quantized"):
        return linear
    return None


def linear_supports_quantized_input(
    linear: torch.nn.Module, n_cols: int, m_rows: Optional[int] = None
) -> bool:
    """Duck-typed check mirroring DSV4's ``_can_reuse_qkv_input_quant``.

    True when ``linear`` (or its DeepGEMM inner, when wrapped by
    ``CudaFp8GEMMLinear``) is an ue8m0 DeepGEMM FP8 linear whose ``K``
    matches the norm width, so the fused (fp8, packed_scale) pair can be
    handed to :func:`deepgemm_forward_quantized`.

    ``m_rows`` matters only for wrappers that keep a small-M FlashInfer
    backend: those dispatch ``m < FLASHINFER_M_THRESHOLD`` (32) inputs to
    FlashInfer, which consumes bf16 — so the quantized pair is only usable
    when DeepGEMM will actually run for this row count.
    """
    inner = _deepgemm_backend(linear)
    if inner is None:
        return False
    if getattr(inner, "scale_ue8m0", None) is not True:
        return False
    if getattr(inner, "K", None) != n_cols:
        return False
    if getattr(linear, "_flashinfer_linear", None) is not None:
        if m_rows is None:
            return False
        threshold = getattr(type(linear), "FLASHINFER_M_THRESHOLD", 32)
        if m_rows < threshold:
            return False
    return True


def deepgemm_forward_quantized(
    linear: torch.nn.Module,
    input_fp8: torch.Tensor,
    input_scales: torch.Tensor,
):
    """``forward_quantized`` on the DeepGEMM inner (unwrap-aware)."""
    inner = _deepgemm_backend(linear)
    if inner is None:
        raise ValueError(
            "linear does not expose a DeepGEMM quantized-input path"
        )
    return inner.forward_quantized(input_fp8, input_scales)


def ue8m0_deepgemm_active() -> bool:
    """True on devices whose DeepGEMM/router quant flavor is packed ue8m0.

    The fused kernel emits the packed ue8m0 layout only. On SM100/SM120 the
    router's ``_do_quant`` and the DeepGEMM linears both use it; on SM90 the
    router quantizes with ``trt_fp8_quantize_128`` (fp32 scales), so the MoE
    fusion must stay off there.
    """
    try:
        from rtp_llm.models_py.kernels.cuda.deepgemm_wrapper import (
            is_deep_gemm_e8m0_used,
        )

        return bool(is_deep_gemm_e8m0_used())
    except Exception:
        return False
