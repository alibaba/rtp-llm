"""Fused RMSNorm + FP8 UE8M0 per-token-group quantization (shared wrapper).

The packed-scale output layout (``[M, K/128/4]`` int32, column-major
TMA-aligned, produced by ``_make_ue8m0_scale_like``) is exactly what
``CudaFp8DeepGEMMLinear.quantize_input`` produces with
``sgl_per_token_group_quant_fp8(..., column_major_scales=True,
scale_tma_aligned=True, scale_ue8m0=True)`` on SM100/SM120, so a
``forward_quantized`` consumer cannot tell the fused result from the
two-kernel chain.

Kernel design (register-resident, single read): the whole H-wide row
(``BLOCK_H = next_pow_2(N) <= 8192``) is loaded once and stays in registers
across all three phases — RMS sum, normed output, and the group-scale
quantization, whose per-group ``absmax`` comes from a ``tl.reshape`` to
``[GROUPS, GROUP_SIZE]`` instead of re-reading x per group. The DSV4
one-CTA-per-row kernel this replaces re-read the row inside its SCALE_PACKS
loop and measured net-negative on Xing4 shapes at every M (decode latency
floor 14.7 us/launch vs ~4.6 us for the separate norm+quant pair); the
register-resident layout is the single-read discipline the
``xing4/hc/mean_norm_triton.py`` kernel validated. Dead pad lanes (BLOCK_H >
N) and their dead scale groups only ever hold deterministic values that no
consumer reads (DeepGEMM reads the ``N // 128`` real groups only).

The wrapper accepts 2D inputs whose rows are strided (``stride(1) == 1``,
``stride(0) >= N``): MlaAttention feeds ``q_a_layernorm`` the strided split
view of the fused QKV projection; the kernel reads rows through
``x_stride_m``, so the strided view goes in without a ``.contiguous()`` copy
and the norm output still lands in a fresh contiguous tensor (matching
flashinfer's strided-input behavior).
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import triton
import triton.language as tl

from rtp_llm.models_py.modules.dsv4._fused_rmsnorm_fp8_quant_triton import (
    _make_ue8m0_scale_like,
)

_MAX_BLOCK_H = 8192


@triton.jit
def _rmsnorm_fp8_quant_reg_kernel(
    x_ptr,
    w_ptr,
    out_norm_ptr,
    out_q_ptr,
    out_scale_ptr,
    eps,
    x_stride_m,
    out_norm_stride_m,
    out_q_stride_m,
    out_scale_stride_k,
    FP8_MIN: tl.constexpr,
    FP8_MAX: tl.constexpr,
    CLAMP_EPS: tl.constexpr,
    N: tl.constexpr,
    BLOCK_H: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    GROUPS: tl.constexpr,
    PACKS_ALL: tl.constexpr,
    SCALE_PACKS: tl.constexpr,
):
    # Cast pid to int64 BEFORE the row-stride multiply.
    row = tl.program_id(0).to(tl.int64)
    offs = tl.arange(0, BLOCK_H)
    mask = offs < N

    # Single read: the row stays register-resident for every phase below.
    x = tl.load(x_ptr + row * x_stride_m + offs, mask=mask, other=0.0).to(
        tl.float32
    )
    inv = tl.math.rsqrt(tl.sum(x * x, axis=0) / N + eps)
    w = tl.load(w_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    y = (x * inv * w).to(tl.bfloat16)
    tl.store(out_norm_ptr + row * out_norm_stride_m + offs, y, mask=mask)

    # Group scales from the register-resident normed values: reshape the
    # BLOCK_H lanes to [GROUPS, GROUP_SIZE] (GROUPS * GROUP_SIZE == BLOCK_H
    # exactly; groups past N//GROUP_SIZE are all-zero pad groups whose
    # nibbles no consumer reads).
    y_f32 = y.to(tl.float32)
    y2d = tl.reshape(y_f32, (GROUPS, GROUP_SIZE))
    absmax = tl.max(tl.abs(y2d), axis=1)
    scale_raw = tl.maximum(absmax, CLAMP_EPS) / FP8_MAX
    exponent = tl.ceil(tl.log2(scale_raw))
    scale = tl.math.exp2(exponent)
    q2d = tl.clamp(y2d / scale[:, None], FP8_MIN, FP8_MAX)
    q = tl.reshape(q2d, (BLOCK_H,))
    tl.store(
        out_q_ptr + row * out_q_stride_m + offs,
        q.to(out_q_ptr.dtype.element_ty),
        mask=mask,
    )

    # Pack 4 biased exponents per int32 word (little-endian nibble order,
    # matching the DSV4/sgl packed ue8m0 layout). GROUPS is padded up to the
    # block size, so the reshape target is GROUPS//4 words; only the
    # ceil(N//GROUP_SIZE/4) real words are stored (dead groups inside the
    # last real word hold deterministic pad exponents no consumer reads).
    e_biased = tl.clamp(exponent + 127.0, 0.0, 255.0).to(tl.int32)
    e2d = tl.reshape(e_biased, (PACKS_ALL, 4))
    shifts = (tl.arange(0, 4) * 8).to(tl.int32)
    packed = tl.sum(e2d << shifts, axis=1)
    offs_p = tl.arange(0, PACKS_ALL)
    tl.store(
        out_scale_ptr + offs_p * out_scale_stride_k + row,
        packed,
        mask=offs_p < SCALE_PACKS,
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
    if n > _MAX_BLOCK_H:
        raise ValueError(
            f"N={n} exceeds the register-resident row cap {_MAX_BLOCK_H}"
        )
    if n < 4 * group_size:
        # The packed-scale phase reshapes [GROUPS] -> [GROUPS//4, 4]; the
        # block-padded GROUPS must stay a multiple of 4.
        raise ValueError(
            f"N={n} is below the minimum {4 * group_size} for the packed-scale layout"
        )

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
    block_h = triton.next_power_of_2(n)
    groups = block_h // group_size
    scale_packs = (n // group_size + 3) // 4
    _rmsnorm_fp8_quant_reg_kernel[(m,)](
        x,
        weight,
        out_norm,
        out_q,
        out_scale,
        eps,
        x.stride(0),
        out_norm.stride(0),
        out_q.stride(0),
        out_scale.stride(1),
        FP8_MIN=finfo.min,
        FP8_MAX=finfo.max,
        CLAMP_EPS=clamp_eps,
        N=n,
        BLOCK_H=block_h,
        GROUP_SIZE=group_size,
        GROUPS=groups,
        PACKS_ALL=groups // 4,
        SCALE_PACKS=scale_packs,
        num_warps=8 if block_h >= 4096 else 4,
        num_stages=1,
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
