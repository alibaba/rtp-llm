"""FusedAddRMSNorm + FP8 per-token-group quant (Qwen3.5 decode group G).

Variant of DSV4 ``_rmsnorm_fp8_quant_kernel`` with FlashInfer residual-add
prepended. Quant / UE8M0 packing is the verified DSV4 formula; do not replace it.

Old path (2 kernels, re-reads BF16 hidden)::

    rtp_llm_ops.fused_add_rmsnorm(hidden, residual, weight, eps)
    sgl_per_token_group_quant_fp8(
        hidden, 128, eps=1e-4,
        column_major_scales=True, scale_tma_aligned=True, scale_ue8m0=True,
    )

Semantics (FlashInfer ``FusedAddRMSNormKernel``)::

    residual = residual + hidden
    hidden_norm = rmsnorm(residual) * weight
"""

from __future__ import annotations

from typing import Optional

import torch
import triton
import triton.language as tl

from rtp_llm.models_py.triton_kernels.qwen35_decode_fusion.env import (
    is_decode_fusion_enabled,
)
from rtp_llm.models_py.triton_kernels.qwen35_decode_fusion.fp8_scale import (
    make_ue8m0_scale_like,
)

_DEFAULT_GROUP = 128
_DEFAULT_EPS = 1.0e-6
_DEFAULT_CLAMP_EPS = 1.0e-4


def _fusion_enabled() -> bool:
    return is_decode_fusion_enabled()


def is_supported(
    hidden: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    group_size: int = _DEFAULT_GROUP,
) -> bool:
    """CUDA-graph-safe shape/dtype/layout gate. Metadata only; no host sync."""
    if not _fusion_enabled():
        return False
    if hidden is None or residual is None or weight is None:
        return False
    if hidden.dim() != 2 or residual.dim() != 2 or weight.dim() != 1:
        return False
    if hidden.shape != residual.shape:
        return False
    n = hidden.shape[1]
    if n == 0 or n % int(group_size) != 0:
        return False
    if weight.shape[0] != n:
        return False
    if hidden.dtype != torch.bfloat16 or residual.dtype != torch.bfloat16:
        return False
    if weight.dtype != torch.bfloat16:
        return False
    if not hidden.is_cuda or not residual.is_cuda or not weight.is_cuda:
        return False
    if hidden.device != residual.device or hidden.device != weight.device:
        return False
    if not hidden.is_contiguous() or not residual.is_contiguous():
        return False
    if not weight.is_contiguous():
        return False
    return True


# One HBM pass: keep (h+r) in registers, write residual / hidden / fp8.
# Quant from the BF16 hidden registers — do not reload the intermediate.
@triton.jit(do_not_specialize=["M", "out_scale_stride_k"])
def _fused_add_rmsnorm_fp8_quant_kernel(
    hidden_ptr,
    residual_ptr,
    w_ptr,
    out_norm_ptr,
    out_q_ptr,
    out_scale_ptr,
    M,
    hidden_stride_m,
    residual_stride_m,
    out_norm_stride_m,
    out_q_stride_m,
    out_scale_stride_k,
    EPS: tl.constexpr,
    CLAMP_EPS: tl.constexpr,
    FP8_MIN: tl.constexpr,
    FP8_MAX: tl.constexpr,
    N: tl.constexpr,
    BLOCK_N: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    SCALE_PACKS: tl.constexpr,
    NUM_GROUPS: tl.constexpr,
    NUM_GROUPS_PAD: tl.constexpr,
    WRITE_NORM: tl.constexpr,
):
    pid_m = tl.program_id(0).to(tl.int64)
    if pid_m >= M:
        return

    offs_n = tl.arange(0, BLOCK_N)
    n_mask = offs_n < N
    hidden_row = hidden_ptr + pid_m * hidden_stride_m
    residual_row = residual_ptr + pid_m * residual_stride_m
    h = tl.load(hidden_row + offs_n, mask=n_mask, other=0.0, eviction_policy="evict_first").to(
        tl.float32
    )
    r = tl.load(residual_row + offs_n, mask=n_mask, other=0.0).to(tl.float32)
    # FlashInfer: residual = residual + input; RMS / affine on the fp32 sum.
    x = h + r
    tl.store(residual_row + offs_n, x.to(tl.bfloat16), mask=n_mask)
    inv = tl.rsqrt(tl.sum(tl.where(n_mask, x * x, 0.0), axis=0) / N + EPS)
    w = tl.load(w_ptr + offs_n, mask=n_mask, other=0.0).to(tl.float32)
    y_bf16 = (x * inv * w).to(tl.bfloat16)
    if WRITE_NORM:
        tl.store(out_norm_ptr + pid_m * out_norm_stride_m + offs_n, y_bf16, mask=n_mask)

    y_g = tl.reshape(y_bf16.to(tl.float32), (NUM_GROUPS_PAD, GROUP_SIZE))
    absmax = tl.max(tl.abs(y_g), axis=1)
    scale_raw = tl.maximum(absmax, CLAMP_EPS) / FP8_MAX
    exponent = tl.ceil(tl.log2(scale_raw))
    scale = tl.math.exp2(exponent)
    q = tl.reshape(tl.clamp(y_g / scale[:, None], FP8_MIN, FP8_MAX), (BLOCK_N))
    tl.store(
        out_q_ptr + pid_m * out_q_stride_m + offs_n,
        q.to(out_q_ptr.dtype.element_ty),
        mask=n_mask,
    )
    exp_i32 = tl.clamp(exponent + 127.0, 0.0, 255.0).to(tl.int32)
    if NUM_GROUPS_PAD == SCALE_PACKS * 4:
        e4 = tl.reshape(exp_i32, (SCALE_PACKS, 4))
        packed = tl.sum(e4 << (tl.arange(0, 4) * 8)[None, :], axis=1)
        pack_ids = tl.arange(0, SCALE_PACKS)
        tl.store(out_scale_ptr + pack_ids * out_scale_stride_k + pid_m, packed)
    else:
        gid = tl.arange(0, NUM_GROUPS_PAD)
        for pack_id in tl.static_range(0, SCALE_PACKS):
            packed_scale = tl.zeros((), dtype=tl.int32)
            for pack_idx in tl.static_range(0, 4):
                group_id = pack_id * 4 + pack_idx
                if group_id < NUM_GROUPS:
                    e = tl.sum(tl.where(gid == group_id, exp_i32, 0))
                    packed_scale = packed_scale | (e << (pack_idx * 8))
            tl.store(out_scale_ptr + pack_id * out_scale_stride_k + pid_m, packed_scale)


def _default_launch_config(n: int) -> tuple[int, int]:
    # Graph-tuned on L20D decode, N=4096, single-pass register fusion.
    if n >= 4096:
        return 8, 3
    return 4, 2


def fused_add_rmsnorm_fp8_quant(
    hidden: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float = _DEFAULT_EPS,
    group_size: int = _DEFAULT_GROUP,
    clamp_eps: float = _DEFAULT_CLAMP_EPS,
    num_warps: Optional[int] = None,
    num_stages: Optional[int] = None,
    out_fp8: Optional[torch.Tensor] = None,
    out_scale: Optional[torch.Tensor] = None,
    write_norm: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fuse residual-add RMSNorm and UE8M0 per-token-group FP8 quant.

    May mutate ``hidden`` / ``residual`` inplace (same as
    ``rtp_llm_ops.fused_add_rmsnorm``). Returns
    ``(hidden_norm, residual_out, fp8, scale)``.
    """
    if not is_supported(hidden, residual, weight, group_size=group_size):
        raise ValueError(
            "fused_add_rmsnorm_fp8_quant: unsupported "
            f"hidden={tuple(getattr(hidden, 'shape', ()))} "
            f"dtype={getattr(hidden, 'dtype', None)}"
        )

    m, n = hidden.shape
    hidden_norm = hidden
    residual_out = residual
    if out_fp8 is None:
        out_q = torch.empty(m, n, device=hidden.device, dtype=torch.float8_e4m3fn)
    else:
        if out_fp8.shape != hidden.shape or out_fp8.dtype != torch.float8_e4m3fn:
            raise ValueError("out_fp8 must be float8_e4m3fn with hidden's shape")
        out_q = out_fp8
    if out_scale is None:
        out_scale = make_ue8m0_scale_like(
            hidden.shape, device=hidden.device, group_size=group_size
        )
    # else: caller-provided TMA scale, written by the kernel
    if (n // group_size) % 4 != 0:
        out_scale.zero_()
    if m == 0:
        return hidden_norm, residual_out, out_q, out_scale

    finfo = torch.finfo(torch.float8_e4m3fn)
    num_groups = n // group_size
    scale_packs = (num_groups + 3) // 4
    default_warps, default_stages = _default_launch_config(n)
    if num_warps is None:
        num_warps = default_warps
    if num_stages is None:
        num_stages = default_stages

    block_n = triton.next_power_of_2(n)
    _fused_add_rmsnorm_fp8_quant_kernel[(m,)](
        hidden,
        residual,
        weight,
        hidden_norm,
        out_q,
        out_scale,
        m,
        hidden.stride(0),
        residual.stride(0),
        hidden_norm.stride(0),
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
        NUM_GROUPS=num_groups,
        NUM_GROUPS_PAD=block_n // group_size,
        WRITE_NORM=write_norm,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return hidden_norm, residual_out, out_q, out_scale


def maybe_fused_add_rmsnorm_fp8_quant(
    hidden: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float = _DEFAULT_EPS,
    group_size: int = _DEFAULT_GROUP,
    clamp_eps: float = _DEFAULT_CLAMP_EPS,
) -> Optional[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Return the fused outputs, or ``None`` if disabled / unsupported."""
    if not _fusion_enabled():
        return None
    if not is_supported(hidden, residual, weight, group_size=group_size):
        return None
    return fused_add_rmsnorm_fp8_quant(
        hidden,
        residual,
        weight,
        eps=eps,
        group_size=group_size,
        clamp_eps=clamp_eps,
    )
