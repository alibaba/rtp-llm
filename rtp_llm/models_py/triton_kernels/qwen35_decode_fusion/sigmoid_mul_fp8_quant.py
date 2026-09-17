"""Fused ``attn * sigmoid(gate)`` + per-token-group FP8 UE8M0 quant.

Replaces the three-kernel full-attention o_proj prep (15 FA layers)::

    y = attn * torch.sigmoid(gate)          # BF16 sigmoid + mul
    fp8, scale = sgl_per_token_group_quant_fp8(
        y, 128, eps=1e-4,
        column_major_scales=True,
        scale_tma_aligned=True,
        scale_ue8m0=True,
    )

This is **not** ``silu_mul_fp8_quant_packed`` (that is ``silu(gate)*up``).
UE8M0 packing / FP8 clamp is copied from
``triton_kernels/moe/silu_mul_fp8_quant.py`` and
``modules/dsv4/_fused_rmsnorm_fp8_quant_triton.py`` so the scale layout
matches ``sgl_per_token_group_quant_fp8``.

397B decode: ``attn``, ``gate`` are ``[M, 8192]`` BF16 contiguous.
``y`` is not written back; only ``(fp8, scale)`` are produced for
``CudaFp8DeepGEMMLinear.forward_quantized``.
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

_GROUP_SIZE = 128
_CLAMP_EPS = 1.0e-4
_FP8_DTYPE = torch.float8_e4m3fn


def _fusion_enabled() -> bool:
    return is_decode_fusion_enabled()


def _tensors_supported(attn: torch.Tensor, gate: torch.Tensor) -> bool:
    if not isinstance(attn, torch.Tensor) or not isinstance(gate, torch.Tensor):
        return False
    if attn.dim() != 2 or gate.dim() != 2:
        return False
    if attn.shape != gate.shape:
        return False
    if attn.dtype != torch.bfloat16 or gate.dtype != torch.bfloat16:
        return False
    if (not attn.is_cuda) or (not gate.is_cuda):
        return False
    if attn.device != gate.device:
        return False
    if (not attn.is_contiguous()) or (not gate.is_contiguous()):
        return False
    n = attn.shape[-1]
    if n == 0 or n % _GROUP_SIZE != 0:
        return False
    return True


def is_supported(attn: torch.Tensor, gate: torch.Tensor) -> bool:
    """Host-only checks; CUDA-graph safe (no ``.item()`` / device sync)."""
    if not _fusion_enabled():
        return False
    return _tensors_supported(attn, gate)


# Packing / clamp match DSV4 ``_rmsnorm_fp8_quant_kernel`` and
# ``_silu_mul_fp8_quant_packed_split_kernel``. CLAMP_EPS=1e-4 matches
# ``sgl_per_token_group_quant_fp8(..., eps=1e-4)``.
@triton.jit(do_not_specialize=["M", "output_scale_stride_k"])
def _sigmoid_mul_fp8_quant_kernel(
    attn_ptr,
    gate_ptr,
    output_q_ptr,
    output_scale_ptr,
    M,
    attn_stride_m,
    gate_stride_m,
    output_q_stride_m,
    output_scale_stride_k,
    N: tl.constexpr,
    NUM_GROUPS: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    FP8_MIN: tl.constexpr,
    FP8_MAX: tl.constexpr,
    CLAMP_EPS: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    pid_pack = tl.program_id(0).to(tl.int64)
    pid_m = tl.program_id(1).to(tl.int64)
    m_offset = pid_m * BLOCK_M
    if m_offset >= M:
        return

    offs_m = tl.arange(0, BLOCK_M).to(tl.int64)
    offs_n = tl.arange(0, GROUP_SIZE)
    row_mask = (m_offset + offs_m) < M

    attn_base = (m_offset + offs_m[:, None]) * attn_stride_m
    gate_base = (m_offset + offs_m[:, None]) * gate_stride_m
    out_base = (m_offset + offs_m[:, None]) * output_q_stride_m

    packed_scale = tl.zeros((BLOCK_M,), dtype=tl.int32)

    for pack_idx in tl.static_range(4):
        group_id = pid_pack * 4 + pack_idx
        if group_id < NUM_GROUPS:
            n_offset = group_id * GROUP_SIZE
            cols = n_offset + offs_n
            mask = row_mask[:, None] & (cols[None, :] < N)

            attn = tl.load(
                attn_ptr + attn_base + cols[None, :],
                mask=mask,
                other=0.0,
                eviction_policy="evict_first",
            ).to(tl.float32)
            gate = tl.load(
                gate_ptr + gate_base + cols[None, :],
                mask=mask,
                other=0.0,
                eviction_policy="evict_first",
            ).to(tl.float32)

            # aten BF16 sigmoid: fp32 sigmoid, then round to BF16, then BF16 mul.
            sig_bf16 = (1.0 / (1.0 + tl.exp(-gate))).to(tl.bfloat16)
            y_bf16 = (attn * sig_bf16.to(tl.float32)).to(tl.bfloat16)
            y = y_bf16.to(tl.float32)

            absmax = tl.max(tl.abs(y), axis=1)
            scale_raw = tl.maximum(absmax, CLAMP_EPS) / FP8_MAX
            exponent = tl.ceil(tl.log2(scale_raw))
            scale = tl.math.exp2(exponent)

            y_q = tl.clamp(y / scale[:, None], FP8_MIN, FP8_MAX)
            tl.store(
                output_q_ptr + out_base + cols[None, :],
                y_q.to(output_q_ptr.dtype.element_ty),
                mask=mask,
            )

            exponent_biased = tl.clamp(exponent + 127.0, 0.0, 255.0).to(tl.int32)
            packed_scale = packed_scale | (exponent_biased << (pack_idx * 8))

    scale_ptrs = output_scale_ptr + pid_pack * output_scale_stride_k + m_offset + offs_m
    tl.store(scale_ptrs, packed_scale, mask=row_mask)


def sigmoid_mul_fp8_quant(
    attn: torch.Tensor,
    gate: torch.Tensor,
    out_q: Optional[torch.Tensor] = None,
    out_scale: Optional[torch.Tensor] = None,
    block_m: Optional[int] = None,
    num_warps: Optional[int] = None,
    num_stages: Optional[int] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fuse ``attn * sigmoid(gate)`` and UE8M0 FP8 quant.

    Args:
        attn: ``[M, N]`` BF16 contiguous CUDA tensor.
        gate: same shape / dtype / device / layout as ``attn``.
        out_q / out_scale: optional CUDA-graph-safe preallocated outputs.

    Returns:
        fp8: ``[M, N]`` ``float8_e4m3fn``
        scale: TMA-aligned MN-major packed UE8M0 int32 from
            ``make_ue8m0_scale_like``
    """
    if not _tensors_supported(attn, gate):
        raise ValueError(
            "sigmoid_mul_fp8_quant expects 2D contiguous CUDA BF16 "
            f"attn/gate with N%{_GROUP_SIZE}==0; got "
            f"attn={tuple(getattr(attn, 'shape', ()))} {getattr(attn, 'dtype', None)} "
            f"gate={tuple(getattr(gate, 'shape', ()))} {getattr(gate, 'dtype', None)}"
        )

    m, n = attn.shape
    if out_q is None:
        out_q = torch.empty((m, n), device=attn.device, dtype=_FP8_DTYPE)
    elif (
        out_q.shape != (m, n)
        or out_q.dtype != _FP8_DTYPE
        or out_q.device != attn.device
        or not out_q.is_contiguous()
    ):
        raise ValueError("out_q must be contiguous [M, N] float8_e4m3fn on attn.device")
    packed_k = triton.cdiv(n // _GROUP_SIZE, 4)
    if out_scale is None:
        out_scale = make_ue8m0_scale_like(
            attn.shape, device=attn.device, group_size=_GROUP_SIZE
        )
    elif (
        out_scale.dtype != torch.int32
        or out_scale.device != attn.device
        or out_scale.shape != (m, packed_k)
    ):
        raise ValueError(
            f"out_scale must be int32 [{m}, {packed_k}] on attn.device "
            f"(make_ue8m0_scale_like layout), got {tuple(out_scale.shape)} {out_scale.dtype}"
        )
    if m == 0:
        return out_q, out_scale

    finfo = torch.finfo(_FP8_DTYPE)
    num_groups = n // _GROUP_SIZE
    num_packed = (num_groups + 3) // 4
    if block_m is None:
        block_m = 8
    if num_warps is None:
        num_warps = 4
    if num_stages is None:
        num_stages = 2
    grid = (num_packed, triton.cdiv(m, block_m))
    _sigmoid_mul_fp8_quant_kernel[grid](
        attn,
        gate,
        out_q,
        out_scale,
        m,
        attn.stride(0),
        gate.stride(0),
        out_q.stride(0),
        out_scale.stride(1),
        N=n,
        NUM_GROUPS=num_groups,
        GROUP_SIZE=_GROUP_SIZE,
        FP8_MIN=float(finfo.min),
        FP8_MAX=float(finfo.max),
        CLAMP_EPS=_CLAMP_EPS,
        BLOCK_M=block_m,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return out_q, out_scale


def maybe_sigmoid_mul_fp8_quant(
    attn: torch.Tensor,
    gate: torch.Tensor,
) -> Optional[tuple[torch.Tensor, torch.Tensor]]:
    """Return ``(fp8, scale)`` when supported; otherwise ``None`` (old path)."""
    if not is_supported(attn, gate):
        return None
    return sigmoid_mul_fp8_quant(attn, gate)
