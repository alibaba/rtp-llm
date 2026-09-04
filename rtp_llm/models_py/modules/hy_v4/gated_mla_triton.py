"""HY4 decode-only fused gated-MLA projection and MXFP8 quantization.

The kernel keeps the existing numerical boundaries while avoiding the
materialized BF16 gate tensor and the separate sigmoid/multiply/quantize
launch. Unsupported shapes return ``None`` so callers retain the established
linear + epilogue path.
"""

from typing import Optional, Tuple

import torch
import triton
import triton.language as tl

from rtp_llm.models_py.kernels.cuda.mxfp8_ops import create_mxfp8_packed_scale
from rtp_llm.models_py.triton_kernels.common.attn_output_gate import (
    _ieee_rn_div_f32,
    _ue8m0_pow2_round_scalar,
)

_MXFP8_GROUP_SIZE = 32
_SCALE_GROUPS_PER_TILE = 4
_BLOCK_M = 16
_BLOCK_N = _MXFP8_GROUP_SIZE * _SCALE_GROUPS_PER_TILE
_BLOCK_K = 256
_MAX_DECODE_TOKENS = 16
_HY4_HIDDEN_SIZE = 6144
_HY4_GATE_SIZE = 16384


@triton.jit
def _gated_mla_proj_mxfp8_kernel(
    hidden_ptr,
    weight_ptr,
    attn_ptr,
    fp8_out_ptr,
    scale_out_ptr,
    M,
    K,
    N,
    stride_hidden_m,
    stride_weight_n,
    stride_weight_k,
    stride_attn_m,
    stride_fp8_m,
    stride_scale_m,
    stride_scale_n,
    fp8_max: tl.constexpr,
    fp8_min: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    block_n = tl.program_id(0)
    offs_m = tl.arange(0, BLOCK_M)
    offs_n = block_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    mask_m = offs_m < M
    mask_n = offs_n < N

    hidden_ptrs = hidden_ptr + offs_m[:, None] * stride_hidden_m + offs_k[None, :]
    weight_ptrs = (
        weight_ptr
        + offs_k[:, None] * stride_weight_k
        + offs_n[None, :] * stride_weight_n
    )
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k_block in range(0, tl.cdiv(K, BLOCK_K)):
        remaining_k = K - k_block * BLOCK_K
        hidden = tl.load(
            hidden_ptrs,
            mask=mask_m[:, None] & (offs_k[None, :] < remaining_k),
            other=0.0,
        )
        weight = tl.load(
            weight_ptrs,
            mask=(offs_k[:, None] < remaining_k) & mask_n[None, :],
            other=0.0,
        )
        acc += tl.dot(hidden, weight, out_dtype=tl.float32)
        hidden_ptrs += BLOCK_K
        weight_ptrs += BLOCK_K * stride_weight_k

    # Preserve the current F.linear BF16 output and elementwise BF16 rounding
    # boundaries before applying MXFP8 quantization.
    gate_bf16 = acc.to(tl.bfloat16)
    attn_bf16 = tl.load(
        attn_ptr + offs_m[:, None] * stride_attn_m + offs_n[None, :],
        mask=mask_m[:, None] & mask_n[None, :],
        other=0.0,
    )
    sigmoid_bf16 = tl.sigmoid(gate_bf16.to(tl.float32)).to(tl.bfloat16)
    gated = (
        (attn_bf16.to(tl.float32) * sigmoid_bf16.to(tl.float32))
        .to(tl.bfloat16)
        .to(tl.float32)
    )

    gated_groups = tl.reshape(gated, (BLOCK_M, BLOCK_N // 32, 32))
    absmax = tl.maximum(tl.max(tl.abs(gated_groups), axis=2), 1e-4)
    scale_init = _ieee_rn_div_f32(absmax, fp8_max)
    scale, exp_bits = _ue8m0_pow2_round_scalar(scale_init)
    scale_broadcast = tl.broadcast_to(
        scale[:, :, None],
        (BLOCK_M, BLOCK_N // 32, 32),
    )
    quantized = tl.clamp(
        _ieee_rn_div_f32(gated_groups, scale_broadcast),
        fp8_min,
        fp8_max,
    ).to(fp8_out_ptr.dtype.element_ty)
    tl.store(
        fp8_out_ptr + offs_m[:, None] * stride_fp8_m + offs_n[None, :],
        tl.reshape(quantized, (BLOCK_M, BLOCK_N)),
        mask=mask_m[:, None] & mask_n[None, :],
    )

    exp_groups = tl.reshape(exp_bits, (BLOCK_M, BLOCK_N // 128, 4))
    shifts = tl.arange(0, 4)[None, None, :] * 8
    packed_scale = tl.sum(exp_groups << shifts, axis=2)
    packed_offsets = block_n * (BLOCK_N // 128) + tl.arange(0, BLOCK_N // 128)
    tl.store(
        scale_out_ptr
        + offs_m[:, None] * stride_scale_m
        + packed_offsets[None, :] * stride_scale_n,
        packed_scale,
        mask=mask_m[:, None],
    )


def maybe_fused_gated_mla_proj_mxfp8(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    attn_output: torch.Tensor,
) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
    """Return fused FP8 activation and packed UE8M0 scales when supported."""
    if hidden.dim() != 2 or weight.dim() != 2 or attn_output.dim() != 2:
        return None
    m, k = hidden.shape
    n, weight_k = weight.shape
    if (
        m < 1
        or m > _MAX_DECODE_TOKENS
        or (k, n) != (_HY4_HIDDEN_SIZE, _HY4_GATE_SIZE)
        or k != weight_k
        or tuple(attn_output.shape) != (m, n)
        or k % _BLOCK_K != 0
        or n % _BLOCK_N != 0
    ):
        return None
    if (
        not hidden.is_cuda
        or hidden.device != weight.device
        or hidden.device != attn_output.device
        or hidden.dtype != torch.bfloat16
        or weight.dtype != torch.bfloat16
        or attn_output.dtype != torch.bfloat16
        or not hidden.is_contiguous()
        or not attn_output.is_contiguous()
        # CudaF16Linear exposes the loader's contiguous [K, N] allocation as
        # a logical [N, K] transpose. The kernel relies on N being coalesced;
        # other layouts keep using F.linear.
        or weight.stride(0) != 1
        or weight.stride(1) != n
    ):
        return None
    capability = torch.cuda.get_device_capability(hidden.device)
    if capability[0] != 10:
        return None

    fp8_out = torch.empty((m, n), dtype=torch.float8_e4m3fn, device=hidden.device)
    packed_scale = create_mxfp8_packed_scale(m, n, hidden.device)
    with torch.cuda.device(hidden.device):
        _gated_mla_proj_mxfp8_kernel[(n // _BLOCK_N,)](
            hidden,
            weight,
            attn_output,
            fp8_out,
            packed_scale,
            m,
            k,
            n,
            hidden.stride(0),
            weight.stride(0),
            weight.stride(1),
            attn_output.stride(0),
            fp8_out.stride(0),
            packed_scale.stride(0),
            packed_scale.stride(1),
            fp8_max=torch.finfo(torch.float8_e4m3fn).max,
            fp8_min=torch.finfo(torch.float8_e4m3fn).min,
            BLOCK_M=_BLOCK_M,
            BLOCK_N=_BLOCK_N,
            BLOCK_K=_BLOCK_K,
            num_warps=8,
            num_stages=4,
        )
    return fp8_out, packed_scale
