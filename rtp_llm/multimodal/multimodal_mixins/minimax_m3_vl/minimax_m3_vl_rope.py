# SPDX-License-Identifier: Apache-2.0

import logging
from typing import Optional, Tuple

import torch

logger = logging.getLogger(__name__)

try:
    import triton
    import triton.language as tl
except Exception:
    triton = None
    tl = None

_TRITON_DISABLED = False
_GROUP_HEADS = 4


if triton is not None:

    @triton.jit
    def _qkv_rope_pack_kernel(
        qkv_ptr,
        cos_ptr,
        sin_ptr,
        output_ptr,
        qkv_stride_t,
        qkv_stride_p,
        qkv_stride_h,
        cos_stride_t,
        cos_stride_d,
        sin_stride_t,
        sin_stride_d,
        OUTPUT_PROJECTION_STRIDE: tl.constexpr,
        OUTPUT_TOKEN_STRIDE: tl.constexpr,
        NUM_HEADS: tl.constexpr,
        HEAD_DIM: tl.constexpr,
        ROT_DIM: tl.constexpr,
        ROT_HALF: tl.constexpr,
        GROUP_HEADS: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        token_index = tl.program_id(0).to(tl.int64)
        head_offsets = (tl.program_id(1) * GROUP_HEADS + tl.arange(0, GROUP_HEADS))[
            :, None
        ]
        dim_offsets = tl.arange(0, BLOCK_D)[None, :]
        value_mask = (head_offsets < NUM_HEADS) & (dim_offsets < HEAD_DIM)
        rotary_mask = value_mask & (dim_offsets < ROT_DIM)

        qkv_offsets = (
            token_index * qkv_stride_t + head_offsets * qkv_stride_h + dim_offsets
        )
        partner_dims = tl.where(
            dim_offsets < ROT_HALF,
            dim_offsets + ROT_HALF,
            dim_offsets - ROT_HALF,
        )
        partner_offsets = (
            token_index * qkv_stride_t + head_offsets * qkv_stride_h + partner_dims
        )

        cos_values = tl.load(
            cos_ptr + token_index * cos_stride_t + dim_offsets * cos_stride_d,
            mask=dim_offsets < ROT_DIM,
            other=1.0,
        ).to(tl.float32)
        sin_values = tl.load(
            sin_ptr + token_index * sin_stride_t + dim_offsets * sin_stride_d,
            mask=dim_offsets < ROT_DIM,
            other=0.0,
        ).to(tl.float32)
        partner_sign = tl.where(dim_offsets < ROT_HALF, -1.0, 1.0)

        output_offsets = (
            token_index * OUTPUT_TOKEN_STRIDE + head_offsets * HEAD_DIM + dim_offsets
        )
        for projection in range(3):
            projection_offset = projection * qkv_stride_p
            values = tl.load(
                qkv_ptr + qkv_offsets + projection_offset,
                mask=value_mask,
                other=0.0,
            ).to(tl.float32)
            if projection < 2:
                partner_values = tl.load(
                    qkv_ptr + partner_offsets + projection_offset,
                    mask=rotary_mask,
                    other=0.0,
                ).to(tl.float32)
                rotated = (
                    values * cos_values + partner_sign * partner_values * sin_values
                )
                values = tl.where(rotary_mask, rotated, values)
            tl.store(
                output_ptr + projection * OUTPUT_PROJECTION_STRIDE + output_offsets,
                values,
                mask=value_mask,
            )


def fused_qkv_rope(
    qkv: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Apply Q/K RoPE and pack strided fused-QKV output in one CUDA kernel."""
    global _TRITON_DISABLED

    if triton is None or _TRITON_DISABLED or not qkv.is_cuda:
        return None
    if qkv.dim() != 4 or qkv.shape[1] != 3:
        raise ValueError(f"qkv must have shape [seq, 3, heads, dim], got {qkv.shape}")
    if cos.shape != sin.shape or cos.shape[:2] != (qkv.shape[0], 1):
        raise ValueError(
            f"invalid RoPE shapes for qkv={qkv.shape}: cos={cos.shape}, sin={sin.shape}"
        )
    if qkv.stride(-1) != 1 or cos.stride(-1) != 1 or sin.stride(-1) != 1:
        return None

    sequence_length, _, num_heads, head_dim = qkv.shape
    rotary_dim = cos.shape[-1]
    if rotary_dim > head_dim or rotary_dim % 2 != 0:
        raise ValueError(
            f"rotary_dim must be even and <= head_dim, got {rotary_dim} and {head_dim}"
        )

    packed = torch.empty(
        (3, sequence_length, num_heads, head_dim),
        dtype=qkv.dtype,
        device=qkv.device,
    )
    grid = (sequence_length, triton.cdiv(num_heads, _GROUP_HEADS))
    try:
        _qkv_rope_pack_kernel[grid](
            qkv,
            cos,
            sin,
            packed,
            qkv.stride(0),
            qkv.stride(1),
            qkv.stride(2),
            cos.stride(0),
            cos.stride(-1),
            sin.stride(0),
            sin.stride(-1),
            OUTPUT_PROJECTION_STRIDE=sequence_length * num_heads * head_dim,
            OUTPUT_TOKEN_STRIDE=num_heads * head_dim,
            NUM_HEADS=num_heads,
            HEAD_DIM=head_dim,
            ROT_DIM=rotary_dim,
            ROT_HALF=rotary_dim // 2,
            GROUP_HEADS=_GROUP_HEADS,
            BLOCK_D=triton.next_power_of_2(head_dim),
            num_warps=4,
        )
    except Exception as error:
        if "out of memory" in str(error).lower():
            raise
        _TRITON_DISABLED = True
        logger.warning(
            "Fused MiniMax M3VL QKV/RoPE kernel unavailable; using eager path: %s",
            error,
        )
        return None

    q, k, v = packed.unbind(dim=0)
    return q, k, v
