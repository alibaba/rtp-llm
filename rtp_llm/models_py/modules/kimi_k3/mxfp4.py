"""Torch correctness fallback for K3's routed-expert MXFP4 weights.

This is intentionally a small, transparent implementation.  It lets model
bring-up and synthetic tests execute before the final Blackwell grouped-GEMM
kernel is connected.  Production serving must replace materialized BF16
expert weights with a native packed MXFP4 MoE kernel.
"""

from __future__ import annotations

import torch


MXFP4_GROUP_SIZE = 32


def dequantize_mxfp4(
    packed: torch.Tensor,
    scale: torch.Tensor,
    *,
    dtype: torch.dtype = torch.bfloat16,
    group_size: int = MXFP4_GROUP_SIZE,
) -> torch.Tensor:
    """Dequantize packed E2M1 values with UE8M0 block scales.

    ``packed[..., j]`` stores logical values ``2*j`` in the low nibble and
    ``2*j+1`` in the high nibble.  A scale byte ``s`` represents ``2**(s-127)``.
    The function supports arbitrary leading dimensions, including a stacked
    expert dimension.
    """

    if packed.dtype not in (torch.uint8, torch.int8):
        raise TypeError(f"packed MXFP4 weight must be byte-valued, got {packed.dtype}")
    if scale.dtype != torch.uint8:
        raise TypeError(f"MXFP4 UE8M0 scale must be torch.uint8, got {scale.dtype}")
    if group_size <= 0 or group_size % 2 != 0:
        raise ValueError("MXFP4 group_size must be a positive even integer")
    if packed.ndim == 0 or scale.ndim == 0:
        raise ValueError("packed weight and scale must have at least one dimension")

    packed_u8 = packed.view(torch.uint8) if packed.dtype == torch.int8 else packed
    codes = torch.empty(
        (*packed_u8.shape[:-1], packed_u8.shape[-1] * 2),
        dtype=torch.uint8,
        device=packed.device,
    )
    codes[..., 0::2] = packed_u8 & 0x0F
    codes[..., 1::2] = (packed_u8 >> 4) & 0x0F

    logical_width = codes.shape[-1]
    if logical_width % group_size != 0:
        raise ValueError(
            f"logical MXFP4 width {logical_width} is not divisible by group_size "
            f"{group_size}"
        )
    expected_scale_shape = (*codes.shape[:-1], logical_width // group_size)
    if tuple(scale.shape) != expected_scale_shape:
        raise ValueError(
            f"MXFP4 scale shape {tuple(scale.shape)} does not match expected "
            f"{expected_scale_shape}"
        )

    magnitude_table = torch.tensor(
        [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0],
        dtype=torch.float32,
        device=packed.device,
    )
    magnitude = magnitude_table[(codes & 0x07).long()]
    sign = 1.0 - 2.0 * ((codes >> 3) & 0x01).float()
    scale_factor = torch.exp2(scale.float() - 127.0).repeat_interleave(
        group_size, dim=-1
    )
    return (sign * magnitude * scale_factor).to(dtype=dtype)


__all__ = ["MXFP4_GROUP_SIZE", "dequantize_mxfp4"]
