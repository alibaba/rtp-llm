"""Tensor layouts shared by quantized modules and backend executors."""

import torch


def swizzle_nvfp4_scales(scale: torch.Tensor) -> torch.Tensor:
    """Interleave NVFP4 block scales, preserving the input device and bytes.

    Accept [rows, blocks] or [experts, rows, blocks]. Padding is zero-filled to
    rows divisible by 128 and blocks divisible by 4. Unlike the legacy device
    wrapper, this operation never implicitly moves to the current CUDA device.
    """
    if scale.dtype != torch.float8_e4m3fn:
        raise ValueError(f"NVFP4 scales require float8_e4m3fn, got {scale.dtype}")
    if scale.ndim not in (2, 3):
        raise ValueError(f"NVFP4 scales require 2 or 3 dimensions, got {scale.ndim}")
    unbatched = scale.ndim == 2
    source = scale.unsqueeze(0) if unbatched else scale
    batches, rows, blocks = source.shape
    padded_rows = (rows + 127) // 128 * 128
    padded_blocks = (blocks + 3) // 4 * 4
    # Work on bytes: this permutation must not interpret/re-quantize FP8 values.
    padded = torch.zeros(
        (batches, padded_rows, padded_blocks), dtype=torch.uint8, device=scale.device
    )
    padded[:, :rows, :blocks] = source.view(torch.uint8)
    result = (
        padded.reshape(batches, padded_rows // 128, 4, 32, padded_blocks // 4, 4)
        .permute(0, 1, 4, 3, 2, 5)
        .contiguous()
        .view(torch.float8_e4m3fn)
    )
    shape = (
        (padded_rows, padded_blocks)
        if unbatched
        else (batches, padded_rows, padded_blocks)
    )
    return result.reshape(shape)
