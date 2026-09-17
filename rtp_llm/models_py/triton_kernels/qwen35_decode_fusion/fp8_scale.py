"""UE8M0 packed scale layout used by DeepGEMM FP8_PER_BLOCK linears."""

from __future__ import annotations

import torch
import triton


def make_ue8m0_scale_like(
    x_shape: torch.Size | tuple[int, int],
    *,
    device: torch.device,
    group_size: int = 128,
) -> torch.Tensor:
    """Allocate the TMA-aligned, MN-major packed UE8M0 scale tensor.

    Matches ``sgl_per_token_group_quant_fp8(..., column_major_scales=True,
    scale_tma_aligned=True, scale_ue8m0=True)``.
    """
    m, n = int(x_shape[0]), int(x_shape[1])
    scale_cols = n // group_size
    aligned_m = triton.cdiv(m, 4) * 4
    aligned_k = triton.cdiv(scale_cols, 4) * 4
    return torch.empty(
        (aligned_k // 4, aligned_m), device=device, dtype=torch.int32
    ).transpose(-1, -2)[:m, :]
