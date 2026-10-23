"""Fixed-tile BF16 indexer projection for the SM120 decode path."""

from typing import Optional

import torch
import triton
import triton.language as tl


@triton.jit
def _pad_projection_tile(
    x,
    output,
    ROWS: tl.constexpr,
    X_ROW_STRIDE: tl.constexpr,
    X_COL_STRIDE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    rows, cols = offsets // 4096, offsets % 4096
    values = tl.load(
        x + rows * X_ROW_STRIDE + cols * X_COL_STRIDE,
        (rows < ROWS) & (offsets < 8 * 4096),
        other=0,
    )
    tl.store(output + offsets, values, offsets < 8 * 4096)


def project_weights_if_supported(
    x: torch.Tensor, weight: torch.Tensor
) -> Optional[torch.Tensor]:
    """Return the specialized projection, or None for the qualified fallback."""
    if not (
        x.is_cuda
        and x.dtype == torch.bfloat16
        and x.ndim >= 2
        and x.shape[-1] == 4096
        and weight.device == x.device
        and weight.dtype == torch.bfloat16
        and weight.shape == (64, 4096)
        and x.numel() > 0
    ):
        return None
    flat = x.reshape(-1, 4096)
    rows = flat.shape[0]
    output = torch.empty(
        (triton.cdiv(rows, 8) * 8, 64), device=x.device, dtype=torch.float32
    )
    for start in range(0, rows, 8):
        tile = flat[start : start + 8]
        if tile.shape[0] < 8:
            padded = torch.empty((8, 4096), device=x.device, dtype=x.dtype)
            _pad_projection_tile[(64,)](
                tile, padded, tile.shape[0], tile.stride(0), tile.stride(1), 512
            )
            tile = padded
        # Preserve the qualified eight-row cuBLAS reduction. Write FP32 tiles
        # directly into one buffer so only the final BF16 conversion is needed.
        torch.mm(
            tile, weight.t(), out_dtype=torch.float32, out=output[start : start + 8]
        )
    return output[:rows].to(x.dtype).reshape(*x.shape[:-1], 64)
