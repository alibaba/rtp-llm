"""Copy routes and fill DeepEP's inactive columns in one launch."""

import torch
import triton
import triton.language as tl


@triton.jit
def _pad_routes(
    indices,
    weights,
    out_indices,
    out_weights,
    ROWS: tl.constexpr,
    WIDTH: tl.constexpr,
    TARGET: tl.constexpr,
    I_ROW: tl.constexpr,
    I_COL: tl.constexpr,
    W_ROW: tl.constexpr,
    W_COL: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    row, column = offsets // TARGET, offsets % TARGET
    valid = (row < ROWS) & (column < WIDTH)
    idx = tl.load(indices + row * I_ROW + column * I_COL, valid, -1)
    weight = tl.load(weights + row * W_ROW + column * W_COL, valid, 0)
    tl.store(out_indices + offsets, idx, row < ROWS)
    tl.store(out_weights + offsets, weight, row < ROWS)


def pad_inactive_routes(indices, weights, target):
    """Preserve route order and FP32 bits; inactive routes are exactly -1/0."""
    if (
        indices.ndim != 2
        or weights.shape != indices.shape
        or indices.dtype != torch.int64
        or weights.dtype != torch.float32
        or not indices.is_cuda
        or weights.device != indices.device
        or indices.shape[1] <= 0
        or target not in (2, 4, 8, 16)
        or target <= indices.shape[1]
    ):
        raise ValueError(
            "Route padding requires CUDA int64/FP32 matrices and a larger DeepEP width"
        )
    shape = (indices.shape[0], target)
    out_indices = torch.empty(shape, dtype=indices.dtype, device=indices.device)
    out_weights = torch.empty(shape, dtype=weights.dtype, device=weights.device)
    if out_indices.numel():
        _pad_routes[(triton.cdiv(out_indices.numel(), 256),)](
            indices,
            weights,
            out_indices,
            out_weights,
            indices.shape[0],
            indices.shape[1],
            target,
            *indices.stride(),
            *weights.stride(),
            256,
            num_warps=4,
        )
    return out_indices, out_weights
