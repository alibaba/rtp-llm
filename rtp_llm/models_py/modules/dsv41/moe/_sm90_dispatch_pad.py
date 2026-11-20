"""Standalone bit-preserving DeepEP transport padding; not a production hook.

This does not choose experts, normalize weights, or alter topk. It only copies
the existing routes and appends the transport's (-1, +0) sentinel slots. The
caller chooses a supported transport width; all outputs own fresh storage,
including when the source width already equals the requested width.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _pad_routes_kernel(
    indices,
    weights,
    out_indices,
    out_weights,
    ROWS: tl.constexpr,
    TOPK: tl.constexpr,
    PAD_TO: tl.constexpr,
    I_ROW: tl.constexpr,
    I_COL: tl.constexpr,
    W_ROW: tl.constexpr,
    W_COL: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offset = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    row, col = offset // PAD_TO, offset % PAD_TO
    valid = (row < ROWS) & (col < TOPK)
    expert = tl.load(indices + row * I_ROW + col * I_COL, valid, other=-1)
    # Integer pointer copies preserve every FP32 bit, including NaN payloads
    # and signed zero. Padded source addresses are never dereferenced.
    weight_bits = tl.load(
        weights.to(tl.pointer_type(tl.int32)) + row * W_ROW + col * W_COL,
        valid,
        other=0,
    )
    tl.store(out_indices + offset, expert, row < ROWS)
    tl.store(
        out_weights.to(tl.pointer_type(tl.int32)) + offset,
        weight_bits,
        row < ROWS,
    )


def pad_deepep_routes(indices, weights, pad_to=8):
    """Fresh contiguous INT64/FP32 routes with exact copy and sentinel padding.

    CUDA 2-D inputs may have different, arbitrary nonnegative strides. No
    device data is read on the host, making changed-input Graph replay safe.
    Unlike the strategy wrapper, this standalone primitive always copies;
    production may keep its existing no-op for already-supported widths.
    """
    if (
        not isinstance(pad_to, int)
        or isinstance(pad_to, bool)
        or pad_to not in (2, 4, 8, 16)
        or indices.ndim != 2
        or weights.ndim != 2
        or indices.shape != weights.shape
        or not indices.is_cuda
        or weights.device != indices.device
        or indices.dtype != torch.int64
        or weights.dtype != torch.float32
        or not 1 <= indices.shape[1] <= pad_to
        or any(stride < 0 for stride in (*indices.stride(), *weights.stride()))
    ):
        raise ValueError("expected matching CUDA INT64/FP32 routes and valid pad width")
    rows, topk = indices.shape
    out_indices = torch.empty((rows, pad_to), device=indices.device, dtype=torch.int64)
    out_weights = torch.empty(
        (rows, pad_to), device=weights.device, dtype=torch.float32
    )
    if rows:
        _pad_routes_kernel[(triton.cdiv(rows * pad_to, 256),)](
            indices,
            weights,
            out_indices,
            out_weights,
            rows,
            topk,
            pad_to,
            *indices.stride(),
            *weights.stride(),
            256,
            num_warps=4,
        )
    return out_indices, out_weights
