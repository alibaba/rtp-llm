"""Fused materialization of a padded sparse-MLA query tensor.

Blackwell's BF16 sparse FlashMLA prefill kernel consumes a 128-head envelope,
while HY4 TP8 produces 64 real query heads.  The eager implementation first
zero-filled the complete padded tensor and then copied the real heads into a
strided prefix.  This kernel writes every destination element once: real heads
are copied and padding heads are zeroed in the same launch.
"""

from __future__ import annotations

import os

import torch
import triton
import triton.language as tl


_FUSE_QUERY_PADDING_ENV = "RTP_LLM_FUSE_SPARSE_QUERY_PADDING"
_BLOCK_ROWS = 32
_BLOCK_D = 256


def _query_padding_enabled() -> bool:
    return os.environ.get(_FUSE_QUERY_PADDING_ENV, "1").strip().lower() not in (
        "0",
        "false",
        "off",
        "no",
    )


@triton.jit
def _pad_query_heads_kernel(
    query_ptr,
    padded_ptr,
    num_tokens,
    src_stride_token,
    src_stride_head,
    dst_stride_token,
    dst_stride_head,
    ACTUAL_HEADS: tl.constexpr,
    PADDED_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    row_start = tl.program_id(0).to(tl.int64) * BLOCK_ROWS
    dim_start = tl.program_id(1) * BLOCK_D
    rows = row_start + tl.arange(0, BLOCK_ROWS).to(tl.int64)
    dims = dim_start + tl.arange(0, BLOCK_D).to(tl.int64)

    token = rows // PADDED_HEADS
    head = rows % PADDED_HEADS
    valid_row = rows < num_tokens * PADDED_HEADS
    valid_dim = dims < HEAD_DIM
    real_head = head < ACTUAL_HEADS

    src_offsets = (
        token[:, None] * src_stride_token
        + head[:, None] * src_stride_head
        + dims[None, :]
    )
    values = tl.load(
        query_ptr + src_offsets,
        mask=valid_row[:, None] & real_head[:, None] & valid_dim[None, :],
        other=0.0,
    )
    dst_offsets = (
        token[:, None] * dst_stride_token
        + head[:, None] * dst_stride_head
        + dims[None, :]
    )
    tl.store(
        padded_ptr + dst_offsets,
        values,
        mask=valid_row[:, None] & valid_dim[None, :],
    )


def maybe_pad_query_heads(
    query: torch.Tensor, padded_heads: int
) -> torch.Tensor | None:
    """Return a zero-padded query tensor, or ``None`` for the eager fallback."""
    if not _query_padding_enabled():
        return None
    if torch.is_grad_enabled() and query.requires_grad:
        return None
    if query.device.type != "cuda" or query.dtype != torch.bfloat16:
        return None
    if query.dim() != 3 or not query.is_contiguous():
        return None

    num_tokens, actual_heads, head_dim = query.shape
    if (
        num_tokens <= 0
        or actual_heads <= 0
        or head_dim <= 0
        or padded_heads <= actual_heads
    ):
        return None

    padded = torch.empty(
        (num_tokens, padded_heads, head_dim),
        dtype=query.dtype,
        device=query.device,
    )
    grid = (
        triton.cdiv(num_tokens * padded_heads, _BLOCK_ROWS),
        triton.cdiv(head_dim, _BLOCK_D),
    )
    with torch.cuda.device(query.device.index):
        _pad_query_heads_kernel[grid](
            query,
            padded,
            num_tokens,
            query.stride(0),
            query.stride(1),
            padded.stride(0),
            padded.stride(1),
            ACTUAL_HEADS=actual_heads,
            PADDED_HEADS=padded_heads,
            HEAD_DIM=head_dim,
            BLOCK_ROWS=_BLOCK_ROWS,
            BLOCK_D=_BLOCK_D,
            num_warps=8,
            num_stages=2,
        )
    return padded


__all__ = ["maybe_pad_query_heads"]
