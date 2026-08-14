"""Triton permutations for equal-block zig-zag token layouts.

For a group of ``P`` ranks, the canonical sequence is split into ``2 * P``
equal blocks.  A normal AllGather of rank-local zig-zag shards produces the
rank-major layout::

    C0, C(2P-1), C1, C(2P-2), ..., C(P-1), CP

The functions in this module convert that layout to/from canonical order.  The
copy is exact: tensor values are never converted or reduced.
"""

from __future__ import annotations

from typing import Optional

import torch
import triton
import triton.language as tl

_BLOCK_WIDTH = 1024


@triton.jit
def _zigzag_permute_kernel(
    input_ptr,
    output_ptr,
    row_width,
    block_tokens,
    WORLD_SIZE: tl.constexpr,
    TO_CANONICAL: tl.constexpr,
    BLOCK_WIDTH: tl.constexpr,
):
    destination_token = tl.program_id(0).to(tl.int64)
    columns = tl.program_id(1) * BLOCK_WIDTH + tl.arange(0, BLOCK_WIDTH)
    column_mask = columns < row_width
    column_offsets = columns.to(tl.int64)

    # destination_token is int64, so every dependent token/index expression is
    # promoted to int64 even when Triton specializes a scalar argument to 1.
    destination_block = destination_token // block_tokens
    token_in_block = destination_token - destination_block * block_tokens

    if TO_CANONICAL:
        # Canonical block d comes from rank-major block 2*d in the first half,
        # and from the odd blocks in reverse rank order in the second half.
        source_block = tl.where(
            destination_block < WORLD_SIZE,
            2 * destination_block,
            2 * (2 * WORLD_SIZE - 1 - destination_block) + 1,
        )
    else:
        # Rank-major destination blocks are [C_rank, C_(2P-1-rank)].
        rank = destination_block // 2
        source_block = tl.where(
            destination_block % 2 == 0,
            rank,
            2 * WORLD_SIZE - 1 - rank,
        )

    source_token = source_block * block_tokens + token_in_block
    # A 1M x 7168 activation has 7.168B elements, beyond both signed and
    # unsigned 32-bit indexing.  Keep the full pointer-offset chain in int64.
    source_offsets = source_token * row_width + column_offsets
    destination_offsets = destination_token * row_width + column_offsets
    values = tl.load(input_ptr + source_offsets, mask=column_mask)
    tl.store(output_ptr + destination_offsets, values, mask=column_mask)


def _validate_layout(tensor: torch.Tensor, world_size: int) -> int:
    if not isinstance(world_size, int) or isinstance(world_size, bool):
        raise TypeError(f"world_size must be an int, got {type(world_size).__name__}")
    if world_size <= 0:
        raise ValueError(f"world_size must be positive, got {world_size}")
    if tensor.ndim == 0:
        raise ValueError("zig-zag permutation requires a tensor with dim0")
    block_count = 2 * world_size
    if tensor.shape[0] == 0 or tensor.shape[0] % block_count:
        raise ValueError(
            "zig-zag permutation requires dim0 divisible by 2 * world_size: "
            f"shape={tuple(tensor.shape)}, world_size={world_size}"
        )
    row_width = tensor.numel() // tensor.shape[0]
    if row_width == 0:
        raise ValueError(
            f"zig-zag permutation requires non-empty trailing dimensions: shape={tuple(tensor.shape)}"
        )
    return row_width


def _prepare_output(
    tensor: torch.Tensor,
    output: Optional[torch.Tensor],
) -> torch.Tensor:
    if output is None:
        return torch.empty_like(tensor, memory_format=torch.contiguous_format)
    if (
        output.shape != tensor.shape
        or output.dtype != tensor.dtype
        or output.device != tensor.device
        or not output.is_contiguous()
    ):
        raise ValueError(
            "zig-zag permutation output mismatch: "
            f"input={tuple(tensor.shape)}/{tensor.dtype}/{tensor.device}, "
            f"output={tuple(output.shape)}/{output.dtype}/{output.device}, "
            f"output_contiguous={output.is_contiguous()}"
        )
    if output.data_ptr() == tensor.data_ptr():
        raise ValueError("zig-zag permutation does not support in-place output")
    return output


def is_triton_zigzag_permute_supported(tensor: torch.Tensor) -> bool:
    """Return whether ``tensor`` can use the Triton fast path."""

    return (
        tensor.is_cuda
        and tensor.is_contiguous()
        and tensor.dtype in (torch.float16, torch.bfloat16, torch.float32)
    )


def _permute(
    tensor: torch.Tensor,
    world_size: int,
    *,
    to_canonical: bool,
    output: Optional[torch.Tensor],
) -> torch.Tensor:
    row_width = _validate_layout(tensor, world_size)
    if not is_triton_zigzag_permute_supported(tensor):
        raise ValueError(
            "Triton zig-zag permutation requires a contiguous CUDA tensor with "
            "dtype float16, bfloat16, or float32: "
            f"device={tensor.device}, dtype={tensor.dtype}, "
            f"contiguous={tensor.is_contiguous()}"
        )
    result = _prepare_output(tensor, output)
    total_tokens = tensor.shape[0]
    block_tokens = total_tokens // (2 * world_size)
    grid = (total_tokens, triton.cdiv(row_width, _BLOCK_WIDTH))
    _zigzag_permute_kernel[grid](
        tensor,
        result,
        row_width,
        block_tokens,
        WORLD_SIZE=world_size,
        TO_CANONICAL=to_canonical,
        BLOCK_WIDTH=_BLOCK_WIDTH,
        num_warps=8,
    )
    return result


def rank_major_zigzag_to_canonical(
    tensor: torch.Tensor,
    world_size: int,
    *,
    output: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Permute normal-AllGather output from rank-major zig-zag to canonical."""

    return _permute(
        tensor,
        world_size,
        to_canonical=True,
        output=output,
    )


def canonical_to_rank_major_zigzag(
    tensor: torch.Tensor,
    world_size: int,
    *,
    output: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Pre-permute canonical partials for a normal ReduceScatter."""

    return _permute(
        tensor,
        world_size,
        to_canonical=False,
        output=output,
    )


__all__ = [
    "canonical_to_rank_major_zigzag",
    "is_triton_zigzag_permute_supported",
    "rank_major_zigzag_to_canonical",
]
