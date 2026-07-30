"""Stage FP8 activation scales for DeepGEMM's fused shared expert."""

from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl
except Exception:
    triton = None
    tl = None


def shared_fp8_scale_row_indices(
    tokens: int,
    block_m: int,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Map token rows to DeepGEMM's shared-L1 UTCCP scale layout."""
    if tokens < 0:
        raise ValueError(f"tokens must be non-negative, got {tokens}")
    if block_m <= 0:
        raise ValueError(f"block_m must be positive, got {block_m}")
    token_idx = torch.arange(tokens, dtype=torch.long, device=device)
    within_block = token_idx % block_m
    aligned_block_m = ((block_m + 127) // 128) * 128
    return (
        token_idx // block_m * aligned_block_m
        + (within_block // 128) * 128
        + (within_block % 32) * 4
        + (within_block % 128) // 32
    )


if triton is not None:

    @triton.jit
    def _stage_shared_fp8_scale_kernel(
        source_ptr,
        destination_ptr,
        tokens,
        active_destination_rows,
        source_stride_m: tl.constexpr,
        source_stride_k: tl.constexpr,
        destination_stride_m: tl.constexpr,
        destination_stride_k: tl.constexpr,
        BLOCK_M: tl.constexpr,
        ALIGNED_BLOCK_M: tl.constexpr,
        BLOCK_ROWS: tl.constexpr,
    ):
        destination_row = tl.program_id(0) * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
        scale_col = tl.program_id(1)
        local_destination_row = destination_row % ALIGNED_BLOCK_M
        row_in_128_tile = local_destination_row % 128
        row_in_block = (
            local_destination_row // 128 * 128
            + (row_in_128_tile % 4) * 32
            + row_in_128_tile // 4
        )
        source_row = destination_row // ALIGNED_BLOCK_M * BLOCK_M + row_in_block
        valid_destination = destination_row < active_destination_rows
        valid_source = (
            valid_destination & (row_in_block < BLOCK_M) & (source_row < tokens)
        )
        value = tl.load(
            source_ptr + source_row * source_stride_m + scale_col * source_stride_k,
            mask=valid_source,
            other=0,
        )
        tl.store(
            destination_ptr
            + destination_row * destination_stride_m
            + scale_col * destination_stride_k,
            value,
            mask=valid_destination,
        )


def stage_shared_fp8_input_scales(
    source: torch.Tensor,
    destination: torch.Tensor,
    tokens: int,
    block_m: int,
) -> None:
    """Copy routed input scales into the layout consumed by shared L1."""
    if tokens == 0:
        destination.zero_()
        return
    if block_m <= 0:
        raise ValueError(f"block_m must be positive, got {block_m}")
    if source.dtype != torch.int32 or destination.dtype != torch.int32:
        raise TypeError(
            "shared FP8 input scales must be packed int32 tensors, got "
            f"source={source.dtype}, destination={destination.dtype}"
        )
    if source.dim() != 2 or destination.dim() != 2:
        raise ValueError(
            "shared FP8 input scales must be 2D, got "
            f"source={tuple(source.shape)}, destination={tuple(destination.shape)}"
        )
    if source.size(1) != destination.size(1):
        raise ValueError(
            "shared FP8 input scale width mismatch: "
            f"source={source.size(1)}, destination={destination.size(1)}"
        )

    aligned_block_m = ((block_m + 127) // 128) * 128
    required_destination_rows = ((tokens + block_m - 1) // block_m) * aligned_block_m
    if tokens > source.size(0) or required_destination_rows > destination.size(0):
        raise ValueError(
            "shared FP8 input scale buffer is too small: "
            f"tokens={tokens}/{source.size(0)}, destination_rows="
            f"{required_destination_rows}/{destination.size(0)}"
        )

    if triton is not None and source.is_cuda and destination.is_cuda:
        block_rows = 128
        destination_rows = destination.size(0)
        grid = (triton.cdiv(destination_rows, block_rows), source.size(1))
        _stage_shared_fp8_scale_kernel[grid](
            source,
            destination,
            tokens,
            destination_rows,
            source.stride(0),
            source.stride(1),
            destination.stride(0),
            destination.stride(1),
            BLOCK_M=block_m,
            ALIGNED_BLOCK_M=aligned_block_m,
            BLOCK_ROWS=block_rows,
            num_warps=4,
        )
        return

    destination.zero_()
    row_indices = shared_fp8_scale_row_indices(tokens, block_m, source.device)
    destination.index_copy_(0, row_indices, source[:tokens])
