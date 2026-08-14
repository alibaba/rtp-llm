"""Fused Top-K index postprocessing for sparse-MLA prefill."""

from __future__ import annotations

from typing import Optional

import torch
import triton
import triton.language as tl


@triton.jit
def _topk_stage1_offset_mask_kernel(
    raw_indices_ptr,
    ragged_offsets_ptr,
    request_output_ptr,
    num_elements,
    N_COLS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    linear = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = linear < num_elements
    raw = tl.load(raw_indices_ptr + linear, mask=mask)
    row = linear // N_COLS
    ragged_offset = tl.load(ragged_offsets_ptr + row, mask=mask)
    request_local = tl.where(raw >= 0, raw + ragged_offset, raw)
    tl.store(request_output_ptr + linear, request_local, mask=mask)


@triton.jit
def _topk_stage2_workspace_mask_kernel(
    request_indices_ptr,
    workspace_offsets_ptr,
    global_output_ptr,
    num_elements,
    N_COLS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    linear = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = linear < num_elements
    request_local = tl.load(request_indices_ptr + linear, mask=mask)
    row = linear // N_COLS
    workspace_offset = tl.load(workspace_offsets_ptr + row, mask=mask)
    attention_global = tl.where(
        request_local < 0,
        -1,
        request_local + workspace_offset,
    )
    tl.store(global_output_ptr + linear, attention_global, mask=mask)


def _supported(
    indices: torch.Tensor,
    row_offsets: torch.Tensor,
    output: Optional[torch.Tensor],
) -> bool:
    if not (
        indices.is_cuda
        and row_offsets.is_cuda
        and indices.dtype == torch.int32
        and row_offsets.dtype == torch.int32
        and indices.ndim == 2
        and row_offsets.ndim == 1
        and indices.is_contiguous()
        and row_offsets.is_contiguous()
        and row_offsets.numel() == indices.shape[0]
        and row_offsets.device == indices.device
    ):
        return False
    if output is None:
        return True
    return (
        output.is_cuda
        and output.dtype == torch.int32
        and output.shape == indices.shape
        and output.is_contiguous()
        and output.device == indices.device
    )


def fused_stage1_request_indices(
    raw_indices: torch.Tensor,
    ragged_offsets: torch.Tensor,
    output: Optional[torch.Tensor] = None,
) -> Optional[torch.Tensor]:
    """Add each row's ragged offset while preserving negative sentinels."""

    if not _supported(raw_indices, ragged_offsets, output):
        return None
    if output is None:
        output = torch.empty_like(raw_indices)
    if raw_indices.numel() == 0:
        return output
    block_size = 256
    _topk_stage1_offset_mask_kernel[(triton.cdiv(raw_indices.numel(), block_size),)](
        raw_indices,
        ragged_offsets,
        output,
        raw_indices.numel(),
        N_COLS=raw_indices.shape[1],
        BLOCK_SIZE=block_size,
        num_warps=4,
    )
    return output


def fused_stage2_global_indices(
    request_indices: torch.Tensor,
    workspace_offsets: torch.Tensor,
    output: Optional[torch.Tensor] = None,
) -> Optional[torch.Tensor]:
    """Add each row's workspace offset and canonicalize invalid values to -1."""

    if not _supported(request_indices, workspace_offsets, output):
        return None
    if output is None:
        output = torch.empty_like(request_indices)
    if request_indices.numel() == 0:
        return output
    block_size = 256
    _topk_stage2_workspace_mask_kernel[
        (triton.cdiv(request_indices.numel(), block_size),)
    ](
        request_indices,
        workspace_offsets,
        output,
        request_indices.numel(),
        N_COLS=request_indices.shape[1],
        BLOCK_SIZE=block_size,
        num_warps=4,
    )
    return output
