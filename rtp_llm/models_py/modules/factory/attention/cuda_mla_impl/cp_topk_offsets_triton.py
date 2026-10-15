"""Fuse CP request offsets with preservation of invalid sparse KV indices."""

import os

import torch
import triton
import triton.language as tl


@triton.jit
def _offset_topk_kernel(
    indices,
    offsets,
    output,
    rows,
    width: tl.constexpr,
    row_stride: tl.constexpr,
    col_stride: tl.constexpr,
    offset_stride: tl.constexpr,
    BLOCK: tl.constexpr,
):
    positions = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    row = positions // width
    col = positions % width
    valid = row < rows
    index = tl.load(indices + row * row_stride + col * col_stride, mask=valid, other=-1)
    offset = tl.load(offsets + row * offset_stride, mask=valid, other=0)
    # Convert before addition to preserve torch's int32/int64 promotion.
    global_index = index.to(output.dtype.element_ty) + offset.to(
        output.dtype.element_ty
    )
    tl.store(output + positions, tl.where(index >= 0, global_index, -1), mask=valid)


def is_supported(indices: torch.Tensor, offsets: torch.Tensor) -> bool:
    return (
        indices.is_cuda
        and offsets.device == indices.device
        and indices.ndim == 2
        and offsets.ndim == 1
        and offsets.shape[0] == indices.shape[0]
        and indices.dtype in (torch.int32, torch.int64)
        and offsets.dtype in (torch.int32, torch.int64)
    )


def offset_cp_topk(indices: torch.Tensor, offsets: torch.Tensor) -> torch.Tensor:
    """Return independent global indices; negative local entries remain -1."""
    if os.environ.get("RTP_CP_FUSED_TOPK_OFFSETS", "1") != "1" or not is_supported(
        indices, offsets
    ):
        result = indices + offsets.unsqueeze(1)
        return result.masked_fill_(indices < 0, -1)

    result = torch.empty(
        indices.shape,
        dtype=torch.promote_types(indices.dtype, offsets.dtype),
        device=indices.device,
    )
    if indices.numel():
        _offset_topk_kernel[(triton.cdiv(indices.numel(), 256),)](
            indices,
            offsets,
            result,
            indices.shape[0],
            indices.shape[1],
            indices.stride(0),
            indices.stride(1),
            offsets.stride(0),
            BLOCK=256,
        )
    return result
