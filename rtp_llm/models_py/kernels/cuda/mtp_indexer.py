"""Integer-only MTP shared-index updates."""

import torch
import triton
import triton.language as tl
from rtp_llm.models_py import mtp_optimization


@triton.jit
def _advance(
    indices, positions, width: tl.constexpr, stride: tl.constexpr, block: tl.constexpr
):
    row = tl.program_id(0)
    col = tl.arange(0, block)
    position = tl.load(positions + row)
    old = tl.load(indices + row * stride + col, col < width, other=-1)
    present = tl.sum(((old == position) & (col < width)).to(tl.int32), 0) > 0
    shifted = tl.load(
        indices + row * stride + col - 1, (col > 0) & (col < width), other=0
    )
    shifted = tl.where(col == 0, position, shifted)
    # The source and destination overlap across warps within each row.
    tl.debug_barrier()
    tl.store(indices + row * stride + col, tl.where(present, old, shifted), col < width)


@triton.jit
def _seed(
    source,
    rows,
    output,
    width: tl.constexpr,
    source_stride: tl.constexpr,
    output_stride: tl.constexpr,
    block: tl.constexpr,
):
    row = tl.program_id(0)
    col = tl.arange(0, block)
    source_row = tl.load(rows + row)
    value = tl.load(source + source_row * source_stride + col, col < width, other=0)
    tl.store(output + row * output_stride + col, value, col < width)


def try_advance(indices, positions):
    if not mtp_optimization.ENABLED:
        return False
    if (
        indices.ndim != 2
        or indices.dtype != torch.int32
        or not indices.is_cuda
        or indices.stride(1) != 1
        or indices.shape[1] <= 0
        or indices.shape[1] > 8192
        or not torch.is_tensor(positions)
        or positions.device != indices.device
        or positions.dtype != torch.int32
        or not positions.is_contiguous()
        or positions.numel() != indices.shape[0]
    ):
        return False
    if indices.shape[0]:
        _advance[(indices.shape[0],)](
            indices,
            positions,
            indices.shape[1],
            indices.stride(0),
            triton.next_power_of_2(indices.shape[1]),
            num_warps=4,
        )
    return True


def try_seed(source, rows, output):
    if not mtp_optimization.ENABLED:
        return False
    if (
        source.ndim != 2
        or output.ndim != 2
        or source.dtype != torch.int32
        or output.dtype != torch.int32
        or not source.is_cuda
        or output.device != source.device
        or rows.device != source.device
        or rows.dtype != torch.int32
        or rows.ndim != 1
        or not rows.is_contiguous()
        or source.stride(1) != 1
        or output.stride(1) != 1
        or source.shape[1] != output.shape[1]
        or output.shape[0] != rows.numel()
        or source.shape[1] <= 0
        or source.shape[1] > 8192
        or source.untyped_storage().data_ptr() == output.untyped_storage().data_ptr()
    ):
        return False
    if rows.numel():
        _seed[(rows.numel(),)](
            source,
            rows,
            output,
            source.shape[1],
            source.stride(0),
            output.stride(0),
            triton.next_power_of_2(source.shape[1]),
            num_warps=4,
        )
    return True
