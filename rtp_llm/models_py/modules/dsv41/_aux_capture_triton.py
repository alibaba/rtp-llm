"""Write the mHC mean directly to the DSpARK feature buffer."""

import torch
import triton
import triton.language as tl


@triton.jit(do_not_specialize=["rows"])
def _capture_kernel(
    source,
    output,
    rows,
    width: tl.constexpr,
    lanes: tl.constexpr,
    source_row_stride: tl.constexpr,
    source_lane_stride: tl.constexpr,
    source_col_stride: tl.constexpr,
    output_stride: tl.constexpr,
    output_col_stride: tl.constexpr,
    column: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = (tl.program_id(0) * 4 + tl.arange(0, 4)).to(tl.int64)
    col = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    valid = (row[:, None] < rows) & (col[None, :] < width)
    offset = row[:, None] * source_row_stride + col[None, :] * source_col_stride
    total = tl.full((4, BLOCK), 0, tl.float32)
    for lane in tl.static_range(lanes):
        total += tl.load(
            source + offset + lane * source_lane_stride, valid, other=0
        ).to(tl.float32)
    mean = total / lanes
    tl.store(
        output
        + row[:, None] * output_stride
        + (column + col[None, :]) * output_col_stride,
        mean,
        valid,
    )


def capture_aux_hidden(
    hidden: torch.Tensor, buffer: torch.Tensor, segment: int
) -> None:
    rows, lanes = hidden.shape[:2]
    width = hidden.shape[-1]
    _capture_kernel[(triton.cdiv(rows, 4), triton.cdiv(width, 256))](
        hidden,
        buffer,
        rows,
        width,
        lanes,
        hidden.stride(0),
        hidden.stride(1),
        hidden.stride(2),
        buffer.stride(0),
        buffer.stride(1),
        segment * width,
        256,
        num_warps=4,
    )
