import torch
import triton
import triton.language as tl


@triton.jit
def _detect_nan(
    values,
    records,
    width: tl.constexpr,
    row_stride: tl.constexpr,
    column_stride: tl.constexpr,
    layer_id,
    stage_id,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    record = records + row * 4
    compute_dtype: tl.constexpr = (
        tl.float64 if values.dtype.element_ty == tl.float64 else tl.float32
    )
    if tl.load(record + 3) == 0:
        first = tl.full((), width, tl.int64)
        offsets = tl.arange(0, BLOCK)
        for start in range(tl.cdiv(width, BLOCK)):
            columns = start * BLOCK + offsets
            value = tl.load(
                values + row * row_stride + columns * column_stride,
                columns < width,
                other=0,
            ).to(compute_dtype)
            bad = (value != value) | (tl.abs(value) == float("inf"))
            tile_first = tl.min(tl.where(bad, columns.to(tl.int64), width), 0)
            first = tl.minimum(first, tile_first)
        if first < width:
            value = tl.load(values + row * row_stride + first * column_stride).to(
                compute_dtype
            )
            kind = tl.where(value != value, 1, tl.where(value > 0, 2, 3))
            tl.store(record, layer_id)
            tl.store(record + 1, stage_id)
            tl.store(record + 2, first)
            tl.store(record + 3, kind)


def detect_nan_cuda(
    values: torch.Tensor, records: torch.Tensor, layer_id: int, stage_id: int
) -> None:
    width = values.shape[1]
    block = min(triton.next_power_of_2(width), 1024)
    _detect_nan[(values.shape[0],)](
        values,
        records,
        width,
        values.stride(0),
        values.stride(1),
        layer_id,
        stage_id,
        block,
        num_warps=4,
    )
