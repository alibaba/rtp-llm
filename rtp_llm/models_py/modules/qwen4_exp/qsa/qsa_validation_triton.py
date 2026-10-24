"""Optional one-launch validation of QSA indexer cache block tables."""

import torch
import triton
import triton.language as tl

_MAX_ELEMENTS = 4096


@triton.jit
def _validate_required_blocks_kernel(
    table_ptr,
    required_ptr,
    result_ptr,
    TABLE_ROW_STRIDE: tl.constexpr,
    TABLE_COL_STRIDE: tl.constexpr,
    REQUIRED_STRIDE: tl.constexpr,
    BATCH: tl.constexpr,
    WIDTH: tl.constexpr,
    POOL_BLOCKS: tl.constexpr,
    N: tl.constexpr,
):
    offset = tl.arange(0, N)
    row = offset // WIDTH
    column = offset % WIDTH
    active = offset < BATCH * WIDTH
    required = tl.load(required_ptr + row * REQUIRED_STRIDE, mask=active, other=0)
    block = tl.load(
        table_ptr + row * TABLE_ROW_STRIDE + column * TABLE_COL_STRIDE,
        mask=active,
        other=0,
    )
    invalid = active & (
        (required < 0)
        | (required > WIDTH)
        | (block >= POOL_BLOCKS)
        | ((column < required) & (block <= 0))
    )
    tl.store(result_ptr, tl.max(invalid.to(tl.int32), 0))


def is_supported(table: torch.Tensor, required_columns: torch.Tensor) -> bool:
    """Keep the one-program reduction bounded; other shapes use the Torch path."""
    return (
        table.is_cuda
        and torch.version.hip is None
        and required_columns.is_cuda
        and table.device == required_columns.device
        and table.dtype == torch.int32
        and required_columns.dtype == torch.int32
        and table.dim() == 2
        and required_columns.dim() == 1
        and table.shape[0] == required_columns.numel()
        and 0 < table.shape[0] * table.shape[1] <= _MAX_ELEMENTS
    )


def required_blocks_are_valid(
    table: torch.Tensor, required_columns: torch.Tensor, pool_blocks: int
) -> bool:
    """Return one device-computed verdict, synchronizing only for that scalar."""
    batch, width = table.shape
    result = torch.empty((), dtype=torch.int32, device=table.device)
    _validate_required_blocks_kernel[(1,)](
        table,
        required_columns,
        result,
        TABLE_ROW_STRIDE=table.stride(0),
        TABLE_COL_STRIDE=table.stride(1),
        REQUIRED_STRIDE=required_columns.stride(0),
        BATCH=batch,
        WIDTH=width,
        POOL_BLOCKS=pool_blocks,
        N=triton.next_power_of_2(batch * width),
    )
    return not bool(result.item())
