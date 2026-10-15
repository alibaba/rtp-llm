"""Fused PLE commit destination verdict and host-mirror payload."""

import torch
import triton
import triton.language as tl


@triton.jit
def _commit_block_verdict_kernel(
    TABLE,
    PAGES,
    OUTPUT,
    B: tl.constexpr,
    COLUMNS: tl.constexpr,
    STRIDE_ROW: tl.constexpr,
    STRIDE_COL: tl.constexpr,
    POOL_ROWS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.arange(0, BLOCK)
    live = row < B
    page = tl.load(PAGES + row, live, other=0)
    safe_page = tl.minimum(tl.maximum(page, 0), COLUMNS - 1)
    block = tl.load(
        TABLE + row.to(tl.int64) * STRIDE_ROW + safe_page * STRIDE_COL, live, other=0
    ).to(tl.int64)
    tl.store(OUTPUT, tl.sum((live & (page < 0)).to(tl.int32), 0) > 0)
    tl.store(OUTPUT + 1, tl.sum((live & (page >= COLUMNS)).to(tl.int32), 0) > 0)
    tl.store(OUTPUT + 2, tl.sum((live & (block <= 0)).to(tl.int32), 0) > 0)
    tl.store(OUTPUT + 3, tl.sum((live & (block >= POOL_ROWS)).to(tl.int32), 0) > 0)
    tl.store(OUTPUT + 4 + row, block, live)


def resolve_commit_blocks(table: torch.Tensor, pages: torch.Tensor, pool_rows: int):
    """Return [four ordered error flags, B block IDs], or None if unsupported.

    This is a device-only operation. The caller must inspect the flags before
    reading or writing persistent pool rows. Clamped lookups never validate an
    invalid logical page.
    """
    if not (
        table.is_cuda
        and pages.is_cuda
        and table.device == pages.device
        and table.ndim == 2
        and table.dtype == torch.int32
        and pages.ndim == 1
        and pages.dtype == torch.int64
        and pages.is_contiguous()
        and 0 < pages.numel() <= 1024
        and table.shape[0] == pages.numel()
        and table.shape[1] > 0
        and pool_rows > 0
    ):
        return None
    result = torch.empty(4 + pages.numel(), dtype=torch.int64, device=pages.device)
    _commit_block_verdict_kernel[(1,)](
        table,
        pages,
        result,
        pages.numel(),
        table.shape[1],
        table.stride(0),
        table.stride(1),
        pool_rows,
        triton.next_power_of_2(pages.numel()),
        num_warps=4,
    )
    return result
