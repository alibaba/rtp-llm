"""Copy selected prefill state pages without materializing the strided pool."""

import torch
import triton
import triton.language as tl

_BLOCK = 4096
_TILE_BYTES = 256 * 1024
_MAX_TILES = 32


@triton.jit
def _gather_pages_kernel(
    pool,
    pages,
    output,
    page_stride,
    row_elems,
    BLOCK: tl.constexpr,
    TILES: tl.constexpr,
):
    row = tl.program_id(0)
    tile = tl.program_id(1)
    page = tl.load(pages + row).to(tl.int64)
    work = tl.cdiv(row_elems, TILES)
    start = tile.to(tl.int64) * work
    end = tl.minimum(start + work, row_elems.to(tl.int64))
    offsets = tl.arange(0, BLOCK)
    for offset in range(start, end, BLOCK):
        mask = offset + offsets < end
        values = tl.load(pool + page * page_stride + offset + offsets, mask=mask)
        tl.store(
            output + row.to(tl.int64) * row_elems + offset + offsets, values, mask=mask
        )


@triton.jit
def _scatter_pages_kernel(
    pool,
    pages,
    source,
    page_stride,
    row_elems,
    BLOCK: tl.constexpr,
    TILES: tl.constexpr,
):
    row = tl.program_id(0)
    tile = tl.program_id(1)
    page = tl.load(pages + row).to(tl.int64)
    work = tl.cdiv(row_elems, TILES)
    start = tile.to(tl.int64) * work
    end = tl.minimum(start + work, row_elems.to(tl.int64))
    offsets = tl.arange(0, BLOCK)
    for offset in range(start, end, BLOCK):
        mask = offset + offsets < end
        values = tl.load(
            source + row.to(tl.int64) * row_elems + offset + offsets, mask=mask
        )
        tl.store(pool + page * page_stride + offset + offsets, values, mask=mask)


def gather_state_pages(pool: torch.Tensor, pages: torch.Tensor) -> torch.Tensor:
    """Gather selected rows; pool inner dimensions and page indices are contiguous."""
    rows = pages.numel()
    output = torch.empty((rows, *pool.shape[1:]), dtype=pool.dtype, device=pool.device)
    if rows == 0:
        return output
    row_elems = output.numel() // rows
    tiles = max(1, min(_MAX_TILES, row_elems * pool.element_size() // _TILE_BYTES))
    _gather_pages_kernel[(rows, tiles)](
        pool, pages, output, pool.stride(0), row_elems, BLOCK=_BLOCK, TILES=tiles
    )
    return output


def scatter_state_pages(
    pool: torch.Tensor, pages: torch.Tensor, source: torch.Tensor
) -> None:
    """Write selected rows, preserving every other page and each page's padding."""
    rows = pages.numel()
    if rows == 0:
        return
    row_elems = source.numel() // rows
    tiles = max(1, min(_MAX_TILES, row_elems * pool.element_size() // _TILE_BYTES))
    _scatter_pages_kernel[(rows, tiles)](
        pool, pages, source, pool.stride(0), row_elems, BLOCK=_BLOCK, TILES=tiles
    )
