"""Views and block-table geometry for a single resident FP8 indexer cache.

RTP allocates one raw 128-token FULL block. Ratio-2 source layers write 64
indexer keys into one 64-entry FP8 page. The ratio-1 source writes 128 keys;
its 16,896-byte raw block is two consecutive DeepGEMM 64-entry pages. Each
page is planar: 8,192 FP8 payload bytes followed by 256 FP32-scale bytes.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


PAGE_ENTRIES = 64
ENTRY_BYTES = 132


@triton.jit
def _initialize_zero_page_kernel(pool):
    tile = tl.program_id(0)
    byte = tile * 256 + tl.arange(0, 256)
    tl.store(pool + byte, 0)
    if tile == 0:
        scale = tl.arange(0, 64)
        tl.store((pool + 64 * 128 + scale * 4).to(tl.pointer_type(tl.float32)), 1.0)


def initialize_zero_page(pages: torch.Tensor) -> None:
    """Make RTP's unallocated block-zero sentinel a numerical zero K page."""
    if pages.dtype != torch.uint8 or pages.shape[1:] != (64, 132):
        raise ValueError("FP8 indexer pages must be uint8 [P,64,132]")
    _initialize_zero_page_kernel[(32,)](pages, num_warps=4)


def dg_pages(pool: torch.Tensor) -> torch.Tensor:
    """Return a zero-copy `[pages,64,132]` view of the primary indexer pool."""
    if pool.dtype != torch.uint8 or pool.ndim != 3 or pool.shape[2] != ENTRY_BYTES:
        raise ValueError("FP8 indexer pool must be uint8 [blocks,entries,132]")
    if pool.shape[1] not in (PAGE_ENTRIES, 2 * PAGE_ENTRIES):
        raise ValueError("FP8 indexer source must have 64 or 128 entries/block")
    if not pool.is_contiguous():
        raise ValueError("FP8 indexer pool must have contiguous planar blocks")
    return pool.view(-1, PAGE_ENTRIES, ENTRY_BYTES)


@triton.jit
def _expand_table_kernel(src, dst, B: tl.constexpr, N: tl.constexpr, STRIDE: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = i < B * N
    b = i // N
    n = i % N
    raw = tl.load(src + b * STRIDE + n, mask=mask, other=0)
    page0 = tl.where(raw > 0, raw * 2, 0)
    tl.store(dst + b * (2 * N) + 2 * n, page0, mask=mask)
    tl.store(dst + b * (2 * N) + 2 * n + 1, tl.where(raw > 0, page0 + 1, 0), mask=mask)


@triton.jit
def _sanitize_table_kernel(src, dst, B: tl.constexpr, N: tl.constexpr, STRIDE: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = i < B * N
    raw = tl.load(src + (i // N) * STRIDE + i % N, mask, other=0)
    tl.store(dst + i, tl.maximum(raw, 0), mask)


def dg_block_table(table: torch.Tensor, pool: torch.Tensor) -> torch.Tensor:
    """Map RTP raw-block IDs to DeepGEMM physical 64-entry page IDs."""
    if table.dtype != torch.int32 or table.ndim != 2 or table.stride(1) != 1:
        raise ValueError("block table must be int32 [batch,raw_blocks]")
    if pool.shape[1] == PAGE_ENTRIES:
        b, n = table.shape
        result = torch.empty((b, n), dtype=torch.int32, device=table.device)
        if b * n:
            _sanitize_table_kernel[(triton.cdiv(b * n, 256),)](
                table, result, B=b, N=n, STRIDE=table.stride(0), BLOCK=256,
                num_warps=4,
            )
        return result
    if pool.shape[1] != 2 * PAGE_ENTRIES:
        raise ValueError("unsupported FP8 indexer page geometry")
    b, n = table.shape
    result = torch.empty((b, 2 * n), dtype=torch.int32, device=table.device)
    if b * n:
        _expand_table_kernel[(triton.cdiv(b * n, 256),)](
            table, result, B=b, N=n, STRIDE=table.stride(0), BLOCK=256,
            num_warps=4,
        )
    return result
