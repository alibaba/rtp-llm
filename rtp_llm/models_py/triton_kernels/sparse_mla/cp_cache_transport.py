"""Byte-preserving CP cache transport with tiled pack and fused consumers."""

from dataclasses import dataclass

import torch
import triton
import triton.language as tl

from rtp_llm.models_py.triton_kernels.sparse_mla.cp_cache_restore import (
    _restore_indexer,
    _restore_mla,
)


@dataclass(frozen=True, slots=True)
class PagedRows:
    """Request/local-token pairs address a paged cache; -1 denotes padding."""

    cache: torch.Tensor
    table: torch.Tensor
    rows: torch.Tensor
    index_dim: int = 0


@dataclass(frozen=True, slots=True)
class PackedRows:
    """A row section within equal-sized, rank-major communication packets."""

    data: torch.Tensor
    rows_per_rank: int
    width: int
    rank_bytes: int
    offset: int = 0


@triton.jit
def _row_map(
    Actual,
    Padded,
    Out,
    N,
    B: tl.constexpr,
    REQUESTS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    req = tl.arange(0, REQUESTS)
    padded = tl.load(Padded + req, req < B, other=0)
    ends = tl.cumsum(padded)
    row = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    batch = tl.sum((row[:, None] >= ends[None, :]) & (req[None, :] < B), 1)
    start = tl.sum(tl.where(req[None, :] < batch[:, None], padded[None, :], 0), 1)
    actual = tl.load(Actual + batch, (row < N) & (batch < B), other=0)
    local = row - start
    encoded = batch.to(tl.int64) << 32 | local.to(tl.int64)
    tl.store(Out + row, tl.where(local < actual, encoded, -1), row < N)


def paged_row_map(
    actual: torch.Tensor, padded: torch.Tensor, count: int
) -> torch.Tensor:
    output = torch.empty(count, dtype=torch.int64, device=actual.device)
    if count:
        _row_map[(triton.cdiv(count, 32),)](
            actual,
            padded,
            output,
            count,
            actual.numel(),
            triton.next_power_of_2(actual.numel()),
            32,
        )
    return output


@triton.jit
def _pack(
    Cache,
    Table,
    Rows,
    Out,
    N,
    WIDTH: tl.constexpr,
    PAGE: tl.constexpr,
    BS: tl.constexpr,
    TS,
    INDEX: tl.constexpr,
    COLS: tl.constexpr,
    TILE: tl.constexpr,
):
    row = tl.program_id(0) * TILE + tl.arange(0, TILE)
    col = tl.arange(0, COLS)
    encoded = tl.load(Rows + row, row < N, other=-1)
    valid = (row < N) & (encoded >= 0)
    request = encoded >> 32
    token = encoded & 0xFFFFFFFF
    block = tl.load(Table + request * TS + token // PAGE, valid, other=0)
    slot = token % PAGE
    if INDEX:
        address = tl.where(
            col[None, :] < INDEX,
            slot[:, None] * INDEX + col[None, :],
            PAGE * INDEX + slot[:, None] * (WIDTH - INDEX) + col[None, :] - INDEX,
        )
    else:
        address = slot[:, None] * WIDTH + col[None, :]
    value = tl.load(
        Cache + block[:, None].to(tl.int64) * BS + address,
        valid[:, None] & (col[None, :] < WIDTH),
        other=0,
    )
    tl.store(
        Out + row[:, None].to(tl.int64) * WIDTH + col[None, :],
        value,
        (row[:, None] < N) & (col[None, :] < WIDTH),
    )


def is_supported(source: PagedRows) -> bool:
    cache = source.cache
    return (
        cache.is_cuda
        and cache.dtype == torch.uint8
        and cache.ndim == 3
        and cache.is_contiguous()
        and cache.shape[-1] > source.index_dim
        and source.table.device == cache.device
        and source.table.ndim == 2
        and source.table.stride(1) == 1
        and source.rows.device == cache.device
        and source.rows.ndim == 1
        and source.rows.dtype == torch.int64
        and source.rows.is_contiguous()
        and not torch.cuda.is_current_stream_capturing()
    )


def pack_rows(source: PagedRows, output: torch.Tensor | None = None) -> torch.Tensor:
    """Read cache bytes straight into their padded or joint-packet send section."""
    width, count = source.cache.shape[-1], source.rows.numel()
    if output is None:
        output = torch.empty(
            (count, width), dtype=torch.uint8, device=source.cache.device
        )
    if count:
        _pack[(triton.cdiv(count, 4),)](
            source.cache,
            source.table,
            source.rows,
            output,
            count,
            width,
            source.cache.shape[1],
            source.cache.stride(0),
            source.table.stride(0),
            source.index_dim,
            triton.next_power_of_2(width),
            4,
        )
    return output


def restore_mla(
    view: PackedRows,
    mapping: torch.Tensor,
    output: torch.Tensor,
    destinations: torch.Tensor | None = None,
) -> None:
    """Dequantize directly to final rows; negative mappings leave new-token holes untouched."""
    if mapping.numel() and view.rows_per_rank:
        _restore_mla[(triton.cdiv(mapping.numel(), 4),)](
            view.data,
            mapping,
            mapping if destinations is None else destinations,
            output,
            mapping.numel(),
            view.rows_per_rank,
            view.rank_bytes,
            view.offset,
            destinations is not None,
            4,
        )


def restore_indexer(
    view: PackedRows, mapping: torch.Tensor, outputs: tuple[torch.Tensor, torch.Tensor]
) -> None:
    """Restore FP8 values and scale bytes without allocating a restored packet."""
    key, scales = outputs
    if mapping.numel() and view.rows_per_rank:
        _restore_indexer[(triton.cdiv(mapping.numel(), 4),)](
            view.data,
            mapping,
            key.view(torch.uint8),
            scales,
            mapping.numel(),
            view.rows_per_rank,
            view.width,
            key.shape[1],
            view.rank_bytes,
            view.offset,
            triton.next_power_of_2(view.width),
            4,
        )
