"""C4 Q3 batch page gather and native K/Q scale preparation."""

import torch
import triton
import triton.language as tl


@triton.jit
def _batch_gather(
    cache,
    table,
    lengths,
    packed,
    prepared,
    WIDTH: tl.constexpr,
    COLUMNS: tl.constexpr,
    PAGES: tl.constexpr,
    BLOCK: tl.constexpr,
):
    request = tl.program_id(1)
    row = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    end = tl.maximum(
        tl.maximum(tl.load(lengths + request * 3), tl.load(lengths + request * 3 + 1)),
        tl.load(lengths + request * 3 + 2),
    )
    valid = (row < WIDTH) & (row < end) & (row // 64 < COLUMNS)
    physical = tl.load(table + request * COLUMNS + row // 64, valid, other=0)
    valid = valid & (physical > 0) & (physical < PAGES)
    page = tl.where(valid, physical, 0).to(tl.int64) * (64 * 68)
    lane = tl.arange(0, 64)
    payload = tl.load(
        cache + page[:, None] + (row % 64)[:, None] * 64 + lane[None, :],
        valid[:, None],
        other=0,
    )
    tl.store(
        packed + (request * WIDTH + row[:, None]) * 64 + lane[None, :],
        payload.to(tl.int8, bitcast=True),
        row[:, None] < WIDTH,
    )
    scale = cache + page + 64 * 64 + (row % 64) * 4
    for pair in tl.static_range(2):
        low = tl.load(scale + pair * 2, valid, other=127).to(tl.uint32)
        high = tl.load(scale + pair * 2 + 1, valid, other=127).to(tl.uint32)
        tl.store(
            prepared + request * 2 * WIDTH + pair * WIDTH + row,
            (low | (high << 8)).to(tl.uint16),
            row < WIDTH,
        )


@triton.jit
def _batch_prepare_q(scales, prepared, starts, BLOCK: tl.constexpr):
    request = tl.program_id(0)
    row = tl.arange(0, BLOCK)
    for pair in tl.static_range(2):
        value = tl.load(scales + (request * 192 + row) * 2 + pair, row < 192, other=0)
        tl.store(prepared + request * 384 + pair * 192 + row, value, row < 192)
    tl.store(starts + request * 3 + row, 0, row < 3)


def can_batch_prepare(cache, table, lengths, scales, width):
    return (
        cache.is_cuda
        and torch.cuda.get_device_name(cache.device) == "ZW-M890P"
        and width == 12288
        and cache.ndim == 3
        and cache.shape[1:] == (64, 68)
        and cache.dtype == torch.uint8
        and cache.is_contiguous()
        and table.shape == (8, 192)
        and lengths.shape == (24,)
        and scales.shape == (24, 64, 1)
        and all(
            x.dtype == torch.int32 and x.device == cache.device and x.is_contiguous()
            for x in (table, lengths, scales)
        )
    )


def gather_batch_prepared(cache, table, lengths, scales, width):
    assert can_batch_prepare(cache, table, lengths, scales, width)
    packed = torch.empty((8, width, 64), dtype=torch.int8, device=cache.device)
    kscale = torch.empty((8, 2, width), dtype=torch.uint16, device=cache.device)
    qscale = torch.empty((8, 2, 192), dtype=torch.uint16, device=cache.device)
    starts = torch.empty((24,), dtype=torch.int32, device=cache.device)
    _batch_gather[(triton.cdiv(width, 16), 8)](
        cache, table, lengths, packed, kscale, width, table.shape[1], cache.shape[0], 16
    )
    _batch_prepare_q[(8,)](scales.view(torch.uint16), qscale, starts, 256)
    return packed, kscale.transpose(1, 2), qscale.transpose(1, 2), starts
