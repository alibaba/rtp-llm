"""Prepare exact BF16 KV operands and membership in one stream launch."""

import torch
import triton
import triton.language as tl

from .membership import _find_request, _write_membership


@triton.jit(
    do_not_specialize=[
        "TOTAL",
        "WIDTH",
        "PAGES",
        "TABLE_ROW",
        "MAX_PAGES",
        "B",
        "WORDS",
        "NQ",
    ]
)
def _prepare_paged_kv(
    Cache,
    Table,
    Scale,
    Cu,
    Prefixes,
    Out,
    OutTable,
    OutScale,
    Ids,
    Mask,
    B,
    WORDS,
    NQ,
    MAKE_MASK: tl.constexpr,
    TOTAL,
    WIDTH,
    PAGES,
    TABLE_ROW,
    MAX_PAGES,
    CACHE_PAGE: tl.constexpr,
    SCALE_PAGE: tl.constexpr,
    SCALE_KV: tl.constexpr,
    PAGE: tl.constexpr,
    BATCHED: tl.constexpr,
):
    # Each CTA owns either one query bitmap or 32 rows of one KV operand.
    row = tl.program_id(0).to(tl.int64)
    if MAKE_MASK and row < NQ:
        local_s = TOTAL
        if BATCHED:
            request = _find_request(Cu, row, B)
            local_s = (
                tl.load(Prefixes + request)
                + tl.load(Cu + request + 1)
                - tl.load(Cu + request)
            )
        _write_membership(Ids, Mask, row, local_s, WORDS, 128)
    else:
        # Page zero remains a safe zero-filled destination for missing pages.
        work = tl.program_id(0) - NQ
        dst = work // (2 * (PAGE // 32))
        packed = tl.maximum(dst - 1, 0)
        request = packed // MAX_PAGES
        logical = packed % MAX_PAGES
        block = work % (2 * (PAGE // 32))
        kv = block // (PAGE // 32)
        page_row = (block % (PAGE // 32)) * 32 + tl.arange(0, 32)
        d = tl.arange(0, 256)
        total = TOTAL
        if BATCHED:
            total = (
                tl.load(Prefixes + request)
                + tl.load(Cu + request + 1)
                - tl.load(Cu + request)
            )
        physical = tl.load(
            Table + request.to(tl.int64) * TABLE_ROW + logical, logical < WIDTH, other=0
        )
        valid = (dst > 0) & (physical > 0) & (physical < PAGES)
        live = valid & (logical * PAGE + page_row < total)
        value = tl.load(
            Cache
            + physical.to(tl.int64) * CACHE_PAGE
            + (kv * PAGE + page_row[:, None]) * 256
            + d[None, :],
            live[:, None],
            other=0.0,
        )
        tl.store(
            Out
            + dst.to(tl.int64) * (2 * PAGE * 256)
            + (kv * PAGE + page_row[:, None]) * 256
            + d[None, :],
            value.to(tl.bfloat16),
        )
        scale = tl.load(
            Scale + physical.to(tl.int64) * SCALE_PAGE + kv * SCALE_KV + page_row,
            live,
            other=0.0,
        )
        tl.store(OutScale + dst.to(tl.int64) * (2 * PAGE) + kv * PAGE + page_row, scale)
        if block == 0:
            tl.store(OutTable + packed, tl.where(valid, dst, 0), dst > 0)


def stage_paged_kv(
    cache,
    table,
    scales,
    total,
    page_size,
    cu=None,
    prefixes=None,
    selected=None,
    membership=None,
):
    # Bound scratch by the validated CPU plan; GPU contents remain dynamic.
    pages = triton.cdiv(total, page_size)
    batch = table.shape[0]
    out = torch.empty(
        (batch * pages + 1, 2 * page_size * 256),
        device=cache.device,
        dtype=torch.bfloat16,
    )
    scale = torch.empty(
        (batch * pages + 1, 2, 1, page_size), device=cache.device, dtype=torch.float32
    )
    new_table = torch.empty((batch, pages), device=cache.device, dtype=torch.int32)
    tokens = 0 if selected is None else selected.numel() // 2051
    _prepare_paged_kv[(tokens + (batch * pages + 1) * 2 * (page_size // 32),)](
        cache,
        table,
        scales,
        table if cu is None else cu,
        table if prefixes is None else prefixes,
        out,
        new_table,
        scale,
        new_table if selected is None else selected,
        new_table if membership is None else membership,
        batch,
        1 if membership is None else membership.shape[1],
        tokens,
        selected is not None,
        total,
        table.shape[1],
        cache.shape[0],
        table.stride(0),
        pages,
        cache.stride(0),
        scales.stride(0),
        scales.stride(1),
        page_size,
        cu is not None,
        num_warps=4,
    )
    return out, new_table, scale
