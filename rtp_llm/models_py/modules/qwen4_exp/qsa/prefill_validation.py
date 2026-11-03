"""Check one invocation's partition, selected tokens and physical page bounds."""

import torch
import triton
import triton.language as tl


@triton.jit(do_not_specialize=["B", "N", "K", "WIDTH", "PAGES"])
def _validate_packed_inputs(
    Ids,
    Cu,
    Lengths,
    Prefixes,
    Table,
    Visible,
    Flag,
    B,
    N,
    K,
    WIDTH,
    PAGES,
    IDS_ROW: tl.constexpr,
    IDS_COL: tl.constexpr,
    TABLE_ROW: tl.constexpr,
    TABLE_COL: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    PADDED: tl.constexpr,
    BLOCK: tl.constexpr,
    BATCH: tl.constexpr,
):
    # Report every failure before the host permits any cache mutation.
    row = tl.program_id(0)
    if row == 0:
        b = tl.arange(0, BATCH)
        start = tl.load(Cu + b, b < B, other=0)
        end = tl.load(Cu + b + 1, b < B, other=0)
        length = tl.load(Lengths + b, b < B, other=0)
        bad = tl.sum(((b < B) & ((end - start) != length)).to(tl.int32), 0) > 0
        bad = bad | (tl.load(Cu) != 0) | (tl.load(Cu + B) != N)
        tl.atomic_or(Flag, 1, bad, sem="relaxed")
    # Each request checks its full table row once, including unused columns.
    if row < B:
        prefix = tl.load(Prefixes + row).to(tl.int64)
        length = tl.load(Lengths + row).to(tl.int64)
        # Widen before rounding up so large invalid prefixes cannot wrap.
        required = tl.cdiv(prefix + length, PAGE_SIZE)
        column = tl.arange(0, 128)
        out_of_range = False
        invalid = False
        active = False
        for tile in range(tl.cdiv(WIDTH, 128)):
            col = tile * 128 + column
            page = tl.load(
                Table + row.to(tl.int64) * TABLE_ROW + col * TABLE_COL,
                col < WIDTH,
                other=0,
            )
            out_of_range = out_of_range | (
                tl.sum(((col < WIDTH) & (page >= PAGES)).to(tl.int32), 0) > 0
            )
            active = active | (
                tl.sum(((col < WIDTH) & (page != 0)).to(tl.int32), 0) > 0
            )
            invalid = invalid | (
                tl.sum(
                    (
                        (col < WIDTH)
                        & (col < required)
                        & ((page <= 0) | (page >= PAGES))
                    ).to(tl.int32),
                    0,
                )
                > 0
            )
        if PADDED:
            required = tl.where(active, required, 0)
            invalid = invalid & active
        flags = (
            out_of_range.to(tl.int32) * 8
            + (required > WIDTH).to(tl.int32) * 16
            + invalid.to(tl.int32) * 32
        )
        tl.atomic_or(Flag, flags, flags != 0, sem="relaxed")
    low = 0
    high = B
    while low < high:
        mid = (low + high) // 2
        if row >= tl.load(Cu + mid + 1):
            low = mid + 1
        else:
            high = mid
    base = tl.load(Cu + low, low < B, other=0)
    prefix = tl.load(Prefixes + low, low < B, other=0)
    visible = prefix + row - base + 1
    tl.store(Visible + row, visible)
    column = tl.arange(0, BLOCK)
    ids = tl.load(
        Ids + row.to(tl.int64) * IDS_ROW + column * IDS_COL, column < K, other=-1
    )
    bad_padding = tl.sum((ids < -1).to(tl.int32), 0) > 0
    bad_visible = tl.sum(((ids >= 0) & (ids >= visible)).to(tl.int32), 0) > 0
    flags = bad_padding.to(tl.int32) * 2 + bad_visible.to(tl.int32) * 4
    tl.atomic_or(Flag, flags, flags != 0, sem="relaxed")


def validate_prefill_inputs(
    selected, cu, lengths, prefixes, table, page_size, pages, padded
):
    tokens, width = selected.shape
    visible = torch.empty(tokens, dtype=torch.int32, device=selected.device)
    flag = torch.zeros(1, dtype=torch.int32, device=selected.device)
    _validate_packed_inputs[(tokens,)](
        selected,
        cu,
        lengths,
        prefixes,
        table,
        visible,
        flag,
        lengths.numel(),
        tokens,
        width,
        table.shape[1],
        pages,
        selected.stride(0),
        selected.stride(1),
        table.stride(0),
        table.stride(1),
        page_size,
        padded,
        triton.next_power_of_2(width),
        triton.next_power_of_2(lengths.numel()),
        num_warps=4,
    )
    flags = int(flag.item())
    if flags & 1:
        raise ValueError(
            "qwen4_exp incremental prefill cu_seqlens do not match input lengths"
        )
    if flags & 8:
        raise RuntimeError(
            "qwen4_exp QSA paged main block table contains an out-of-range physical block"
        )
    if flags & 16:
        raise RuntimeError(
            "qwen4_exp QSA paged main block table does not cover the visible KV"
        )
    if flags & 32:
        raise RuntimeError(
            "qwen4_exp QSA paged visible KV resolves to an unallocated or out-of-range physical block"
        )
    if flags & 2:
        raise ValueError("selected indices may only use -1 as padding")
    if flags & 4:
        raise ValueError(
            "incremental prefill selected indices contain a token outside the row's visible KV"
        )
    return visible
