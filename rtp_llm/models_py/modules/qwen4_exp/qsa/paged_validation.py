"""Check paged geometry once per invocation before cache mutation."""

import torch
import triton
import triton.language as tl


@triton.jit(do_not_specialize=["WIDTH", "PAGES", "TABLE_ROW"])
def _validate_geometry(
    Lengths,
    Prefixes,
    Table,
    Flags,
    WIDTH,
    PAGES,
    TABLE_ROW,
    LD_ROW: tl.constexpr,
    PD_ROW: tl.constexpr,
    TABLE_COL: tl.constexpr,
    PAGE: tl.constexpr,
    PADDED: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    prefix = tl.load(Prefixes + row * PD_ROW).to(tl.int64)
    length = tl.load(Lengths + row * LD_ROW).to(tl.int64)
    bad_sequence = (prefix < 0) | (length <= 0)
    # Compare against remaining capacity before adding potentially invalid lengths.
    exceeds = prefix > WIDTH.to(tl.int64) * PAGE - length
    required = tl.cdiv(prefix + length, PAGE)
    column = tl.arange(0, 128)
    bad_pool = False
    invalid = False
    active = False
    for tile in range(tl.cdiv(WIDTH, 128)):
        col = (tile * 128 + column).to(tl.int64)
        page = tl.load(Table + row * TABLE_ROW + col * TABLE_COL, col < WIDTH, other=0)
        bad_pool = bad_pool | (
            tl.sum(((col < WIDTH) & (page >= PAGES)).to(tl.int32), 0) > 0
        )
        active = active | (tl.sum(((col < WIDTH) & (page != 0)).to(tl.int32), 0) > 0)
        invalid = invalid | (
            tl.sum(
                ((col < WIDTH) & (col < required) & ((page <= 0) | (page >= PAGES))).to(
                    tl.int32
                ),
                0,
            )
            > 0
        )
    if PADDED:
        exceeds = exceeds & active
        invalid = invalid & active
    flags = (
        bad_sequence.to(tl.int32)
        + bad_pool.to(tl.int32) * 8
        + exceeds.to(tl.int32) * 16
        + invalid.to(tl.int32) * 32
    )
    tl.store(Flags + row, flags)


def validate_geometry(lengths, prefixes, table, page_size, pages, padded):
    flags = torch.empty(lengths.numel(), dtype=torch.int32, device=table.device)
    _validate_geometry[(lengths.numel(),)](
        lengths,
        prefixes,
        table,
        flags,
        table.shape[1],
        pages,
        table.stride(0),
        lengths.stride(0),
        prefixes.stride(0),
        table.stride(1),
        page_size,
        padded,
        num_warps=4,
    )
    verdict = 0
    for flag in flags.cpu().tolist():
        verdict |= flag
    if verdict & 1:
        raise RuntimeError(
            "qwen4_exp QSA paged sequence bases must be non-negative and query lengths positive"
        )
    if verdict & 8:
        raise RuntimeError(
            "qwen4_exp QSA paged main block table contains an out-of-range physical block"
        )
    if verdict & 16:
        raise RuntimeError(
            "qwen4_exp QSA paged main block table does not cover the visible KV"
        )
    if verdict & 32:
        raise RuntimeError(
            "qwen4_exp QSA paged visible KV resolves to an unallocated or out-of-range physical block"
        )
