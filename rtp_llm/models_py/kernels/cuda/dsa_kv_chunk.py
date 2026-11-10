"""Map final global TopK positions to one KV workspace and normalize empties."""

import torch
import triton
import triton.language as tl


@triton.jit
def _map_slots(
    Indices,
    Requests,
    Slices,
    Table,
    Slots,
    Lengths,
    Counts,
    TOPK: tl.constexpr,
    PREFILL: tl.constexpr,
    PAGE: tl.constexpr,
    SHARDS: tl.constexpr,
    RANK: tl.constexpr,
    START: tl.constexpr,
    WIDTH: tl.constexpr,
    RAW_CACHE: tl.constexpr,
    KERNEL_PAGE: tl.constexpr,
    TABLE_WIDTH: tl.constexpr,
    TABLE_ROW_STRIDE: tl.constexpr,
    TABLE_COL_STRIDE: tl.constexpr,
    BLOCK: tl.constexpr,
    HAS_LENGTHS: tl.constexpr,
    COMPACT: tl.constexpr,
):
    row = tl.program_id(0)
    col = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    pos = tl.load(Indices + row * TOPK + col, col < TOPK, other=-1)
    if HAS_LENGTHS:
        tl.store(Lengths + row, tl.sum((pos >= 0).to(tl.int32), axis=0))
    request = tl.load(Requests + row)
    if PREFILL:
        start = tl.load(Slices + request * 3)
        end = tl.load(Slices + request * 3 + 1)
        offset = tl.load(Slices + request * 3 + 2)
        valid = (pos >= start) & (pos < end) & (pos >= 0)
        slot = pos - start + offset
    else:
        page = pos // PAGE
        local = (page // SHARDS) * PAGE + pos % PAGE
        valid = (pos >= 0) & (page % SHARDS == RANK)
        valid = valid & (local >= START) & (local < START + WIDTH)
        if RAW_CACHE:
            column = local // KERNEL_PAGE
            valid = valid & (column >= 0) & (column < TABLE_WIDTH)
            block = tl.load(
                Table + request * TABLE_ROW_STRIDE + column * TABLE_COL_STRIDE,
                valid & (col < TOPK),
                other=0,
            ).to(tl.int64)
            valid = valid & (block > 0)
            slot = block * KERNEL_PAGE + local % KERNEL_PAGE
        else:
            slot = request * WIDTH + local - START
    if COMPACT:
        # One program covers the whole row, so a block-wide prefix sum packs this
        # owner's slots into [0, count) and `Counts` becomes the kernel's
        # topk_length. The two stores cover disjoint columns.
        owned = (valid & (col < TOPK)).to(tl.int32)
        count = tl.sum(owned, axis=0)
        rank = tl.cumsum(owned, axis=0) - 1
        tl.store(Slots + row * TOPK + rank, slot, owned == 1)
        tl.store(Slots + row * TOPK + col, -1, (col >= count) & (col < TOPK))
        tl.store(Counts + row, count)
    else:
        tl.store(Slots + row * TOPK + col, tl.where(valid, slot, -1), col < TOPK)


def map_dsa_chunk_slots(
    indices, params, block, out, *, block_table=None, lengths=None, counts=None
):
    """Translate global TopK positions to this owner's slots.

    ``lengths`` keeps its existing meaning: how many globally valid TopK entries
    a row has, regardless of ownership. ``counts`` instead asks for this owner's
    valid slots to be packed at the front of each row and counted, which is the
    ``topk_length`` contract of the metadata-free FP8 decode kernel.
    """
    if indices.shape != out.shape or not indices.is_contiguous():
        raise ValueError("DSA final TopK must be contiguous [query,K]")
    requests = params.query_requests if params.is_prefill else params.batch_indice_d
    if indices.numel():
        block_size = (
            max(256, triton.next_power_of_2(indices.shape[1]))
            if lengths is not None or counts is not None
            else 256
        )
        num_warps = 8 if block_size >= 2048 else 4
        _map_slots[(indices.shape[0], triton.cdiv(indices.shape[1], block_size))](
            indices,
            requests,
            block.slices if block.slices is not None else requests,
            block_table if block_table is not None else requests,
            out,
            lengths if lengths is not None else requests,
            counts if counts is not None else requests,
            indices.shape[1],
            params.is_prefill,
            params.page_size,
            params.shard_size,
            params.shard_rank,
            # Prefill bounds come from Slices; keep unused Decode keys fixed.
            0 if params.is_prefill else block.start,
            0 if params.is_prefill else block.width,
            block_table is not None,
            params.kernel_page_size if block_table is not None else 1,
            block_table.shape[1] if block_table is not None else 0,
            block_table.stride(0) if block_table is not None else 0,
            block_table.stride(1) if block_table is not None else 0,
            block_size,
            lengths is not None,
            counts is not None,
            num_warps=num_warps,
        )
    else:
        if lengths is not None and lengths.numel():
            lengths.zero_()
        if counts is not None and counts.numel():
            counts.zero_()


@triton.jit
def _normalize_partial(
    Output,
    LSE,
    NormalLSE,
    Lengths,
    LSE_SCALE: tl.constexpr,
    o0: tl.constexpr,
    o1: tl.constexpr,
    l0: tl.constexpr,
    l1: tl.constexpr,
    HEADS: tl.constexpr,
    DIM: tl.constexpr,
    BLOCK: tl.constexpr,
    HAS_LENGTHS: tl.constexpr,
):
    row, head = tl.program_id(0), tl.program_id(1)
    lse = tl.load(LSE + row * l0 + head * l1)
    valid = (lse != float("inf")) & (lse != -float("inf")) & (lse == lse)
    if HAS_LENGTHS:
        valid = valid & (tl.load(Lengths + row) > 0)
    if not valid:
        d = tl.arange(0, BLOCK)
        tl.store(Output + row * o0 + head * o1 + d, 0, d < DIM)
    tl.store(
        NormalLSE + row * HEADS + head,
        tl.where(valid, lse * LSE_SCALE, -float("inf")),
    )


def normalize_sparse_partial(output, lse, heads, normal_lse, lengths=None, *, lse_scale=1.0):
    """FlashMLA empty rows use +inf; canonical natural-log state uses -inf."""
    if normal_lse.shape != (output.shape[0], heads):
        raise ValueError("partial attention LSE shape differs from its head layout")
    if output.shape[0]:
        _normalize_partial[(output.shape[0], heads)](
            output,
            lse,
            normal_lse,
            lengths if lengths is not None else lse,
            lse_scale,
            output.stride(0),
            output.stride(1),
            lse.stride(0),
            lse.stride(1),
            heads,
            output.shape[2],
            triton.next_power_of_2(output.shape[2]),
            lengths is not None,
        )
