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
    ROWS: tl.constexpr,
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
    tl.store(Slots + row * TOPK + col, tl.where(valid, slot, -1), col < TOPK)


def map_dsa_chunk_slots(indices, params, block, out, *, block_table=None, lengths=None):
    if indices.shape != out.shape or not indices.is_contiguous():
        raise ValueError("DSA final TopK must be contiguous [query,K]")
    requests = params.query_requests if params.is_prefill else params.batch_indice_d
    if indices.numel():
        block_size = (
            max(256, triton.next_power_of_2(indices.shape[1]))
            if lengths is not None
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
            indices.shape[0],
            indices.shape[1],
            params.is_prefill,
            params.page_size,
            params.shard_size,
            params.shard_rank,
            block.start,
            block.width,
            block_table is not None,
            params.kernel_page_size if block_table is not None else 1,
            block_table.shape[1] if block_table is not None else 0,
            block_table.stride(0) if block_table is not None else 0,
            block_table.stride(1) if block_table is not None else 0,
            block_size,
            lengths is not None,
            num_warps=num_warps,
        )
    elif lengths is not None and lengths.numel():
        lengths.zero_()


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


@triton.jit
def _zero_sparse_empty_rows(Output, LSE, O0: tl.constexpr, O1: tl.constexpr,
                            L0: tl.constexpr, L1: tl.constexpr,
                            HEADS: tl.constexpr, DIM: tl.constexpr,
                            HEAD_BLOCK: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    head = tl.program_id(1) * HEAD_BLOCK + tl.arange(0, HEAD_BLOCK)
    lse = tl.load(LSE + row * L0 + head * L1, head < HEADS, other=0.0)
    empty = ((lse == float("inf")) | (lse == -float("inf"))) & (head < HEADS)
    unexpected_nan = (lse != lse) & (head < HEADS)
    write = empty | unexpected_nan
    if tl.sum(write.to(tl.int32), 0) > 0:
        col = tl.arange(0, BLOCK)
        tl.store(Output + row * O0 + head[:, None] * O1 + col[None, :],
                 tl.where(unexpected_nan[:, None], float("nan"), 0.0),
                 write[:, None] & (col[None, :] < DIM))


def zero_sparse_empty_rows(output, lse, heads):
    """Single-block path: leave valid latent rows and unused LSE untouched."""
    if lse.shape[0] != output.shape[0] or min(output.shape[1], lse.shape[1]) < heads:
        raise ValueError("Sparse output and LSE head layouts differ")
    if output.shape[0]:
        _zero_sparse_empty_rows[(output.shape[0], triton.cdiv(heads, 16))](
            output, lse, output.stride(0), output.stride(1),
            lse.stride(0), lse.stride(1), heads, output.shape[2], 16,
            triton.next_power_of_2(output.shape[2]), num_warps=4)


def check_sparse_lse(lse):
    """Reject unexpected NaNs asynchronously; +/-inf remain empty sentinels.

    Called once per Prefill layer, after the online merge when chunked. This
    checks only the small LSE state and adds no host synchronization or KV pass.
    """
    torch._assert_async(torch.all(lse == lse), "DSA sparse attention produced NaN LSE")
