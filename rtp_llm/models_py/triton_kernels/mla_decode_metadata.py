"""Device metadata for fixed-query-length paged MLA decode and MTP verify."""

import triton
import triton.language as tl


@triton.jit
def _mla_decode_metadata(
    Prefix,
    BlockTable,
    SeqLens,
    DenseTable,
    Positions,
    BatchIndices,
    SlotMapping,
    PREFIX_STRIDE: tl.constexpr,
    TABLE_ROW_STRIDE: tl.constexpr,
    TABLE_COL_STRIDE: tl.constexpr,
    SOURCE_WIDTH: tl.constexpr,
    DEST_WIDTH: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    QUERY_LENGTH: tl.constexpr,
    MAX_SEQ_LEN: tl.constexpr,
    BLOCK: tl.constexpr,
    QUERY_BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    prefix = tl.load(Prefix + row * PREFIX_STRIDE).to(tl.int64)
    kv_len = prefix + QUERY_LENGTH
    # This guards invalid device input, not ordinary graph-capacity selection.
    # Do not silently truncate a live sequence or rely on TRITON_DEBUG.
    if (prefix < 0) | (kv_len > MAX_SEQ_LEN):
        tl.inline_asm_elementwise(
            "trap; mov.u32 $0, 0;", "=r", [], tl.int32, is_pure=False, pack=1
        )
    pages = tl.cdiv(kv_len, PAGE_SIZE)
    tl.store(SeqLens + row, kv_len.to(tl.int32))
    for start in range(tl.cdiv(DEST_WIDTH, BLOCK)):
        col = start * BLOCK + tl.arange(0, BLOCK)
        page = tl.load(
            BlockTable + row * TABLE_ROW_STRIDE + col * TABLE_COL_STRIDE,
            mask=(col < pages) & (col < SOURCE_WIDTH),
            other=0,
        )
        tl.store(DenseTable + row * DEST_WIDTH + col, page, col < DEST_WIDTH)
    token = tl.arange(0, QUERY_BLOCK)
    position = prefix + token
    page = tl.load(
        BlockTable
        + row * TABLE_ROW_STRIDE
        + (position // PAGE_SIZE) * TABLE_COL_STRIDE,
        mask=token < QUERY_LENGTH,
        other=0,
    ).to(tl.int64)
    offset = row * QUERY_LENGTH + token
    tl.store(Positions + offset, position.to(tl.int32), token < QUERY_LENGTH)
    tl.store(BatchIndices + offset, row, token < QUERY_LENGTH)
    tl.store(
        SlotMapping + offset,
        page * PAGE_SIZE + position % PAGE_SIZE,
        token < QUERY_LENGTH,
    )


def prepare_mla_decode_metadata(
    prefix,
    block_table,
    seq_lens,
    dense_table,
    positions,
    batch_indices,
    slot_mapping,
    page_size: int,
    query_length: int,
    max_seq_len: int,
) -> None:
    _mla_decode_metadata[(prefix.numel(),)](
        prefix,
        block_table,
        seq_lens,
        dense_table,
        positions,
        batch_indices,
        slot_mapping,
        prefix.stride(0),
        block_table.stride(0),
        block_table.stride(1),
        block_table.size(1),
        dense_table.size(1),
        page_size,
        query_length,
        max_seq_len,
        256,
        triton.next_power_of_2(query_length),
    )
