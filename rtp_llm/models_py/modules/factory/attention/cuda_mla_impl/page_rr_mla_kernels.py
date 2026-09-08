import triton
import triton.language as tl


@triton.jit
def _page_rr_coordinates(
    position,
    tokens_per_block: tl.constexpr,
    kernel_tokens_per_block: tl.constexpr,
    cp_size: tl.constexpr,
    cp_rank: tl.constexpr,
):
    global_page = position // tokens_per_block
    offset = position % tokens_per_block
    column = global_page // cp_size * (tokens_per_block // kernel_tokens_per_block)
    column += offset // kernel_tokens_per_block
    owned = (position >= 0) & (global_page % cp_size == cp_rank)
    return column, offset % kernel_tokens_per_block, owned


@triton.jit
def _prefill_page_rr_slots(
    positions,
    requests,
    table,
    output,
    count,
    table_rows,
    table_width,
    table_stride0,
    table_stride1,
    tokens_per_block: tl.constexpr,
    kernel_tokens_per_block: tl.constexpr,
    cp_size: tl.constexpr,
    cp_rank: tl.constexpr,
    block: tl.constexpr,
):
    row = tl.program_id(0) * block + tl.arange(0, block)
    valid = row < count
    position = tl.load(positions + row, valid, other=0).to(tl.int64)
    request = tl.load(requests + row, valid, other=0).to(tl.int64)
    column, offset, owned = _page_rr_coordinates(
        position, tokens_per_block, kernel_tokens_per_block, cp_size, cp_rank
    )
    readable = (
        valid & owned & (request >= 0) & (request < table_rows)
        & (column >= 0) & (column < table_width)
    )
    physical = tl.load(
        table + request * table_stride0 + column * table_stride1,
        readable, other=-1,
    ).to(tl.int64)
    slot = physical * kernel_tokens_per_block + offset
    tl.store(output + row, tl.where(readable & (physical > 0), slot, -1), valid)
