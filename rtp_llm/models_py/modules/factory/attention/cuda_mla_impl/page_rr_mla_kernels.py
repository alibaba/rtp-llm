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
