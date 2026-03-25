"""Triton kernel: filter topk indices for sharded CP cache and map to local cache addresses.

Given request-local topk token positions, this kernel:
1. Filters out positions not owned by the current CP rank (sets to -1).
2. Maps owned positions to sharded cache addresses:
   indices_in_kvcache = block_table[req][vblock_idx] * block_size + local_offset
   where vblock_idx = position // virtual_block_size
         local_offset = (position % virtual_block_size) // cp_size
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _filter_topk_for_sharded_cache_kernel(
    req_id_ptr,           # int32 [num_tokens]
    block_table_ptr,      # int32 [num_requests, max_blocks_per_req]
    token_indices_ptr,    # int32 [num_tokens, NUM_TOPK]
    out_ptr,              # int32 [num_tokens, NUM_TOPK]
    cp_rank: tl.constexpr,
    cp_size: tl.constexpr,
    block_size: tl.constexpr,
    virtual_block_size: tl.constexpr,
    max_blocks_per_req: tl.constexpr,
    BLOCK_N: tl.constexpr,
    # strides
    bt_stride0,
    bt_stride1,
    ti_stride0,
    ti_stride1,
    out_stride0,
    out_stride1,
):
    token_id = tl.program_id(0)
    tile_id = tl.program_id(1)
    col_ids = tile_id * BLOCK_N + tl.arange(0, BLOCK_N)

    req = tl.load(req_id_ptr + token_id)

    ti_ptr = token_indices_ptr + token_id * ti_stride0 + col_ids * ti_stride1
    tok_pos = tl.load(ti_ptr)  # request-local token position

    is_invalid = tok_pos < 0

    # Check ownership: position p belongs to rank (p % virtual_block_size) % cp_size
    target_rank = (tok_pos % virtual_block_size) % cp_size
    not_owned = target_rank != cp_rank

    # Compute sharded cache address for owned positions
    vblock_idx = tok_pos // virtual_block_size
    local_offset = (tok_pos % virtual_block_size) // cp_size

    valid_block = (vblock_idx >= 0) & (vblock_idx < max_blocks_per_req)
    bt_ptr = block_table_ptr + req * bt_stride0 + vblock_idx * bt_stride1
    phys_block = tl.load(bt_ptr, mask=valid_block & ~is_invalid & ~not_owned, other=0)
    out_val = phys_block * block_size + local_offset

    # Set invalid or not-owned to -1
    out_val = tl.where(is_invalid | not_owned | ~valid_block, -1, out_val)

    out_ptr_ij = out_ptr + token_id * out_stride0 + col_ids * out_stride1
    tl.store(out_ptr_ij, out_val)


def triton_filter_topk_for_sharded_cache(
    req_id: torch.Tensor,          # int32 [num_tokens]
    block_table: torch.Tensor,     # int32 [num_requests, max_blocks_per_req]
    token_indices: torch.Tensor,   # int32 [num_tokens, NUM_TOPK]
    cp_rank: int,
    cp_size: int,
    block_size: int,
    BLOCK_N: int = 128,
) -> torch.Tensor:
    """Filter topk indices for local CP rank and map to sharded cache addresses.

    Args:
        req_id: Request id per q token, int32 [num_tokens].
        block_table: Sharded cache block table, int32 [num_requests, max_blocks].
            Each entry is a physical block id; block indices are virtual block indices.
        token_indices: Request-local topk token positions, int32 [num_tokens, topk].
        cp_rank: Current CP rank.
        cp_size: Total CP size.
        block_size: Physical block size (tokens per physical block).
        BLOCK_N: Tile width for triton grid.

    Returns:
        out: int32 [num_tokens, topk]. For owned positions:
            block_table[req][vblock] * block_size + local_offset.
            For non-owned or invalid positions: -1.
    """
    assert req_id.dtype == torch.int32
    assert block_table.dtype == torch.int32
    assert token_indices.dtype == torch.int32

    num_tokens = req_id.shape[0]
    num_topk = token_indices.shape[1]
    virtual_block_size = block_size * cp_size
    max_blocks_per_req = block_table.shape[1]

    assert num_topk % BLOCK_N == 0, (
        f"NUM_TOPK ({num_topk}) must be divisible by BLOCK_N ({BLOCK_N})"
    )

    req_id_c = req_id.contiguous()
    block_table_c = block_table.contiguous()
    token_indices_c = token_indices.contiguous()
    out = torch.empty_like(token_indices_c)

    bt_stride0, bt_stride1 = block_table_c.stride()
    ti_stride0, ti_stride1 = token_indices_c.stride()
    out_stride0, out_stride1 = out.stride()

    tiles_per_row = num_topk // BLOCK_N
    grid = (num_tokens, tiles_per_row)

    _filter_topk_for_sharded_cache_kernel[grid](
        req_id_c,
        block_table_c,
        token_indices_c,
        out,
        cp_rank,
        cp_size,
        block_size,
        virtual_block_size,
        max_blocks_per_req,
        BLOCK_N,
        bt_stride0,
        bt_stride1,
        ti_stride0,
        ti_stride1,
        out_stride0,
        out_stride1,
    )
    return out
