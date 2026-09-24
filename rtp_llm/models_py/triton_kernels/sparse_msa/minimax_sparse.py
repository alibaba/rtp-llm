# Copyright 2025 XunhaoLai. All rights reserved.

from typing import Optional

import torch

from .common.index import topk_index_reduce
from .decode.flash_with_topk_idx import flash_decode_with_topk_idx_paged
from .decode.topk_sparse import flash_decode_with_gqa_share_sparse_paged
from .prefill.score_chunk import m3_index_score_chunk_enabled, m3_index_score_chunk_rows
from .prefill.topk_bt_fused import flash_prefill_with_fmha

# fmha_sm100 OnlyScore (flash_prefill_with_fmha step1) materializes a maxscore
# buffer [num_idx_heads, max_k_tiles, total_q] and addresses it with int32. When
# its element count exceeds 2**31, _fmha_sm100_plan degrades (max_k_tiles = -1)
# and _fmha_sm100 returns maxscore=None -> flash_prefill_topk_to_block_tables
# crashes on maxscore.transpose(). Reject unsupported geometry before launch.
_FMHA_MAXSCORE_INT32_LIMIT = 1 << 31


def _fmha_onlyscore_overflows_int32(
    num_idx_heads: int, max_seqlen_k: int, total_q: int
) -> bool:
    # mirror fmha_sm100 api.py: max_k_tiles = ceil(ceil(max_kv/128)/128)*128
    n_blocks = (max_seqlen_k + 127) // 128
    max_k_tiles = ((n_blocks + 127) // 128) * 128
    return num_idx_heads * max_k_tiles * total_q > _FMHA_MAXSCORE_INT32_LIMIT


def m3_fmha_prefill_enabled(
    *,
    sparse_attn_plan: Optional[object],
    num_idx_heads: int,
    num_kv_heads: int,
    disable_index_value: bool,
    has_idx_sink: bool,
    has_sink: bool,
    max_seqlen_k: int,
    total_q: int,
) -> bool:
    """Return whether the FMHA index-score and sparse-attention path is usable."""
    fmha_score_rows = total_q
    if m3_index_score_chunk_enabled(total_q):
        fmha_score_rows = min(total_q, m3_index_score_chunk_rows())
    fmha_score_fits = not _fmha_onlyscore_overflows_int32(
        num_idx_heads, max_seqlen_k, fmha_score_rows
    )
    return (
        sparse_attn_plan is not None
        and num_idx_heads == num_kv_heads
        and disable_index_value
        and not has_idx_sink
        and not has_sink
        and fmha_score_fits
    )


def minimax_sparse_prefill(
    q: torch.Tensor,
    idx_q: torch.Tensor,
    idx_k_cache: torch.Tensor,
    req_to_token: torch.Tensor,
    cu_seqlens: torch.Tensor,
    seq_lens: torch.Tensor,
    prefix_lens: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    block_size_k: int,
    topk: int,
    init_blocks: int,
    local_blocks: int,
    k_paged_cache: torch.Tensor,
    v_paged_cache: torch.Tensor,
    disable_index_value: bool,
    sm_scale: Optional[float] = None,
    index_score_plan=None,
    sparse_attn_plan=None,
    kv_indices=None,
):
    """Run sparse prefill from HND working pages and compact idx_K pages."""
    if k_paged_cache is None or v_paged_cache is None:
        raise RuntimeError("sparse prefill requires paged main K/V")
    if k_paged_cache.shape != v_paged_cache.shape:
        raise ValueError("sparse prefill paged K/V shapes differ")
    num_idx_heads = int(idx_q.shape[1])
    num_kv_heads = int(k_paged_cache.shape[1])
    if not m3_fmha_prefill_enabled(
        sparse_attn_plan=sparse_attn_plan,
        num_idx_heads=num_idx_heads,
        num_kv_heads=num_kv_heads,
        disable_index_value=disable_index_value,
        has_idx_sink=False,
        has_sink=False,
        max_seqlen_k=max_seqlen_k,
        total_q=int(idx_q.shape[0]),
    ):
        raise RuntimeError("paged sparse prefill requires the native FMHA plan")
    scale = sm_scale if sm_scale is not None else q.shape[-1] ** -0.5
    output = flash_prefill_with_fmha(
        q=q,
        idx_q=idx_q,
        idx_k_cache=idx_k_cache,
        req_to_token=req_to_token,
        cu_seqlens=cu_seqlens,
        seq_lens=seq_lens,
        prefix_lens=prefix_lens,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        block_size_k=block_size_k,
        topk=topk,
        init_blocks=init_blocks,
        local_blocks=local_blocks,
        sm_scale=scale,
        index_score_plan=index_score_plan,
        sparse_attn_plan=sparse_attn_plan,
        kv_indices=kv_indices,
        k_paged_cache=k_paged_cache,
        v_paged_cache=v_paged_cache,
    )
    return None, output


def minimax_paged_sparse_decode(
    q: torch.Tensor,  # [batch_size, num_q_heads, qk_head_dim]
    sink: Optional[torch.Tensor],
    idx_q: torch.Tensor,  # [batch_size, num_idx_heads, idx_head_dim]
    seq_lens: torch.Tensor,  # [batch_size]
    max_seqlen: int,
    block_size_k: int,
    topk: int,
    init_blocks: int,
    local_blocks: int,
    paged_main_k: torch.Tensor,  # [block, kh, page, dim]
    paged_main_v: torch.Tensor,  # [block, kh, page, dim]
    phys_block_table: torch.Tensor,  # [batch, max_blocks]
    paged_idx_k: torch.Tensor,  # [block, page, idx_dim]
    paged_idx_scale: Optional[torch.Tensor] = None,  # [block, page]
    sm_scale: Optional[float] = None,
    idx_sm_scale: Optional[float] = None,
    score_type: str = "max",
    disable_index_value: bool = False,
    score_block_table: Optional[torch.Tensor] = None,
    score_seq_lens: Optional[torch.Tensor] = None,
    decode_query_len: int = 1,
):
    """Paged-only sparse decode that never consumes token-major scratch caches."""
    if not disable_index_value:
        raise RuntimeError(
            "minimax_paged_sparse_decode requires disable_index_value=True; "
            "idx value decode is not implemented for paged attention."
        )
    if paged_main_k is None or paged_main_v is None or paged_idx_k is None:
        raise RuntimeError("paged sparse decode requires paged main K/V and idx_K")
    if phys_block_table is None:
        raise RuntimeError("paged sparse decode requires a physical block table")
    if int(paged_main_k.shape[2]) != int(block_size_k):
        raise RuntimeError(
            f"paged main K/V page_size={int(paged_main_k.shape[2])} "
            f"must equal block_size_k={block_size_k}"
        )
    if int(paged_idx_k.shape[1]) != int(block_size_k):
        raise RuntimeError(
            f"paged idx_K page_size={int(paged_idx_k.shape[1])} "
            f"must equal block_size_k={block_size_k}"
        )

    num_idx_heads = idx_q.shape[1]
    num_kv_heads = paged_main_k.shape[1]
    idx_group_size = num_idx_heads // num_kv_heads

    idx_o, topk_idx = flash_decode_with_topk_idx_paged(
        q=idx_q,
        k_paged=paged_idx_k,
        block_table=(
            phys_block_table if score_block_table is None else score_block_table
        ),
        seq_lens=seq_lens if score_seq_lens is None else score_seq_lens,
        max_seqlen=max_seqlen,
        block_size=block_size_k,
        topk=topk,
        init_blocks=init_blocks,
        local_blocks=local_blocks,
        sm_scale=idx_sm_scale,
        score_type=score_type,
        decode_query_len=decode_query_len,
        token_seq_lens=seq_lens,
        k_scale=paged_idx_scale,
    )
    if idx_group_size > 1:
        topk_idx = topk_index_reduce(
            topk_idx.view(num_kv_heads, idx_group_size, -1, topk), dim=1
        )

    o = flash_decode_with_gqa_share_sparse_paged(
        q=q,
        sink=sink,
        k_paged=paged_main_k,
        v_paged=paged_main_v,
        block_table=phys_block_table,
        seq_lens=seq_lens,
        block_size=block_size_k,
        topk_idx=topk_idx,
        sm_scale=sm_scale,
        num_topk_chunks=4 if decode_query_len > 1 else None,
    )
    return idx_o, o
