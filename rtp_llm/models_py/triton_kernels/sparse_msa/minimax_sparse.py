# Copyright 2025 XunhaoLai. All rights reserved.

from typing import Optional

import torch

try:
    from rtp_llm.ops.compute_ops import (
        cuda_graph_capture_forward_enabled,
        cuda_graph_warmup_forward_enabled,
    )
except ImportError:

    def cuda_graph_capture_forward_enabled() -> bool:
        return False

    def cuda_graph_warmup_forward_enabled() -> bool:
        return False


def _cuda_graph_forward_active() -> bool:
    return cuda_graph_capture_forward_enabled() or cuda_graph_warmup_forward_enabled()


from .common.index import topk_index_reduce
from .decode.flash_with_topk_idx import (
    flash_decode_with_topk_idx,
    flash_decode_with_topk_idx_paged,
)
from .decode.topk_sparse import (
    flash_decode_with_gqa_share_sparse,
    flash_decode_with_gqa_share_sparse_paged,
)
from .prefill.flash_with_topk_idx import flash_prefill_with_topk_index
from .prefill.topk_bt_fused import (
    flash_decode_with_trtllm_gen,
    flash_prefill_with_fmha,
    flash_prefill_with_fused_topk_index,
)
from .prefill.topk_sparse import flash_prefill_with_gqa_share_sparse

# trtllm-gen's sparse-decode kernel launches with grid_dim_x = total_q *
# num_kv_heads. The CUDA grid_dim_x hardware cap is 2**16 - 1 = 65535, so a
# single call covers up to 65535 // num_kv_heads queries (16383 for NUM_KV=4).
# flash_decode_with_trtllm_gen chunks beyond this internally, so the caller
# does NOT need a max_q gate any more — any q_len works.

# fmha_sm100 OnlyScore (flash_prefill_with_fmha step1) materializes a maxscore
# buffer [num_idx_heads, max_k_tiles, total_q] and addresses it with int32. When
# its element count exceeds 2**31, _fmha_sm100_plan degrades (max_k_tiles = -1)
# and _fmha_sm100 returns maxscore=None -> flash_prefill_topk_to_block_tables
# crashes on maxscore.transpose(). Detect that up front and route to the Triton
# block-score path instead (64-bit indexing, no such buffer).
_FMHA_MAXSCORE_INT32_LIMIT = 1 << 31


def _fmha_onlyscore_overflows_int32(
    num_idx_heads: int, max_seqlen_k: int, total_q: int
) -> bool:
    # mirror fmha_sm100 api.py: max_k_tiles = ceil(ceil(max_kv/128)/128)*128
    n_blocks = (max_seqlen_k + 127) // 128
    max_k_tiles = ((n_blocks + 127) // 128) * 128
    return num_idx_heads * max_k_tiles * total_q > _FMHA_MAXSCORE_INT32_LIMIT


def minimax_sparse_prefill(
    q: torch.Tensor,  # [total_extend_tokens, num_q_heads, qk_head_dim]
    k_cache: torch.Tensor,  # [max_slots, num_kv_heads, head_dim] (paged main)
    v_cache: torch.Tensor,  # [max_slots, num_kv_heads, head_dim] (paged main)
    sink: Optional[torch.Tensor],  # [num_q_heads, qk_head_dim]
    idx_q: torch.Tensor,  # [total_extend_tokens, num_idx_heads, idx_head_dim]
    idx_k_cache: torch.Tensor,  # [max_slots, 1, idx_head_dim] (paged index)
    idx_v_cache: Optional[
        torch.Tensor
    ],  # [max_slots, 1, idx_head_dim] (paged index); None when disable_index_value
    idx_sink: Optional[torch.Tensor],  # [num_idx_heads, idx_head_dim]
    req_to_token: torch.Tensor,  # [max_reqs, max_kv_len]
    slot_ids: torch.Tensor,  # [batch_size, ]
    cu_seqlens: torch.Tensor,  # [batch_size + 1, ] (Q-side cumulative)
    seq_lens: torch.Tensor,  # [batch_size, ] total K length (prefix + chunk)
    prefix_lens: torch.Tensor,  # [batch_size, ]
    max_seqlen_q: int,
    max_seqlen_k: int,
    block_size_q: int,
    block_size_k: int,
    topk: int,
    init_blocks: int,
    local_blocks: int,
    sm_scale: Optional[float] = None,
    idx_sm_scale: Optional[float] = None,
    score_type: str = "max",
    disable_index_value: bool = False,
    workspace: Optional[torch.Tensor] = None,
    index_score_plan=None,
    sparse_attn_plan=None,
    kv_indices=None,
):
    # All seqlen is less than topk, use full attention
    # Step 1: Flash attention with topk index (using index head)
    num_idx_heads = idx_q.shape[1]
    num_kv_heads = k_cache.shape[1]
    idx_group_size = num_idx_heads // num_kv_heads

    # Fastest path: mega topk-to-block-tables + trtllm-gen sparse decode.
    # The fused-topk_idx and legacy 3-stage paths stay as fallbacks:
    # idx_group_size > 1 routes there via topk_index_reduce. The (cu_seqlens-1 > 2)
    # multi-request restriction applies ONLY to the trtllm-gen block-table (bt) path,
    # whose page-id layout (pid_h * num_pages + block_idx, no per-segment offset)
    # assumes a single contiguous KV slice. The fmha step3 path (sparse_attn_plan set)
    # consumes batch-local topk_idx + a per-request page_table and handles any number of
    # requests (verified bit-identical vs per-request runs), so it is NOT capped.
    # No per-call q_len cap — flash_prefill_with_fmha/decode chunk internally to stay
    # under CUDA's grid_dim_x = 65535 limit.
    # fmha OnlyScore step1 would overflow its int32-indexed maxscore buffer on
    # long context (num_idx_heads * max_k_tiles * total_q > 2**31) -> fall back to
    # the Triton block-score path (use_fused below) to avoid the maxscore=None crash.
    # total_q here is the EXTEND (reuse-after) query token count == idx_q.shape[0]
    # == cu_seqlens[-1] == fmha nnz_qo/total_qo_len (the exact 3rd dim of the buffer
    # that overflows). max_seqlen_k is the FULL KV len (prefix+extend, reuse-before),
    # matching how fmha derives max_k_tiles -- so the product is same-source as fmha's.
    total_q = idx_q.shape[0]
    fmha_score_fits = not _fmha_onlyscore_overflows_int32(
        num_idx_heads, max_seqlen_k, total_q
    )
    use_trtllm = (
        workspace is not None
        and sparse_attn_plan is not None
        and idx_group_size == 1
        and disable_index_value
        and idx_sink is None
        and sink is None
        and fmha_score_fits
    )
    if use_trtllm:
        sm_scale_v = sm_scale if sm_scale is not None else q.shape[-1] ** -0.5
        o = flash_prefill_with_fmha(
            q=q,
            k_cache=k_cache,
            v_cache=v_cache,
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
            sm_scale=sm_scale_v,
            index_score_plan=index_score_plan,
            sparse_attn_plan=sparse_attn_plan,
            kv_indices=kv_indices,
        )
        return None, o

    # Slower path: fused topk_bt_fused step 1+2 emitting topk_idx + legacy
    # triton step 3 sparse attention. Avoids the second kernel launch in the
    # original 3-stage flow but keeps the triton sparse_fwd kernel.
    use_fused = idx_group_size == 1 and disable_index_value and idx_sink is None
    if use_fused:
        idx_o, topk_idx = flash_prefill_with_fused_topk_index(
            idx_q=idx_q,
            idx_k_cache=idx_k_cache,
            req_to_token=req_to_token,
            slot_ids=slot_ids,
            cu_seqlens=cu_seqlens,
            seq_lens=seq_lens,
            prefix_lens=prefix_lens,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            block_size_k=block_size_k,
            topk=topk,
            init_blocks=init_blocks,
            local_blocks=local_blocks,
            sm_scale=idx_sm_scale,
            score_type=score_type,
        )
    else:
        idx_o, topk_idx = flash_prefill_with_topk_index(
            q=idx_q,
            k_cache=idx_k_cache,
            v_cache=idx_v_cache,
            sink=idx_sink,
            req_to_token=req_to_token,
            slot_ids=slot_ids,
            cu_seqlens=cu_seqlens,
            seq_lens=seq_lens,
            prefix_lens=prefix_lens,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            block_size_q=block_size_q,
            block_size_k=block_size_k,
            topk=topk,
            init_blocks=init_blocks,
            local_blocks=local_blocks,
            sm_scale=idx_sm_scale,
            score_type=score_type,
            disable_index_value=disable_index_value,
        )
        if idx_group_size > 1:
            topk_idx = topk_index_reduce(
                topk_idx.view(num_kv_heads, idx_group_size, -1, topk), dim=1
            )
    # Step 3: Sparse attention using topk index (main head)
    o = flash_prefill_with_gqa_share_sparse(
        q=q,
        k_cache=k_cache,
        v_cache=v_cache,
        sink=sink,
        req_to_token=req_to_token,
        slot_ids=slot_ids,
        topk_idx=topk_idx,
        block_size_q=block_size_q,
        block_size_k=block_size_k,
        cu_seqlens=cu_seqlens,
        seq_lens=seq_lens,
        prefix_lens=prefix_lens,
        max_seqlen_q=max_seqlen_q,
        sm_scale=sm_scale,
    )
    return idx_o, o


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
            "idx value decode still uses the token-major scratch path."
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


def minimax_sparse_decode(
    q: torch.Tensor,  # [batch_size, num_q_heads, qk_head_dim]
    sink: Optional[torch.Tensor],  # [num_q_heads, qk_head_dim]
    k_cache: torch.Tensor,  # [max_slots, num_kv_heads, head_dim] (paged)
    v_cache: torch.Tensor,  # [max_slots, num_kv_heads, head_dim] (paged)
    idx_q: torch.Tensor,  # [batch_size, num_idx_heads, idx_head_dim], num_idx_heads >= num_kv_heads
    idx_sink: Optional[torch.Tensor],  # [num_idx_heads, idx_head_dim]
    idx_k_cache: torch.Tensor,  # [max_slots, 1, idx_head_dim] (paged)
    idx_v_cache: Optional[
        torch.Tensor
    ],  # [max_slots, 1, idx_head_dim] (paged); None when disable_index_value
    req_to_token: torch.Tensor,  # [max_reqs, max_kv_len]
    slot_ids: torch.Tensor,  # [batch_size, ]
    seq_lens: torch.Tensor,  # [batch_size, ]
    max_seqlen: int,  # max of seq_lens, passed from caller to avoid sync during CUDA graph capture
    block_size_k: int,
    topk: int,
    init_blocks: int,
    local_blocks: int,
    sm_scale: Optional[float] = None,
    idx_sm_scale: Optional[float] = None,
    score_type: str = "max",
    disable_index_value: bool = False,
    workspace: Optional[torch.Tensor] = None,
    cu_seqlens: Optional[torch.Tensor] = None,
    prefix_lens: Optional[torch.Tensor] = None,
):
    num_idx_heads = idx_q.shape[1]
    num_kv_heads = k_cache.shape[1]
    idx_group_size = num_idx_heads // num_kv_heads
    batch = q.shape[0]

    # Fast path: reuse the trtllm-gen sparse-decode kernel (same op the prefill
    # fast path uses; it treats every query independently so decode = total_q==batch
    # queries). Mirrors minimax_sparse_prefill's use_trtllm gate. Single-request
    # only for now (the fused topk-to-block-table kernel assumes a single
    # contiguous KV slice: page id = pid_h*num_pages + block_idx, no per-request
    # offset); multi-request decode falls back to the legacy triton path.
    use_trtllm = (
        workspace is not None
        and cu_seqlens is not None
        and prefix_lens is not None
        and idx_group_size == 1
        and disable_index_value
        and idx_sink is None
        and sink is None
        and batch <= 1
        and not _cuda_graph_forward_active()
    )
    if use_trtllm:
        sm_scale_v = sm_scale if sm_scale is not None else q.shape[-1] ** -0.5
        o = flash_decode_with_trtllm_gen(
            q=q,
            k_cache=k_cache,
            v_cache=v_cache,
            idx_q=idx_q,
            idx_k_cache=idx_k_cache,
            req_to_token=req_to_token,
            cu_seqlens=cu_seqlens,
            seq_lens=seq_lens,
            prefix_lens=prefix_lens,
            max_seqlen_q=1,
            max_seqlen_k=max_seqlen,
            block_size_k=block_size_k,
            topk=topk,
            init_blocks=init_blocks,
            local_blocks=local_blocks,
            sm_scale=sm_scale_v,
            workspace=workspace,
        )
        return None, o

    # Step 1: Flash decode with topk index (using index head).
    idx_o, topk_idx = flash_decode_with_topk_idx(
        q=idx_q,
        sink=idx_sink,
        k_cache=idx_k_cache,
        v_cache=idx_v_cache,
        req_to_token=req_to_token,
        seq_lens=seq_lens,
        max_seqlen=max_seqlen,
        slot_ids=slot_ids,
        block_size=block_size_k,
        topk=topk,
        init_blocks=init_blocks,
        local_blocks=local_blocks,
        sm_scale=idx_sm_scale,
        score_type=score_type,
        disable_index_value=disable_index_value,
    )
    # Step 2: Reduce topk idx if num_idx_heads > num_kv_heads
    if idx_group_size > 1:
        topk_idx = topk_index_reduce(
            topk_idx.view(num_kv_heads, idx_group_size, -1, topk), dim=1
        )
    # Step 3: Sparse attention using topk index (main head).
    o = flash_decode_with_gqa_share_sparse(
        q=q,
        sink=sink,
        k_cache=k_cache,
        v_cache=v_cache,
        req_to_token=req_to_token,
        seq_lens=seq_lens,
        slot_ids=slot_ids,
        block_size=block_size_k,
        topk_idx=topk_idx,
        sm_scale=sm_scale,
    )
    return idx_o, o
