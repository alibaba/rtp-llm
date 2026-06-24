# Copyright 2025 XunhaoLai. All rights reserved.

import os
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
from .decode.flash_with_topk_idx import flash_decode_with_topk_idx
from .decode.topk_sparse import (
    flash_decode_with_gqa_share_sparse,
    flash_decode_with_gqa_share_sparse_paged,
)
from .prefill.flash_with_topk_idx import flash_prefill_with_topk_index
from .prefill.topk_bt_fused import (
    flash_prefill_with_fused_topk_index,
    flash_prefill_with_trtllm_gen,
)
from .prefill.topk_sparse import flash_prefill_with_gqa_share_sparse

# trtllm-gen's sparse-decode kernel launches with grid_dim_x = total_q *
# num_kv_heads. The CUDA grid_dim_x hardware cap is 2**16 - 1 = 65535, so a
# single call covers up to 65535 // num_kv_heads queries (16383 for NUM_KV=4).
# flash_prefill_with_trtllm_gen chunks beyond this internally, so the caller
# does NOT need a max_q gate any more — any q_len works.


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
):
    # All seqlen is less than topk, use full attention
    # Step 1: Flash attention with topk index (using index head)
    num_idx_heads = idx_q.shape[1]
    num_kv_heads = k_cache.shape[1]
    idx_group_size = num_idx_heads // num_kv_heads

    # Fastest path: mega topk-to-block-tables + trtllm-gen sparse decode.
    # The fused-topk_idx and legacy 3-stage paths stay as fallbacks:
    # idx_group_size > 1 routes there via topk_index_reduce, and so does any
    # multi-request batch (cu_seqlens-1 > 2) because the mega kernel's page
    # id layout (pid_h * num_pages + block_idx, no per-segment offset)
    # requires a single contiguous KV cache slice across all segments.
    # No per-call q_len cap — flash_prefill_with_trtllm_gen chunks internally
    # to stay under CUDA's grid_dim_x = 65535 limit.
    use_trtllm = (
        workspace is not None
        and idx_group_size == 1
        and disable_index_value
        and idx_sink is None
        and sink is None
        and (cu_seqlens.numel() - 1) <= 2
    )
    if use_trtllm:
        sm_scale_v = sm_scale if sm_scale is not None else q.shape[-1] ** -0.5
        o = flash_prefill_with_trtllm_gen(
            q=q,
            k_cache=k_cache,
            v_cache=v_cache,
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
            sm_scale=sm_scale_v,
            workspace=workspace,
            score_type=score_type,
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
    block_size_q: int,  # useless for now, will always be 1
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
    paged_main_k: Optional[torch.Tensor] = None,  # [block, kh, page, dim] base[:,0]
    paged_main_v: Optional[torch.Tensor] = None,  # [block, kh, page, dim] base[:,1]
    phys_block_table: Optional[torch.Tensor] = None,  # [batch, max_blocks] phys pages
    paged_idx_k: Optional[torch.Tensor] = None,  # [block, page, idx_dim] scale view
):
    num_idx_heads = idx_q.shape[1]
    # k_cache is None on the zero-copy decode path (the gather scratch was
    # skipped); fall back to the paged pool view for the head count.
    if k_cache is not None:
        num_kv_heads = k_cache.shape[1]
    else:
        num_kv_heads = paged_main_k.shape[1]
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
        and k_cache is not None  # zero-copy decode: k_cache None -> use legacy paged path
    )
    if use_trtllm:
        sm_scale_v = sm_scale if sm_scale is not None else q.shape[-1] ** -0.5
        o = flash_prefill_with_trtllm_gen(
            q=q,
            k_cache=k_cache,
            v_cache=v_cache,
            idx_q=idx_q,
            idx_k_cache=idx_k_cache,
            req_to_token=req_to_token,
            slot_ids=slot_ids,
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
            score_type=score_type,
        )
        return None, o

    # Step 1: Flash decode with topk index (using index head).
    # Zero-copy paged idx scoring (M3_MSA_ZEROCOPY): when the caller supplies the
    # paged scale-region idx view + physical block table (and page==block), the
    # indexer reads idx_K straight from the paged pool, dropping the per-step
    # idx_K gather+scatter scratch. Falls back to the token-major scratch path
    # when these are not provided.
    use_paged_idx = (
        disable_index_value
        and paged_idx_k is not None
        and phys_block_table is not None
        and int(paged_idx_k.shape[1]) == int(block_size_k)
    )
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
        k_paged=paged_idx_k if use_paged_idx else None,
        block_table=phys_block_table if use_paged_idx else None,
    )
    # Step 2: Reduce topk idx if num_idx_heads > num_kv_heads
    num_idx_heads = idx_q.shape[1]
    if k_cache is not None:
        num_kv_heads = k_cache.shape[1]
    else:
        num_kv_heads = paged_main_k.shape[1]
    idx_group_size = num_idx_heads // num_kv_heads
    if idx_group_size > 1:
        topk_idx = topk_index_reduce(
            topk_idx.view(num_kv_heads, idx_group_size, -1, topk), dim=1
        )
    # Step 3: Sparse attention using topk index (main head).
    # Zero-copy paged path (M3_MSA_ZEROCOPY): when the caller supplies the
    # persistent paged K/V pool views + physical block table (and page==block),
    # read the pool directly instead of the token-major gather scratch. This
    # removes the per-step main K/V gather under PD / CP-replicated decode.
    if (
        paged_main_k is not None
        and paged_main_v is not None
        and phys_block_table is not None
        and int(paged_main_k.shape[2]) == int(block_size_k)
    ):
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
        )
    else:
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
