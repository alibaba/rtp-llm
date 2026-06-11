# Copyright 2025. All rights reserved.
"""Fused bitonic top-k + emission kernel for M3 sparse prefill.

Reuses kernel-1 (``_flash_attn_fwd_with_block_score_kernel``) for QK score,
then in one kernel-2 (``_topk_to_block_table_kernel``) does bitonic top-k and
emits either trtllm block_tables+seq_lens (``EMIT_BLOCK_TABLE``) or the
``topk_idx`` layout topk_sparse.py expects (``EMIT_TOPK_IDX``). Assumes
``idx_group_size == 1`` (M3 production: num_idx_heads == num_kv_heads).
"""

import torch
import triton
import triton.language as tl

from ..common.utils import get_cu_seqblocks, robust_allocator
from .flash_with_topk_idx import (
    _bitonic_merge,
    _flash_attn_fwd_with_block_score_kernel,
)


_HEUR_topk_to_block_table_kernel = {
    "BLOCK_SIZE_T": lambda args: triton.next_power_of_2(args["topk"]),
}


@triton.heuristics(_HEUR_topk_to_block_table_kernel)
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE_K": 2048}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_SIZE_K": 1024}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_SIZE_K": 512}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_SIZE_K": 256}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_SIZE_K": 256}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_SIZE_K": 128}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_SIZE_K": 128}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_SIZE_K": 64}, num_warps=2, num_stages=2),
    ],
    key=["BLOCK_SIZE_T"],
)
@triton.jit
def _topk_to_block_table_kernel(
    s_ptr,  # Score: h x n x max_seqblock
    bt_ptr,  # block_tables: (n*NKV) x topk
    seqlen_ptr,  # seq_lens: (n*NKV)
    ti_ptr,  # topk_idx: NKV x n x topk (raw block indices, -1 padding)
    sample_interval: tl.constexpr,
    block_size: tl.constexpr,
    cu_seqlens,
    cu_seqblocks_q,
    prefix_lens,
    topk,
    init_blocks: tl.constexpr,
    local_blocks: tl.constexpr,
    num_pages,
    stride_s_h,
    stride_s_n,
    stride_s_k,
    stride_bt_r,
    stride_bt_t,
    stride_ti_h,
    stride_ti_n,
    stride_ti_t,
    NKV: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    BLOCK_SIZE_T: tl.constexpr,
    MASK_INIT: tl.constexpr,
    MASK_LOCAL: tl.constexpr,
    EMIT_BLOCK_TABLE: tl.constexpr,
    EMIT_TOPK_IDX: tl.constexpr,
):
    tl.static_assert(BLOCK_SIZE_K > BLOCK_SIZE_T)
    pid_q = tl.program_id(0)
    pid_b = tl.program_id(1)
    pid_h = tl.program_id(2)
    seq_start = tl.load(cu_seqlens + pid_b)
    block_start = tl.load(cu_seqblocks_q + pid_b)
    block_num = tl.load(cu_seqblocks_q + pid_b + 1) - block_start
    prefix_len = tl.load(prefix_lens + pid_b)
    if pid_q >= block_num:
        return
    off_k = tl.arange(0, BLOCK_SIZE_K)
    off_t = tl.arange(0, BLOCK_SIZE_T)
    s_ptrs = (
        s_ptr
        + (seq_start + pid_q * sample_interval) * stride_s_n
        + pid_h * stride_s_h
        + off_k * stride_s_k
    )
    topk_score = tl.full((BLOCK_SIZE_K,), -1e30, dtype=tl.float32)
    topk_idx = tl.full((BLOCK_SIZE_K,), 0, dtype=tl.int32)
    left_half_mask = tl.arange(0, BLOCK_SIZE_K) < BLOCK_SIZE_K // 2
    valid_blocks = (prefix_len + pid_q * sample_interval + block_size) // block_size
    for i in tl.range(0, valid_blocks, BLOCK_SIZE_K):
        causal_mask = i + off_k < valid_blocks
        local_mask = i + off_k >= max(0, valid_blocks - local_blocks)
        init_mask = i + off_k < init_blocks
        score = tl.load(s_ptrs, mask=causal_mask, other=-1e30).to(tl.float32)
        score = tl.where(score != score, -1e30, score)
        s_ptrs = s_ptrs + stride_s_k * BLOCK_SIZE_K
        if MASK_INIT:
            score = tl.where(causal_mask & init_mask, score - 1e29, score)
        else:
            score = tl.where(causal_mask & init_mask, 1e30, score)
        if MASK_LOCAL:
            score = tl.where(causal_mask & local_mask, score - 1e28, score)
        else:
            score = tl.where(causal_mask & local_mask, 1e29, score)
        topk_score, last_topk_score = score, topk_score
        topk_idx, last_topk_idx = (tl.where(causal_mask, i + off_k + 1, 0), topk_idx)
        n_dims: tl.constexpr = tl.standard._log2(BLOCK_SIZE_K)
        for j in tl.static_range(1, n_dims):
            topk_score, topk_idx = _bitonic_merge(
                topk_score, topk_idx.to(tl.int32), j, 2, n_dims
            )
        if i != 0:
            topk_score, topk_idx = _bitonic_merge(
                topk_score, topk_idx.to(tl.int32), n_dims, False, n_dims
            )
            topk_score_new = last_topk_score * left_half_mask + topk_score * (
                1 - left_half_mask
            )
            topk_idx_new = last_topk_idx * left_half_mask + topk_idx * (
                1 - left_half_mask
            )
            topk_score, topk_idx = _bitonic_merge(
                topk_score_new, topk_idx_new.to(tl.int32), n_dims, True, n_dims
            )
        else:
            topk_score, topk_idx = _bitonic_merge(
                topk_score, topk_idx.to(tl.int32), n_dims, True, n_dims
            )
    # reduce to the top BLOCK_SIZE_T block indices (>=0; -1 padding)
    sel_mask = tl.arange(0, BLOCK_SIZE_K // BLOCK_SIZE_T) == 0
    t = tl.sum(
        sel_mask[:, None]
        * tl.reshape(topk_idx - 1, [BLOCK_SIZE_K // BLOCK_SIZE_T, BLOCK_SIZE_T]),
        axis=0,
    )  # [BLOCK_SIZE_T] block indices

    # ---- fused emission: optional block-table + optional topk_idx ----
    n_sel = min(topk, valid_blocks)
    valid = (off_t < n_sel) & (t >= 0)
    if EMIT_BLOCK_TABLE:
        local_blk = (prefix_len + pid_q * sample_interval) // block_size
        nvalid = tl.sum(valid.to(tl.int32))
        is_local = valid & (t == local_blk)
        has_local = tl.sum(is_local.to(tl.int32)) > 0
        non_local = valid & (t != local_blk)
        rank = tl.cumsum(non_local.to(tl.int32)) - 1
        out_pos = tl.where(is_local, nvalid - 1, rank)
        page = pid_h * num_pages + t
        row = (block_start + pid_q) * NKV + pid_h
        tl.store(bt_ptr + row * stride_bt_r + out_pos * stride_bt_t, page, mask=valid)
        partial = (prefix_len + pid_q * sample_interval) % block_size + 1
        sl_val = tl.where(
            has_local, (nvalid - 1) * block_size + partial, nvalid * block_size
        )
        tl.store(seqlen_ptr + row, sl_val)
    if EMIT_TOPK_IDX:
        # topk_idx [NKV, total_q, topk] in the layout topk_sparse expects.
        # Padding is -1 (matches existing flash_prefill_with_topk_index contract).
        ti_val = tl.where(valid, t, -1)
        ti_offset = (
            (block_start + pid_q) * stride_ti_n
            + pid_h * stride_ti_h
            + off_t * stride_ti_t
        )
        tl.store(
            ti_ptr + ti_offset,
            ti_val.to(ti_ptr.dtype.element_ty),
            mask=off_t < topk,
        )


@torch.no_grad()
def flash_prefill_topk_to_block_tables(
    idx_q: torch.Tensor,  # [total_q, num_idx_heads(=num_kv_heads), d]
    idx_k_cache: torch.Tensor,  # paged [max_slots, 1, d]
    req_to_token: torch.Tensor,
    slot_ids: torch.Tensor,
    cu_seqlens: torch.Tensor,
    seq_lens: torch.Tensor,
    prefix_lens: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    block_size_k: int,
    topk: int,
    num_pages: int,
    init_blocks: int = 0,
    local_blocks: int = 1,
    sm_scale=None,
    score_type: str = "max",
):
    """Returns (block_tables [total_q*NKV, topk] int32, seq_lens [total_q*NKV] int32)."""
    triton.set_allocator(robust_allocator)
    total_q, num_heads, qk_head_dim = idx_q.shape
    max_slots, idx_kv_heads, _ = idx_k_cache.shape  # idx K is usually single-head
    gqa_group_size = num_heads // idx_kv_heads  # QK kernel: index-heads share idx K
    # block-table layout: in production idx_group_size==1 so each index head maps to
    # one main-attention KV head -> NKV == num_heads.
    num_kv_heads = num_heads
    batch_size = cu_seqlens.shape[0] - 1
    block_size_q = 1
    if sm_scale is None:
        sm_scale = qk_head_dim**-0.5

    cu_seqblocks_q, max_seqblock_q, all_seqblock_q, _, _, _ = get_cu_seqblocks(
        cu_seqlens, max_seqlen_q, block_size_q, block_size_k
    )
    max_seqblock_k = triton.cdiv(max_seqlen_k, block_size_k)
    v_head_dim = qk_head_dim  # V never loaded (disable_index_value)

    score = torch.full(
        (num_heads, total_q, max_seqblock_k),
        float("-inf"),
        dtype=torch.float32,
        device=idx_q.device,
    )

    def grid(META):
        return (triton.cdiv(max_seqlen_q, META["BLOCK_SIZE_Q"]), batch_size * num_heads)

    _flash_attn_fwd_with_block_score_kernel[grid](
        idx_q, idx_k_cache, None, None, None, score, req_to_token, cu_seqlens,
        seq_lens, prefix_lens, slot_ids, max_slots, num_heads, gqa_group_size,
        qk_head_dim, v_head_dim, block_size_k, sm_scale, False, 1,
        idx_q.stride(0), idx_q.stride(1), idx_q.stride(2),
        idx_k_cache.stride(0), idx_k_cache.stride(1), idx_k_cache.stride(2),
        0, 0, 0,
        0, 0,
        0, 0, 0,
        score.stride(0), score.stride(1), score.stride(2),
        req_to_token.stride(0),
        SCORE_TYPE=score_type, DISABLE_INDEX_VALUE=True,
    )

    bt = torch.zeros(total_q * num_kv_heads, topk, dtype=torch.int32, device=idx_q.device)
    sl = torch.zeros(total_q * num_kv_heads, dtype=torch.int32, device=idx_q.device)
    # ti_ptr unused; bt reused as a non-null placeholder
    grid2 = (max_seqblock_q, batch_size, num_heads)
    _topk_to_block_table_kernel[grid2](
        score, bt, sl, bt,
        block_size_q, block_size_k, cu_seqlens, cu_seqblocks_q,
        prefix_lens, topk, init_blocks, local_blocks, num_pages,
        score.stride(0), score.stride(1), score.stride(2),
        bt.stride(0), bt.stride(1),
        0, 0, 0,
        NKV=num_kv_heads, MASK_INIT=False, MASK_LOCAL=False,
        EMIT_BLOCK_TABLE=True, EMIT_TOPK_IDX=False,
    )
    return bt, sl


@torch.no_grad()
def flash_prefill_with_trtllm_gen(
    q: torch.Tensor,                # [total_q, num_q_heads, head_dim] bf16
    k_cache: torch.Tensor,          # [max_slots, num_kv_heads, head_dim] FLAT
    v_cache: torch.Tensor,          # [max_slots, num_kv_heads, head_dim] FLAT
    idx_q: torch.Tensor,            # [total_q, num_idx_heads, idx_head_dim]
    idx_k_cache: torch.Tensor,      # [max_slots, 1, idx_head_dim]
    req_to_token: torch.Tensor,
    slot_ids: torch.Tensor,
    cu_seqlens: torch.Tensor,
    seq_lens: torch.Tensor,
    prefix_lens: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    block_size_k: int,
    topk: int,
    init_blocks: int,
    local_blocks: int,
    sm_scale: float,
    workspace: torch.Tensor,
    score_type: str = "max",
) -> torch.Tensor:
    """Fast sparse prefill: mega topk + trtllm-gen sparse-decode attention.

    Replaces the legacy 3-stage all-triton pipeline (flash_prefill_with_topk_index
    + flash_prefill_with_gqa_share_sparse). Verified ~1.76x faster than the
    triton ``_gqa_share_sparse_fwd_kernel`` at q_len=8192 on L20D (see
    m3_test/test_mega_pipeline.py).

    Constraints (caller must check, otherwise fall back to legacy):
      * idx_group_size == 1  (num_idx_heads == num_kv_heads)
      * max_seqlen_q <= 8192 (trtllm-gen kernel crashes with INVALID_VALUE on
        L20D for q_len in [16384, 32768])
      * The fused topk-to-block-table kernel assumes a single contiguous KV
        cache slice (page id = pid_h * num_pages + block_idx with no per-batch
        offset). CP prefill on a single request satisfies this; multi-request
        batching would need a kernel extension.
    """
    from flashinfer.decode import trtllm_batch_decode_with_kv_cache

    total_q, num_q_heads, head_dim = q.shape
    max_slots, num_kv_heads, _ = k_cache.shape
    gqa = num_q_heads // num_kv_heads
    num_pages = (max_seqlen_k + block_size_k - 1) // block_size_k

    # Step 1+2: fused QK score + bitonic topk + trtllm-style block_table emit.
    bt, sl = flash_prefill_topk_to_block_tables(
        idx_q=idx_q, idx_k_cache=idx_k_cache,
        req_to_token=req_to_token, slot_ids=slot_ids,
        cu_seqlens=cu_seqlens, seq_lens=seq_lens, prefix_lens=prefix_lens,
        max_seqlen_q=max_seqlen_q, max_seqlen_k=max_seqlen_k,
        block_size_k=block_size_k, topk=topk, num_pages=num_pages,
        init_blocks=init_blocks, local_blocks=local_blocks,
        score_type=score_type,
    )

    # Permute flat MSA side cache to trtllm's paged layout:
    #   flat  [num_pages * block_size_k, num_kv_heads, head_dim]
    # -> paged [num_kv_heads * num_pages, 1, block_size_k, head_dim]
    # The trailing unsqueeze(1) is trtllm's blocks-per-token dim (= 1 here).
    # `.contiguous()` after permute is one DtoD memcpy ~16us per cache per
    # layer at 8k tokens — far less than the legacy step3's 1.4ms savings.
    paged_slots = num_pages * block_size_k

    def _to_paged(cache: torch.Tensor) -> torch.Tensor:
        return (
            cache[:paged_slots]
            .view(num_pages, block_size_k, num_kv_heads, head_dim)
            .permute(2, 0, 1, 3)
            .contiguous()
            .view(num_kv_heads * num_pages, block_size_k, head_dim)
            .unsqueeze(1)
        )

    k_paged = _to_paged(k_cache)
    v_paged = _to_paged(v_cache)

    # Pack Q for trtllm-gen's GQA layout:
    #   [total_q, num_q_heads, head_dim] -> [total_q * num_kv_heads, gqa, head_dim]
    q_packed = (
        q.view(total_q, num_kv_heads, gqa, head_dim)
        .reshape(total_q * num_kv_heads, gqa, head_dim)
        .contiguous()
    )

    out = trtllm_batch_decode_with_kv_cache(
        query=q_packed,
        kv_cache=(k_paged, v_paged),
        workspace_buffer=workspace,
        block_tables=bt,
        seq_lens=sl,
        max_seq_len=max_seqlen_q,
        bmm1_scale=sm_scale,
        bmm2_scale=1.0,
        backend="trtllm-gen",
        out_dtype=torch.bfloat16,
    )

    return (
        out.view(total_q, num_kv_heads, gqa, head_dim)
        .reshape(total_q, num_q_heads, head_dim)
    )


@torch.no_grad()
def flash_prefill_with_fused_topk_index(
    idx_q: torch.Tensor,  # [total_q, num_idx_heads, idx_head_dim]
    idx_k_cache: torch.Tensor,  # [max_slots, 1, idx_head_dim]
    req_to_token: torch.Tensor,
    slot_ids: torch.Tensor,
    cu_seqlens: torch.Tensor,
    seq_lens: torch.Tensor,
    prefix_lens: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    block_size_k: int,
    topk: int,
    init_blocks: int = 0,
    local_blocks: int = 1,
    sm_scale=None,
    score_type: str = "max",
):
    """Drop-in for ``flash_prefill_with_topk_index`` (M3 sparse, disable_index_value,
    idx_group_size==1). Returns ``(None, topk_idx[num_idx_heads, total_q, topk] int32)``
    with -1 padding — same contract, fed straight into topk_sparse step 3.
    """
    triton.set_allocator(robust_allocator)
    total_q, num_heads, qk_head_dim = idx_q.shape
    max_slots, idx_kv_heads, _ = idx_k_cache.shape
    gqa_group_size = num_heads // idx_kv_heads
    batch_size = cu_seqlens.shape[0] - 1
    block_size_q = 1
    if sm_scale is None:
        sm_scale = qk_head_dim**-0.5

    cu_seqblocks_q, max_seqblock_q, _all_seqblock_q, _, _, _ = get_cu_seqblocks(
        cu_seqlens, max_seqlen_q, block_size_q, block_size_k
    )
    max_seqblock_k = triton.cdiv(max_seqlen_k, block_size_k)
    v_head_dim = qk_head_dim  # V never loaded (disable_index_value)

    score = torch.full(
        (num_heads, total_q, max_seqblock_k),
        float("-inf"),
        dtype=torch.float32,
        device=idx_q.device,
    )

    def grid(META):
        return (triton.cdiv(max_seqlen_q, META["BLOCK_SIZE_Q"]), batch_size * num_heads)

    _flash_attn_fwd_with_block_score_kernel[grid](
        idx_q, idx_k_cache, None, None, None, score, req_to_token, cu_seqlens,
        seq_lens, prefix_lens, slot_ids, max_slots, num_heads, gqa_group_size,
        qk_head_dim, v_head_dim, block_size_k, sm_scale, False, 1,
        idx_q.stride(0), idx_q.stride(1), idx_q.stride(2),
        idx_k_cache.stride(0), idx_k_cache.stride(1), idx_k_cache.stride(2),
        0, 0, 0,
        0, 0,
        0, 0, 0,
        score.stride(0), score.stride(1), score.stride(2),
        req_to_token.stride(0),
        SCORE_TYPE=score_type, DISABLE_INDEX_VALUE=True,
    )

    topk_idx = torch.full(
        (num_heads, total_q, topk),
        fill_value=-1,
        dtype=torch.int32,
        device=idx_q.device,
    )
    # bt/sl unused; tiny dummies as non-null pointer placeholders
    bt_dummy = torch.empty(1, 1, dtype=torch.int32, device=idx_q.device)
    sl_dummy = torch.empty(1, dtype=torch.int32, device=idx_q.device)
    grid2 = (max_seqblock_q, batch_size, num_heads)
    _topk_to_block_table_kernel[grid2](
        score, bt_dummy, sl_dummy, topk_idx,
        block_size_q, block_size_k, cu_seqlens, cu_seqblocks_q,
        prefix_lens, topk, init_blocks, local_blocks, 0,
        score.stride(0), score.stride(1), score.stride(2),
        bt_dummy.stride(0), bt_dummy.stride(1),
        topk_idx.stride(0), topk_idx.stride(1), topk_idx.stride(2),
        NKV=num_heads, MASK_INIT=False, MASK_LOCAL=False,
        EMIT_BLOCK_TABLE=False, EMIT_TOPK_IDX=True,
    )
    return None, topk_idx
