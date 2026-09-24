# Copyright 2025 XunhaoLai. All rights reserved.

from typing import Optional

import torch
import triton
import triton.language as tl

from ..common.utils import robust_allocator

_HEUR_gqa_share_sparse_decode_kernel = {
    "BLOCK_SIZE_H": lambda args: max(
        16, triton.next_power_of_2(args["gqa_group_size"])
    ),
    "BLOCK_SIZE_D": lambda args: triton.next_power_of_2(args["head_dim"]),
    "BLOCK_SIZE_T": lambda args: triton.next_power_of_2(args["max_topk"]),
    "HAS_SINK": lambda args: args["sink_ptr"] is not None,
    "BATCH_SIZE_BUCKET": lambda args: triton.next_power_of_2(args["batch_size"]),
}


@triton.heuristics(_HEUR_gqa_share_sparse_decode_kernel)
@triton.autotune(
    configs=[
        triton.Config({}, num_warps=nw, num_stages=ns)
        for nw in [4, 8]
        for ns in [2, 3, 4, 5]
    ],
    key=["BATCH_SIZE_BUCKET", "gqa_group_size", "head_dim", "block_size", "HAS_SINK"],
)
@triton.jit
def _gqa_share_sparse_decode_paged_kernel(
    q_ptr,  # Q: b x qh x d
    sink_ptr,  # Sink: qh x d
    k_paged_ptr,  # K paged pool view: block x kh x page x d
    v_paged_ptr,  # V paged pool view: block x kh x page x d
    block_table_ptr,  # physical block table: b x max_blocks (logical blk -> phys page)
    idx_ptr,  # topk index: qh x b x topk (logical block ids)
    o_ptr,  # O partial: c x b x qh x d
    lse_ptr,  # lse partial: c x b x qh
    seq_lens,
    # shape
    batch_size,
    gqa_group_size,
    head_dim,
    max_topk,
    max_kv_len,
    num_phys_blocks,
    # sm_scale
    sm_scale,
    # stride
    stride_q_b,
    stride_q_h,
    stride_q_d,
    stride_sink_h,
    stride_sink_d,
    stride_k_blk,
    stride_k_h,
    stride_k_p,
    stride_k_d,
    stride_v_blk,
    stride_v_h,
    stride_v_p,
    stride_v_d,
    stride_bt_b,
    stride_bt_blk,
    stride_ti_h,
    stride_ti_b,
    stride_ti_t,
    stride_o_c,
    stride_o_b,
    stride_o_h,
    stride_o_d,
    stride_l_c,
    stride_l_b,
    stride_l_h,
    # META parameters
    BATCH_SIZE_BUCKET: tl.constexpr,
    BLOCK_SIZE_H: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,  # MUST equal the paged page_size (== block_size)
    BLOCK_SIZE_D: tl.constexpr,
    BLOCK_SIZE_T: tl.constexpr,
    NUM_TOPK_CHUNKS: tl.constexpr,
    HAS_SINK: tl.constexpr,
):
    """Paged sparse decode kernel.

    Reads the persistent paged K/V pool ([block, kh, page, dim] view of the
    standard [block, 2, kh, page, dim] cache) directly via the physical block
    table, instead of a token-major gather scratch. Requires the sparse block
    size to equal the physical page size (``BLOCK_SIZE_N == page_size``) so each
    topk logical block ``c`` is exactly physical page ``block_table[b, c]``,
    tokens at within-page offsets ``0 .. BLOCK_SIZE_N-1``. This removes the
    per-step main K/V gather under PD/CP-replicated decode (the paged pool
    already holds the full replicated history)."""
    pid_bc, pid_kh = tl.program_id(0), tl.program_id(1)
    pid_b = pid_bc % batch_size
    pid_c = pid_bc // batch_size
    pid_h = pid_kh * gqa_group_size
    chunk_size_topk = (max_topk + NUM_TOPK_CHUNKS - 1) // NUM_TOPK_CHUNKS
    chunk_start_topk = pid_c * chunk_size_topk
    chunk_end_topk_compiletime = chunk_start_topk + chunk_size_topk
    seq_len = tl.minimum(tl.load(seq_lens + pid_b), max_kv_len)
    # get real topk
    off_t = tl.arange(0, BLOCK_SIZE_T)
    idx_base = idx_ptr + pid_kh * stride_ti_h + pid_b * stride_ti_b
    topk_idx = tl.load(idx_base + off_t * stride_ti_t, mask=off_t < max_topk, other=-1)
    valid_idx = tl.where(topk_idx >= 0, off_t, -1)
    real_topk = tl.sum(valid_idx != -1, axis=0)
    chunk_end_topk = tl.minimum(chunk_end_topk_compiletime, real_topk)
    # init pointer
    off_n = tl.arange(0, BLOCK_SIZE_N)
    off_d = tl.arange(0, BLOCK_SIZE_D)
    dim_mask = off_d < head_dim
    bt_row = block_table_ptr + pid_b * stride_bt_b
    if HAS_SINK and pid_c == 0:
        q_ptrs = tl.make_block_ptr(
            base=q_ptr + pid_b * stride_q_b + pid_h * stride_q_h,
            shape=(gqa_group_size, head_dim),
            strides=(stride_q_h, stride_q_d),
            offsets=(0, 0),
            block_shape=(BLOCK_SIZE_H, BLOCK_SIZE_D),
            order=(1, 0),
        )
        q = tl.load(q_ptrs, boundary_check=(0, 1), padding_option="zero")
        sink_ptrs = tl.make_block_ptr(
            base=sink_ptr + pid_h * stride_sink_h,
            shape=(gqa_group_size, head_dim),
            strides=(stride_sink_h, stride_sink_d),
            offsets=(0, 0),
            block_shape=(BLOCK_SIZE_H, BLOCK_SIZE_D),
            order=(1, 0),
        )
        sink = tl.load(sink_ptrs, boundary_check=(0, 1), padding_option="zero").to(
            tl.float32
        )
        qsink = tl.sum(q.to(tl.float32) * sink, axis=1) * sm_scale
        m_i = qsink
        lse_i = qsink
    else:
        m_i = tl.full((BLOCK_SIZE_H,), float("-inf"), dtype=tl.float32)
        lse_i = tl.full((BLOCK_SIZE_H,), float("-inf"), dtype=tl.float32)
        q_ptrs = tl.make_block_ptr(
            base=q_ptr + pid_b * stride_q_b + pid_h * stride_q_h,
            shape=(gqa_group_size, head_dim),
            strides=(stride_q_h, stride_q_d),
            offsets=(0, 0),
            block_shape=(BLOCK_SIZE_H, BLOCK_SIZE_D),
            order=(1, 0),
        )
        q = tl.load(q_ptrs, boundary_check=(0, 1), padding_option="zero")
    acc_o = tl.full((BLOCK_SIZE_H, BLOCK_SIZE_D), 0, dtype=tl.float32)
    cur_idx_ptr = idx_base + chunk_start_topk * stride_ti_t
    max_blocks = (max_kv_len + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
    for _ in tl.range(chunk_start_topk, chunk_end_topk):
        # logical block id -> physical page via the block table; within-page
        # offset is off_n because page_size == block_size == BLOCK_SIZE_N.
        c = tl.load(cur_idx_ptr).to(tl.int32)
        cur_idx_ptr = cur_idx_ptr + stride_ti_t
        valid_logical_block = (c >= 0) & (c < max_blocks)
        safe_c = tl.where(valid_logical_block, c, 0)
        page = tl.load(
            bt_row + safe_c * stride_bt_blk, mask=valid_logical_block, other=-1
        ).to(tl.int64)
        valid_page = valid_logical_block & (page >= 0) & (page < num_phys_blocks)
        safe_page = tl.where(valid_page, page, 0)
        base_pos = safe_c * BLOCK_SIZE_N
        pos_mask = valid_page & ((base_pos + off_n) < seq_len)
        # load K as (head_dim, BLOCK_SIZE_N)
        k_off = (
            safe_page * stride_k_blk
            + pid_kh * stride_k_h
            + off_n[None, :] * stride_k_p
            + off_d[:, None] * stride_k_d
        )
        k = tl.load(
            k_paged_ptr + k_off,
            mask=valid_page & dim_mask[:, None] & pos_mask[None, :],
            other=0.0,
        )
        # load V as (BLOCK_SIZE_N, head_dim)
        v_off = (
            safe_page * stride_v_blk
            + pid_kh * stride_v_h
            + off_n[:, None] * stride_v_p
            + off_d[None, :] * stride_v_d
        )
        v = tl.load(
            v_paged_ptr + v_off,
            mask=valid_page & pos_mask[:, None] & dim_mask[None, :],
            other=0.0,
        )
        block_active = tl.sum(tl.where(pos_mask, 1, 0), axis=0) > 0
        # Upconvert to Q's dtype so an FP8 (e4m3) paged pool feeds the bf16/fp16
        # matmuls (no-op when the pool already matches Q). The M3 sparse KV is a
        # "no-scale" e4m3 cast, so a plain widening cast reconstructs it.
        k = k.to(q.dtype)
        v = v.to(q.dtype)
        qk = tl.zeros((BLOCK_SIZE_H, BLOCK_SIZE_N), dtype=tl.float32)
        qk += tl.where(pos_mask[None, :], 0, float("-inf"))
        qk += tl.dot(q, k) * sm_scale
        m_ij_candidate = tl.maximum(m_i, tl.max(qk, axis=1))
        m_ij = tl.where(block_active, m_ij_candidate, m_i)
        p = tl.where(block_active, tl.exp(qk - m_ij[:, None]), 0.0)
        l_ij = tl.sum(p, axis=1)
        acc_o_scale = tl.where(block_active, tl.exp(m_i - m_ij), 1.0)
        acc_o = acc_o * acc_o_scale[:, None]
        p = p.to(v.dtype)
        acc_o += tl.dot(p.to(v.dtype), v)
        m_i = m_ij
        lse_i = tl.where(
            block_active,
            m_ij + tl.log(tl.exp(lse_i - m_ij) + l_ij),
            lse_i,
        )
    scale = tl.where(
        lse_i > float("-inf"),
        tl.exp(m_i - lse_i),
        tl.zeros_like(lse_i),
    )
    acc_o = acc_o * scale[:, None]
    o_ptrs = tl.make_block_ptr(
        base=o_ptr + pid_c * stride_o_c + pid_b * stride_o_b + pid_h * stride_o_h,
        shape=(gqa_group_size, head_dim),
        strides=(stride_o_h, stride_o_d),
        offsets=(0, 0),
        block_shape=(BLOCK_SIZE_H, BLOCK_SIZE_D),
        order=(1, 0),
    )
    tl.store(o_ptrs, acc_o.to(o_ptr.dtype.element_ty), boundary_check=(0, 1))
    lse_ptrs = tl.make_block_ptr(
        base=lse_ptr + pid_c * stride_l_c + pid_b * stride_l_b + pid_h * stride_l_h,
        shape=(gqa_group_size,),
        strides=(stride_l_h,),
        offsets=(0,),
        block_shape=(BLOCK_SIZE_H,),
        order=(0,),
    )
    tl.store(lse_ptrs, lse_i.to(lse_ptr.dtype.element_ty), boundary_check=(0,))


_HEUR_merge_topk_attn_out_kernel = {
    "BLOCK_SIZE_D": lambda args: triton.next_power_of_2(args["head_dim"]),
}


@triton.heuristics(_HEUR_merge_topk_attn_out_kernel)
@triton.jit
def _merge_topk_attn_out_kernel(
    o_ptr,  # [NUM_TOPK_CHUNKS, BS, NQH, D] — partials in, merged out at chunk 0
    lse_ptr,  # [NUM_TOPK_CHUNKS, BS, NQH]
    head_dim,
    stride_o_c,
    stride_o_b,
    stride_o_h,
    stride_o_d,
    stride_l_c,
    stride_l_b,
    stride_l_h,
    NUM_TOPK_CHUNKS: tl.constexpr,
    BLOCK_SIZE_D: tl.constexpr,
):
    pid_b, pid_h = tl.program_id(0), tl.program_id(1)
    off_c = tl.arange(0, NUM_TOPK_CHUNKS)
    off_d = tl.arange(0, BLOCK_SIZE_D)
    o_ptrs = tl.make_block_ptr(
        base=o_ptr + pid_b * stride_o_b + pid_h * stride_o_h,
        shape=(NUM_TOPK_CHUNKS, head_dim),
        strides=(stride_o_c, stride_o_d),
        offsets=(0, 0),
        block_shape=(NUM_TOPK_CHUNKS, BLOCK_SIZE_D),
        order=(1, 0),
    )
    lse_ptrs = lse_ptr + pid_b * stride_l_b + pid_h * stride_l_h + off_c * stride_l_c
    o = tl.load(o_ptrs, boundary_check=(0, 1), padding_option="zero")
    lse = tl.load(lse_ptrs)  # empty chunks contribute -inf
    # standard flash-decoding merge in linear (not log2) space, matching the
    # decode kernel which uses tl.exp / tl.log.
    lse_max = tl.max(lse, axis=0)
    has_valid_chunk = lse_max > float("-inf")
    safe_lse_max = tl.where(has_valid_chunk, lse_max, 0.0)
    weights = tl.where(has_valid_chunk, tl.exp(lse - safe_lse_max), 0.0)
    weight_sum = tl.sum(weights, axis=0)
    weights = weights / tl.where(has_valid_chunk, weight_sum, 1.0)
    o_merged = tl.sum(o * weights[:, None], axis=0)
    o_out_ptrs = o_ptr + pid_b * stride_o_b + pid_h * stride_o_h + off_d * stride_o_d
    tl.store(o_out_ptrs, o_merged.to(o_ptr.dtype.element_ty), mask=off_d < head_dim)


def _merge_topk_attn_out(
    o_partial: torch.Tensor, lse_partial: torch.Tensor
) -> torch.Tensor:
    batch_size = int(o_partial.shape[1])
    num_q_heads = int(o_partial.shape[2])
    head_dim = int(o_partial.shape[3])
    num_topk_chunks = int(o_partial.shape[0])
    _merge_topk_attn_out_kernel[(batch_size, num_q_heads)](
        o_partial,
        lse_partial,
        head_dim,
        o_partial.stride(0),
        o_partial.stride(1),
        o_partial.stride(2),
        o_partial.stride(3),
        lse_partial.stride(0),
        lse_partial.stride(1),
        lse_partial.stride(2),
        NUM_TOPK_CHUNKS=num_topk_chunks,
    )
    return o_partial[0].contiguous()


@torch.no_grad()
def flash_decode_with_gqa_share_sparse_paged(
    q: torch.Tensor,  # [batch_size, num_q_heads, head_dim]
    sink: Optional[torch.Tensor],
    k_paged: torch.Tensor,  # [block, num_kv_heads, page, head_dim] (base[:,0] view)
    v_paged: torch.Tensor,  # [block, num_kv_heads, page, head_dim] (base[:,1] view)
    block_table: torch.Tensor,  # [batch_size, max_blocks] physical page ids
    seq_lens: torch.Tensor,  # [batch_size, ]
    block_size: int,  # sparse block size; MUST equal page size (k_paged.shape[2])
    topk_idx: torch.Tensor,  # [num_kv_heads, batch_size, topk] logical block ids
    sm_scale: Optional[float] = None,
    num_topk_chunks: Optional[int] = None,
) -> torch.Tensor:
    """Paged sparse decode. Reads the persistent paged K/V pool
    directly via the physical block table, so the caller does NOT need to gather
    history into a token-major scratch. Requires ``block_size == page_size`` so
    each topk logical block maps 1:1 to a physical page."""
    triton.set_allocator(robust_allocator)
    assert q.dtype in (torch.bfloat16, torch.float16)
    # k_paged may be an FP8 (e4m3) paged pool; the kernel upconverts to q.dtype.
    assert k_paged.dtype in (q.dtype, torch.float8_e4m3fn)
    assert v_paged.dtype == k_paged.dtype
    batch_size, num_q_heads, head_dim = q.shape
    num_phys_blocks, num_kv_heads, page_size, _ = k_paged.shape
    assert (
        page_size == block_size
    ), f"paged decode requires page_size({page_size}) == block_size({block_size})"
    assert seq_lens.shape[0] == batch_size
    assert topk_idx.shape[0] == num_kv_heads
    assert block_table.shape[0] == batch_size
    assert (
        triton.next_power_of_2(block_size) == block_size
    ), f"block_size must be a power of 2, but got {block_size}"
    max_kv_len = block_table.shape[1] * page_size
    assert num_q_heads % num_kv_heads == 0
    gqa_group_size = num_q_heads // num_kv_heads
    max_topk = topk_idx.shape[2]
    if sm_scale is None:
        sm_scale = head_dim**-0.5
    if num_topk_chunks is None:
        target = max(
            1,
            min(max_topk, 256 // max(1, batch_size * num_kv_heads)),
        )
        num_topk_chunks = 1 << (target.bit_length() - 1)
    if (
        num_topk_chunks <= 0
        or num_topk_chunks > max_topk
        or num_topk_chunks & (num_topk_chunks - 1)
    ):
        raise ValueError(
            "num_topk_chunks must be a positive power of two no larger than topk"
        )
    NUM_TOPK_CHUNKS = num_topk_chunks
    o_partial = torch.empty(
        NUM_TOPK_CHUNKS,
        batch_size,
        num_q_heads,
        head_dim,
        dtype=q.dtype,
        device=q.device,
    )
    lse_partial = torch.empty(
        NUM_TOPK_CHUNKS,
        batch_size,
        num_q_heads,
        dtype=torch.float32,
        device=q.device,
    )
    grid = (batch_size * NUM_TOPK_CHUNKS, num_kv_heads)
    kernel_args = (
        q,
        sink,
        k_paged,
        v_paged,
        block_table,
        topk_idx,
        o_partial,
        lse_partial,
        seq_lens,
        batch_size,
        gqa_group_size,
        head_dim,
        max_topk,
        max_kv_len,
        num_phys_blocks,
        sm_scale,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        sink.stride(0) if sink is not None else 0,
        sink.stride(1) if sink is not None else 0,
        k_paged.stride(0),
        k_paged.stride(1),
        k_paged.stride(2),
        k_paged.stride(3),
        v_paged.stride(0),
        v_paged.stride(1),
        v_paged.stride(2),
        v_paged.stride(3),
        block_table.stride(0),
        block_table.stride(1),
        topk_idx.stride(0),
        topk_idx.stride(1),
        topk_idx.stride(2),
        o_partial.stride(0),
        o_partial.stride(1),
        o_partial.stride(2),
        o_partial.stride(3),
        lse_partial.stride(0),
        lse_partial.stride(1),
        lse_partial.stride(2),
    )
    _gqa_share_sparse_decode_paged_kernel[grid](
        *kernel_args,
        BLOCK_SIZE_N=block_size,
        NUM_TOPK_CHUNKS=NUM_TOPK_CHUNKS,
    )
    return _merge_topk_attn_out(o_partial, lse_partial)
