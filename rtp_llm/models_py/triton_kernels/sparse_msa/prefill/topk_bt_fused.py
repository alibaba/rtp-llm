# Copyright 2025. All rights reserved.
"""Fused bitonic top-k + emission kernel for M3 sparse prefill.

Reuses kernel-1 (``_flash_attn_fwd_with_block_score_kernel``) for QK score,
then in one kernel-2 (``_topk_to_block_table_kernel``) does bitonic top-k and
emits either trtllm block_tables+seq_lens (``EMIT_BLOCK_TABLE``) or the
``topk_idx`` layout topk_sparse.py expects (``EMIT_TOPK_IDX``). Assumes
``idx_group_size == 1`` (M3 production: num_idx_heads == num_kv_heads).
"""

import os
import subprocess

import torch
import triton
import triton.language as tl

from ..common.utils import get_cu_seqblocks, robust_allocator
from .flash_with_topk_idx import _bitonic_merge, _flash_attn_fwd_with_block_score_kernel

# Opt-1+: bypass the fmha_sm100 adapter in step3, preallocating the CSR/schedule/
# page_table buffers once per forward (reused across all sparse layers) so the
# per-layer ``.tolist()`` DtoH sync + 6 torch.empty allocs + schedule-capacity
# recompute (~250us/layer of GPU idle) collapse to once-per-forward. The native CSR
# kernel + schedule are reused verbatim -> output is bit-identical + deterministic.
_M3_MSA_FUSED_CSR = os.environ.get("M3_MSA_FUSED_CSR", "1") == "1"


def _patch_fmha_sm100_cxx_standard():
    """GCC < 11 crashes (ICE) on CUDA 13's libcudacxx C++20 templates.
    The fmha_sm100 JIT kernels compile fine with C++17; patch the flag."""
    try:
        ver = subprocess.check_output(["g++", "-dumpversion"], text=True).strip()
        if int(ver.split(".")[0]) >= 11:
            return
    except Exception:
        return
    try:
        import fmha_sm100.jit as _jit

        _orig = _jit._get_nvcc_flags

        def _patched(cache_dir, fmha=True):
            return _orig(cache_dir, fmha).replace("-std=c++20", "-std=c++17")

        _jit._get_nvcc_flags = _patched
    except Exception:
        pass


_patch_fmha_sm100_cxx_standard()


@triton.jit
def _kv_flat_to_paged_kernel(
    k_in,
    v_in,
    k_out,
    v_out,
    block,
    nkv,
    dim,
    BLOCK_B: tl.constexpr,
    DIM: tl.constexpr,
    NKV: tl.constexpr,
):
    """Fused K+V copy: flat scratch page [block, nkv, dim] -> paged [nkv, block, dim]
    (out[p,h,b,d] = in[p,b,h,d]). One launch for both caches; coalesced read (the
    input page is contiguous [block,nkv,dim]) + coalesced write (nkv contiguous
    [BLOCK_B,dim] chunks). ~4x faster than two aten permute+contiguous copies."""
    pid_p = tl.program_id(0)
    b0 = tl.program_id(1) * BLOCK_B
    offs_b = b0 + tl.arange(0, BLOCK_B)
    offs_h = tl.arange(0, NKV)
    offs_d = tl.arange(0, DIM)
    mask_b = offs_b < block
    in_off = (
        pid_p * (block * nkv * dim)
        + offs_b[:, None, None] * (nkv * dim)
        + offs_h[None, :, None] * dim
        + offs_d[None, None, :]
    )
    k_tile = tl.load(k_in + in_off, mask=mask_b[:, None, None], other=0)
    v_tile = tl.load(v_in + in_off, mask=mask_b[:, None, None], other=0)
    out_off = (
        pid_p * (nkv * block * dim)
        + offs_h[None, :, None] * (block * dim)
        + offs_b[:, None, None] * dim
        + offs_d[None, None, :]
    )
    tl.store(k_out + out_off, k_tile, mask=mask_b[:, None, None])
    tl.store(v_out + out_off, v_tile, mask=mask_b[:, None, None])


def _kv_flat_to_paged(
    k_cache, v_cache, num_paged, block_size_k, num_kv_heads, head_dim
):
    """flat [slots, nkv, dim] -> paged [num_paged, nkv, block, dim] for K and V in one
    fused launch. Returns (k_paged, v_paged) contiguous. Replaces two aten
    permute(0,2,1,3).contiguous() copies (~4x faster, bit-identical)."""
    shape = (num_paged, num_kv_heads, block_size_k, head_dim)
    k_paged = torch.empty(shape, dtype=k_cache.dtype, device=k_cache.device)
    v_paged = torch.empty(shape, dtype=v_cache.dtype, device=v_cache.device)
    # BLOCK_B=8 / num_warps=8 / num_stages=2 tuned best; the copy is HBM-bandwidth-bound
    # (~6 TB/s achievable here) so it already runs at the pure-contiguous-copy ceiling.
    BLOCK_B = 8
    grid = (num_paged, triton.cdiv(block_size_k, BLOCK_B))
    _kv_flat_to_paged_kernel[grid](
        k_cache,
        v_cache,
        k_paged,
        v_paged,
        block_size_k,
        num_kv_heads,
        head_dim,
        BLOCK_B=BLOCK_B,
        DIM=head_dim,
        NKV=num_kv_heads,
        num_warps=8,
        num_stages=2,
    )
    return k_paged, v_paged


def build_index_score_plan(
    cu_seqlens, seq_lens, prefix_lens, num_idx_heads, idx_kv_heads, block_size_k
):
    """Build the fmha_sm100 OnlyScore plan for the index QK score.

    The plan depends ONLY on the per-forward segment geometry (qo/kv segment
    lengths, per-segment offsets, head/page config) -- identical across every
    sparse layer in one model forward. Build it once per forward and pass it to
    ``flash_prefill_topk_to_block_tables(..., index_score_plan=plan)`` so only the
    first sparse layer pays the build cost (no module-global cache needed)."""
    from fmha_sm100.api import _fmha_sm100_plan

    qo_seg = (cu_seqlens[1:] - cu_seqlens[:-1]).cpu().to(torch.int32)
    kv_seg = seq_lens.cpu().to(torch.int32)
    qo_off = prefix_lens.cpu().to(
        torch.int32
    )  # qo_offset=prefix -> bottom-right causal
    return _fmha_sm100_plan(
        qo_seg,
        kv_seg,
        num_idx_heads,
        num_kv_heads=idx_kv_heads,
        qo_offset=qo_off,
        page_size=block_size_k,
        output_maxscore=True,
        causal=True,
        num_kv_splits=1,
    )


def _attach_direct_csr(plan, kv_seg_cpu, block_size_k, device):
    """Opt-1+: preallocate CSR/schedule/page_table buffers ONCE per forward (stored on
    the cached plan, reused across all sparse layers) so step3 can bypass the adapter.
    batch-general. Native CSR builder + schedule are reused as-is -> bit-identical."""
    from src.sm100.prepare_k2q_csr import SparseK2qCsrBuilderSm100
    from src.sm100.prepare_scheduler import SparseAttentionSchedule

    head_kv = int(plan["num_kv_heads"])
    total_q = int(plan["cu_seqlens_q"][-1].item())
    topk = int(plan["kv_block_num"])
    total_rows = int(plan["total_rows"])
    blk = int(plan["blk_kv"])
    pages_per_batch = [
        (int(s) + block_size_k - 1) // block_size_k for s in kv_seg_cpu.tolist()
    ]
    batch, max_pages = len(pages_per_batch), max(pages_per_batch)
    cap = int(plan["scheduler_metadata_capacity"])

    builder = SparseK2qCsrBuilderSm100()
    builder._ensure_loaded()
    q_ind = torch.empty((head_kv, total_q * topk), dtype=torch.int32, device=device)
    # 16-byte aligned page_table [batch, max_pages] (the cute kernel asserts %16 == 0)
    buf = torch.empty(batch * max_pages + 4, dtype=torch.int32, device=device)
    shift = ((-buf.data_ptr()) % 16) // 4
    page_table = buf[shift : shift + batch * max_pages].view(batch, max_pages)
    plan["_csr_direct"] = dict(
        builder=builder,
        row_ptr=torch.empty(
            (head_kv, total_rows + 1), dtype=torch.int32, device=device
        ),
        q_ind=q_ind,
        sched=SparseAttentionSchedule(
            enabled=True,
            scheduler_metadata=torch.empty((cap, 6), dtype=torch.int32, device=device),
            work_count=torch.empty((1,), dtype=torch.int32, device=device),
            qsplit_indices=torch.empty_like(q_ind),
            split_counts=torch.empty(
                (total_q, head_kv), dtype=torch.int32, device=device
            ),
            target_q_per_cta=int(plan["target_q_per_cta"]),
        ),
        page_table=page_table,
        pages_per_batch=pages_per_batch,
        max_kv_blocks=(max(int(plan["max_seqlen_k"]), blk) + blk - 1) // blk,
        target_q_per_cta=int(plan["target_q_per_cta"]),
    )


def build_sparse_attn_plan(
    cu_seqlens, seq_lens, prefix_lens, num_q_heads, num_kv_heads, block_size_k, topk
):
    """Build the fmha_sm100 sparse-attention (step3) plan.

    Like build_index_score_plan it depends only on the per-forward segment geometry
    (so build once per forward, reuse across sparse layers). ``topk`` becomes
    ``kv_block_num`` (must be in {4,8,16,32}). Consumed by sparse_fmha for the main
    GQA sparse attention over the top-k selected blocks."""
    from fmha_sm100.api import sparse_fmha_plan

    qo_seg = (cu_seqlens[1:] - cu_seqlens[:-1]).cpu().to(torch.int32)
    kv_seg = seq_lens.cpu().to(torch.int32)
    qo_off = prefix_lens.cpu().to(
        torch.int32
    )  # qo_offset=prefix -> bottom-right causal
    plan = sparse_fmha_plan(
        qo_seg,
        kv_seg,
        num_qo_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
        qo_offset=qo_off,
        page_size=block_size_k,
        output_maxscore=False,
        kv_block_num=topk,
        causal=True,
    )
    if _M3_MSA_FUSED_CSR:
        _attach_direct_csr(plan, kv_seg, block_size_k, cu_seqlens.device)
    return plan


def build_kv_page_indices(req_to_token, seq_lens, block_size_k):
    """Flat per-segment physical page table for fmha (page = slot // block_size_k).

    Depends only on req_to_token + seq_lens (per-forward constant, identical across
    sparse layers AND shared by the index-score and step3 fmha calls), so the caller
    builds it once per forward (stored on _cp_shared_meta) and threads it in. Built
    on the fly when not supplied."""
    nblocks = (seq_lens.to(torch.int64) + block_size_k - 1) // block_size_k  # [batch]
    page_starts = req_to_token[
        :, ::block_size_k
    ]  # [batch, max_pages] block-start slots
    cols = torch.arange(page_starts.shape[1], device=req_to_token.device)
    valid = cols[None, :] < nblocks[:, None]
    return (page_starts // block_size_k)[valid].to(torch.int32)


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
    # int64 base offset: score is [num_idx_heads, total_q, max_seqblock_k], so the
    # per-head slab (total_q * max_seqblock_k) times pid_h overflows int32 for long
    # context. Promote the n/head terms to int64 before scaling by the strides.
    s_ptrs = (
        s_ptr
        + (seq_start + pid_q * sample_interval).to(tl.int64) * stride_s_n
        + pid_h.to(tl.int64) * stride_s_h
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
    index_score_plan=None,
    kv_indices=None,
    emit_block_table: bool = True,
):
    """Returns (block_tables [total_q*NKV, topk] int32, seq_lens [total_q*NKV] int32).

    ``index_score_plan``: a prebuilt fmha_sm100 OnlyScore plan (from
    ``build_index_score_plan``). It depends only on the per-forward segment shape,
    so the caller builds it once per forward (stored on MSAAttention._cp_shared_meta)
    and passes it here, avoiding a rebuild per sparse layer. When None it is built
    on the fly (decode fast-path / non-CP callers)."""
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

    # Index QK score via fmha_sm100 OnlyScore (SM100/Blackwell) instead of the
    # Triton block-score kernel. fmha emits the same per-128-block max score that
    # the bitonic topk->block_table kernel below consumes, but is MMA-efficient on
    # the skinny index QK -> ~2.5x faster at 64k ctx. JIT-compiled+cached on first
    # use (needs a gcc>=11 host compiler at COMPILE time; the cached .so is reused
    # afterwards with no compiler dependency).
    from fmha_sm100.api import _fmha_sm100

    # paged MQA view of the idx-K cache: [num_total_pages, idx_kv_heads, page, d]
    num_total_pages = max_slots // block_size_k
    k_pages = idx_k_cache[: num_total_pages * block_size_k].view(
        num_total_pages, idx_kv_heads, block_size_k, qk_head_dim
    )
    # flat per-segment physical page table (page = slot // block_size_k). Built once
    # per forward by the caller and threaded in (shared with step3); else built here.
    if kv_indices is None:
        kv_indices = build_kv_page_indices(req_to_token, seq_lens, block_size_k)
    # The fmha plan depends only on the per-forward segment shape (identical across
    # every sparse layer). The caller builds it once per forward and passes it in;
    # build on the fly only when not supplied (decode fast-path / non-CP callers).
    plan = index_score_plan
    if plan is None:
        plan = build_index_score_plan(
            cu_seqlens, seq_lens, prefix_lens, num_heads, idx_kv_heads, block_size_k
        )
    _o, maxscore = _fmha_sm100(
        idx_q,
        k_pages,
        k_pages,
        plan,
        kv_indices=kv_indices,
        output_o=False,
        output_maxscore=True,
        sm_scale=sm_scale,
    )
    # [num_heads, max_block, total_q] -> [num_heads, total_q, max_seqblock_k] fp32
    score = maxscore.transpose(1, 2)[:, :, :max_seqblock_k].contiguous().float()

    # bt/sl (compacted trtllm-gen block table) are consumed ONLY by the trtllm-gen
    # step3 path; the fmha step3 path uses topk_idx + page_table and never reads them.
    # When the caller won't use them (emit_block_table=False) skip the alloc and the
    # kernel's block-table emission, passing 1-elem dummies just to satisfy the strides.
    if emit_block_table:
        bt = torch.zeros(
            total_q * num_kv_heads, topk, dtype=torch.int32, device=idx_q.device
        )
        sl = torch.zeros(total_q * num_kv_heads, dtype=torch.int32, device=idx_q.device)
    else:
        bt = torch.empty(1, 1, dtype=torch.int32, device=idx_q.device)
        sl = torch.empty(1, dtype=torch.int32, device=idx_q.device)
    # topk_idx (-1 padded, raw block ids for fmha step3) is allocated directly in the
    # [nkv, total_q, topk] q2k layout the native CSR builder / sparse_atten_func consume,
    # so the fmha step3 feeds it straight in with NO permute-copy. The kernel writes via
    # the (h, n, t) strides (passed in natural order for this layout).
    topk_idx = torch.full(
        (num_kv_heads, total_q, topk), -1, dtype=torch.int32, device=idx_q.device
    )
    grid2 = (max_seqblock_q, batch_size, num_heads)
    _topk_to_block_table_kernel[grid2](
        score,
        bt,
        sl,
        topk_idx,
        block_size_q,
        block_size_k,
        cu_seqlens,
        cu_seqblocks_q,
        prefix_lens,
        topk,
        init_blocks,
        local_blocks,
        num_pages,
        score.stride(0),
        score.stride(1),
        score.stride(2),
        bt.stride(0),
        bt.stride(1),
        topk_idx.stride(0),
        topk_idx.stride(1),
        topk_idx.stride(2),  # h, n, t
        NKV=num_kv_heads,
        MASK_INIT=False,
        MASK_LOCAL=False,
        EMIT_BLOCK_TABLE=emit_block_table,
        EMIT_TOPK_IDX=True,
    )
    return bt, sl, topk_idx


@torch.no_grad()
def flash_prefill_with_fmha(
    q: torch.Tensor,  # [total_q, num_q_heads, head_dim] bf16
    k_cache: torch.Tensor,  # [max_slots, num_kv_heads, head_dim] FLAT
    v_cache: torch.Tensor,  # [max_slots, num_kv_heads, head_dim] FLAT
    idx_q: torch.Tensor,  # [total_q, num_idx_heads, idx_head_dim]
    idx_k_cache: torch.Tensor,  # [max_slots, 1, idx_head_dim]
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
    sm_scale: float,
    index_score_plan=None,
    sparse_attn_plan=None,
    kv_indices=None,
) -> torch.Tensor:
    """Fast sparse prefill: mega topk (index score + bitonic) -> fmha_sm100 sparse attn.

    Step 1+2 emit ``topk_idx`` (raw top-k block ids); step 3 runs the main GQA sparse
    attention through fmha_sm100 (~1.6x faster than trtllm-gen at 64k, cos>=0.9999).
    No trtllm block table (bt/sl) is built -- the fmha path uses topk_idx + page_table.

    ``sparse_attn_plan`` (from ``build_sparse_attn_plan``) is required. When it carries
    the Opt-1+ ``_csr_direct`` buffers, step3 bypasses the fmha adapter (prealloc CSR/
    schedule + GPU page_table + direct ``sparse_atten_func``, bit-identical + determi-
    nistic); otherwise it uses the adapter ``sparse_fmha``.

    Decode uses ``flash_decode_with_trtllm_gen`` instead (trtllm-gen sparse-decode).
    Constraint: idx_group_size == 1 (num_idx_heads == num_kv_heads).
    """
    from fmha_sm100.api import sparse_fmha

    if sparse_attn_plan is None:
        raise ValueError("flash_prefill_with_fmha requires a sparse_attn_plan")

    total_q, num_q_heads, head_dim = q.shape
    max_slots, num_kv_heads, _ = k_cache.shape
    num_pages = (max_seqlen_k + block_size_k - 1) // block_size_k

    # Physical page table: build once here (or take the per-forward cached one) and
    # share between the index-score kernel and step3 -- avoids the double build.
    if kv_indices is None:
        kv_indices = build_kv_page_indices(req_to_token, seq_lens, block_size_k)

    # Step 1+2: fused QK score + bitonic topk -> topk_idx. emit_block_table=False: the
    # fmha step3 path uses topk_idx + page_table and never reads the trtllm block table.
    _bt, _sl, topk_idx = flash_prefill_topk_to_block_tables(
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
        num_pages=num_pages,
        init_blocks=init_blocks,
        local_blocks=local_blocks,
        index_score_plan=index_score_plan,
        kv_indices=kv_indices,
        emit_block_table=False,
    )

    # Step 3 (fmha): read the flat MSA scratch as paged [num_paged, nkv, page, dim] +
    # the raw top-k block ids. Paginate the FULL gather scratch (max_slots) so multi-
    # request page_table indices resolve; single request: max_slots // block == num_pages.
    # One fused Triton kernel does both K and V (coalesced, ~4x faster + 1 launch vs the
    # two aten permute+contiguous copies; bit-identical).
    num_paged = max_slots // block_size_k
    k_paged_f, v_paged_f = _kv_flat_to_paged(
        k_cache, v_cache, num_paged, block_size_k, num_kv_heads, head_dim
    )
    # Opt-1+ direct path: when the per-forward CSR buffers are attached, bypass the
    # adapter -- feed topk_idx (already [nkv,Q,topk] = q2k) straight to the native CSR
    # _run into prealloc row_ptr/q_indices, GPU page_table (no .tolist() sync), direct
    # sparse_atten_func reusing the native schedule. Same native kernel/schedule ->
    # bit-identical + deterministic. batch-general.
    csr = (
        sparse_attn_plan.get("_csr_direct")
        if isinstance(sparse_attn_plan, dict)
        else None
    )
    if csr is not None:
        from interface import sparse_atten_func

        p = sparse_attn_plan
        # topk_idx is already [nkv, total_q, topk] contiguous = the q2k layout the native
        # builder wants -> feed it straight in (no permute-copy, no q2k staging buffer).
        pt, off = csr["page_table"], 0
        for b, n in enumerate(csr["pages_per_batch"]):
            pt[b, :n] = kv_indices[off : off + n]
            off += n
        s = csr["sched"]
        csr["builder"]._run_with_schedule(
            topk_idx,
            p["cu_seqlens_q"],
            p["cu_seqlens_k"],
            csr["row_ptr"],
            csr["q_ind"],
            s.scheduler_metadata,
            s.work_count,
            s.qsplit_indices,
            s.split_counts,
            topk,
            block_size_k,
            p["total_rows"],
            csr["max_kv_blocks"],
            csr["target_q_per_cta"],
            s.work_capacity,
            p["max_seqlen_q"],
        )
        out_f = sparse_atten_func(
            q,
            k_paged_f,
            v_paged_f,
            csr["row_ptr"],
            csr["q_ind"],
            topk,
            cu_seqlens_q=p["cu_seqlens_q"],
            cu_seqlens_k=p["cu_seqlens_k"],
            max_seqlen_q=p["max_seqlen_q"],
            max_seqlen_k=p["max_seqlen_k"],
            blk_kv=block_size_k,
            causal=p["causal"],
            softmax_scale=sm_scale,
            return_softmax_lse=False,
            page_table=pt,
            seqused_k=p["seqused_k"],
            schedule=s,
            usable_SM_count=int(p.get("usable_SM_count", -1)),
        )
        if isinstance(out_f, tuple):
            out_f = out_f[0]
        return out_f.view(total_q, num_q_heads, head_dim)
    # Adapter fallback: sparse_fmha's kv_block_indexes wants [total_q, nkv, topk]; topk_idx
    # is now [nkv, total_q, topk], so transpose the view back (adapter then re-permutes +
    # makes it contiguous internally). kv_indices shared with the index-score kernel.
    out_f, _ = sparse_fmha(
        q,
        k_paged_f,
        v_paged_f,
        sparse_attn_plan,
        kv_indices=kv_indices,
        kv_block_indexes=topk_idx.permute(1, 0, 2),
        output_o=True,
        output_maxscore=False,
        sm_scale=sm_scale,
    )
    return out_f.view(total_q, num_q_heads, head_dim)


@torch.no_grad()
def flash_decode_with_trtllm_gen(
    q: torch.Tensor,  # [total_q, num_q_heads, head_dim] bf16
    k_cache: torch.Tensor,  # [max_slots, num_kv_heads, head_dim] FLAT
    v_cache: torch.Tensor,  # [max_slots, num_kv_heads, head_dim] FLAT
    idx_q: torch.Tensor,  # [total_q, num_idx_heads, idx_head_dim]
    idx_k_cache: torch.Tensor,  # [max_slots, 1, idx_head_dim]
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
    sm_scale: float,
    workspace: torch.Tensor,
    index_score_plan=None,
    kv_indices=None,
) -> torch.Tensor:
    """Fast sparse decode: mega topk (index score + bitonic) -> trtllm-gen sparse-decode.

    Step 1+2 emit bt/sl (compacted trtllm block table) + topk_idx; step 3 runs
    flashinfer's trtllm-gen sparse-decode kernel over the paged KV.

    trtllm-gen's kernel launches with grid_dim_x = total_q * num_kv_heads, capped at
    2**16 - 1 = 65535; when total_q exceeds ``65535 // num_kv_heads`` we slice the
    q_packed / block_tables / seq_lens rows and issue multiple calls over the shared
    paged KV (per-query independence -> no LSE merge). Verified bit-equivalent up to
    q=65536 in m3_test/test_trtllm_gen_q_limit.py. Constraint: idx_group_size == 1.
    """
    from flashinfer.decode import trtllm_batch_decode_with_kv_cache

    total_q, num_q_heads, head_dim = q.shape
    max_slots, num_kv_heads, _ = k_cache.shape
    gqa = num_q_heads // num_kv_heads
    num_pages = (max_seqlen_k + block_size_k - 1) // block_size_k

    if kv_indices is None:
        kv_indices = build_kv_page_indices(req_to_token, seq_lens, block_size_k)

    # Step 1+2: fused QK score + bitonic topk; emit bt/sl (compacted trtllm block table).
    bt, sl, _topk_idx = flash_prefill_topk_to_block_tables(
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
        num_pages=num_pages,
        init_blocks=init_blocks,
        local_blocks=local_blocks,
        index_score_plan=index_score_plan,
        kv_indices=kv_indices,
        emit_block_table=True,
    )

    # Permute flat MSA side cache to trtllm's paged layout:
    #   flat  [num_pages * block_size_k, num_kv_heads, head_dim]
    # -> paged [num_kv_heads * num_pages, 1, block_size_k, head_dim]
    # The trailing unsqueeze(1) is trtllm's blocks-per-token dim (= 1 here).
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

    # Chunk q_packed along axis 0 (= (token, kv_head) interleaved rows) so each
    # call's grid_dim_x stays <= 65535. block_tables and seq_lens share the same
    # row layout so they slice identically. KV cache + workspace are shared across
    # chunks; sparse-decode queries are mutually independent, no LSE merge needed.
    CUDA_GRID_MAX = 65535
    rows_per_chunk = (CUDA_GRID_MAX // num_kv_heads) * num_kv_heads
    nrows = q_packed.shape[0]

    if nrows <= rows_per_chunk:
        out = trtllm_batch_decode_with_kv_cache(
            query=q_packed,
            kv_cache=(k_paged, v_paged),
            workspace_buffer=workspace,
            block_tables=bt,
            seq_lens=sl,
            max_seq_len=max_seqlen_k,
            bmm1_scale=sm_scale,
            bmm2_scale=1.0,
            backend="trtllm-gen",
            out_dtype=torch.bfloat16,
        )
    else:
        out_chunks = []
        for start in range(0, nrows, rows_per_chunk):
            end = min(start + rows_per_chunk, nrows)
            out_chunks.append(
                trtllm_batch_decode_with_kv_cache(
                    query=q_packed[start:end],
                    kv_cache=(k_paged, v_paged),
                    workspace_buffer=workspace,
                    block_tables=bt[start:end],
                    seq_lens=sl[start:end],
                    max_seq_len=max_seqlen_k,
                    bmm1_scale=sm_scale,
                    bmm2_scale=1.0,
                    backend="trtllm-gen",
                    out_dtype=torch.bfloat16,
                )
            )
        out = torch.cat(out_chunks, dim=0)

    return out.view(total_q, num_kv_heads, gqa, head_dim).reshape(
        total_q, num_q_heads, head_dim
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
        idx_q,
        idx_k_cache,
        None,
        None,
        None,
        score,
        req_to_token,
        cu_seqlens,
        seq_lens,
        prefix_lens,
        slot_ids,
        max_slots,
        num_heads,
        gqa_group_size,
        qk_head_dim,
        v_head_dim,
        block_size_k,
        sm_scale,
        False,
        1,
        idx_q.stride(0),
        idx_q.stride(1),
        idx_q.stride(2),
        idx_k_cache.stride(0),
        idx_k_cache.stride(1),
        idx_k_cache.stride(2),
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        score.stride(0),
        score.stride(1),
        score.stride(2),
        req_to_token.stride(0),
        SCORE_TYPE=score_type,
        DISABLE_INDEX_VALUE=True,
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
        score,
        bt_dummy,
        sl_dummy,
        topk_idx,
        block_size_q,
        block_size_k,
        cu_seqlens,
        cu_seqblocks_q,
        prefix_lens,
        topk,
        init_blocks,
        local_blocks,
        0,
        score.stride(0),
        score.stride(1),
        score.stride(2),
        bt_dummy.stride(0),
        bt_dummy.stride(1),
        topk_idx.stride(0),
        topk_idx.stride(1),
        topk_idx.stride(2),
        NKV=num_heads,
        MASK_INIT=False,
        MASK_LOCAL=False,
        EMIT_BLOCK_TABLE=False,
        EMIT_TOPK_IDX=True,
    )
    return None, topk_idx
