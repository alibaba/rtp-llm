"""Fused Triton kernels for DSV4 FP8 decode metadata preparation.

Called unconditionally from ``update_decode_metadata_in_place_fp8``.
Requires CUDA tensors — no CPU/fallback path exists.

1. ``fused_update_decode_meta_pure`` — replaces the pure-``start_pos``
   arithmetic (~60 aten ops → 1 kernel). Supports compress ratio subsets
   {4,128}, {4}, {128}, or SWA-only via ``HAS_CMP_4``/``HAS_CMP_128`` constexpr.

2. ``fused_phase2b_pool_slot_mapping`` — replaces the 4×
   ``compute_kv_pool_slot_mapping`` calls for paged pool write slots.
   Handles any subset of {SWA, CSA, INDEXER, HCA} pools via constexpr flags.

Correctness verified by ``test/test_fused_decode_meta_comprehensive.py``
(1342 cases: bs=1-256, q_len=1-6, seqlen=1-1M, ratio=4/128 boundary
starts, and heterogeneous per-request start_pos).
"""

from __future__ import annotations

from typing import Any

import torch
import triton
import triton.language as tl


@triton.jit
def _fused_prepare_meta_kernel(
    # inputs
    start_pos_in_ptr,  # [bs] i32
    # persistent outputs (max-sized; we write [:bs] prefix only)
    start_pos_out_ptr,  # [B_max] i32
    cmp_len_4_ptr,  # [B_max] i32 — None when HAS_CMP_4=False
    cmp_len_128_ptr,  # [B_max] i32 — None when HAS_CMP_128=False
    cmp_len_pt_4_ptr,  # [B_max, q_len] i32
    cmp_len_pt_128_ptr,  # [B_max, q_len] i32
    slot_swa_ptr,  # [B_max * q_len] i32
    slot_cmp_4_ptr,  # [B_max * q_len] i32
    slot_cmp_128_ptr,  # [B_max * q_len] i32
    topk_window_ptr,  # [B_max, q_len, WINDOW] i32
    swa_abs_idx_ptr,  # [B_max, q_len, WINDOW] i32
    swa_plus_csa_indexer_indices_ptr,  # [B_max, q_len, WINDOW + INDEX_TOPK] i32
    swa_plus_hca_dense_indices_ptr,  # [B_max, q_len, WINDOW + HCA_DENSE_WIDTH] i32
    # runtime sizes
    bs,
    # constants
    Q_LEN: tl.constexpr,
    WINDOW: tl.constexpr,
    INDEX_TOPK: tl.constexpr,
    HCA_DENSE_WIDTH: tl.constexpr,
    STRIDE_CMP_4: tl.constexpr,  # = max_seq_len // 4
    STRIDE_CMP_128: tl.constexpr,  # = max_seq_len // 128
    BLOCK_W: tl.constexpr,  # >= WINDOW, power of 2
    BLOCK_K: tl.constexpr,  # >= INDEX_TOPK, power of 2
    BLOCK_HCA: tl.constexpr,  # >= HCA_DENSE_WIDTH, power of 2
    HAS_CMP_4: tl.constexpr,  # False for SWA-only (MTP draft)
    HAS_CMP_128: tl.constexpr,  # False when no HCA layers
):
    r = tl.program_id(0)
    q = tl.program_id(1)
    if r >= bs:
        return

    sp = tl.load(start_pos_in_ptr + r)  # i32
    abs_pos = sp + q  # i32 scalar
    abs_pos_p1 = abs_pos + 1  # i32 scalar
    in_ring = abs_pos % WINDOW  # i32 scalar
    out_rq = r * Q_LEN + q

    # ---- Scalar outputs (write once per request; gate on q==0) ----
    if q == 0:
        tl.store(start_pos_out_ptr + r, sp)
        if HAS_CMP_4:
            tl.store(cmp_len_4_ptr + r, (sp + Q_LEN) // 4)
        if HAS_CMP_128:
            tl.store(cmp_len_128_ptr + r, (sp + Q_LEN) // 128)

    # ---- SWA slot mapping: r * WINDOW + (abs_pos % WINDOW) ----
    swa_slot = r * WINDOW + in_ring
    tl.store(slot_swa_ptr + out_rq, swa_slot)

    # ---- Window half (shared across topk_total_by_ratio writes below) ----
    k = tl.arange(0, BLOCK_W)
    mask_w = k < WINDOW
    k_i32 = k.to(tl.int32)

    # Full ring: (in_ring + 1 + k) % WINDOW
    ring_full = (in_ring + 1 + k_i32) % WINDOW
    # Partial (early decode): k if k <= abs_pos else -1
    partial = tl.where(k_i32 <= abs_pos, k_i32, -1)
    is_full = abs_pos >= (WINDOW - 1)
    wi = tl.where(is_full, ring_full, partial).to(tl.int32)

    base_w = r * Q_LEN * WINDOW + q * WINDOW
    tl.store(topk_window_ptr + base_w + k, wi, mask=mask_w)

    # swa_abs_idx[r, q, :] = left-aligned abs positions in
    # [max(0, abs_pos - WINDOW + 1) .. abs_pos], pad -1.
    win_start = tl.maximum(abs_pos - WINDOW + 1, 0)
    candidate = win_start + k_i32
    valid_pos = candidate <= abs_pos
    swa_abs = tl.where(valid_pos, candidate, -1).to(tl.int32)
    tl.store(swa_abs_idx_ptr + base_w + k, swa_abs, mask=mask_w)

    # ---- Per-ratio compressed work: slot mapping + per-token lens
    #      + topk_total (window half = wi, ratio-specific dense half).
    #      cmp_len_R = abs_pos_p1 // R is computed ONCE and reused for
    #      slot index (cmp_len_R - 1), per-token len, and HCA mask.
    if HAS_CMP_4:
        cmp_len_4 = abs_pos_p1 // 4
        on_b4 = (abs_pos_p1 % 4) == 0
        cmp_slot_4 = tl.where(on_b4, r * STRIDE_CMP_4 + cmp_len_4 - 1, -1)
        tl.store(slot_cmp_4_ptr + out_rq, cmp_slot_4)
        tl.store(cmp_len_pt_4_ptr + out_rq, cmp_len_4)

        WK_TOTAL_4 = WINDOW + INDEX_TOPK
        base_tt_4 = r * Q_LEN * WK_TOTAL_4 + q * WK_TOTAL_4
        tl.store(swa_plus_csa_indexer_indices_ptr + base_tt_4 + k, wi, mask=mask_w)
        # CSA dense half: all -1 (indexer fills per-call)
        k2 = tl.arange(0, BLOCK_K)
        mask_k = k2 < INDEX_TOPK
        tl.store(
            swa_plus_csa_indexer_indices_ptr + base_tt_4 + WINDOW + k2,
            tl.full((BLOCK_K,), -1, tl.int32),
            mask=mask_k,
        )

    if HAS_CMP_128:
        cmp_len_128 = abs_pos_p1 // 128
        on_b128 = (abs_pos_p1 % 128) == 0
        cmp_slot_128 = tl.where(on_b128, r * STRIDE_CMP_128 + cmp_len_128 - 1, -1)
        tl.store(slot_cmp_128_ptr + out_rq, cmp_slot_128)
        tl.store(cmp_len_pt_128_ptr + out_rq, cmp_len_128)

        WK_TOTAL_128 = WINDOW + HCA_DENSE_WIDTH
        base_tt_128 = r * Q_LEN * WK_TOTAL_128 + q * WK_TOTAL_128
        tl.store(swa_plus_hca_dense_indices_ptr + base_tt_128 + k, wi, mask=mask_w)
        # HCA dense half is causal per query token: a speculative
        # multi-token step must not let earlier q positions see the
        # HCA block completed by a later q in the same step.
        # When max_seq_len < 128 the dense width is 0 and the destination
        # tensor has shape ``[..., WINDOW]`` — skip the dense write
        # entirely to avoid OOB on ``base + WINDOW + 0``.
        if HCA_DENSE_WIDTH > 0:
            h2 = tl.arange(0, BLOCK_HCA)
            mask_h = h2 < HCA_DENSE_WIDTH
            h2_i32 = h2.to(tl.int32)
            valid_h = h2_i32 < cmp_len_128
            dense_h = tl.where(valid_h, h2_i32, -1).to(tl.int32)
            tl.store(
                swa_plus_hca_dense_indices_ptr + base_tt_128 + WINDOW + h2,
                dense_h,
                mask=mask_h,
            )


def _next_pow2(n: int) -> int:
    p = 1
    while p < n:
        p <<= 1
    return p


def fused_update_decode_meta_pure(
    meta: Any,
    start_pos: torch.Tensor,
    max_seq_len: int,
) -> None:
    """Fuse the pure-start_pos metadata preparation into a single Triton kernel.

    Requirements:
      - All tensors on CUDA
      - Supported ratio subsets: {4,128}, {4}, {128}, or {} (SWA-only)

    Writes start_pos, slot_mapping_swa, slot_mapping_compressed,
    compressed_lens, compressed_lens_per_token, topk_window_idxs,
    topk_total_by_ratio, swa_abs_idx.
    Callers handle pool_write_slot_mappings, swa_global_slots,
    hca_cmp_global_slots, and topk_buffer_compressed reset separately.
    """
    assert start_pos.is_cuda, (
        "fused_update_decode_meta_pure requires CUDA tensors, "
        f"got start_pos on {start_pos.device}"
    )
    for r in meta.slot_mapping_compressed:
        assert r in (4, 128), (
            f"fused_update_decode_meta_pure only supports compress ratios "
            f"{{4, 128}} or SWA-only, got ratio={r}"
        )

    bs = int(start_pos.shape[0])
    q_len = meta.q_len_per_req
    window_size = meta.window_size
    index_topk = meta.topk_buffer_compressed.shape[-1]

    has_4 = 4 in meta.slot_mapping_compressed
    has_128 = 128 in meta.topk_total_by_ratio
    hca_dense_width = (
        meta.topk_total_by_ratio[128].shape[-1] - window_size if has_128 else 0
    )
    # When a ratio is absent (SWA-only MTP draft: compress_ratios=[0]),
    # the corresponding kernel pointer is None — Triton lowers it to a
    # null pointer, and the HAS_CMP_* constexpr gates eliminate any
    # load/store referencing it at compile time.
    slot_cmp_4 = meta.slot_mapping_compressed[4] if has_4 else None
    cmp_len_4 = meta.compressed_lens[4] if has_4 else None
    cmp_len_pt_4 = meta.compressed_lens_per_token[4] if has_4 else None
    topk_total_4 = meta.topk_total_by_ratio[4] if has_4 else None
    stride_cmp_4 = meta.compressed_buffer_t_dim_per_ratio[4] if has_4 else 1

    slot_cmp_128 = meta.slot_mapping_compressed[128] if has_128 else None
    cmp_len_128 = meta.compressed_lens[128] if has_128 else None
    cmp_len_pt_128 = meta.compressed_lens_per_token[128] if has_128 else None
    topk_total_128 = meta.topk_total_by_ratio[128] if has_128 else None
    stride_cmp_128 = meta.compressed_buffer_t_dim_per_ratio[128] if has_128 else 1

    BLOCK_W = _next_pow2(window_size)
    BLOCK_K = _next_pow2(max(index_topk, 1))
    # Pass the real ``hca_dense_width`` as the HCA_DENSE_WIDTH constexpr
    # (may be 0); the kernel's ``if HCA_DENSE_WIDTH > 0`` gate elides the
    # whole dense write at compile time, so BLOCK_HCA (=_next_pow2(0)==1)
    # is never actually consumed by ``tl.arange`` on the width=0 path.
    BLOCK_HCA = _next_pow2(hca_dense_width)

    grid = (bs, q_len)
    _fused_prepare_meta_kernel[grid](
        start_pos,
        meta.start_pos,
        cmp_len_4,
        cmp_len_128,
        cmp_len_pt_4,
        cmp_len_pt_128,
        meta.slot_mapping_swa,
        slot_cmp_4,
        slot_cmp_128,
        meta.topk_window_idxs,
        meta.swa_abs_idx,
        topk_total_4,
        topk_total_128,
        bs,
        Q_LEN=q_len,
        WINDOW=window_size,
        INDEX_TOPK=max(index_topk, 1),
        HCA_DENSE_WIDTH=hca_dense_width,
        STRIDE_CMP_4=stride_cmp_4,
        STRIDE_CMP_128=stride_cmp_128,
        BLOCK_W=BLOCK_W,
        BLOCK_K=BLOCK_K,
        BLOCK_HCA=BLOCK_HCA,
        HAS_CMP_4=has_4,
        HAS_CMP_128=has_128,
    )


# ---------------------------------------------------------------------------
# Phase 2b: fused pool_write_slot_mapping.
# ---------------------------------------------------------------------------
# Before fusion, each prepare call invoked ``compute_kv_pool_slot_mapping``
# four times (once per pool: SWA / CSA / INDEXER / HCA).  Each call expanded
# to ~15 eager aten ops (arange/expand/to/where/floor_div/mul/sub/clamp_/
# index/...), plus per-ratio ``on_boundary`` / ``cmp_idx`` prep in the
# caller — ~60 small ops total for ~1.4ms per prepare at bs=128/64K.
#
# This kernel replaces all 4 calls with one grid of (bs, q_len) programs,
# each of which:
#   1. loads start_pos[r], derives abs_pos and abs_pos+1
#   2. for each pool, computes the per-pool abs_pos (token pos for SWA;
#      compressed index with -1 skip for CSA/INDEXER/HCA), gathers
#      block_id from the pool's block_table, and writes the int64 slot
#      (``block_id * E + in_block``, or ``-1`` sentinel when skipped)
#
# Ratio constants are hard-coded to V4-Flash values (CSA/INDEXER=4,
# HCA=128). Pool existence is controlled by HAS_SWA/HAS_CSA/HAS_IDX/HAS_HCA.
# ---------------------------------------------------------------------------


@triton.jit
def _fused_phase2b_pool_slot_mapping_kernel(
    # start_pos
    start_pos_ptr,  # [bs] i32
    # per-pool block tables (i32, [B_max, BT_STRIDE_*])
    bt_swa_ptr,
    bt_csa_ptr,
    bt_idx_ptr,
    bt_hca_ptr,
    # per-pool output slot buffers (i64, [B_max * q_len])
    slot_swa_ptr,
    slot_csa_ptr,
    slot_idx_ptr,
    slot_hca_ptr,
    # runtime
    bs,
    # compile-time constants
    Q_LEN: tl.constexpr,
    # per-pool entries_per_block (E) + block_table row stride (max_blocks_per_req)
    SWA_E: tl.constexpr,
    SWA_TOKENS_PER_BLOCK: tl.constexpr,
    SWA_BT_STRIDE: tl.constexpr,
    CSA_E: tl.constexpr,
    CSA_TOKENS_PER_BLOCK: tl.constexpr,
    CSA_BT_STRIDE: tl.constexpr,
    IDX_E: tl.constexpr,
    IDX_TOKENS_PER_BLOCK: tl.constexpr,
    IDX_BT_STRIDE: tl.constexpr,
    HCA_E: tl.constexpr,
    HCA_TOKENS_PER_BLOCK: tl.constexpr,
    HCA_BT_STRIDE: tl.constexpr,
    # pool existence flags — skip store when pool absent
    HAS_SWA: tl.constexpr,
    HAS_CSA: tl.constexpr,
    HAS_IDX: tl.constexpr,
    HAS_HCA: tl.constexpr,
):
    r = tl.program_id(0)
    q = tl.program_id(1)
    if r >= bs:
        return

    sp = tl.load(start_pos_ptr + r)  # i32 scalar
    abs_pos = sp + q  # token abs pos
    abs_pos_p1 = abs_pos + 1  # used by compressed pools
    out_idx = r * Q_LEN + q

    # ---------- SWA: ratio=1, every token writes ----------
    if HAS_SWA:
        bis_swa_raw = abs_pos // SWA_TOKENS_PER_BLOCK
        in_blk_swa = abs_pos % SWA_E
        bis_swa = tl.maximum(tl.minimum(bis_swa_raw, SWA_BT_STRIDE - 1), 0)
        bid_swa = tl.load(bt_swa_ptr + r * SWA_BT_STRIDE + bis_swa).to(tl.int64)
        slot_swa = tl.where(bid_swa <= 0, -1, bid_swa * SWA_E + in_blk_swa)
        tl.store(slot_swa_ptr + out_idx, slot_swa)

    # ---------- CSA: ratio=4, boundary tokens only ----------
    if HAS_CSA or HAS_IDX:
        on_b4 = (abs_pos_p1 % 4) == 0
        cmp_idx_4 = abs_pos_p1 // 4 - 1
        skip_4 = (on_b4 == 0) | (cmp_idx_4 < 0)
        safe_4 = tl.where(skip_4, 0, cmp_idx_4)

    if HAS_CSA:
        bis_csa_raw = safe_4 // CSA_TOKENS_PER_BLOCK
        in_blk_csa = safe_4 % CSA_E
        bis_csa = tl.maximum(tl.minimum(bis_csa_raw, CSA_BT_STRIDE - 1), 0)
        bid_csa = tl.load(bt_csa_ptr + r * CSA_BT_STRIDE + bis_csa).to(tl.int64)
        skip_csa = skip_4 | (bid_csa <= 0)
        slot_csa = tl.where(skip_csa, -1, bid_csa * CSA_E + in_blk_csa)
        tl.store(slot_csa_ptr + out_idx, slot_csa)

    # ---------- INDEXER: ratio=4, shares boundary with CSA ----------
    if HAS_IDX:
        bis_idx_raw = safe_4 // IDX_TOKENS_PER_BLOCK
        in_blk_idx = safe_4 % IDX_E
        bis_idx = tl.maximum(tl.minimum(bis_idx_raw, IDX_BT_STRIDE - 1), 0)
        bid_idx = tl.load(bt_idx_ptr + r * IDX_BT_STRIDE + bis_idx).to(tl.int64)
        skip_idx = skip_4 | (bid_idx <= 0)
        slot_idx = tl.where(skip_idx, -1, bid_idx * IDX_E + in_blk_idx)
        tl.store(slot_idx_ptr + out_idx, slot_idx)

    # ---------- HCA: ratio=128, boundary tokens only ----------
    if HAS_HCA:
        on_b128 = (abs_pos_p1 % 128) == 0
        cmp_idx_128 = abs_pos_p1 // 128 - 1
        skip_128 = (on_b128 == 0) | (cmp_idx_128 < 0)
        safe_128 = tl.where(skip_128, 0, cmp_idx_128)
        bis_hca_raw = safe_128 // HCA_TOKENS_PER_BLOCK
        in_blk_hca = safe_128 % HCA_E
        bis_hca = tl.maximum(tl.minimum(bis_hca_raw, HCA_BT_STRIDE - 1), 0)
        bid_hca = tl.load(bt_hca_ptr + r * HCA_BT_STRIDE + bis_hca).to(tl.int64)
        skip_hca = skip_128 | (bid_hca <= 0)
        slot_hca = tl.where(skip_hca, -1, bid_hca * HCA_E + in_blk_hca)
        tl.store(slot_hca_ptr + out_idx, slot_hca)


def fused_phase2b_pool_slot_mapping(
    meta: Any,
    start_pos: torch.Tensor,
    bs: int,
    paged_pool_entries_per_block: Any,
    paged_pool_tokens_per_block: Any,
) -> None:
    """Fuse paged pool write slot mapping for all pools into one Triton kernel.

    Requirements:
      - All tensors on CUDA
      - Pool subset from {SWA, CSA, INDEXER, HCA} (any combination)

    Writes pool_write_slot_mappings[:bs*q_len] for present pools.
    Absent pools are compile-time eliminated via constexpr flags.
    """
    assert start_pos.is_cuda, (
        "fused_phase2b_pool_slot_mapping requires CUDA tensors, "
        f"got start_pos on {start_pos.device}"
    )

    from rtp_llm.models_py.modules.dsv4.attn_type import (
        CSA_KV,
        HCA_KV,
        INDEXER_KV,
        SWA_KV,
    )

    q_len = meta.q_len_per_req

    has_swa = (
        SWA_KV in meta.pool_block_tables and SWA_KV in meta.pool_write_slot_mappings
    )
    has_csa = (
        CSA_KV in meta.pool_block_tables and CSA_KV in meta.pool_write_slot_mappings
    )
    has_idx = (
        INDEXER_KV in meta.pool_block_tables
        and INDEXER_KV in meta.pool_write_slot_mappings
    )
    has_hca = (
        HCA_KV in meta.pool_block_tables and HCA_KV in meta.pool_write_slot_mappings
    )

    if not (has_swa or has_csa or has_idx or has_hca):
        return

    # For absent pools, pass the SWA buffer as a dummy (kernel won't touch it)
    bt_swa = meta.pool_block_tables.get(
        SWA_KV, meta.pool_block_tables.get(next(iter(meta.pool_block_tables)))
    )
    bt_csa = meta.pool_block_tables.get(CSA_KV, bt_swa)
    bt_idx = meta.pool_block_tables.get(INDEXER_KV, bt_swa)
    bt_hca = meta.pool_block_tables.get(HCA_KV, bt_swa)

    slot_swa = meta.pool_write_slot_mappings.get(
        SWA_KV,
        meta.pool_write_slot_mappings.get(next(iter(meta.pool_write_slot_mappings))),
    )
    slot_csa = meta.pool_write_slot_mappings.get(CSA_KV, slot_swa)
    slot_idx = meta.pool_write_slot_mappings.get(INDEXER_KV, slot_swa)
    slot_hca = meta.pool_write_slot_mappings.get(HCA_KV, slot_swa)

    swa_e = int(paged_pool_entries_per_block[SWA_KV]) if has_swa else 1
    csa_e = int(paged_pool_entries_per_block[CSA_KV]) if has_csa else 1
    idx_e = int(paged_pool_entries_per_block[INDEXER_KV]) if has_idx else 1
    hca_e = int(paged_pool_entries_per_block[HCA_KV]) if has_hca else 1

    def _raw_tokens(attn_type: int) -> int:
        return int(paged_pool_tokens_per_block[attn_type])

    def _compressed_tokens(raw_tokens_per_block: int, ratio: int) -> int:
        if raw_tokens_per_block % ratio != 0:
            raise ValueError(
                "compressed pool tokens_per_block must be divisible by "
                f"ratio, got tokens={raw_tokens_per_block}, ratio={ratio}"
            )
        return raw_tokens_per_block // ratio

    swa_tokens = _raw_tokens(SWA_KV) if has_swa else 1
    csa_tokens = _compressed_tokens(_raw_tokens(CSA_KV), 4) if has_csa else 1
    idx_tokens = _compressed_tokens(_raw_tokens(INDEXER_KV), 4) if has_idx else 1
    hca_tokens = _compressed_tokens(_raw_tokens(HCA_KV), 128) if has_hca else 1

    grid = (bs, q_len)
    _fused_phase2b_pool_slot_mapping_kernel[grid](
        start_pos,
        bt_swa,
        bt_csa,
        bt_idx,
        bt_hca,
        slot_swa,
        slot_csa,
        slot_idx,
        slot_hca,
        bs,
        Q_LEN=q_len,
        SWA_E=swa_e,
        SWA_TOKENS_PER_BLOCK=swa_tokens,
        SWA_BT_STRIDE=bt_swa.shape[1],
        CSA_E=csa_e,
        CSA_TOKENS_PER_BLOCK=csa_tokens,
        CSA_BT_STRIDE=bt_csa.shape[1],
        IDX_E=idx_e,
        IDX_TOKENS_PER_BLOCK=idx_tokens,
        IDX_BT_STRIDE=bt_idx.shape[1],
        HCA_E=hca_e,
        HCA_TOKENS_PER_BLOCK=hca_tokens,
        HCA_BT_STRIDE=bt_hca.shape[1],
        HAS_SWA=has_swa,
        HAS_CSA=has_csa,
        HAS_IDX=has_idx,
        HAS_HCA=has_hca,
    )
