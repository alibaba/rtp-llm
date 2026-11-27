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
    # optional fused metadata outputs (elided at compile time when HAS_*=False)
    position_ids_ptr,  # [B_max * q_len] i32
    position_ids_long_ptr,  # [B_max * q_len] i64
    decode_seq_start_ptr,  # [B_max] i64
    topk_buffer_compressed_ptr,  # [B_max, q_len, INDEX_TOPK] i32
    swa_topk_len_ptr,  # [B_max] i32
    cmp_topk_len_4_ptr,  # [B_max] i32
    cmp_topk_len_128_ptr,  # [B_max] i32
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
    HAS_POS_IDS: tl.constexpr,
    HAS_POS_IDS_LONG: tl.constexpr,
    HAS_DECODE_SEQ_START: tl.constexpr,
    HAS_TOPK_BUF: tl.constexpr,
    HAS_SWA_TOPK_LEN: tl.constexpr,
    HAS_CMP_TOPK_LEN_4: tl.constexpr,
    HAS_CMP_TOPK_LEN_128: tl.constexpr,
    PAD_POS_IDS_LONG_16: tl.constexpr,
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

    if HAS_POS_IDS:
        tl.store(position_ids_ptr + out_rq, abs_pos)
    if HAS_POS_IDS_LONG:
        tl.store(position_ids_long_ptr + out_rq, abs_pos.to(tl.int64))
        if PAD_POS_IDS_LONG_16:
            if (r == bs - 1) and (q == Q_LEN - 1):
                t_act = bs * Q_LEN
                k16 = tl.arange(0, 16)
                tl.store(
                    position_ids_long_ptr + t_act + k16,
                    0,
                    mask=(t_act + k16) < 16,
                )

    # ---- Scalar outputs (write once per request; gate on q==0) ----
    if q == 0:
        tl.store(start_pos_out_ptr + r, sp)
        if HAS_DECODE_SEQ_START:
            tl.store(decode_seq_start_ptr + r, sp.to(tl.int64))
        if HAS_SWA_TOPK_LEN:
            swa_tl = tl.minimum(tl.maximum(sp + Q_LEN, 0), WINDOW)
            tl.store(swa_topk_len_ptr + r, swa_tl)
        if HAS_CMP_4:
            len_4 = (sp + Q_LEN) // 4
            tl.store(cmp_len_4_ptr + r, len_4)
            if HAS_CMP_TOPK_LEN_4:
                tl.store(cmp_topk_len_4_ptr + r, tl.minimum(len_4, INDEX_TOPK))
        if HAS_CMP_128:
            len_128 = (sp + Q_LEN) // 128
            tl.store(cmp_len_128_ptr + r, len_128)
            if HAS_CMP_TOPK_LEN_128:
                tl.store(cmp_topk_len_128_ptr + r, tl.minimum(len_128, HCA_DENSE_WIDTH))

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

    if HAS_CMP_4 or HAS_TOPK_BUF:
        k2 = tl.arange(0, BLOCK_K)
        mask_k = k2 < INDEX_TOPK
        neg_k = tl.full((BLOCK_K,), -1, tl.int32)

    if HAS_TOPK_BUF:
        base_tb = r * Q_LEN * INDEX_TOPK + q * INDEX_TOPK
        tl.store(topk_buffer_compressed_ptr + base_tb + k2, neg_k, mask=mask_k)

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
        tl.store(
            swa_plus_csa_indexer_indices_ptr + base_tt_4 + WINDOW + k2,
            neg_k,
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
    *,
    update_extra_fields: bool = False,
    update_topk_lengths: bool = False,
) -> None:
    """Fuse the pure-start_pos metadata preparation into a single Triton kernel.

    Requirements:
      - All tensors on CUDA
      - Supported ratio subsets: {4,128}, {4}, {128}, or {} (SWA-only)

    Writes start_pos, slot_mapping_swa, slot_mapping_compressed,
    compressed_lens, compressed_lens_per_token, topk_window_idxs,
    topk_total_by_ratio, swa_abs_idx, and optionally position_ids,
    position_ids_long, decode_seq_start_per_req, topk_buffer_compressed,
    and per-request effective topk_length buffers.
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
    topk_buf = getattr(meta, "topk_buffer_compressed", None)
    index_topk = int(topk_buf.shape[-1]) if topk_buf is not None else 0

    has_4 = 4 in meta.slot_mapping_compressed
    has_128 = 128 in meta.topk_total_by_ratio
    hca_dense_width = (
        meta.topk_total_by_ratio[128].shape[-1] - window_size if has_128 else 0
    )
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

    pos_ids = getattr(meta, "position_ids", None) if update_extra_fields else None
    pos_ids_long = (
        getattr(meta, "position_ids_long", None) if update_extra_fields else None
    )
    decode_seq_start = (
        getattr(meta, "decode_seq_start_per_req", None) if update_extra_fields else None
    )
    topk_buf_out = topk_buf if (update_extra_fields and index_topk > 0) else None

    swa_topk_len = (
        getattr(meta, "swa_topk_length", None) if update_topk_lengths else None
    )
    cmp_topk_map = (
        getattr(meta, "compressed_topk_length_by_ratio", None)
        if update_topk_lengths
        else None
    )
    cmp_topk_4 = cmp_topk_map.get(4) if (cmp_topk_map and has_4) else None
    cmp_topk_128 = cmp_topk_map.get(128) if (cmp_topk_map and has_128) else None

    BLOCK_W = _next_pow2(window_size)
    BLOCK_K = _next_pow2(max(index_topk, 1))
    BLOCK_HCA = _next_pow2(hca_dense_width)

    pos16_backing = getattr(meta, "hca_front_position_ids_16", None)
    pad_pos_ids_long_16 = bool(
        pos_ids_long is not None
        and pos16_backing is not None
        and pos16_backing.data_ptr() == pos_ids_long.data_ptr()
        and int(pos16_backing.numel()) >= 16
        and bs * q_len < 16
    )

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
        pos_ids,
        pos_ids_long,
        decode_seq_start,
        topk_buf_out,
        swa_topk_len,
        cmp_topk_4,
        cmp_topk_128,
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
        HAS_POS_IDS=pos_ids is not None,
        HAS_POS_IDS_LONG=pos_ids_long is not None,
        HAS_DECODE_SEQ_START=decode_seq_start is not None,
        HAS_TOPK_BUF=topk_buf_out is not None,
        HAS_SWA_TOPK_LEN=swa_topk_len is not None,
        HAS_CMP_TOPK_LEN_4=cmp_topk_4 is not None,
        HAS_CMP_TOPK_LEN_128=cmp_topk_128 is not None,
        PAD_POS_IDS_LONG_16=pad_pos_ids_long_16,
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
    bt_csa_state_ptr,
    bt_idx_state_ptr,
    bt_hca_state_ptr,
    # per-pool output slot buffers (i64, [B_max * q_len])
    slot_swa_ptr,
    slot_csa_ptr,
    slot_idx_ptr,
    slot_hca_ptr,
    slot_csa_state_ptr,
    slot_idx_state_ptr,
    slot_hca_state_ptr,
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
    CSA_STATE_E: tl.constexpr,
    CSA_STATE_TOKENS_PER_BLOCK: tl.constexpr,
    CSA_STATE_BT_STRIDE: tl.constexpr,
    IDX_STATE_E: tl.constexpr,
    IDX_STATE_TOKENS_PER_BLOCK: tl.constexpr,
    IDX_STATE_BT_STRIDE: tl.constexpr,
    HCA_STATE_E: tl.constexpr,
    HCA_STATE_TOKENS_PER_BLOCK: tl.constexpr,
    HCA_STATE_BT_STRIDE: tl.constexpr,
    # pool existence flags — skip store when pool absent
    HAS_SWA: tl.constexpr,
    HAS_CSA: tl.constexpr,
    HAS_IDX: tl.constexpr,
    HAS_HCA: tl.constexpr,
    HAS_CSA_STATE: tl.constexpr,
    HAS_IDX_STATE: tl.constexpr,
    HAS_HCA_STATE: tl.constexpr,
    PAD_HCA_STATE_16: tl.constexpr,
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

    # ---------- Compressor state pools: cyclic per-token ring ----------
    if HAS_CSA_STATE or HAS_IDX_STATE or HAS_HCA_STATE:
        abs_pos_i64 = abs_pos.to(tl.int64)
        r_i64 = r.to(tl.int64)

    if HAS_CSA_STATE:
        bis_csa_st = (abs_pos_i64 // CSA_STATE_TOKENS_PER_BLOCK) % CSA_STATE_BT_STRIDE
        in_blk_csa_st = abs_pos_i64 % CSA_STATE_E
        bid_csa_st = tl.load(
            bt_csa_state_ptr + r_i64 * CSA_STATE_BT_STRIDE + bis_csa_st
        ).to(tl.int64)
        slot_csa_st = tl.where(
            bid_csa_st <= 0, -1, bid_csa_st * CSA_STATE_E + in_blk_csa_st
        )
        tl.store(slot_csa_state_ptr + out_idx, slot_csa_st)

    if HAS_IDX_STATE:
        bis_idx_st = (abs_pos_i64 // IDX_STATE_TOKENS_PER_BLOCK) % IDX_STATE_BT_STRIDE
        in_blk_idx_st = abs_pos_i64 % IDX_STATE_E
        bid_idx_st = tl.load(
            bt_idx_state_ptr + r_i64 * IDX_STATE_BT_STRIDE + bis_idx_st
        ).to(tl.int64)
        slot_idx_st = tl.where(
            bid_idx_st <= 0, -1, bid_idx_st * IDX_STATE_E + in_blk_idx_st
        )
        tl.store(slot_idx_state_ptr + out_idx, slot_idx_st)

    if HAS_HCA_STATE:
        bis_hca_st = (abs_pos_i64 // HCA_STATE_TOKENS_PER_BLOCK) % HCA_STATE_BT_STRIDE
        in_blk_idx_hca = abs_pos_i64 % HCA_STATE_E
        bid_hca_st = tl.load(
            bt_hca_state_ptr + r_i64 * HCA_STATE_BT_STRIDE + bis_hca_st
        ).to(tl.int64)
        slot_hca_st = tl.where(
            bid_hca_st <= 0, -1, bid_hca_st * HCA_STATE_E + in_blk_idx_hca
        )
        tl.store(slot_hca_state_ptr + out_idx, slot_hca_st)
        if PAD_HCA_STATE_16:
            if (r == bs - 1) and (q == Q_LEN - 1):
                t_act = bs * Q_LEN
                k16 = tl.arange(0, 16)
                tl.store(
                    slot_hca_state_ptr + t_act + k16,
                    -1,
                    mask=(t_act + k16) < 16,
                )


def fused_phase2b_pool_slot_mapping(
    meta: Any,
    start_pos: torch.Tensor,
    bs: int,
    paged_pool_entries_per_block: Any,
    paged_pool_tokens_per_block: Any,
    *,
    update_state_slots: bool = False,
) -> None:
    """Fuse paged pool write slot mapping for all pools into one Triton kernel.

    Requirements:
      - All tensors on CUDA
      - Pool subset from {SWA, CSA, INDEXER, HCA} and optionally compressor
        state pools {CSA_STATE, INDEXER_STATE, HCA_STATE}

    Writes pool_write_slot_mappings[:bs*q_len] (and optionally
    compressor_state_slot_mappings[:bs*q_len]) for present pools.
    Absent pools are compile-time eliminated via constexpr flags.
    """
    assert start_pos.is_cuda, (
        "fused_phase2b_pool_slot_mapping requires CUDA tensors, "
        f"got start_pos on {start_pos.device}"
    )

    from rtp_llm.models_py.modules.dsv4.kv_cache_utils import (
        CSA_KV,
        CSA_STATE,
        HCA_KV,
        HCA_STATE,
        INDEXER_KV,
        INDEXER_STATE,
        SWA_KV,
    )

    q_len = meta.q_len_per_req
    state_slots_map = getattr(meta, "compressor_state_slot_mappings", {}) or {}

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
    has_csa_state = bool(
        update_state_slots
        and CSA_STATE in meta.pool_block_tables
        and CSA_STATE in state_slots_map
    )
    has_idx_state = bool(
        update_state_slots
        and INDEXER_STATE in meta.pool_block_tables
        and INDEXER_STATE in state_slots_map
    )
    has_hca_state = bool(
        update_state_slots
        and HCA_STATE in meta.pool_block_tables
        and HCA_STATE in state_slots_map
    )

    if not (
        has_swa
        or has_csa
        or has_idx
        or has_hca
        or has_csa_state
        or has_idx_state
        or has_hca_state
    ):
        return

    # For absent pools, pass a valid buffer as a dummy (kernel won't touch it)
    fallback_bt = meta.pool_block_tables.get(
        SWA_KV, meta.pool_block_tables.get(next(iter(meta.pool_block_tables)))
    )
    bt_swa = meta.pool_block_tables.get(SWA_KV, fallback_bt)
    bt_csa = meta.pool_block_tables.get(CSA_KV, fallback_bt)
    bt_idx = meta.pool_block_tables.get(INDEXER_KV, fallback_bt)
    bt_hca = meta.pool_block_tables.get(HCA_KV, fallback_bt)
    bt_csa_state = meta.pool_block_tables.get(CSA_STATE, fallback_bt)
    bt_idx_state = meta.pool_block_tables.get(INDEXER_STATE, fallback_bt)
    bt_hca_state = meta.pool_block_tables.get(HCA_STATE, fallback_bt)

    if meta.pool_write_slot_mappings:
        fallback_slot = meta.pool_write_slot_mappings.get(
            SWA_KV,
            meta.pool_write_slot_mappings.get(
                next(iter(meta.pool_write_slot_mappings))
            ),
        )
    else:
        fallback_slot = state_slots_map[next(iter(state_slots_map))]
    slot_swa = meta.pool_write_slot_mappings.get(SWA_KV, fallback_slot)
    slot_csa = meta.pool_write_slot_mappings.get(CSA_KV, fallback_slot)
    slot_idx = meta.pool_write_slot_mappings.get(INDEXER_KV, fallback_slot)
    slot_hca = meta.pool_write_slot_mappings.get(HCA_KV, fallback_slot)
    slot_csa_state = state_slots_map.get(CSA_STATE, fallback_slot)
    slot_idx_state = state_slots_map.get(INDEXER_STATE, fallback_slot)
    slot_hca_state = state_slots_map.get(HCA_STATE, fallback_slot)

    swa_e = int(paged_pool_entries_per_block[SWA_KV]) if has_swa else 1
    csa_e = int(paged_pool_entries_per_block[CSA_KV]) if has_csa else 1
    idx_e = int(paged_pool_entries_per_block[INDEXER_KV]) if has_idx else 1
    hca_e = int(paged_pool_entries_per_block[HCA_KV]) if has_hca else 1
    csa_state_e = int(paged_pool_entries_per_block[CSA_STATE]) if has_csa_state else 1
    idx_state_e = (
        int(paged_pool_entries_per_block[INDEXER_STATE]) if has_idx_state else 1
    )
    hca_state_e = int(paged_pool_entries_per_block[HCA_STATE]) if has_hca_state else 1

    def _raw_tokens(tag: str) -> int:
        return int(paged_pool_tokens_per_block[tag])

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
    csa_state_tokens = _raw_tokens(CSA_STATE) if has_csa_state else 1
    idx_state_tokens = _raw_tokens(INDEXER_STATE) if has_idx_state else 1
    hca_state_tokens = _raw_tokens(HCA_STATE) if has_hca_state else 1

    hca_state16_backing = getattr(meta, "hca_front_state_rows_16", None)
    pad_hca_state_16 = bool(
        has_hca_state
        and hca_state16_backing is not None
        and hca_state16_backing.data_ptr() == slot_hca_state.data_ptr()
        and int(hca_state16_backing.numel()) >= 16
        and bs * q_len < 16
    )

    grid = (bs, q_len)
    _fused_phase2b_pool_slot_mapping_kernel[grid](
        start_pos,
        bt_swa,
        bt_csa,
        bt_idx,
        bt_hca,
        bt_csa_state,
        bt_idx_state,
        bt_hca_state,
        slot_swa,
        slot_csa,
        slot_idx,
        slot_hca,
        slot_csa_state,
        slot_idx_state,
        slot_hca_state,
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
        CSA_STATE_E=csa_state_e,
        CSA_STATE_TOKENS_PER_BLOCK=csa_state_tokens,
        CSA_STATE_BT_STRIDE=bt_csa_state.shape[1],
        IDX_STATE_E=idx_state_e,
        IDX_STATE_TOKENS_PER_BLOCK=idx_state_tokens,
        IDX_STATE_BT_STRIDE=bt_idx_state.shape[1],
        HCA_STATE_E=hca_state_e,
        HCA_STATE_TOKENS_PER_BLOCK=hca_state_tokens,
        HCA_STATE_BT_STRIDE=bt_hca_state.shape[1],
        HAS_SWA=has_swa,
        HAS_CSA=has_csa,
        HAS_IDX=has_idx,
        HAS_HCA=has_hca,
        HAS_CSA_STATE=has_csa_state,
        HAS_IDX_STATE=has_idx_state,
        HAS_HCA_STATE=has_hca_state,
        PAD_HCA_STATE_16=pad_hca_state_16,
    )
