"""Fused Triton kernels for ``update_decode_metadata_in_place_fp8``.

Two production prepare shapes are supported:

* full V4: compressed ratios ``{4, 128}``;
* SWA-only: no compressed ratios.

Any other ratio set is a configuration error, not a slow-path fallback.

Block-table-dependent outputs (``pool_write_slot_mappings``,
``swa_global_slots``, ``hca_cmp_global_slots``) are handled by separate
phase2b kernels so the two prepare paths stay explicit.
"""

from __future__ import annotations

from typing import Any

import torch

import triton
import triton.language as tl


FUSED_PREPARE_FULL_V4 = "full_v4"
FUSED_PREPARE_SWA_ONLY = "swa_only"


@triton.jit
def _fused_prepare_meta_kernel(
    # inputs
    start_pos_in_ptr,  # [bs] i32
    # persistent outputs (max-sized; we write [:bs] prefix only)
    start_pos_out_ptr,  # [B_max] i32
    cache_seqlens_ptr,  # [B_max] i32
    cmp_len_4_ptr,  # [B_max] i32
    cmp_len_128_ptr,  # [B_max] i32
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
):
    r = tl.program_id(0)
    q = tl.program_id(1)
    if r >= bs:
        return

    sp = tl.load(start_pos_in_ptr + r)  # i32
    abs_pos = sp + q  # i32 scalar
    abs_pos_p1 = abs_pos + 1  # i32 scalar
    in_ring = abs_pos % WINDOW  # i32 scalar

    # ---- Scalar outputs (write once per request; gate on q==0) ----
    if q == 0:
        tl.store(start_pos_out_ptr + r, sp)
        tl.store(cache_seqlens_ptr + r, sp + Q_LEN)
        # compressed_lens uses (start_pos + q_len) // ratio
        tl.store(cmp_len_4_ptr + r, (sp + Q_LEN) // 4)
        tl.store(cmp_len_128_ptr + r, (sp + Q_LEN) // 128)

    # ---- SWA slot mapping: r * WINDOW + (abs_pos % WINDOW) ----
    swa_slot = r * WINDOW + in_ring
    tl.store(slot_swa_ptr + r * Q_LEN + q, swa_slot)

    # ---- Per-ratio compressed slot mappings ----
    on_b4 = (abs_pos_p1 % 4) == 0
    in_req_4 = abs_pos_p1 // 4 - 1
    cmp_slot_4 = tl.where(on_b4, r * STRIDE_CMP_4 + in_req_4, -1)
    tl.store(slot_cmp_4_ptr + r * Q_LEN + q, cmp_slot_4)

    on_b128 = (abs_pos_p1 % 128) == 0
    in_req_128 = abs_pos_p1 // 128 - 1
    cmp_slot_128 = tl.where(on_b128, r * STRIDE_CMP_128 + in_req_128, -1)
    tl.store(slot_cmp_128_ptr + r * Q_LEN + q, cmp_slot_128)

    # ---- Window half: topk_window_idxs + topk_total_{4,128}[:WINDOW] ----
    # Mirrors _build_window_topk_idxs: left-aligned.
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

    WK_TOTAL_4 = WINDOW + INDEX_TOPK
    WK_TOTAL_128 = WINDOW + HCA_DENSE_WIDTH
    base_tt_4 = r * Q_LEN * WK_TOTAL_4 + q * WK_TOTAL_4
    base_tt_128 = r * Q_LEN * WK_TOTAL_128 + q * WK_TOTAL_128
    tl.store(swa_plus_csa_indexer_indices_ptr + base_tt_4 + k, wi, mask=mask_w)
    tl.store(swa_plus_hca_dense_indices_ptr + base_tt_128 + k, wi, mask=mask_w)

    # ---- HCA dense half (r=128): [WINDOW : WINDOW + HCA_DENSE_WIDTH] ----
    # valid if k2 < cmp_len_128; else -1.  CSA dense half: all -1.
    k2 = tl.arange(0, BLOCK_K)
    mask_k = k2 < INDEX_TOPK
    h2 = tl.arange(0, BLOCK_HCA)
    mask_h = h2 < HCA_DENSE_WIDTH
    h2_i32 = h2.to(tl.int32)
    cmp_len_128_val = abs_pos_p1 // 128
    valid_h = h2_i32 < cmp_len_128_val
    dense_h = tl.where(valid_h, h2_i32, -1).to(tl.int32)
    tl.store(
        swa_plus_hca_dense_indices_ptr + base_tt_128 + WINDOW + h2,
        dense_h,
        mask=mask_h,
    )
    tl.store(
        swa_plus_csa_indexer_indices_ptr + base_tt_4 + WINDOW + k2,
        tl.full((BLOCK_K,), -1, tl.int32),
        mask=mask_k,
    )


def _next_pow2(n: int) -> int:
    p = 1
    while p < n:
        p <<= 1
    return p


@triton.jit
def _fused_prepare_meta_swa_only_kernel(
    # inputs
    start_pos_in_ptr,  # [bs] i32
    # persistent outputs
    start_pos_out_ptr,  # [B_max] i32
    cache_seqlens_ptr,  # [B_max] i32
    slot_swa_ptr,  # [B_max * q_len] i32
    topk_window_ptr,  # [B_max, q_len, WINDOW] i32
    swa_abs_idx_ptr,  # [B_max, q_len, WINDOW] i32
    # runtime sizes
    bs,
    # constants
    Q_LEN: tl.constexpr,
    WINDOW: tl.constexpr,
    BLOCK_W: tl.constexpr,
):
    r = tl.program_id(0)
    q = tl.program_id(1)
    if r >= bs:
        return

    sp = tl.load(start_pos_in_ptr + r)
    abs_pos = sp + q
    in_ring = abs_pos % WINDOW

    if q == 0:
        tl.store(start_pos_out_ptr + r, sp)
        tl.store(cache_seqlens_ptr + r, sp + Q_LEN)

    swa_slot = r * WINDOW + in_ring
    tl.store(slot_swa_ptr + r * Q_LEN + q, swa_slot)

    k = tl.arange(0, BLOCK_W)
    mask_w = k < WINDOW
    k_i32 = k.to(tl.int32)

    ring_full = (in_ring + 1 + k_i32) % WINDOW
    partial = tl.where(k_i32 <= abs_pos, k_i32, -1)
    is_full = abs_pos >= (WINDOW - 1)
    wi = tl.where(is_full, ring_full, partial).to(tl.int32)

    base_w = r * Q_LEN * WINDOW + q * WINDOW
    tl.store(topk_window_ptr + base_w + k, wi, mask=mask_w)

    win_start = tl.maximum(abs_pos - WINDOW + 1, 0)
    candidate = win_start + k_i32
    valid_pos = candidate <= abs_pos
    swa_abs = tl.where(valid_pos, candidate, -1).to(tl.int32)
    tl.store(swa_abs_idx_ptr + base_w + k, swa_abs, mask=mask_w)


def fused_update_decode_meta_full_v4(
    meta: Any,
    start_pos: torch.Tensor,
    max_seq_len: int,
) -> None:
    """Replaces the "pure start_pos" portion of
    :func:`update_decode_metadata_in_place_fp8` with a single Triton kernel.

    Writes ONLY the outputs listed in the module docstring; callers are
    still responsible for the block_table-dependent sections
    (``pool_write_slot_mappings``, ``swa_global_slots``,
    ``hca_cmp_global_slots``) and the ``topk_buffer_compressed`` reset.

    Contract: every output tensor on ``meta`` keeps its original storage
    (graph-safe).  ``start_pos.device`` and ``.dtype`` must already
    match (the caller coerces).

    Requires: ratios {4, 128} present in ``meta.slot_mapping_compressed``.
    """

    bs = int(start_pos.shape[0])
    q_len = meta.q_len_per_req
    window_size = meta.window_size
    index_topk = meta.topk_buffer_compressed.shape[-1]
    hca_dense_width = meta.topk_total_by_ratio[128].shape[-1] - window_size

    # Assumption: {4, 128} present — mirrors V4-Flash compress_ratios.
    slot_cmp_4 = meta.slot_mapping_compressed[4]
    slot_cmp_128 = meta.slot_mapping_compressed[128]
    cmp_len_4 = meta.compressed_lens[4]
    cmp_len_128 = meta.compressed_lens[128]
    topk_total_4 = meta.topk_total_by_ratio[4]
    topk_total_128 = meta.topk_total_by_ratio[128]

    # Stride constants match allocate_decode_metadata_fp8:
    # compressed_buffer_t_dim_per_ratio[r] = max_seq_len // r.
    stride_cmp_4 = meta.compressed_buffer_t_dim_per_ratio[4]
    stride_cmp_128 = meta.compressed_buffer_t_dim_per_ratio[128]

    BLOCK_W = _next_pow2(window_size)
    BLOCK_K = _next_pow2(index_topk)
    BLOCK_HCA = _next_pow2(hca_dense_width)

    grid = (bs, q_len)
    _fused_prepare_meta_kernel[grid](
        start_pos,
        meta.start_pos,
        meta.cache_seqlens_i32,
        cmp_len_4,
        cmp_len_128,
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
        INDEX_TOPK=index_topk,
        HCA_DENSE_WIDTH=hca_dense_width,
        STRIDE_CMP_4=stride_cmp_4,
        STRIDE_CMP_128=stride_cmp_128,
        BLOCK_W=BLOCK_W,
        BLOCK_K=BLOCK_K,
        BLOCK_HCA=BLOCK_HCA,
    )


def fused_update_decode_meta_swa_only(
    meta: Any,
    start_pos: torch.Tensor,
) -> None:
    """SWA-only prepare kernel. This path intentionally ignores all
    compressed metadata because ``fused_prepare_kind`` requires those
    dictionaries to be empty for SWA-only metadata.
    """
    bs = int(start_pos.shape[0])
    q_len = meta.q_len_per_req
    window_size = meta.window_size
    BLOCK_W = _next_pow2(window_size)

    grid = (bs, q_len)
    _fused_prepare_meta_swa_only_kernel[grid](
        start_pos,
        meta.start_pos,
        meta.cache_seqlens_i32,
        meta.slot_mapping_swa,
        meta.topk_window_idxs,
        meta.swa_abs_idx,
        bs,
        Q_LEN=q_len,
        WINDOW=window_size,
        BLOCK_W=BLOCK_W,
    )


def fused_prepare_kind(meta: Any) -> str:
    """Return the supported fused prepare kind or raise.

    Runtime only supports the two graph-safe fused shapes. Unsupported
    ratio sets must fail clearly so they cannot silently route to a slow
    eager fallback.
    """
    ratios = set(meta.slot_mapping_compressed.keys())
    if meta.cache_seqlens_i32 is None:
        raise RuntimeError("fused prepare requires cache_seqlens_i32")
    if not meta.cache_seqlens_i32.is_cuda:
        raise RuntimeError("fused prepare requires CUDA metadata tensors")
    if meta.swa_abs_idx is None:
        raise RuntimeError("fused prepare requires swa_abs_idx")

    if ratios == set():
        if meta.compressed_lens:
            raise RuntimeError("SWA-only fused prepare requires empty compressed_lens")
        if meta.compressed_lens_per_token:
            raise RuntimeError(
                "SWA-only fused prepare requires empty compressed_lens_per_token"
            )
        if meta.topk_total_by_ratio:
            raise RuntimeError("SWA-only fused prepare requires empty topk_total")
        if meta.compressed_buffer_t_dim_per_ratio:
            raise RuntimeError(
                "SWA-only fused prepare requires empty compressed buffer strides"
            )
        return FUSED_PREPARE_SWA_ONLY

    if ratios != {4, 128}:
        raise RuntimeError(
            "unsupported fused_prepare_kind: expected SWA-only or ratios {4, 128}, "
            f"got {sorted(ratios)}"
        )

    for r in (4, 128):
        if r not in meta.compressed_lens:
            raise RuntimeError(f"full V4 fused prepare missing compressed_lens[{r}]")
        if r not in meta.compressed_lens_per_token:
            raise RuntimeError(
                f"full V4 fused prepare missing compressed_lens_per_token[{r}]"
            )
        if r not in meta.topk_total_by_ratio:
            raise RuntimeError(f"full V4 fused prepare missing topk_total[{r}]")
        if r not in meta.compressed_buffer_t_dim_per_ratio:
            raise RuntimeError(
                f"full V4 fused prepare missing compressed stride for ratio {r}"
            )

    want4 = meta.window_size + meta.topk_buffer_compressed.shape[-1]
    hca_dense_width = ((meta.compressed_buffer_t_dim_per_ratio[128] + 63) // 64) * 64
    want128 = meta.window_size + hca_dense_width
    if meta.topk_total_by_ratio[4].shape[-1] != want4:
        raise RuntimeError(
            "full V4 fused prepare invalid ratio-4 topk width: "
            f"expected {want4}, got {meta.topk_total_by_ratio[4].shape[-1]}"
        )
    if meta.topk_total_by_ratio[128].shape[-1] != want128:
        raise RuntimeError(
            "full V4 fused prepare invalid ratio-128 topk width: "
            f"expected {want128}, got {meta.topk_total_by_ratio[128].shape[-1]}"
        )
    return FUSED_PREPARE_FULL_V4


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
# HCA=128); ``is_full_v4_phase2b_fused_supported`` enforces the same
# ratio set.
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
    SWA_BT_STRIDE: tl.constexpr,
    CSA_E: tl.constexpr,
    CSA_BT_STRIDE: tl.constexpr,
    IDX_E: tl.constexpr,
    IDX_BT_STRIDE: tl.constexpr,
    HCA_E: tl.constexpr,
    HCA_BT_STRIDE: tl.constexpr,
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
    # NOTE: Python reference computes ``in_block`` from UNCLAMPED
    # ``block_in_seq`` (= ``abs_pos % E``), then separately clamps
    # ``block_in_seq`` only for the gather.  Mirror that.
    bis_swa_raw = abs_pos // SWA_E
    in_blk_swa = abs_pos - bis_swa_raw * SWA_E
    bis_swa = tl.maximum(tl.minimum(bis_swa_raw, SWA_BT_STRIDE - 1), 0)
    bid_swa = tl.load(bt_swa_ptr + r * SWA_BT_STRIDE + bis_swa).to(tl.int64)
    slot_swa = tl.where(bid_swa <= 0, -1, bid_swa * SWA_E + in_blk_swa)
    tl.store(slot_swa_ptr + out_idx, slot_swa)

    # ---------- CSA: ratio=4, boundary tokens only ----------
    on_b4 = (abs_pos_p1 % 4) == 0
    cmp_idx_4 = abs_pos_p1 // 4 - 1
    skip_4 = (on_b4 == 0) | (cmp_idx_4 < 0)
    safe_4 = tl.where(skip_4, 0, cmp_idx_4)
    bis_csa_raw = safe_4 // CSA_E
    in_blk_csa = safe_4 - bis_csa_raw * CSA_E
    bis_csa = tl.maximum(tl.minimum(bis_csa_raw, CSA_BT_STRIDE - 1), 0)
    bid_csa = tl.load(bt_csa_ptr + r * CSA_BT_STRIDE + bis_csa).to(tl.int64)
    skip_csa = skip_4 | (bid_csa <= 0)
    slot_csa = tl.where(skip_csa, -1, bid_csa * CSA_E + in_blk_csa)
    tl.store(slot_csa_ptr + out_idx, slot_csa)

    # ---------- INDEXER: ratio=4, shares boundary with CSA but has own E/bt ----------
    bis_idx_raw = safe_4 // IDX_E
    in_blk_idx = safe_4 - bis_idx_raw * IDX_E
    bis_idx = tl.maximum(tl.minimum(bis_idx_raw, IDX_BT_STRIDE - 1), 0)
    bid_idx = tl.load(bt_idx_ptr + r * IDX_BT_STRIDE + bis_idx).to(tl.int64)
    skip_idx = skip_4 | (bid_idx <= 0)
    slot_idx = tl.where(skip_idx, -1, bid_idx * IDX_E + in_blk_idx)
    tl.store(slot_idx_ptr + out_idx, slot_idx)

    # ---------- HCA: ratio=128, boundary tokens only ----------
    on_b128 = (abs_pos_p1 % 128) == 0
    cmp_idx_128 = abs_pos_p1 // 128 - 1
    skip_128 = (on_b128 == 0) | (cmp_idx_128 < 0)
    safe_128 = tl.where(skip_128, 0, cmp_idx_128)
    bis_hca_raw = safe_128 // HCA_E
    in_blk_hca = safe_128 - bis_hca_raw * HCA_E
    bis_hca = tl.maximum(tl.minimum(bis_hca_raw, HCA_BT_STRIDE - 1), 0)
    bid_hca = tl.load(bt_hca_ptr + r * HCA_BT_STRIDE + bis_hca).to(tl.int64)
    skip_hca = skip_128 | (bid_hca <= 0)
    slot_hca = tl.where(skip_hca, -1, bid_hca * HCA_E + in_blk_hca)
    tl.store(slot_hca_ptr + out_idx, slot_hca)


def fused_phase2b_full_v4_pool_slot_mapping(
    meta: Any,
    start_pos: torch.Tensor,
    bs: int,
    paged_pool_entries_per_block: Any,
) -> None:
    """Replaces the 4 ``compute_kv_pool_slot_mapping`` calls in
    :func:`update_decode_metadata_in_place_fp8` phase2b with a single
    Triton kernel launch.

    Writes ``meta.pool_write_slot_mappings[{SWA,CSA,INDEXER,HCA}]``'s
    ``[:bs * q_len]`` prefix; tail left at its sentinel from allocate.

    Requires: all 4 pool types present in ``meta.pool_block_tables``
    AND in ``meta.pool_write_slot_mappings``.  Call
    :func:`is_full_v4_phase2b_fused_supported` first.

    Graph-safe: every output buffer reuses its existing storage.
    """
    from rtp_llm.models_py.modules.dsv4.attn_type import (
        CSA_KV,
        HCA_KV,
        INDEXER_KV,
        SWA_KV,
    )

    q_len = meta.q_len_per_req

    bt_swa = meta.pool_block_tables[SWA_KV]
    bt_csa = meta.pool_block_tables[CSA_KV]
    bt_idx = meta.pool_block_tables[INDEXER_KV]
    bt_hca = meta.pool_block_tables[HCA_KV]

    slot_swa = meta.pool_write_slot_mappings[SWA_KV]
    slot_csa = meta.pool_write_slot_mappings[CSA_KV]
    slot_idx = meta.pool_write_slot_mappings[INDEXER_KV]
    slot_hca = meta.pool_write_slot_mappings[HCA_KV]

    swa_e = int(paged_pool_entries_per_block.get(SWA_KV, meta.window_size))
    csa_e = int(paged_pool_entries_per_block.get(CSA_KV, 1))
    idx_e = int(paged_pool_entries_per_block.get(INDEXER_KV, 1))
    hca_e = int(paged_pool_entries_per_block.get(HCA_KV, 1))

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
        SWA_BT_STRIDE=bt_swa.shape[1],
        CSA_E=csa_e,
        CSA_BT_STRIDE=bt_csa.shape[1],
        IDX_E=idx_e,
        IDX_BT_STRIDE=bt_idx.shape[1],
        HCA_E=hca_e,
        HCA_BT_STRIDE=bt_hca.shape[1],
    )


def is_full_v4_phase2b_fused_supported(
    meta: Any,
    paged_block_tables: Any,
    paged_pool_entries_per_block: Any,
) -> bool:
    """Guard: all 4 V4-Flash pool types (SWA/CSA/INDEXER/HCA) must be
    present in both ``meta.pool_block_tables`` (allocated buffers) and
    ``paged_block_tables`` (framework-supplied runtime block tables),
    plus entries_per_block available for each.

    The fused kernel assumes hard-coded ratios {CSA:4, INDEXER:4,
    HCA:128}; :func:`fused_prepare_kind` checks the meta shape.
    """
    from rtp_llm.models_py.modules.dsv4.attn_type import (
        CSA_KV,
        HCA_KV,
        INDEXER_KV,
        SWA_KV,
    )

    required = (SWA_KV, CSA_KV, INDEXER_KV, HCA_KV)
    for at in required:
        if at not in meta.pool_block_tables:
            return False
        if at not in meta.pool_write_slot_mappings:
            return False
        if paged_block_tables is None or at not in paged_block_tables:
            return False
        if (
            paged_pool_entries_per_block is None
            or at not in paged_pool_entries_per_block
        ):
            return False
    return True


@triton.jit
def _fused_phase2b_swa_pool_slot_mapping_kernel(
    # start_pos
    start_pos_ptr,  # [bs] i32
    # SWA block table (i32, [B_max, BT_STRIDE])
    bt_swa_ptr,
    # SWA output slot buffer (i64, [B_max * q_len])
    slot_swa_ptr,
    # runtime
    bs,
    # compile-time constants
    Q_LEN: tl.constexpr,
    SWA_E: tl.constexpr,
    SWA_BT_STRIDE: tl.constexpr,
):
    r = tl.program_id(0)
    q = tl.program_id(1)
    if r >= bs:
        return

    sp = tl.load(start_pos_ptr + r)
    abs_pos = sp + q
    out_idx = r * Q_LEN + q

    bis_swa_raw = abs_pos // SWA_E
    in_blk_swa = abs_pos - bis_swa_raw * SWA_E
    bis_swa = tl.maximum(tl.minimum(bis_swa_raw, SWA_BT_STRIDE - 1), 0)
    bid_swa = tl.load(bt_swa_ptr + r * SWA_BT_STRIDE + bis_swa).to(tl.int64)
    slot_swa = tl.where(bid_swa <= 0, -1, bid_swa * SWA_E + in_blk_swa)
    tl.store(slot_swa_ptr + out_idx, slot_swa)


def fused_phase2b_swa_pool_slot_mapping(
    meta: Any,
    start_pos: torch.Tensor,
    bs: int,
    paged_pool_entries_per_block: Any,
) -> None:
    """SWA-only phase2b write-slot mapping."""
    from rtp_llm.models_py.modules.dsv4.attn_type import SWA_KV

    q_len = meta.q_len_per_req
    bt_swa = meta.pool_block_tables[SWA_KV]
    slot_swa = meta.pool_write_slot_mappings[SWA_KV]
    swa_e = int(paged_pool_entries_per_block.get(SWA_KV, meta.window_size))

    grid = (bs, q_len)
    _fused_phase2b_swa_pool_slot_mapping_kernel[grid](
        start_pos,
        bt_swa,
        slot_swa,
        bs,
        Q_LEN=q_len,
        SWA_E=swa_e,
        SWA_BT_STRIDE=bt_swa.shape[1],
    )


def is_swa_phase2b_fused_supported(
    meta: Any,
    paged_block_tables: Any,
    paged_pool_entries_per_block: Any,
) -> bool:
    """Guard for SWA-only phase2b."""
    from rtp_llm.models_py.modules.dsv4.attn_type import SWA_KV

    if SWA_KV not in meta.pool_block_tables:
        return False
    if SWA_KV not in meta.pool_write_slot_mappings:
        return False
    if paged_block_tables is None or SWA_KV not in paged_block_tables:
        return False
    if paged_pool_entries_per_block is None or SWA_KV not in paged_pool_entries_per_block:
        return False
    if meta.compressor_state_slot_mappings:
        return False
    return True
