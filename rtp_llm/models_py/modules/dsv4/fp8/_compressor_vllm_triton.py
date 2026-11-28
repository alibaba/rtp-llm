"""DSV4 CompressorFP8 Triton kernels (ported from vLLM, RTP-LLM-adapted).

Three kernels backing the DSV4 sparse-attention compressor. The state
pool layout is page-aligned: the framework supplies sparse absolute
logical-block tables over physical blocks of ``entries_per_block`` (=256)
fp32 slots; ``slot_mapping[t]`` resolves to the slot reserved for token
``t`` when its logical block is present.

  * ``_save_partial_states_kernel`` — per-token write of (kv | score+ape)
    into the fp32 state cache. One program per token; non-boundary tokens
    still write so that the boundary writer can read them later.
    Source: vLLM ``deepseek_compressor.py``.

  * ``_fused_kv_compress_norm_rope_insert_sparse_attn`` — head=512 fused
    boundary writer (gather → softmax-reduce → RMSNorm → FP8 quant nope →
    RoPE bf16 → 584B KV-pool slot store). Source: vLLM
    ``fused_compress_quant_cache.py``.

  * ``_fused_kv_compress_norm_rope_insert_indexer_attn`` — head=128 fused
    boundary writer (same pipeline; whole token is one quant block, 132B
    per KV-pool slot).

Boundary writer source dispatch (raw vs state cache):

  The writer needs the prior ``(1+OVERLAP)*COMPRESS_RATIO`` positions of
  kv/score for the boundary token. There are two sources:

    raw   — ``kv_raw[flat_idx]`` / ``score_raw[flat_idx]`` where
             ``flat_idx = pos - seq_start``. ``kv_raw/score_raw`` is the
             authoritative copy of every token whose kv/score the caller
             materialised for this launch (prefill: the current chunk
             passed through linear projections). Used whenever
             ``0 <= flat_idx < n_raw``.

    cache — ``state_cache[block_table[req_idx, pos // block_size], ...]``.
             Used for ``flat_idx < 0``, i.e. positions belonging to a
             prefix-cache hit: they were written into the framework state pool
             by a prior request, and the framework reuses those physical
             blocks via this request's sparse absolute ``block_table``.
             Positions routed to cache must resolve to a non-negative physical
             block; negative ids are invalid.

  Decode passes ``disable_raw_path=True`` → ``n_raw == 0``, so every
  position routes through the cache branch.

Both fused writers self-skip non-boundary tokens via early-exit on
``(position+1) % COMPRESS_RATIO != 0``, so the caller hands in a flat
``[N_tok]`` slot_mapping without pre-filtering.

The thin wrappers ``run_save_partial_states`` / ``run_fused_compress_kv_write``
hide the per-``head_dim`` constexpr table.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import triton
import triton.language as tl


@triton.jit
def _trap_invalid_kv_access(TRAP_INVALID_KV_ACCESS: tl.constexpr) -> None:
    if TRAP_INVALID_KV_ACCESS:
        tl.inline_asm_elementwise(
            "trap; // dummy $0",
            "=r",
            [],
            dtype=tl.int32,
            is_pure=False,
            pack=1,
        )


# =============================================================================
# Per-token state-cache writer (fp32). ape is added to score in-kernel.
# =============================================================================
_SAVE_STATES_TOKENS_PER_CTA = 1


@triton.jit(
    do_not_specialize=[
        "N",
        "kv_stride",
        "score_stride",
        "ape_stride",
        "state_cache_stride0",
        "state_cache_stride1",
        "block_size",
        "num_state_blocks",
    ]
)
def _save_partial_states_kernel(
    kv_ptr,
    kv_stride,
    score_ptr,
    score_stride,
    ape_ptr,
    ape_stride,
    positions_ptr,
    state_cache_ptr,
    state_cache_stride0,
    state_cache_stride1,
    slot_mapping_ptr,
    unpad_restore_ptr,
    active_indices_ptr,
    N,
    block_size,
    num_state_blocks,
    HEAD_SIZE: tl.constexpr,
    TRITON_BLOCK_SIZE: tl.constexpr,
    STATE_WIDTH: tl.constexpr,
    COMPRESS_RATIO: tl.constexpr,
    TRAP_INVALID_KV_ACCESS: tl.constexpr,
    USE_UNPAD_RESTORE: tl.constexpr,
    USE_ACTIVE_INDICES: tl.constexpr,
    ALIGN4: tl.constexpr,
):
    idx = tl.program_id(0).to(tl.int64)
    if idx >= N:
        return
    if USE_ACTIVE_INDICES:
        token_idx = tl.load(active_indices_ptr + idx).to(tl.int64)
    else:
        token_idx = idx
    slot_id = tl.load(slot_mapping_ptr + token_idx)
    if slot_id < 0:
        return

    block_idx = (slot_id // block_size).to(tl.int64)
    pos_in_block = slot_id % block_size
    if block_idx < 0:
        _trap_invalid_kv_access(TRAP_INVALID_KV_ACCESS)
    if block_idx >= num_state_blocks:
        _trap_invalid_kv_access(TRAP_INVALID_KV_ACCESS)

    position = tl.load(positions_ptr + token_idx)
    if USE_UNPAD_RESTORE:
        raw_row_idx = tl.load(unpad_restore_ptr + token_idx).to(tl.int64)
    else:
        raw_row_idx = token_idx

    block = tl.arange(0, TRITON_BLOCK_SIZE)
    if ALIGN4:
        block = tl.multiple_of(block, 4)
    mask = None if HEAD_SIZE == TRITON_BLOCK_SIZE else (block < HEAD_SIZE)
    kv_s = (
        tl.multiple_of(kv_stride.to(tl.int64), 4) if ALIGN4 else kv_stride.to(tl.int64)
    )
    sc_s = (
        tl.multiple_of(score_stride.to(tl.int64), 4)
        if ALIGN4
        else score_stride.to(tl.int64)
    )
    ap_s = (
        tl.multiple_of(ape_stride.to(tl.int64), 4)
        if ALIGN4
        else ape_stride.to(tl.int64)
    )
    st_s0 = (
        tl.multiple_of(state_cache_stride0.to(tl.int64), 4)
        if ALIGN4
        else state_cache_stride0.to(tl.int64)
    )
    st_s1 = (
        tl.multiple_of(state_cache_stride1.to(tl.int64), 4)
        if ALIGN4
        else state_cache_stride1.to(tl.int64)
    )

    base_ptr = state_cache_ptr + block_idx * st_s0 + pos_in_block * st_s1
    ape_row = position % COMPRESS_RATIO
    kv = tl.load(kv_ptr + raw_row_idx * kv_s + block, mask=mask)
    ape = tl.load(ape_ptr + ape_row * ap_s + block, mask=mask)
    score = tl.load(score_ptr + raw_row_idx * sc_s + block, mask=mask)
    tl.store(base_ptr + block, kv, mask=mask)
    tl.store(base_ptr + STATE_WIDTH + block, score + ape, mask=mask)


# =============================================================================
# Shared boundary compression helpers (split score-first + kv-second to halve
# peak register pressure; source dispatch + online softmax + norm)
# =============================================================================
@triton.jit
def _resolve_tile_sources(
    block_table_ptr,
    block_table_stride,
    block_size,
    req_idx,
    unpad_restore_ptr,
    req_seq_start,
    req_cu_lo,
    req_n_raw,
    start,
    tokens,
    NUM_STATE_BLOCKS,
    COMPRESS_RATIO: tl.constexpr,
    WINDOW_SIZE: tl.constexpr,
    TRAP_INVALID_KV_ACCESS: tl.constexpr,
    STATE_RING_ENTRIES: tl.constexpr,
    MAY_USE_CACHE: tl.constexpr,
    USE_UNPAD_RESTORE: tl.constexpr,
):
    pos = start + tokens.to(tl.int64)
    mask_pos = (pos >= 0) & (tokens < WINDOW_SIZE)
    head_offset = (tokens >= COMPRESS_RATIO).to(tl.int32)
    flat_idx_in_req = pos - req_seq_start
    use_raw = mask_pos & (flat_idx_in_req >= 0) & (flat_idx_in_req < req_n_raw)
    flat_idx_safe = tl.where(use_raw, req_cu_lo + flat_idx_in_req, 0)

    if USE_UNPAD_RESTORE:
        raw_row_idx = tl.load(
            unpad_restore_ptr + flat_idx_safe, mask=use_raw, other=0
        ).to(tl.int64)
    else:
        raw_row_idx = flat_idx_safe

    if MAY_USE_CACHE:
        use_cache = mask_pos & ~use_raw
        block_indices = (pos // block_size) % tl.maximum(block_table_stride, 1)
        block_indices_safe = tl.where(use_cache, block_indices, 0)
        block_numbers = tl.load(
            block_table_ptr + req_idx * block_table_stride + block_indices_safe,
            mask=use_cache,
            other=0,
        )
        invalid_state_block = use_cache & (block_numbers >= NUM_STATE_BLOCKS)
        if tl.max(invalid_state_block.to(tl.int32), axis=0) != 0:
            _trap_invalid_kv_access(TRAP_INVALID_KV_ACCESS)
        valid_block = use_cache & (block_numbers > 0)
        block_numbers_i64 = tl.where(valid_block, block_numbers, 0).to(tl.int64)
        block_offsets = tl.where(mask_pos, pos % STATE_RING_ENTRIES, 0).to(tl.int64)
        final_valid = use_raw | (use_cache & valid_block)
    else:
        valid_block = mask_pos & False
        block_numbers_i64 = tl.zeros_like(pos)
        block_offsets = tl.zeros_like(pos)
        final_valid = use_raw

    return (
        pos,
        head_offset,
        use_raw,
        valid_block,
        raw_row_idx,
        block_numbers_i64,
        block_offsets,
        final_valid,
    )


@triton.jit
def _load_score_tile(
    state_cache_ptr,
    state_cache_stride0,
    state_cache_stride1,
    score_raw_ptr,
    score_raw_stride,
    ape_ptr,
    ape_stride,
    pos,
    head_offset,
    use_raw,
    valid_block,
    raw_row_idx,
    block_numbers_i64,
    block_offsets,
    block,
    mask,
    HEAD_SIZE: tl.constexpr,
    STATE_WIDTH: tl.constexpr,
    COMPRESS_RATIO: tl.constexpr,
    MAY_USE_CACHE: tl.constexpr,
    ALIGN4: tl.constexpr,
):
    h_off = head_offset * HEAD_SIZE
    sc_s = (
        tl.multiple_of(score_raw_stride.to(tl.int64), 4)
        if ALIGN4
        else score_raw_stride.to(tl.int64)
    )
    ap_s = (
        tl.multiple_of(ape_stride.to(tl.int64), 4)
        if ALIGN4
        else ape_stride.to(tl.int64)
    )
    raw_base = score_raw_ptr + raw_row_idx * sc_s + h_off
    if MAY_USE_CACHE:
        st_s0 = (
            tl.multiple_of(state_cache_stride0.to(tl.int64), 4)
            if ALIGN4
            else state_cache_stride0.to(tl.int64)
        )
        st_s1 = (
            tl.multiple_of(state_cache_stride1.to(tl.int64), 4)
            if ALIGN4
            else state_cache_stride1.to(tl.int64)
        )
        cache_base = (
            state_cache_ptr
            + block_numbers_i64 * st_s0
            + block_offsets * st_s1
            + STATE_WIDTH
            + h_off
        )
        load_mask = (use_raw | valid_block)[:, None] & mask[None, :]
        row_base = tl.where(use_raw, raw_base, cache_base)
    else:
        load_mask = use_raw[:, None] & mask[None, :]
        row_base = raw_base

    score = tl.load(
        row_base[:, None] + block[None, :],
        mask=load_mask,
        other=0.0,
        eviction_policy="evict_first",
    ).to(tl.float32)
    ape_rows_safe = tl.where(use_raw, pos % COMPRESS_RATIO, 0)
    ape_vals = tl.load(
        ape_ptr + ape_rows_safe[:, None] * ap_s + h_off[:, None] + block[None, :],
        mask=use_raw[:, None] & mask[None, :],
        other=0.0,
    ).to(tl.float32)
    return score + ape_vals


@triton.jit
def _load_kv_tile(
    state_cache_ptr,
    state_cache_stride0,
    state_cache_stride1,
    kv_raw_ptr,
    kv_raw_stride,
    head_offset,
    use_raw,
    valid_block,
    raw_row_idx,
    block_numbers_i64,
    block_offsets,
    block,
    mask,
    HEAD_SIZE: tl.constexpr,
    MAY_USE_CACHE: tl.constexpr,
    ALIGN4: tl.constexpr,
):
    h_off = head_offset * HEAD_SIZE
    kv_s = (
        tl.multiple_of(kv_raw_stride.to(tl.int64), 4)
        if ALIGN4
        else kv_raw_stride.to(tl.int64)
    )
    raw_base = kv_raw_ptr + raw_row_idx * kv_s + h_off
    if MAY_USE_CACHE:
        st_s0 = (
            tl.multiple_of(state_cache_stride0.to(tl.int64), 4)
            if ALIGN4
            else state_cache_stride0.to(tl.int64)
        )
        st_s1 = (
            tl.multiple_of(state_cache_stride1.to(tl.int64), 4)
            if ALIGN4
            else state_cache_stride1.to(tl.int64)
        )
        cache_base = (
            state_cache_ptr + block_numbers_i64 * st_s0 + block_offsets * st_s1 + h_off
        )
        load_mask = (use_raw | valid_block)[:, None] & mask[None, :]
        row_base = tl.where(use_raw, raw_base, cache_base)
    else:
        load_mask = use_raw[:, None] & mask[None, :]
        row_base = raw_base
    return tl.load(
        row_base[:, None] + block[None, :],
        mask=load_mask,
        other=0.0,
        eviction_policy="evict_first",
    ).to(tl.float32)


@triton.jit
def _compress_and_rmsnorm(
    state_cache_ptr,
    state_cache_stride0,
    state_cache_stride1,
    req_idx,
    position,
    block_table_ptr,
    block_table_stride,
    block_size,
    kv_raw_ptr,
    kv_raw_stride,
    score_raw_ptr,
    score_raw_stride,
    ape_ptr,
    ape_stride,
    unpad_restore_ptr,
    seq_start,
    n_raw,
    seq_start_per_req_ptr,
    cu_seq_per_req_ptr,
    rms_norm_weight_ptr,
    rms_norm_eps,
    block,
    mask,
    HEAD_SIZE: tl.constexpr,
    TRITON_BLOCK_SIZE: tl.constexpr,
    STATE_WIDTH: tl.constexpr,
    COMPRESS_RATIO: tl.constexpr,
    OVERLAP: tl.constexpr,
    NUM_STATE_BLOCKS,
    BATCHED: tl.constexpr,
    TRAP_INVALID_KV_ACCESS: tl.constexpr,
    STATE_RING_ENTRIES: tl.constexpr,
    MAY_USE_CACHE: tl.constexpr,
    USE_UNPAD_RESTORE: tl.constexpr,
    TILE_T: tl.constexpr,
    ALIGN4: tl.constexpr,
):
    WINDOW_SIZE: tl.constexpr = (1 + OVERLAP) * COMPRESS_RATIO
    start = position - WINDOW_SIZE + 1

    if BATCHED:
        req_seq_start = tl.load(seq_start_per_req_ptr + req_idx).to(tl.int64)
        req_cu_lo = tl.load(cu_seq_per_req_ptr + req_idx).to(tl.int64)
        req_cu_hi = tl.load(cu_seq_per_req_ptr + req_idx + 1).to(tl.int64)
        req_n_raw = req_cu_hi - req_cu_lo
    else:
        req_seq_start = tl.zeros((), dtype=tl.int64) + seq_start
        req_cu_lo = tl.zeros((), dtype=tl.int64)
        req_n_raw = tl.zeros((), dtype=tl.int64) + n_raw

    if WINDOW_SIZE < 64:
        tokens = tl.arange(0, WINDOW_SIZE)
        (
            pos,
            head_offset,
            use_raw,
            valid_block,
            raw_row_idx,
            block_numbers_i64,
            block_offsets,
            final_valid,
        ) = _resolve_tile_sources(
            block_table_ptr,
            block_table_stride,
            block_size,
            req_idx,
            unpad_restore_ptr,
            req_seq_start,
            req_cu_lo,
            req_n_raw,
            start,
            tokens,
            NUM_STATE_BLOCKS,
            COMPRESS_RATIO=COMPRESS_RATIO,
            WINDOW_SIZE=WINDOW_SIZE,
            TRAP_INVALID_KV_ACCESS=TRAP_INVALID_KV_ACCESS,
            STATE_RING_ENTRIES=STATE_RING_ENTRIES,
            MAY_USE_CACHE=MAY_USE_CACHE,
            USE_UNPAD_RESTORE=USE_UNPAD_RESTORE,
        )
        combined_mask = final_valid[:, None] & mask[None, :]
        # Score + ape -> softmax weights BEFORE loading KV to cut peak registers.
        score = _load_score_tile(
            state_cache_ptr,
            state_cache_stride0,
            state_cache_stride1,
            score_raw_ptr,
            score_raw_stride,
            ape_ptr,
            ape_stride,
            pos,
            head_offset,
            use_raw,
            valid_block,
            raw_row_idx,
            block_numbers_i64,
            block_offsets,
            block,
            mask,
            HEAD_SIZE=HEAD_SIZE,
            STATE_WIDTH=STATE_WIDTH,
            COMPRESS_RATIO=COMPRESS_RATIO,
            MAY_USE_CACHE=MAY_USE_CACHE,
            ALIGN4=ALIGN4,
        )
        score = tl.where(combined_mask, score, float("-inf"))
        weights = tl.softmax(score, dim=0)
        kv = _load_kv_tile(
            state_cache_ptr,
            state_cache_stride0,
            state_cache_stride1,
            kv_raw_ptr,
            kv_raw_stride,
            head_offset,
            use_raw,
            valid_block,
            raw_row_idx,
            block_numbers_i64,
            block_offsets,
            block,
            mask,
            HEAD_SIZE=HEAD_SIZE,
            MAY_USE_CACHE=MAY_USE_CACHE,
            ALIGN4=ALIGN4,
        )
        kv = tl.where(combined_mask, kv, 0.0)
        compressed_kv = tl.sum(kv * weights, axis=0)
    else:
        # Large-ratio path (HCA ratio=128): stream TILE_T=4 with num_stages=1 and
        # load score -> softmax weights BEFORE loading KV in every tile.
        m_i = tl.full((TRITON_BLOCK_SIZE,), float("-inf"), dtype=tl.float32)
        d_i = tl.zeros((TRITON_BLOCK_SIZE,), dtype=tl.float32)
        acc = tl.zeros((TRITON_BLOCK_SIZE,), dtype=tl.float32)
        for tile_start in tl.range(0, WINDOW_SIZE, TILE_T, num_stages=1):
            tokens = tile_start + tl.arange(0, TILE_T)
            (
                pos,
                head_offset,
                use_raw,
                valid_block,
                raw_row_idx,
                block_numbers_i64,
                block_offsets,
                valid_t,
            ) = _resolve_tile_sources(
                block_table_ptr,
                block_table_stride,
                block_size,
                req_idx,
                unpad_restore_ptr,
                req_seq_start,
                req_cu_lo,
                req_n_raw,
                start,
                tokens,
                NUM_STATE_BLOCKS,
                COMPRESS_RATIO=COMPRESS_RATIO,
                WINDOW_SIZE=WINDOW_SIZE,
                TRAP_INVALID_KV_ACCESS=TRAP_INVALID_KV_ACCESS,
                STATE_RING_ENTRIES=STATE_RING_ENTRIES,
                MAY_USE_CACHE=MAY_USE_CACHE,
                USE_UNPAD_RESTORE=USE_UNPAD_RESTORE,
            )
            combined_mask_t = valid_t[:, None] & mask[None, :]
            score_t = _load_score_tile(
                state_cache_ptr,
                state_cache_stride0,
                state_cache_stride1,
                score_raw_ptr,
                score_raw_stride,
                ape_ptr,
                ape_stride,
                pos,
                head_offset,
                use_raw,
                valid_block,
                raw_row_idx,
                block_numbers_i64,
                block_offsets,
                block,
                mask,
                HEAD_SIZE=HEAD_SIZE,
                STATE_WIDTH=STATE_WIDTH,
                COMPRESS_RATIO=COMPRESS_RATIO,
                MAY_USE_CACHE=MAY_USE_CACHE,
                ALIGN4=ALIGN4,
            )
            score_t = tl.where(combined_mask_t, score_t, float("-inf"))
            tile_max = tl.max(score_t, axis=0)
            m_new = tl.maximum(m_i, tile_max)
            m_new_safe = tl.where(m_new > float("-inf"), m_new, 0.0)
            alpha = tl.where(m_i > float("-inf"), tl.exp(m_i - m_new_safe), 0.0)
            p_t = tl.where(combined_mask_t, tl.exp(score_t - m_new_safe[None, :]), 0.0)
            d_i = d_i * alpha + tl.sum(p_t, axis=0)

            kv_t = _load_kv_tile(
                state_cache_ptr,
                state_cache_stride0,
                state_cache_stride1,
                kv_raw_ptr,
                kv_raw_stride,
                head_offset,
                use_raw,
                valid_block,
                raw_row_idx,
                block_numbers_i64,
                block_offsets,
                block,
                mask,
                HEAD_SIZE=HEAD_SIZE,
                MAY_USE_CACHE=MAY_USE_CACHE,
                ALIGN4=ALIGN4,
            )
            kv_t = tl.where(combined_mask_t, kv_t, 0.0)
            acc = acc * alpha + tl.sum(kv_t * p_t, axis=0)
            m_i = m_new
        d_safe = tl.where(d_i > 0.0, d_i, 1.0)
        compressed_kv = tl.where(d_i > 0.0, acc / d_safe, 0.0)

    rms_w = tl.load(rms_norm_weight_ptr + block, mask=mask, other=0.0)
    variance = tl.sum(compressed_kv * compressed_kv, axis=0) / HEAD_SIZE
    rrms = tl.rsqrt(variance + rms_norm_eps)
    return compressed_kv * rrms * rms_w


@triton.jit
def _quant_rope_store_584b(
    normed,
    position,
    kv_block_idx,
    kv_pos_in_block,
    cos_sin_cache_ptr,
    cos_sin_stride,
    k_cache_ptr,
    kv_cache_block_size,
    block,
    mask,
    HEAD_SIZE: tl.constexpr,
    TRITON_BLOCK_SIZE: tl.constexpr,
    COMPRESS_RATIO: tl.constexpr,
    ROPE_HEAD_DIM: tl.constexpr,
    FP8_MAX: tl.constexpr,
    QUANT_BLOCK: tl.constexpr,
    TOKEN_STRIDE: tl.constexpr,
    SCALE_DIM: tl.constexpr,
    KV_BLOCK_STRIDE,
):
    cache_block_ptr = k_cache_ptr + kv_block_idx.to(tl.int64) * KV_BLOCK_STRIDE.to(
        tl.int64
    )
    fp8_ptr = cache_block_ptr + kv_pos_in_block * TOKEN_STRIDE
    scale_ptr = (
        cache_block_ptr
        + kv_cache_block_size * TOKEN_STRIDE
        + kv_pos_in_block * SCALE_DIM
    )

    NOPE_HEAD_DIM: tl.constexpr = HEAD_SIZE - ROPE_HEAD_DIM
    HALF_ROPE: tl.constexpr = ROPE_HEAD_DIM // 2

    N_QUANT_BLOCKS: tl.constexpr = TRITON_BLOCK_SIZE // QUANT_BLOCK
    N_NOPE_BLOCKS: tl.constexpr = NOPE_HEAD_DIM // QUANT_BLOCK
    INV_FP8_MAX: tl.constexpr = 1.0 / FP8_MAX

    quant_input = normed.to(tl.bfloat16).to(tl.float32)
    quant_2d = tl.reshape(quant_input, (N_QUANT_BLOCKS, QUANT_BLOCK))
    abs_2d = tl.abs(quant_2d)
    block_absmax = tl.max(abs_2d, axis=1)
    block_absmax = tl.maximum(block_absmax, 1e-4)

    raw_scales = block_absmax * INV_FP8_MAX
    exponents = tl.ceil(tl.log2(raw_scales))
    inv_scales = tl.exp2(-exponents)
    inv_scales_col = tl.reshape(inv_scales, (N_QUANT_BLOCKS, 1))
    x_scaled = quant_2d * inv_scales_col
    x_clamped = tl.clamp(x_scaled, -FP8_MAX, FP8_MAX)
    x_fp8 = x_clamped.to(tl.float8e4nv)
    x_uint8 = x_fp8.to(tl.uint8, bitcast=True)
    x_uint8_flat = tl.reshape(x_uint8, (TRITON_BLOCK_SIZE,))

    nope_mask = block < NOPE_HEAD_DIM
    tl.store(fp8_ptr + block, x_uint8_flat, mask=nope_mask)

    scale_idx = tl.arange(0, N_QUANT_BLOCKS)
    encoded = exponents + 127.0
    encoded = tl.maximum(tl.minimum(encoded, 255.0), 0.0)
    tl.store(
        scale_ptr + scale_idx,
        encoded.to(tl.uint8),
        mask=scale_idx < N_NOPE_BLOCKS,
    )
    tl.store(scale_ptr + N_NOPE_BLOCKS, tl.zeros((), dtype=tl.uint8))

    NUM_PAIRS: tl.constexpr = TRITON_BLOCK_SIZE // 2
    NOPE_PAIRS: tl.constexpr = NOPE_HEAD_DIM // 2

    pair_2d = tl.reshape(normed, (NUM_PAIRS, 2))
    even, odd = tl.split(pair_2d)

    pair_idx = tl.arange(0, NUM_PAIRS)
    rope_pair_local = pair_idx - NOPE_PAIRS
    is_rope_pair = rope_pair_local >= 0
    cs_idx = tl.maximum(rope_pair_local, 0)

    compressed_pos = (position // COMPRESS_RATIO) * COMPRESS_RATIO
    cache_base = cos_sin_cache_ptr + compressed_pos * cos_sin_stride
    cos_v = tl.load(cache_base + cs_idx, mask=is_rope_pair, other=1.0)
    sin_v = tl.load(cache_base + HALF_ROPE + cs_idx, mask=is_rope_pair, other=0.0)

    new_even = even * cos_v - odd * sin_v
    new_odd = odd * cos_v + even * sin_v
    result = tl.interleave(new_even, new_odd)

    bf16_ptr = (fp8_ptr + NOPE_HEAD_DIM).to(tl.pointer_type(tl.bfloat16))
    rope_local = block - NOPE_HEAD_DIM
    is_rope = (block >= NOPE_HEAD_DIM) & mask
    tl.store(bf16_ptr + rope_local, result.to(tl.bfloat16), mask=is_rope)


_HCA_NUM_T_SPLITS = 4
_HCA_SPLIT_T = 32
_HCA_CHUNK_D = 64
_HCA_D_CHUNKS = 8
_HCA_SPLIT_WS_CACHE: dict[tuple[int, int], torch.Tensor] = {}


def _get_hca_split_ws(
    num_work: int, head_dim: int, device: torch.device
) -> torch.Tensor:
    needed = 3 * num_work * _HCA_NUM_T_SPLITS * head_dim
    if device.type == "cuda" and torch.cuda.is_current_stream_capturing():
        return torch.empty(
            (3, num_work, _HCA_NUM_T_SPLITS, head_dim),
            dtype=torch.float32,
            device=device,
        )
    stream_id = (
        int(torch.cuda.current_stream(device).cuda_stream)
        if device.type == "cuda"
        else 0
    )
    key = (device.index if device.index is not None else 0, stream_id)
    buf = _HCA_SPLIT_WS_CACHE.get(key)
    if buf is None or int(buf.numel()) < needed:
        buf = torch.empty((needed,), dtype=torch.float32, device=device)
        _HCA_SPLIT_WS_CACHE[key] = buf
    return buf[:needed].view(3, num_work, _HCA_NUM_T_SPLITS, head_dim)


@triton.jit(
    do_not_specialize=[
        "num_work",
        "ws_plane",
        "kv_raw_stride",
        "score_raw_stride",
        "ape_stride",
        "state_cache_stride0",
        "state_cache_stride1",
        "block_table_stride",
        "block_size",
        "seq_start",
        "n_raw",
        "kv_cache_block_size",
        "NUM_STATE_BLOCKS",
    ]
)
def _hca_compress_split_stage1_kernel(
    state_cache_ptr,
    state_cache_stride0,
    state_cache_stride1,
    token_to_req_indices_ptr,
    positions_ptr,
    block_table_ptr,
    block_table_stride,
    block_size,
    kv_raw_ptr,
    kv_raw_stride,
    score_raw_ptr,
    score_raw_stride,
    ape_ptr,
    ape_stride,
    seq_start,
    n_raw,
    seq_start_per_req_ptr,
    cu_seq_per_req_ptr,
    kv_slot_mapping_ptr,
    kv_cache_block_size,
    unpad_restore_ptr,
    active_indices_ptr,
    ws_ptr,
    ws_plane,
    num_work,
    HEAD_SIZE: tl.constexpr,
    STATE_WIDTH: tl.constexpr,
    COMPRESS_RATIO: tl.constexpr,
    NUM_T_SPLITS: tl.constexpr,
    SPLIT_T: tl.constexpr,
    CHUNK_D: tl.constexpr,
    D_CHUNKS: tl.constexpr,
    NUM_STATE_BLOCKS,
    BATCHED: tl.constexpr,
    TRAP_INVALID_KV_ACCESS: tl.constexpr,
    STATE_RING_ENTRIES: tl.constexpr,
    MAY_USE_CACHE: tl.constexpr,
    USE_UNPAD_RESTORE: tl.constexpr,
    USE_ACTIVE_INDICES: tl.constexpr,
    ALIGN4: tl.constexpr,
):
    row_id = tl.program_id(0).to(tl.int64)
    if row_id >= num_work:
        return
    if USE_ACTIVE_INDICES:
        token_idx = tl.load(active_indices_ptr + row_id).to(tl.int64)
    else:
        token_idx = row_id

    kv_slot_idx = tl.load(kv_slot_mapping_ptr + token_idx)
    if kv_slot_idx < 0:
        return

    position = tl.load(positions_ptr + token_idx).to(tl.int64)
    if (position + 1) % COMPRESS_RATIO != 0:
        return

    pid_sd = tl.program_id(1)
    split_t = pid_sd // D_CHUNKS
    d_chunk = pid_sd % D_CHUNKS

    req_idx = tl.load(token_to_req_indices_ptr + token_idx).to(tl.int64)
    if BATCHED:
        req_seq_start = tl.load(seq_start_per_req_ptr + req_idx).to(tl.int64)
        req_cu_lo = tl.load(cu_seq_per_req_ptr + req_idx).to(tl.int64)
        req_cu_hi = tl.load(cu_seq_per_req_ptr + req_idx + 1).to(tl.int64)
        req_n_raw = req_cu_hi - req_cu_lo
    else:
        req_seq_start = tl.zeros((), dtype=tl.int64) + seq_start
        req_cu_lo = tl.zeros((), dtype=tl.int64)
        req_n_raw = tl.zeros((), dtype=tl.int64) + n_raw

    start = position - COMPRESS_RATIO + 1
    tokens = split_t * SPLIT_T + tl.arange(0, SPLIT_T)
    cur_pos = start + tokens.to(tl.int64)
    mask_pos = cur_pos >= 0
    flat_idx_in_req = cur_pos - req_seq_start
    use_raw = mask_pos & (flat_idx_in_req >= 0) & (flat_idx_in_req < req_n_raw)
    flat_idx_safe = tl.where(use_raw, req_cu_lo + flat_idx_in_req, 0)

    if USE_UNPAD_RESTORE:
        raw_row_idx = tl.load(
            unpad_restore_ptr + flat_idx_safe, mask=use_raw, other=0
        ).to(tl.int64)
    else:
        raw_row_idx = flat_idx_safe

    kv_s = (
        tl.multiple_of(kv_raw_stride.to(tl.int64), 4)
        if ALIGN4
        else kv_raw_stride.to(tl.int64)
    )
    sc_s = (
        tl.multiple_of(score_raw_stride.to(tl.int64), 4)
        if ALIGN4
        else score_raw_stride.to(tl.int64)
    )
    ap_s = (
        tl.multiple_of(ape_stride.to(tl.int64), 4)
        if ALIGN4
        else ape_stride.to(tl.int64)
    )

    raw_score_ptrs = score_raw_ptr + raw_row_idx * sc_s
    raw_kv_ptrs = kv_raw_ptr + raw_row_idx * kv_s

    if MAY_USE_CACHE:
        use_cache = mask_pos & ~use_raw
        block_indices = (cur_pos // block_size) % tl.maximum(block_table_stride, 1)
        block_indices_safe = tl.where(use_cache, block_indices, 0)
        block_numbers = tl.load(
            block_table_ptr + req_idx * block_table_stride + block_indices_safe,
            mask=use_cache,
            other=0,
        )
        invalid_state_block = use_cache & (block_numbers >= NUM_STATE_BLOCKS)
        if tl.max(invalid_state_block.to(tl.int32), axis=0) != 0:
            _trap_invalid_kv_access(TRAP_INVALID_KV_ACCESS)
        valid_block = use_cache & (block_numbers > 0)
        block_numbers_i64 = tl.where(valid_block, block_numbers, 0).to(tl.int64)
        block_offsets = tl.where(mask_pos, cur_pos % STATE_RING_ENTRIES, 0).to(tl.int64)
        st_s0 = (
            tl.multiple_of(state_cache_stride0.to(tl.int64), 4)
            if ALIGN4
            else state_cache_stride0.to(tl.int64)
        )
        st_s1 = (
            tl.multiple_of(state_cache_stride1.to(tl.int64), 4)
            if ALIGN4
            else state_cache_stride1.to(tl.int64)
        )
        cache_kv_ptrs = (
            state_cache_ptr + block_numbers_i64 * st_s0 + block_offsets * st_s1
        )
        cache_score_ptrs = cache_kv_ptrs + STATE_WIDTH
        load_mask_1d = use_raw | valid_block
        score_row_ptrs = tl.where(use_raw, raw_score_ptrs, cache_score_ptrs)
        kv_row_ptrs = tl.where(use_raw, raw_kv_ptrs, cache_kv_ptrs)
    else:
        load_mask_1d = use_raw
        score_row_ptrs = raw_score_ptrs
        kv_row_ptrs = raw_kv_ptrs

    d_offs = d_chunk * CHUNK_D + tl.arange(0, CHUNK_D)
    if ALIGN4:
        d_offs = tl.multiple_of(d_offs, 4)

    load_mask_2d = load_mask_1d[:, None]
    score = tl.load(
        score_row_ptrs[:, None] + d_offs[None, :],
        mask=load_mask_2d,
        other=0.0,
        eviction_policy="evict_first",
    ).to(tl.float32)
    ape_rows_safe = tl.where(use_raw, cur_pos % COMPRESS_RATIO, 0)
    ape_vals = tl.load(
        ape_ptr + ape_rows_safe[:, None] * ap_s + d_offs[None, :],
        mask=use_raw[:, None],
        other=0.0,
    ).to(tl.float32)
    score = tl.where(load_mask_2d, score + ape_vals, float("-inf"))

    m_s = tl.max(score, axis=0)
    m_safe = tl.where(m_s > float("-inf"), m_s, 0.0)
    p_s = tl.where(load_mask_2d, tl.exp(score - m_safe[None, :]), 0.0)
    l_s = tl.sum(p_s, axis=0)

    kv_tile = tl.load(
        kv_row_ptrs[:, None] + d_offs[None, :],
        mask=load_mask_2d,
        other=0.0,
        eviction_policy="evict_first",
    ).to(tl.float32)
    acc_s = tl.sum(p_s * kv_tile, axis=0)

    ws_off = (row_id * NUM_T_SPLITS + split_t) * HEAD_SIZE + d_offs
    if ALIGN4:
        ws_off = tl.multiple_of(ws_off, 4)
    plane = ws_plane.to(tl.int64)
    tl.store(ws_ptr + ws_off, m_s)
    tl.store(ws_ptr + plane + ws_off, l_s)
    tl.store(ws_ptr + 2 * plane + ws_off, acc_s)


@triton.jit(
    do_not_specialize=[
        "num_work",
        "ws_plane",
        "kv_cache_block_size",
        "KV_BLOCK_STRIDE",
        "NUM_KV_BLOCKS",
        "cos_sin_stride",
    ]
)
def _hca_compress_split_finalize_kernel(
    positions_ptr,
    rms_norm_weight_ptr,
    rms_norm_eps,
    cos_sin_cache_ptr,
    cos_sin_stride,
    k_cache_ptr,
    kv_slot_mapping_ptr,
    kv_cache_block_size,
    active_indices_ptr,
    ws_ptr,
    ws_plane,
    num_work,
    HEAD_SIZE: tl.constexpr,
    TRITON_BLOCK_SIZE: tl.constexpr,
    COMPRESS_RATIO: tl.constexpr,
    ROPE_HEAD_DIM: tl.constexpr,
    FP8_MAX: tl.constexpr,
    QUANT_BLOCK: tl.constexpr,
    TOKEN_STRIDE: tl.constexpr,
    SCALE_DIM: tl.constexpr,
    NUM_T_SPLITS: tl.constexpr,
    KV_BLOCK_STRIDE,
    NUM_KV_BLOCKS,
    TRAP_INVALID_KV_ACCESS: tl.constexpr,
    USE_ACTIVE_INDICES: tl.constexpr,
    ALIGN4: tl.constexpr,
):
    row_id = tl.program_id(0).to(tl.int64)
    if row_id >= num_work:
        return
    if USE_ACTIVE_INDICES:
        token_idx = tl.load(active_indices_ptr + row_id).to(tl.int64)
    else:
        token_idx = row_id

    kv_slot_idx = tl.load(kv_slot_mapping_ptr + token_idx)
    if kv_slot_idx < 0:
        return

    position = tl.load(positions_ptr + token_idx).to(tl.int64)
    if (position + 1) % COMPRESS_RATIO != 0:
        return

    kv_block_idx = kv_slot_idx // kv_cache_block_size
    kv_pos_in_block = kv_slot_idx % kv_cache_block_size
    if kv_block_idx < 0:
        _trap_invalid_kv_access(TRAP_INVALID_KV_ACCESS)
    if kv_block_idx >= NUM_KV_BLOCKS:
        _trap_invalid_kv_access(TRAP_INVALID_KV_ACCESS)

    block = tl.arange(0, TRITON_BLOCK_SIZE)
    if ALIGN4:
        block = tl.multiple_of(block, 4)
    mask = block < HEAD_SIZE

    plane = ws_plane.to(tl.int64)
    row_ws_base = row_id * NUM_T_SPLITS * HEAD_SIZE

    m_g = tl.full((TRITON_BLOCK_SIZE,), float("-inf"), dtype=tl.float32)
    l_g = tl.zeros((TRITON_BLOCK_SIZE,), dtype=tl.float32)
    acc_g = tl.zeros((TRITON_BLOCK_SIZE,), dtype=tl.float32)

    for s in tl.static_range(NUM_T_SPLITS):
        ws_off = row_ws_base + s * HEAD_SIZE + block
        m_s = tl.load(ws_ptr + ws_off, eviction_policy="evict_first")
        l_s = tl.load(ws_ptr + plane + ws_off, eviction_policy="evict_first")
        acc_s = tl.load(ws_ptr + 2 * plane + ws_off, eviction_policy="evict_first")
        m_new = tl.maximum(m_g, m_s)
        m_new_safe = tl.where(m_new > float("-inf"), m_new, 0.0)
        alpha = tl.where(m_g > float("-inf"), tl.exp(m_g - m_new_safe), 0.0)
        beta = tl.where(m_s > float("-inf"), tl.exp(m_s - m_new_safe), 0.0)
        l_g = l_g * alpha + l_s * beta
        acc_g = acc_g * alpha + acc_s * beta
        m_g = m_new

    d_safe = tl.where(l_g > 0.0, l_g, 1.0)
    compressed_kv = tl.where(l_g > 0.0, acc_g / d_safe, 0.0)

    rms_w = tl.load(rms_norm_weight_ptr + block, mask=mask, other=0.0)
    variance = tl.sum(compressed_kv * compressed_kv, axis=0) / HEAD_SIZE
    rrms = tl.rsqrt(variance + rms_norm_eps)
    normed = compressed_kv * rrms * rms_w

    _quant_rope_store_584b(
        normed,
        position,
        kv_block_idx,
        kv_pos_in_block,
        cos_sin_cache_ptr,
        cos_sin_stride,
        k_cache_ptr,
        kv_cache_block_size,
        block,
        mask,
        HEAD_SIZE=HEAD_SIZE,
        TRITON_BLOCK_SIZE=TRITON_BLOCK_SIZE,
        COMPRESS_RATIO=COMPRESS_RATIO,
        ROPE_HEAD_DIM=ROPE_HEAD_DIM,
        FP8_MAX=FP8_MAX,
        QUANT_BLOCK=QUANT_BLOCK,
        TOKEN_STRIDE=TOKEN_STRIDE,
        SCALE_DIM=SCALE_DIM,
        KV_BLOCK_STRIDE=KV_BLOCK_STRIDE,
    )


# =============================================================================
# DeepseekV4 attention path: head_dim=512 (CSA / HCA), 584B per slot
# =============================================================================
@triton.jit(
    do_not_specialize=[
        "num_work",
        "kv_raw_stride",
        "score_raw_stride",
        "ape_stride",
        "state_cache_stride0",
        "state_cache_stride1",
        "block_table_stride",
        "seq_start",
        "n_raw",
        "kv_cache_block_size",
        "KV_BLOCK_STRIDE",
        "NUM_STATE_BLOCKS",
        "NUM_KV_BLOCKS",
    ]
)
def _fused_kv_compress_norm_rope_insert_sparse_attn(
    state_cache_ptr,
    state_cache_stride0,
    state_cache_stride1,
    token_to_req_indices_ptr,
    positions_ptr,
    slot_mapping_ptr,
    block_table_ptr,
    block_table_stride,
    block_size,
    kv_raw_ptr,
    kv_raw_stride,
    score_raw_ptr,
    score_raw_stride,
    ape_ptr,
    ape_stride,
    seq_start,
    n_raw,
    seq_start_per_req_ptr,
    cu_seq_per_req_ptr,
    rms_norm_weight_ptr,
    rms_norm_eps,
    cos_sin_cache_ptr,
    cos_sin_stride,
    k_cache_ptr,
    kv_slot_mapping_ptr,
    kv_cache_block_size,
    unpad_restore_ptr,
    active_indices_ptr,
    num_work,
    HEAD_SIZE: tl.constexpr,
    TRITON_BLOCK_SIZE: tl.constexpr,
    STATE_WIDTH: tl.constexpr,
    COMPRESS_RATIO: tl.constexpr,
    OVERLAP: tl.constexpr,
    ROPE_HEAD_DIM: tl.constexpr,
    FP8_MAX: tl.constexpr,
    QUANT_BLOCK: tl.constexpr,
    TOKEN_STRIDE: tl.constexpr,
    SCALE_DIM: tl.constexpr,
    KV_BLOCK_STRIDE,
    NUM_STATE_BLOCKS,
    NUM_KV_BLOCKS,
    BATCHED: tl.constexpr,
    TRAP_INVALID_KV_ACCESS: tl.constexpr,
    STATE_RING_ENTRIES: tl.constexpr,
    MAY_USE_CACHE: tl.constexpr,
    USE_UNPAD_RESTORE: tl.constexpr,
    USE_ACTIVE_INDICES: tl.constexpr,
    TILE_T: tl.constexpr,
    ALIGN4: tl.constexpr,
):
    row_id = tl.program_id(0).to(tl.int64)
    if row_id >= num_work:
        return
    if USE_ACTIVE_INDICES:
        token_idx = tl.load(active_indices_ptr + row_id).to(tl.int64)
    else:
        token_idx = row_id

    # Early exit before position/raw/cache loads: skips non-boundary tokens and
    # CP-non-owned boundary tokens (sharded CP sets kv_slot_mapping=-1).
    kv_slot_idx = tl.load(kv_slot_mapping_ptr + token_idx)
    if kv_slot_idx < 0:
        return

    position = tl.load(positions_ptr + token_idx).to(tl.int64)
    if (position + 1) % COMPRESS_RATIO != 0:
        return

    kv_block_idx = kv_slot_idx // kv_cache_block_size
    kv_pos_in_block = kv_slot_idx % kv_cache_block_size
    if kv_block_idx < 0:
        _trap_invalid_kv_access(TRAP_INVALID_KV_ACCESS)
    if kv_block_idx >= NUM_KV_BLOCKS:
        _trap_invalid_kv_access(TRAP_INVALID_KV_ACCESS)

    req_idx = tl.load(token_to_req_indices_ptr + token_idx).to(tl.int64)
    block = tl.arange(0, TRITON_BLOCK_SIZE)
    if ALIGN4:
        block = tl.multiple_of(block, 4)
    mask = block < HEAD_SIZE

    normed = _compress_and_rmsnorm(
        state_cache_ptr,
        state_cache_stride0,
        state_cache_stride1,
        req_idx,
        position,
        block_table_ptr,
        block_table_stride,
        block_size,
        kv_raw_ptr,
        kv_raw_stride,
        score_raw_ptr,
        score_raw_stride,
        ape_ptr,
        ape_stride,
        unpad_restore_ptr,
        seq_start,
        n_raw,
        seq_start_per_req_ptr,
        cu_seq_per_req_ptr,
        rms_norm_weight_ptr,
        rms_norm_eps,
        block,
        mask,
        HEAD_SIZE=HEAD_SIZE,
        TRITON_BLOCK_SIZE=TRITON_BLOCK_SIZE,
        STATE_WIDTH=STATE_WIDTH,
        COMPRESS_RATIO=COMPRESS_RATIO,
        OVERLAP=OVERLAP,
        NUM_STATE_BLOCKS=NUM_STATE_BLOCKS,
        BATCHED=BATCHED,
        TRAP_INVALID_KV_ACCESS=TRAP_INVALID_KV_ACCESS,
        STATE_RING_ENTRIES=STATE_RING_ENTRIES,
        MAY_USE_CACHE=MAY_USE_CACHE,
        USE_UNPAD_RESTORE=USE_UNPAD_RESTORE,
        TILE_T=TILE_T,
        ALIGN4=ALIGN4,
    )

    _quant_rope_store_584b(
        normed,
        position,
        kv_block_idx,
        kv_pos_in_block,
        cos_sin_cache_ptr,
        cos_sin_stride,
        k_cache_ptr,
        kv_cache_block_size,
        block,
        mask,
        HEAD_SIZE=HEAD_SIZE,
        TRITON_BLOCK_SIZE=TRITON_BLOCK_SIZE,
        COMPRESS_RATIO=COMPRESS_RATIO,
        ROPE_HEAD_DIM=ROPE_HEAD_DIM,
        FP8_MAX=FP8_MAX,
        QUANT_BLOCK=QUANT_BLOCK,
        TOKEN_STRIDE=TOKEN_STRIDE,
        SCALE_DIM=SCALE_DIM,
        KV_BLOCK_STRIDE=KV_BLOCK_STRIDE,
    )


# =============================================================================
# Indexer path: head_dim=128, 4 boundary tokens/CTA, single quant block, 132B
# =============================================================================
_INDEXER_TOKENS_PER_CTA = 4


@triton.jit(
    do_not_specialize=[
        "num_work",
        "kv_raw_stride",
        "score_raw_stride",
        "ape_stride",
        "state_cache_stride0",
        "state_cache_stride1",
        "block_table_stride",
        "seq_start",
        "n_raw",
        "kv_cache_block_size",
        "KV_BLOCK_STRIDE",
        "NUM_STATE_BLOCKS",
        "NUM_KV_BLOCKS",
    ]
)
def _fused_kv_compress_norm_rope_insert_indexer_attn(
    state_cache_ptr,
    state_cache_stride0,
    state_cache_stride1,
    token_to_req_indices_ptr,
    positions_ptr,
    slot_mapping_ptr,
    block_table_ptr,
    block_table_stride,
    block_size,
    kv_raw_ptr,
    kv_raw_stride,
    score_raw_ptr,
    score_raw_stride,
    ape_ptr,
    ape_stride,
    seq_start,
    n_raw,
    seq_start_per_req_ptr,
    cu_seq_per_req_ptr,
    rms_norm_weight_ptr,
    rms_norm_eps,
    cos_sin_cache_ptr,
    cos_sin_stride,
    k_cache_ptr,
    kv_slot_mapping_ptr,
    kv_cache_block_size,
    unpad_restore_ptr,
    active_indices_ptr,
    num_work,
    HEAD_SIZE: tl.constexpr,
    TRITON_BLOCK_SIZE: tl.constexpr,
    STATE_WIDTH: tl.constexpr,
    COMPRESS_RATIO: tl.constexpr,
    OVERLAP: tl.constexpr,
    ROPE_HEAD_DIM: tl.constexpr,
    FP8_MAX: tl.constexpr,
    QUANT_BLOCK: tl.constexpr,
    TOKEN_STRIDE: tl.constexpr,
    SCALE_DIM: tl.constexpr,
    KV_BLOCK_STRIDE,
    NUM_STATE_BLOCKS,
    NUM_KV_BLOCKS,
    BATCHED: tl.constexpr,
    TRAP_INVALID_KV_ACCESS: tl.constexpr,
    STATE_RING_ENTRIES: tl.constexpr,
    MAY_USE_CACHE: tl.constexpr,
    USE_UNPAD_RESTORE: tl.constexpr,
    USE_ACTIVE_INDICES: tl.constexpr,
    BLOCK_M: tl.constexpr,
    ALIGN4: tl.constexpr,
):
    row_offsets = tl.program_id(0).to(tl.int64) * BLOCK_M + tl.arange(0, BLOCK_M)
    tok_valid = row_offsets < num_work
    if USE_ACTIVE_INDICES:
        token_idx = tl.load(
            active_indices_ptr + row_offsets, mask=tok_valid, other=0
        ).to(tl.int64)
    else:
        token_idx = row_offsets

    kv_slot_idx = tl.load(kv_slot_mapping_ptr + token_idx, mask=tok_valid, other=-1).to(
        tl.int64
    )
    position = tl.load(positions_ptr + token_idx, mask=tok_valid, other=0).to(tl.int64)
    row_active = (
        tok_valid & (kv_slot_idx >= 0) & (((position + 1) % COMPRESS_RATIO) == 0)
    )
    if tl.max(row_active.to(tl.int32), axis=0) == 0:
        return

    kv_block_idx = tl.where(row_active, kv_slot_idx // kv_cache_block_size, 0)
    kv_pos_in_block = tl.where(row_active, kv_slot_idx % kv_cache_block_size, 0)
    bad_kv_block = row_active & ((kv_block_idx < 0) | (kv_block_idx >= NUM_KV_BLOCKS))
    if tl.max(bad_kv_block.to(tl.int32), axis=0) != 0:
        _trap_invalid_kv_access(TRAP_INVALID_KV_ACCESS)

    req_idx = tl.load(
        token_to_req_indices_ptr + token_idx, mask=row_active, other=0
    ).to(tl.int64)
    if BATCHED:
        req_seq_start = tl.load(
            seq_start_per_req_ptr + req_idx, mask=row_active, other=0
        ).to(tl.int64)
        req_cu_lo = tl.load(cu_seq_per_req_ptr + req_idx, mask=row_active, other=0).to(
            tl.int64
        )
        req_cu_hi = tl.load(
            cu_seq_per_req_ptr + req_idx + 1, mask=row_active, other=0
        ).to(tl.int64)
        req_n_raw = req_cu_hi - req_cu_lo
    else:
        req_seq_start = tl.zeros((BLOCK_M,), dtype=tl.int64) + seq_start
        req_cu_lo = tl.zeros((BLOCK_M,), dtype=tl.int64)
        req_n_raw = tl.zeros((BLOCK_M,), dtype=tl.int64) + n_raw

    col = tl.arange(0, TRITON_BLOCK_SIZE)
    if ALIGN4:
        col = tl.multiple_of(col, 4)
    col_mask = col < HEAD_SIZE

    kv_s = (
        tl.multiple_of(kv_raw_stride.to(tl.int64), 4)
        if ALIGN4
        else kv_raw_stride.to(tl.int64)
    )
    sc_s = (
        tl.multiple_of(score_raw_stride.to(tl.int64), 4)
        if ALIGN4
        else score_raw_stride.to(tl.int64)
    )
    ap_s = (
        tl.multiple_of(ape_stride.to(tl.int64), 4)
        if ALIGN4
        else ape_stride.to(tl.int64)
    )
    st_s0 = (
        tl.multiple_of(state_cache_stride0.to(tl.int64), 4)
        if ALIGN4
        else state_cache_stride0.to(tl.int64)
    )
    st_s1 = (
        tl.multiple_of(state_cache_stride1.to(tl.int64), 4)
        if ALIGN4
        else state_cache_stride1.to(tl.int64)
    )

    WINDOW_SIZE: tl.constexpr = (1 + OVERLAP) * COMPRESS_RATIO
    start = position - WINDOW_SIZE + 1

    m_i = tl.full((BLOCK_M, TRITON_BLOCK_SIZE), float("-inf"), dtype=tl.float32)
    d_i = tl.zeros((BLOCK_M, TRITON_BLOCK_SIZE), dtype=tl.float32)
    acc = tl.zeros((BLOCK_M, TRITON_BLOCK_SIZE), dtype=tl.float32)

    for step in tl.range(0, WINDOW_SIZE, num_stages=1):
        cur_pos = start + step
        mask_pos = row_active & (cur_pos >= 0)
        head_offset = tl.where(step >= COMPRESS_RATIO, HEAD_SIZE, 0)
        flat_idx_in_req = cur_pos - req_seq_start
        use_raw = mask_pos & (flat_idx_in_req >= 0) & (flat_idx_in_req < req_n_raw)
        flat_idx_safe = tl.where(use_raw, req_cu_lo + flat_idx_in_req, 0)

        if USE_UNPAD_RESTORE:
            raw_row_idx = tl.load(
                unpad_restore_ptr + flat_idx_safe, mask=use_raw, other=0
            ).to(tl.int64)
        else:
            raw_row_idx = flat_idx_safe

        raw_score_base = score_raw_ptr + raw_row_idx * sc_s + head_offset
        raw_kv_base = kv_raw_ptr + raw_row_idx * kv_s + head_offset
        if MAY_USE_CACHE:
            use_cache = mask_pos & ~use_raw
            block_indices = (cur_pos // block_size) % tl.maximum(block_table_stride, 1)
            block_indices_safe = tl.where(use_cache, block_indices, 0)
            block_numbers = tl.load(
                block_table_ptr + req_idx * block_table_stride + block_indices_safe,
                mask=use_cache,
                other=0,
            )
            invalid_state_block = use_cache & (block_numbers >= NUM_STATE_BLOCKS)
            if tl.max(invalid_state_block.to(tl.int32), axis=0) != 0:
                _trap_invalid_kv_access(TRAP_INVALID_KV_ACCESS)
            valid_block = use_cache & (block_numbers > 0)
            block_numbers_i64 = tl.where(valid_block, block_numbers, 0).to(tl.int64)
            block_offsets = tl.where(mask_pos, cur_pos % STATE_RING_ENTRIES, 0).to(
                tl.int64
            )
            cache_base = (
                state_cache_ptr
                + block_numbers_i64 * st_s0
                + block_offsets * st_s1
                + head_offset
            )
            final_valid = use_raw | (use_cache & valid_block)
            load_mask = (use_raw | valid_block)[:, None] & col_mask[None, :]
            score_row_base = tl.where(use_raw, raw_score_base, cache_base + STATE_WIDTH)
            kv_row_base = tl.where(use_raw, raw_kv_base, cache_base)
        else:
            final_valid = use_raw
            load_mask = use_raw[:, None] & col_mask[None, :]
            score_row_base = raw_score_base
            kv_row_base = raw_kv_base

        # Score + ape -> softmax weights BEFORE loading KV.
        score_t = tl.load(
            score_row_base[:, None] + col[None, :], mask=load_mask, other=0.0
        ).to(tl.float32)
        ape_rows_safe = tl.where(use_raw, cur_pos % COMPRESS_RATIO, 0)
        ape_t = tl.load(
            ape_ptr + ape_rows_safe[:, None] * ap_s + head_offset + col[None, :],
            mask=use_raw[:, None] & col_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        score_t = score_t + ape_t
        tile_mask = final_valid[:, None] & col_mask[None, :]
        score_t = tl.where(tile_mask, score_t, float("-inf"))
        m_new = tl.maximum(m_i, score_t)
        m_new_safe = tl.where(m_new > float("-inf"), m_new, 0.0)
        alpha = tl.where(m_i > float("-inf"), tl.exp(m_i - m_new_safe), 0.0)
        p_t = tl.where(tile_mask, tl.exp(score_t - m_new_safe), 0.0)
        d_i = d_i * alpha + p_t

        kv_t = tl.load(
            kv_row_base[:, None] + col[None, :], mask=load_mask, other=0.0
        ).to(tl.float32)
        kv_t = tl.where(tile_mask, kv_t, 0.0)
        acc = acc * alpha + kv_t * p_t
        m_i = m_new

    d_safe = tl.where(d_i > 0.0, d_i, 1.0)
    compressed_kv = tl.where(d_i > 0.0, acc / d_safe, 0.0)

    rms_w = tl.load(rms_norm_weight_ptr + col, mask=col_mask, other=0.0)
    variance = tl.sum(compressed_kv * compressed_kv, axis=1) / HEAD_SIZE
    rrms = tl.rsqrt(variance + rms_norm_eps)
    normed = compressed_kv * rrms[:, None] * rms_w[None, :]

    if ROPE_HEAD_DIM > 0:
        NOPE_HEAD_DIM: tl.constexpr = HEAD_SIZE - ROPE_HEAD_DIM
        HALF_ROPE: tl.constexpr = ROPE_HEAD_DIM // 2
        NUM_PAIRS: tl.constexpr = TRITON_BLOCK_SIZE // 2
        NOPE_PAIRS: tl.constexpr = NOPE_HEAD_DIM // 2

        normed_pairs = tl.reshape(normed, (BLOCK_M, NUM_PAIRS, 2))
        even, odd = tl.split(normed_pairs)

        pair_idx = tl.arange(0, NUM_PAIRS)
        rope_pair_local = pair_idx - NOPE_PAIRS
        is_rope_pair = rope_pair_local >= 0
        cs_idx = tl.maximum(rope_pair_local, 0)

        compressed_pos = (position // COMPRESS_RATIO) * COMPRESS_RATIO
        cache_base = cos_sin_cache_ptr + compressed_pos[:, None] * cos_sin_stride
        rope_mask = row_active[:, None] & is_rope_pair[None, :]
        cos_v = tl.load(cache_base + cs_idx[None, :], mask=rope_mask, other=1.0)
        sin_v = tl.load(
            cache_base + HALF_ROPE + cs_idx[None, :], mask=rope_mask, other=0.0
        )

        new_even = even * cos_v - odd * sin_v
        new_odd = odd * cos_v + even * sin_v
        result = tl.interleave(new_even, new_odd)
    else:
        result = normed

    tl.static_assert(
        TRITON_BLOCK_SIZE == QUANT_BLOCK,
        "Indexer expects one quant block (QUANT_BLOCK == TRITON_BLOCK_SIZE)",
    )
    INV_FP8_MAX: tl.constexpr = 1.0 / FP8_MAX

    result_bf16 = result.to(tl.bfloat16).to(tl.float32)
    absmax = tl.max(tl.abs(result_bf16), axis=1)
    absmax = tl.maximum(absmax, 1e-4)
    raw_scale = absmax * INV_FP8_MAX
    exponent = tl.ceil(tl.log2(raw_scale))
    inv_scale = tl.exp2(-exponent)

    x_scaled = result_bf16 * inv_scale[:, None]
    x_clamped = tl.clamp(x_scaled, -FP8_MAX, FP8_MAX)
    x_fp8 = x_clamped.to(tl.float8e4nv)
    x_uint8 = x_fp8.to(tl.uint8, bitcast=True)

    cache_block_ptrs = k_cache_ptr + kv_block_idx * KV_BLOCK_STRIDE.to(tl.int64)
    fp8_ptrs = cache_block_ptrs + kv_pos_in_block * TOKEN_STRIDE
    store_mask = row_active[:, None] & col_mask[None, :]
    tl.store(fp8_ptrs[:, None] + col[None, :], x_uint8, mask=store_mask)

    scale_ptrs = (
        cache_block_ptrs
        + kv_cache_block_size * TOKEN_STRIDE
        + kv_pos_in_block * SCALE_DIM
    )
    scale_val = tl.exp2(exponent)
    tl.store(scale_ptrs.to(tl.pointer_type(tl.float32)), scale_val, mask=row_active)


# =============================================================================
# Python wrappers
# =============================================================================
_FUSED_CONSTEXPR_BY_HEAD_DIM = {
    512: dict(quant_block=64, token_stride=576, scale_dim=8, num_warps=8),
    128: dict(quant_block=128, token_stride=128, scale_dim=4, num_warps=4),
}


def _fused_launch_config(head_dim: int, compress_ratio: int, overlap: bool, cfg: dict):
    """Select (num_warps, tile_t, num_stages) per compressor flavor."""
    if head_dim == 512:
        window_size = (1 + int(overlap)) * int(compress_ratio)
        if window_size >= 64:
            # HCA streaming online softmax: TILE_T=4 + num_stages=1 cuts register spills.
            return 4, 4, 1
        # CSA non-loop window (WINDOW_SIZE=8): 8 warps = 1 warp/window row.
        return cfg["num_warps"], 4, None
    return cfg["num_warps"], 4, None


def _is_16b_aligned(
    kv: torch.Tensor,
    score: torch.Tensor,
    ape: torch.Tensor,
    state_cache: Optional[torch.Tensor],
    head_size: int,
) -> bool:
    if (head_size % 4) != 0:
        return False
    if (
        (kv.stride(0) % 4) != 0
        or (score.stride(0) % 4) != 0
        or (ape.stride(0) % 4) != 0
        or (kv.data_ptr() % 16) != 0
        or (score.data_ptr() % 16) != 0
        or (ape.data_ptr() % 16) != 0
    ):
        return False
    if state_cache is not None and (
        (state_cache.stride(0) % 4) != 0
        or (state_cache.stride(1) % 4) != 0
        or (state_cache.data_ptr() % 16) != 0
    ):
        return False
    return True


def run_save_partial_states(
    kv: torch.Tensor,  # [N, coff*head_dim] bf16/fp32 (or padded gathered when unpad_restore set)
    score: torch.Tensor,  # [N, coff*head_dim] bf16/fp32 (or padded gathered when unpad_restore set)
    ape: torch.Tensor,  # [compress_ratio, coff*head_dim] fp32
    positions: torch.Tensor,  # [N] int64
    state_cache: torch.Tensor,  # [num_blocks, block_size, 2*coff*head_dim] fp32
    slot_mapping: torch.Tensor,  # [N] int64; -1 = skip
    compress_ratio: int,
    unpad_restore: Optional[torch.Tensor] = None,
    active_indices: Optional[torch.Tensor] = None,
) -> None:
    """Per-token write of (kv | score+ape) into the fp32 state cache."""
    N_total = int(slot_mapping.shape[0])
    if N_total == 0:
        return
    use_active = active_indices is not None
    if use_active:
        assert active_indices is not None
        num_work = int(active_indices.shape[0])
        if num_work == 0:
            return
        active_indices_arg = active_indices.contiguous()
        grid = (num_work,)
    else:
        num_work = N_total
        active_indices_arg = positions
        grid = (triton.cdiv(num_work, _SAVE_STATES_TOKENS_PER_CTA),)

    use_unpad_restore = unpad_restore is not None
    if use_unpad_restore:
        assert unpad_restore is not None
        if int(unpad_restore.shape[0]) != N_total:
            raise ValueError(
                f"unpad_restore length {int(unpad_restore.shape[0])} != slot_mapping length {N_total}"
            )
        unpad_restore_arg = unpad_restore.contiguous()
    else:
        unpad_restore_arg = positions
    head_size = int(kv.shape[-1])
    state_width = int(state_cache.shape[-1] // 2)
    block_size = int(state_cache.shape[1])
    align4 = _is_16b_aligned(kv, score, ape, state_cache, head_size)
    _save_partial_states_kernel[grid](
        kv,
        kv.stride(0),
        score,
        score.stride(0),
        ape,
        ape.stride(0),
        positions,
        state_cache,
        state_cache.stride(0),
        state_cache.stride(1),
        slot_mapping,
        unpad_restore_arg,
        active_indices_arg,
        num_work,
        block_size,
        num_state_blocks=int(state_cache.shape[0]),
        HEAD_SIZE=head_size,
        TRITON_BLOCK_SIZE=triton.next_power_of_2(head_size),
        STATE_WIDTH=state_width,
        COMPRESS_RATIO=compress_ratio,
        TRAP_INVALID_KV_ACCESS=True,
        USE_UNPAD_RESTORE=use_unpad_restore,
        USE_ACTIVE_INDICES=use_active,
        ALIGN4=align4,
        num_warps=1 if head_size <= 256 else (2 if head_size <= 512 else 4),
    )


def run_fused_compress_kv_write(
    state_cache: Optional[
        torch.Tensor
    ],  # [num_blocks, block_size, 2*coff*head_dim] fp32
    token_to_req_indices: torch.Tensor,  # [N] int32
    positions: torch.Tensor,  # [N] int64
    slot_mapping: torch.Tensor,  # [N] int64 — state-pool slot per token
    block_table: Optional[
        torch.Tensor
    ],  # [B, max_state_blocks] int — state-pool block table
    rms_norm_weight: torch.Tensor,  # [head_dim] bf16
    rms_norm_eps: float,
    cos_sin_cache: torch.Tensor,  # [max_pos, rope_head_dim] fp32
    kv_cache: torch.Tensor,  # [num_blocks, kv_block_size, ENTRY_BYTES] uint8 (per-block padded)
    kv_slot_mapping: torch.Tensor,  # [N] int64 — KV-pool slot per token (-1 if non-boundary)
    kv_raw: torch.Tensor,  # [n_raw, (1+overlap)*head_dim], row-strided OK
    score_raw: torch.Tensor,  # [n_raw, (1+overlap)*head_dim], row-strided OK
    ape: torch.Tensor,  # [compress_ratio, (1+overlap)*head_dim] fp32
    seq_start: int,  # absolute position of kv_raw[0] (prefill: == sp_int)
    disable_raw_path: bool = False,  # decode path: skip raw, read cache only
    *,
    head_dim: int,
    rope_head_dim: int,
    compress_ratio: int,
    overlap: bool,
    seq_start_per_req: Optional[torch.Tensor] = None,  # [B] int64
    cu_seq_per_req: Optional[torch.Tensor] = None,  # [B+1] int64
    state_tokens_per_block: int,
    may_use_cache: bool = True,
    unpad_restore: Optional[torch.Tensor] = None,
    active_token_indices: Optional[torch.Tensor] = None,
) -> None:
    """Boundary-token compress→norm→rope→fp8 quant→KV-pool store."""
    N = int(slot_mapping.shape[0])
    if N == 0:
        return
    use_active = active_token_indices is not None
    if use_active:
        assert active_token_indices is not None
        num_work = int(active_token_indices.shape[0])
        if num_work == 0:
            return
        active_indices_arg = active_token_indices.contiguous()
    else:
        num_work = N
        active_indices_arg = positions

    cfg = _FUSED_CONSTEXPR_BY_HEAD_DIM.get(head_dim)
    if cfg is None:
        raise ValueError(f"Unsupported head_dim {head_dim} for fused compressor write")

    if kv_raw.dtype != torch.float32 or score_raw.dtype != torch.float32:
        raise TypeError(
            "run_fused_compress_kv_write requires float32 kv_raw and score_raw, "
            f"got kv_raw={kv_raw.dtype}, score_raw={score_raw.dtype}"
        )

    effective_may_use_cache = bool(may_use_cache)
    if (
        effective_may_use_cache
        and block_table is not None
        and (
            block_table.dim() != 2
            or int(block_table.shape[1]) <= 0
            or int(block_table.stride(0)) <= 0
        )
    ):
        effective_may_use_cache = False

    if not effective_may_use_cache or state_cache is None or block_table is None:
        if effective_may_use_cache:
            raise ValueError(
                "state_cache and block_table are required when may_use_cache=True"
            )
        state_cache_arg = kv_raw
        state_stride0 = 0
        state_stride1 = 0
        state_width = head_dim
        state_ring_entries = state_tokens_per_block
        num_state_blocks = 0
        block_table_arg = positions
        block_table_stride = 1
        state_for_align = None
    else:
        if state_cache.dtype != kv_raw.dtype or state_cache.dtype != score_raw.dtype:
            raise TypeError(
                "state_cache, kv_raw, and score_raw must share dtype "
                f"(got state_cache={state_cache.dtype}, kv_raw={kv_raw.dtype}, score_raw={score_raw.dtype})"
            )
        state_cache_arg = state_cache
        state_stride0 = int(state_cache.stride(0))
        state_stride1 = int(state_cache.stride(1))
        state_width = int(state_cache.shape[-1] // 2)
        state_ring_entries = int(state_cache.shape[1])
        num_state_blocks = int(state_cache.shape[0])
        block_table_arg = block_table
        block_table_stride = int(block_table.stride(0))
        state_for_align = state_cache

    state_block_size = state_tokens_per_block
    kv_block_size = int(kv_cache.shape[1])
    kv_block_stride = int(kv_cache.stride(0))
    use_unpad_restore = (unpad_restore is not None) and (not disable_raw_path)
    if use_unpad_restore:
        assert unpad_restore is not None
        if int(unpad_restore.shape[0]) != N:
            raise ValueError(
                f"unpad_restore length {int(unpad_restore.shape[0])} != slot_mapping length {N}"
            )
        n_raw = int(unpad_restore.shape[0])
        unpad_restore_arg = unpad_restore.contiguous()
    else:
        n_raw = 0 if disable_raw_path else int(kv_raw.shape[0])
        unpad_restore_arg = positions

    batched = (
        not disable_raw_path
        and seq_start_per_req is not None
        and cu_seq_per_req is not None
    )
    if batched:
        seq_start_per_req = seq_start_per_req.contiguous()
        cu_seq_per_req = cu_seq_per_req.contiguous()
    else:
        seq_start_per_req = positions
        cu_seq_per_req = positions

    align4 = _is_16b_aligned(kv_raw, score_raw, ape, state_for_align, head_dim)
    num_warps, tile_t, num_stages = _fused_launch_config(
        head_dim, compress_ratio, overlap, cfg
    )
    launch_kwargs: dict[str, object] = {"num_warps": num_warps}
    if num_stages is not None:
        launch_kwargs["num_stages"] = num_stages

    if head_dim == 512 and compress_ratio == 128 and (not overlap) and use_active:
        ws_plane = num_work * _HCA_NUM_T_SPLITS * head_dim
        ws = _get_hca_split_ws(num_work, head_dim, kv_cache.device)
        _hca_compress_split_stage1_kernel[
            (num_work, _HCA_NUM_T_SPLITS * _HCA_D_CHUNKS)
        ](
            state_cache_arg,
            state_stride0,
            state_stride1,
            token_to_req_indices,
            positions,
            block_table_arg,
            block_table_stride,
            state_block_size,
            kv_raw,
            kv_raw.stride(0),
            score_raw,
            score_raw.stride(0),
            ape,
            ape.stride(0),
            seq_start,
            n_raw,
            seq_start_per_req,
            cu_seq_per_req,
            kv_slot_mapping,
            kv_block_size,
            unpad_restore_arg,
            active_indices_arg,
            ws,
            ws_plane,
            num_work,
            HEAD_SIZE=head_dim,
            STATE_WIDTH=state_width,
            COMPRESS_RATIO=compress_ratio,
            NUM_T_SPLITS=_HCA_NUM_T_SPLITS,
            SPLIT_T=_HCA_SPLIT_T,
            CHUNK_D=_HCA_CHUNK_D,
            D_CHUNKS=_HCA_D_CHUNKS,
            NUM_STATE_BLOCKS=num_state_blocks,
            BATCHED=batched,
            TRAP_INVALID_KV_ACCESS=True,
            STATE_RING_ENTRIES=state_ring_entries,
            MAY_USE_CACHE=effective_may_use_cache,
            USE_UNPAD_RESTORE=use_unpad_restore,
            USE_ACTIVE_INDICES=True,
            ALIGN4=align4,
            num_warps=4,
            num_stages=1,
        )
        _hca_compress_split_finalize_kernel[(num_work,)](
            positions,
            rms_norm_weight,
            rms_norm_eps,
            cos_sin_cache,
            cos_sin_cache.stride(0),
            kv_cache,
            kv_slot_mapping,
            kv_block_size,
            active_indices_arg,
            ws,
            ws_plane,
            num_work,
            HEAD_SIZE=head_dim,
            TRITON_BLOCK_SIZE=triton.next_power_of_2(head_dim),
            COMPRESS_RATIO=compress_ratio,
            ROPE_HEAD_DIM=rope_head_dim,
            FP8_MAX=448.0,
            QUANT_BLOCK=cfg["quant_block"],
            TOKEN_STRIDE=cfg["token_stride"],
            SCALE_DIM=cfg["scale_dim"],
            NUM_T_SPLITS=_HCA_NUM_T_SPLITS,
            KV_BLOCK_STRIDE=kv_block_stride,
            NUM_KV_BLOCKS=int(kv_cache.shape[0]),
            TRAP_INVALID_KV_ACCESS=True,
            USE_ACTIVE_INDICES=True,
            ALIGN4=align4,
            num_warps=4,
        )
    elif head_dim == 512:
        _fused_kv_compress_norm_rope_insert_sparse_attn[(num_work,)](
            state_cache_arg,
            state_stride0,
            state_stride1,
            token_to_req_indices,
            positions,
            slot_mapping,
            block_table_arg,
            block_table_stride,
            state_block_size,
            kv_raw,
            kv_raw.stride(0),
            score_raw,
            score_raw.stride(0),
            ape,
            ape.stride(0),
            seq_start,
            n_raw,
            seq_start_per_req,
            cu_seq_per_req,
            rms_norm_weight,
            rms_norm_eps,
            cos_sin_cache,
            cos_sin_cache.stride(0),
            kv_cache,
            kv_slot_mapping,
            kv_block_size,
            unpad_restore_arg,
            active_indices_arg,
            num_work,
            HEAD_SIZE=head_dim,
            TRITON_BLOCK_SIZE=triton.next_power_of_2(head_dim),
            STATE_WIDTH=state_width,
            COMPRESS_RATIO=compress_ratio,
            OVERLAP=overlap,
            ROPE_HEAD_DIM=rope_head_dim,
            FP8_MAX=448.0,
            QUANT_BLOCK=cfg["quant_block"],
            TOKEN_STRIDE=cfg["token_stride"],
            SCALE_DIM=cfg["scale_dim"],
            KV_BLOCK_STRIDE=kv_block_stride,
            NUM_STATE_BLOCKS=num_state_blocks,
            NUM_KV_BLOCKS=int(kv_cache.shape[0]),
            BATCHED=batched,
            TRAP_INVALID_KV_ACCESS=True,
            STATE_RING_ENTRIES=state_ring_entries,
            MAY_USE_CACHE=effective_may_use_cache,
            USE_UNPAD_RESTORE=use_unpad_restore,
            USE_ACTIVE_INDICES=use_active,
            TILE_T=tile_t,
            ALIGN4=align4,
            **launch_kwargs,
        )
    else:
        grid = (triton.cdiv(num_work, _INDEXER_TOKENS_PER_CTA),)
        _fused_kv_compress_norm_rope_insert_indexer_attn[grid](
            state_cache_arg,
            state_stride0,
            state_stride1,
            token_to_req_indices,
            positions,
            slot_mapping,
            block_table_arg,
            block_table_stride,
            state_block_size,
            kv_raw,
            kv_raw.stride(0),
            score_raw,
            score_raw.stride(0),
            ape,
            ape.stride(0),
            seq_start,
            n_raw,
            seq_start_per_req,
            cu_seq_per_req,
            rms_norm_weight,
            rms_norm_eps,
            cos_sin_cache,
            cos_sin_cache.stride(0),
            kv_cache,
            kv_slot_mapping,
            kv_block_size,
            unpad_restore_arg,
            active_indices_arg,
            num_work,
            HEAD_SIZE=head_dim,
            TRITON_BLOCK_SIZE=triton.next_power_of_2(head_dim),
            STATE_WIDTH=state_width,
            COMPRESS_RATIO=compress_ratio,
            OVERLAP=overlap,
            ROPE_HEAD_DIM=rope_head_dim,
            FP8_MAX=448.0,
            QUANT_BLOCK=cfg["quant_block"],
            TOKEN_STRIDE=cfg["token_stride"],
            SCALE_DIM=cfg["scale_dim"],
            KV_BLOCK_STRIDE=kv_block_stride,
            NUM_STATE_BLOCKS=num_state_blocks,
            NUM_KV_BLOCKS=int(kv_cache.shape[0]),
            BATCHED=batched,
            TRAP_INVALID_KV_ACCESS=True,
            STATE_RING_ENTRIES=state_ring_entries,
            MAY_USE_CACHE=effective_may_use_cache,
            USE_UNPAD_RESTORE=use_unpad_restore,
            USE_ACTIVE_INDICES=use_active,
            BLOCK_M=_INDEXER_TOKENS_PER_CTA,
            ALIGN4=align4,
            **launch_kwargs,
        )


def build_cos_sin_cache(freqs_cis: torch.Tensor) -> Tuple[torch.Tensor, int]:
    """Pack ``freqs_cis [max_pos, rope_head_dim/2] complex64`` into the
    ``[max_pos, rope_head_dim]`` fp32 cache that the fused kernels expect.

    Layout (matches vLLM ``RotaryEmbedding.cos_sin_cache``):
      - first ``HALF_ROPE = rope_head_dim // 2`` cols  → cos
      - next  ``HALF_ROPE`` cols                       → sin
    """
    assert (
        freqs_cis.dtype == torch.complex64
    ), f"freqs_cis must be complex64, got {freqs_cis.dtype}"
    cos = freqs_cis.real.contiguous().to(torch.float32)
    sin = freqs_cis.imag.contiguous().to(torch.float32)
    cache = torch.cat([cos, sin], dim=-1).contiguous()
    rope_head_dim = int(cache.shape[-1])
    return cache, rope_head_dim
