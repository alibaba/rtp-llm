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
@triton.jit(do_not_specialize=["num_state_blocks"])
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
    block_size,
    num_state_blocks,
    HEAD_SIZE: tl.constexpr,
    TRITON_BLOCK_SIZE: tl.constexpr,
    STATE_WIDTH: tl.constexpr,
    COMPRESS_RATIO: tl.constexpr,
    TRAP_INVALID_KV_ACCESS: tl.constexpr,
    USE_UNPAD_RESTORE: tl.constexpr,
):
    token_idx = tl.program_id(0).to(tl.int64)
    slot_id = tl.load(slot_mapping_ptr + token_idx)
    if slot_id < 0:
        return

    block_idx = (slot_id // block_size).to(tl.int64)
    pos_in_block = slot_id % block_size
    if block_idx < 0:
        _trap_invalid_kv_access(TRAP_INVALID_KV_ACCESS)
    if block_idx >= num_state_blocks:
        _trap_invalid_kv_access(TRAP_INVALID_KV_ACCESS)

    base_ptr = (
        state_cache_ptr
        + block_idx * state_cache_stride0
        + pos_in_block * state_cache_stride1
    )

    block = tl.arange(0, TRITON_BLOCK_SIZE)
    mask = block < HEAD_SIZE

    if USE_UNPAD_RESTORE:
        raw_row_idx = tl.load(unpad_restore_ptr + token_idx).to(tl.int64)
    else:
        raw_row_idx = token_idx

    kv = tl.load(kv_ptr + raw_row_idx * kv_stride + block, mask=mask)
    tl.store(base_ptr + block, kv, mask=mask)

    position = tl.load(positions_ptr + token_idx)
    ape_row = position % COMPRESS_RATIO
    ape = tl.load(ape_ptr + ape_row * ape_stride + block, mask=mask)
    score = tl.load(score_ptr + raw_row_idx * score_stride + block, mask=mask)
    tl.store(base_ptr + STATE_WIDTH + block, score + ape, mask=mask)


# =============================================================================
# Shared boundary compression helpers (source dispatch + online softmax + norm)
# =============================================================================
@triton.jit
def _load_raw_tile(
    kv_raw_ptr,
    kv_raw_stride,
    score_raw_ptr,
    score_raw_stride,
    ape_ptr,
    ape_stride,
    unpad_restore_ptr,
    pos,
    use_raw,
    flat_idx_safe,
    head_offset,
    block,
    mask,
    COMPRESS_RATIO: tl.constexpr,
    USE_UNPAD_RESTORE: tl.constexpr,
):
    if USE_UNPAD_RESTORE:
        raw_row_idx = tl.load(
            unpad_restore_ptr + flat_idx_safe, mask=use_raw, other=0
        ).to(tl.int64)
    else:
        raw_row_idx = flat_idx_safe
    raw_mask = use_raw[:, None] & mask[None, :]
    kv_from_raw = tl.load(
        kv_raw_ptr
        + raw_row_idx[:, None] * kv_raw_stride
        + head_offset[:, None]
        + block[None, :],
        mask=raw_mask,
        other=0.0,
    ).to(tl.float32)
    score_from_raw = tl.load(
        score_raw_ptr
        + raw_row_idx[:, None] * score_raw_stride
        + head_offset[:, None]
        + block[None, :],
        mask=raw_mask,
        other=0.0,
    ).to(tl.float32)
    # state_cache stores score+ape pre-summed; raw path adds ape here.
    ape_rows = pos % COMPRESS_RATIO
    ape_rows_safe = tl.where(use_raw, ape_rows, 0)
    ape_vals = tl.load(
        ape_ptr
        + ape_rows_safe[:, None] * ape_stride
        + head_offset[:, None]
        + block[None, :],
        mask=raw_mask,
        other=0.0,
    ).to(tl.float32)
    score_from_raw = score_from_raw + ape_vals
    return kv_from_raw, score_from_raw


@triton.jit
def _load_kv_score_tile(
    state_cache_ptr,
    state_cache_stride0,
    state_cache_stride1,
    req_idx,
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
    req_seq_start,
    req_cu_lo,
    req_n_raw,
    start,
    tokens,
    block,
    mask,
    NUM_STATE_BLOCKS,
    HEAD_SIZE: tl.constexpr,
    TRITON_BLOCK_SIZE: tl.constexpr,
    STATE_WIDTH: tl.constexpr,
    COMPRESS_RATIO: tl.constexpr,
    WINDOW_SIZE: tl.constexpr,
    TRAP_INVALID_KV_ACCESS: tl.constexpr,
    STATE_RING_ENTRIES: tl.constexpr,
    MAY_USE_CACHE: tl.constexpr,
    USE_UNPAD_RESTORE: tl.constexpr,
):
    pos = start + tokens.to(tl.int64)
    mask_pos = (pos >= 0) & (tokens < WINDOW_SIZE)
    head_offset = (tokens >= COMPRESS_RATIO).to(tl.int32) * HEAD_SIZE

    flat_idx_in_req = pos - req_seq_start
    use_raw = mask_pos & (flat_idx_in_req >= 0) & (flat_idx_in_req < req_n_raw)
    flat_idx = req_cu_lo + flat_idx_in_req
    flat_idx_safe = tl.where(use_raw, flat_idx, 0)

    if MAY_USE_CACHE:
        # Unified raw/cache load: pick the source per row on 1-D row-base
        # pointers, then issue ONE load per tensor. The previous dual-load +
        # where-merge kept four live [TILE_ROWS, TRITON_BLOCK_SIZE] fp32 tiles
        # (raw + cache, kv + score) alive across the ``has_raw``/``has_cache``
        # regions, which capped resident CTAs/SM on chunked-prefill steps 2+.
        # Rows that are neither raw nor a valid cache block stay masked and
        # load ``other=0.0`` — the same zeros the old zeros-init + merge
        # produced. Predicated loads issue no memory traffic for fully
        # masked rows, so decode (n_raw == 0) and pure-prefix tiles still
        # skip the raw global loads they never performed.
        if USE_UNPAD_RESTORE:
            raw_row_idx = tl.load(
                unpad_restore_ptr + flat_idx_safe, mask=use_raw, other=0
            ).to(tl.int64)
        else:
            raw_row_idx = flat_idx_safe

        # State-cache path: prefix-cache hit region.
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
        block_offsets = tl.where(mask_pos, pos % STATE_RING_ENTRIES, 0)

        cache_row_base = (
            state_cache_ptr
            + block_numbers_i64 * state_cache_stride0.to(tl.int64)
            + block_offsets.to(tl.int64) * state_cache_stride1.to(tl.int64)
            + head_offset
        )

        # Per-row source select. Both pointer sets must share the fp32
        # pointee type (kv_raw/score_raw are fp32 everywhere today); use_raw
        # and valid_block are disjoint, so every row is loaded from exactly
        # the source the old where-merge would have picked.
        load_mask = (use_raw | valid_block)[:, None] & mask[None, :]
        kv_row_base = tl.where(
            use_raw,
            kv_raw_ptr + raw_row_idx * kv_raw_stride + head_offset,
            cache_row_base,
        )
        kv = tl.load(
            kv_row_base[:, None] + block[None, :], mask=load_mask, other=0.0
        ).to(tl.float32)

        score_row_base = tl.where(
            use_raw,
            score_raw_ptr + raw_row_idx * score_raw_stride + head_offset,
            cache_row_base + STATE_WIDTH,
        )
        score = tl.load(
            score_row_base[:, None] + block[None, :], mask=load_mask, other=0.0
        ).to(tl.float32)

        # state_cache stores score+ape pre-summed; raw path adds ape here.
        # ape_vals is 0.0 on non-raw rows, so cache rows keep their value.
        ape_rows_safe = tl.where(use_raw, pos % COMPRESS_RATIO, 0)
        ape_vals = tl.load(
            ape_ptr
            + ape_rows_safe[:, None] * ape_stride
            + head_offset[:, None]
            + block[None, :],
            mask=use_raw[:, None] & mask[None, :],
            other=0.0,
        ).to(tl.float32)
        score = score + ape_vals

        final_valid = use_raw | (use_cache & valid_block)
    else:
        kv, score = _load_raw_tile(
            kv_raw_ptr,
            kv_raw_stride,
            score_raw_ptr,
            score_raw_stride,
            ape_ptr,
            ape_stride,
            unpad_restore_ptr,
            pos,
            use_raw,
            flat_idx_safe,
            head_offset,
            block,
            mask,
            COMPRESS_RATIO=COMPRESS_RATIO,
            USE_UNPAD_RESTORE=USE_UNPAD_RESTORE,
        )
        final_valid = use_raw

    return kv, score, final_valid


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
        kv, score, final_valid = _load_kv_score_tile(
            state_cache_ptr,
            state_cache_stride0,
            state_cache_stride1,
            req_idx,
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
            req_seq_start,
            req_cu_lo,
            req_n_raw,
            start,
            tokens,
            block,
            mask,
            NUM_STATE_BLOCKS,
            HEAD_SIZE=HEAD_SIZE,
            TRITON_BLOCK_SIZE=TRITON_BLOCK_SIZE,
            STATE_WIDTH=STATE_WIDTH,
            COMPRESS_RATIO=COMPRESS_RATIO,
            WINDOW_SIZE=WINDOW_SIZE,
            TRAP_INVALID_KV_ACCESS=TRAP_INVALID_KV_ACCESS,
            STATE_RING_ENTRIES=STATE_RING_ENTRIES,
            MAY_USE_CACHE=MAY_USE_CACHE,
            USE_UNPAD_RESTORE=USE_UNPAD_RESTORE,
        )
        combined_mask = final_valid[:, None] & mask[None, :]
        score = tl.where(combined_mask, score, float("-inf"))
        score = tl.softmax(score, dim=0)
        kv = tl.where(combined_mask, kv, 0.0)
        compressed_kv = tl.sum(kv * score, axis=0)
    else:
        # Large-ratio path (e.g. HCA ratio=128): stream tiles with online
        # softmax so live tile buffers stay at [TILE_T, 512] instead of [128, 512].
        m_i = tl.full((TRITON_BLOCK_SIZE,), float("-inf"), dtype=tl.float32)
        d_i = tl.zeros((TRITON_BLOCK_SIZE,), dtype=tl.float32)
        acc = tl.zeros((TRITON_BLOCK_SIZE,), dtype=tl.float32)
        for tile_start in tl.range(0, WINDOW_SIZE, TILE_T):
            tokens = tile_start + tl.arange(0, TILE_T)
            kv_t, score_t, valid_t = _load_kv_score_tile(
                state_cache_ptr,
                state_cache_stride0,
                state_cache_stride1,
                req_idx,
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
                req_seq_start,
                req_cu_lo,
                req_n_raw,
                start,
                tokens,
                block,
                mask,
                NUM_STATE_BLOCKS,
                HEAD_SIZE=HEAD_SIZE,
                TRITON_BLOCK_SIZE=TRITON_BLOCK_SIZE,
                STATE_WIDTH=STATE_WIDTH,
                COMPRESS_RATIO=COMPRESS_RATIO,
                WINDOW_SIZE=WINDOW_SIZE,
                TRAP_INVALID_KV_ACCESS=TRAP_INVALID_KV_ACCESS,
                STATE_RING_ENTRIES=STATE_RING_ENTRIES,
                MAY_USE_CACHE=MAY_USE_CACHE,
                USE_UNPAD_RESTORE=USE_UNPAD_RESTORE,
            )
            combined_mask_t = valid_t[:, None] & mask[None, :]
            score_t = tl.where(combined_mask_t, score_t, float("-inf"))
            kv_t = tl.where(combined_mask_t, kv_t, 0.0)

            tile_max = tl.max(score_t, axis=0)
            m_new = tl.maximum(m_i, tile_max)
            m_new_safe = tl.where(m_new > float("-inf"), m_new, 0.0)
            alpha = tl.where(m_i > float("-inf"), tl.exp(m_i - m_new_safe), 0.0)
            p_t = tl.where(combined_mask_t, tl.exp(score_t - m_new_safe[None, :]), 0.0)
            d_i = d_i * alpha + tl.sum(p_t, axis=0)
            acc = acc * alpha + tl.sum(kv_t * p_t, axis=0)
            m_i = m_new
        d_safe = tl.where(d_i > 0.0, d_i, 1.0)
        compressed_kv = tl.where(d_i > 0.0, acc / d_safe, 0.0)

    rms_w = tl.load(rms_norm_weight_ptr + block, mask=mask, other=0.0)
    variance = tl.sum(compressed_kv * compressed_kv, axis=0) / HEAD_SIZE
    rrms = tl.rsqrt(variance + rms_norm_eps)
    return compressed_kv * rrms * rms_w


# =============================================================================
# DeepseekV4 attention path: head_dim=512 (CSA / HCA), 584B per slot
# =============================================================================
@triton.jit(
    do_not_specialize=[
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
    # Raw kv/score for the tokens supplied to this launch + ape table.
    # ``seq_start`` = absolute position of ``kv_raw[0]`` (prefill: the
    # current chunk's start = sp_int). ``n_raw`` = ``kv_raw.shape[0]``,
    # i.e. how many leading positions ``kv_raw`` covers (NOT a batch
    # size — DSV4 prefill runs bsz=1, so this is the chunk length).
    # See module docstring for the source-dispatch rules.
    kv_raw_ptr,
    kv_raw_stride,
    score_raw_ptr,
    score_raw_stride,
    ape_ptr,
    ape_stride,
    seq_start,
    n_raw,
    # Phase-3a part 4c — varlen raw path. When ``BATCHED=True`` these
    # arrays carry per-request data so the kernel can compute the flat
    # ``kv_raw`` offset for each request independently:
    #   seq_start_per_req[b]  = absolute position of req b's first new token
    #   cu_seq_per_req[b+1]   = end offset of req b in the flat kv_raw axis
    # ``BATCHED=False`` falls back to scalar ``seq_start`` + global
    # ``[0, n_raw)`` check, byte-equal to the legacy single-request path.
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
    TILE_T: tl.constexpr,
):
    token_idx = tl.program_id(0).to(tl.int64)

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
    )

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


# =============================================================================
# Indexer path: head_dim=128, single quant block, 132B per slot
# =============================================================================
@triton.jit(
    do_not_specialize=[
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
    # Raw kv/score for the tokens supplied to this launch + ape table.
    # ``seq_start`` = absolute position of ``kv_raw[0]`` (prefill: the
    # current chunk's start = sp_int). ``n_raw`` = ``kv_raw.shape[0]``,
    # i.e. how many leading positions ``kv_raw`` covers (NOT a batch
    # size — DSV4 prefill runs bsz=1, so this is the chunk length).
    # See module docstring for the source-dispatch rules.
    kv_raw_ptr,
    kv_raw_stride,
    score_raw_ptr,
    score_raw_stride,
    ape_ptr,
    ape_stride,
    seq_start,
    n_raw,
    # Phase-3a part 4c — varlen raw path. When ``BATCHED=True`` these
    # arrays carry per-request data so the kernel can compute the flat
    # ``kv_raw`` offset for each request independently:
    #   seq_start_per_req[b]  = absolute position of req b's first new token
    #   cu_seq_per_req[b+1]   = end offset of req b in the flat kv_raw axis
    # ``BATCHED=False`` falls back to scalar ``seq_start`` + global
    # ``[0, n_raw)`` check, byte-equal to the legacy single-request path.
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
    TILE_T: tl.constexpr,
):
    token_idx = tl.program_id(0).to(tl.int64)

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
    )

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

    NUM_PAIRS: tl.constexpr = TRITON_BLOCK_SIZE // 2
    NOPE_PAIRS: tl.constexpr = NOPE_HEAD_DIM // 2

    normed_2d = tl.reshape(normed, (NUM_PAIRS, 2))
    even, odd = tl.split(normed_2d)

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

    tl.static_assert(
        TRITON_BLOCK_SIZE == QUANT_BLOCK,
        "Indexer expects one quant block (QUANT_BLOCK == TRITON_BLOCK_SIZE)",
    )
    INV_FP8_MAX: tl.constexpr = 1.0 / FP8_MAX

    result_bf16 = result.to(tl.bfloat16).to(tl.float32)
    absmax = tl.max(tl.abs(result_bf16), axis=0)
    absmax = tl.maximum(absmax, 1e-4)
    raw_scale = absmax * INV_FP8_MAX
    exponent = tl.ceil(tl.log2(raw_scale))
    inv_scale = tl.exp2(-exponent)

    x_scaled = result_bf16 * inv_scale
    x_clamped = tl.clamp(x_scaled, -FP8_MAX, FP8_MAX)
    x_fp8 = x_clamped.to(tl.float8e4nv)
    x_uint8 = x_fp8.to(tl.uint8, bitcast=True)

    tl.store(fp8_ptr + block, x_uint8, mask=mask)

    scale_val = tl.exp2(exponent)
    tl.store(scale_ptr.to(tl.pointer_type(tl.float32)), scale_val)


# =============================================================================
# Python wrappers
# =============================================================================
# Per-head_dim constexpr table for the boundary fused kernel.
#   head_dim 512 (CSA / HCA): nope=448 fp8 + rope=64 bf16, 7 ue8m0 scales (+1 pad)
#   head_dim 128 (indexer)  : 128 fp8 + 1 fp32 scale (4 bytes)
_FUSED_CONSTEXPR_BY_HEAD_DIM = {
    512: dict(quant_block=64, token_stride=576, scale_dim=8, num_warps=4),
    128: dict(quant_block=128, token_stride=128, scale_dim=4, num_warps=2),
}


def _fused_num_warps(head_dim: int, compress_ratio: int, cfg: dict) -> int:
    """Pick num_warps for the boundary fused kernel.

    CSA (ratio=4, overlap=1 -> 8x512) and tiled HCA (ratio>=64 streaming
    over TILE_T=8 -> 8x512 per tile) both fit in 4 warps (128 threads,
    coalesced 4-fp32 vector loads, zero register spilling). Indexer
    (head_dim=128) uses 2 warps (64 threads, 2 fp32/thread and 100% SM warp
    occupancy ceiling vs 50% at 1 warp).
    """
    del head_dim, compress_ratio
    return cfg["num_warps"]


def run_save_partial_states(
    kv: torch.Tensor,  # [N, coff*head_dim] bf16/fp32 (or padded gathered when unpad_restore set)
    score: torch.Tensor,  # [N, coff*head_dim] bf16/fp32 (or padded gathered when unpad_restore set)
    ape: torch.Tensor,  # [compress_ratio, coff*head_dim] fp32
    positions: torch.Tensor,  # [N] int64
    state_cache: torch.Tensor,  # [num_blocks, block_size, 2*coff*head_dim] fp32
    slot_mapping: torch.Tensor,  # [N] int64; -1 = skip
    compress_ratio: int,
    unpad_restore: Optional[torch.Tensor] = None,
) -> None:
    """Per-token write of (kv | score+ape) into the fp32 state cache."""
    N = int(slot_mapping.shape[0])
    if N == 0:
        return
    use_unpad_restore = unpad_restore is not None
    if use_unpad_restore:
        assert unpad_restore is not None
        if int(unpad_restore.shape[0]) != N:
            raise ValueError(
                f"unpad_restore length {int(unpad_restore.shape[0])} != slot_mapping length {N}"
            )
        unpad_restore_arg = unpad_restore.contiguous()
    else:
        unpad_restore_arg = positions
    head_size = int(kv.shape[-1])
    state_width = int(state_cache.shape[-1] // 2)
    block_size = int(state_cache.shape[1])
    _save_partial_states_kernel[(N,)](
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
        block_size,
        num_state_blocks=int(state_cache.shape[0]),
        HEAD_SIZE=head_size,
        TRITON_BLOCK_SIZE=triton.next_power_of_2(head_size),
        STATE_WIDTH=state_width,
        COMPRESS_RATIO=compress_ratio,
        TRAP_INVALID_KV_ACCESS=True,
        USE_UNPAD_RESTORE=use_unpad_restore,
        num_warps=2 if head_size <= 256 else 4,
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
    # Raw kv/score for the tokens supplied to this launch + ape table.
    # In prefill these cover the current chunk (kv_raw.shape[0] = chunk
    # length, NOT a batch dim — DSV4 prefill is bsz=1). The kernel reads
    # from them whenever ``flat_idx = pos - seq_start`` falls in
    # ``[0, kv_raw.shape[0])``; positions before ``seq_start`` are
    # prefix-cache hits and route to the state pool via block_table.
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
    # Phase-3a part 4c — varlen raw path. When both arrays are passed
    # (and ``disable_raw_path=False``) the kernel computes per-request
    # ``flat_idx`` so B>1 batched prefill keeps the raw fast path that
    # avoids state-cache readback. Scalar ``seq_start`` is
    # ignored in that mode.
    seq_start_per_req: Optional[torch.Tensor] = None,  # [B] int64
    cu_seq_per_req: Optional[torch.Tensor] = None,  # [B+1] int64
    state_tokens_per_block: int,
    may_use_cache: bool = True,
    unpad_restore: Optional[torch.Tensor] = None,
) -> None:
    """Boundary-token compress→norm→rope→fp8 quant→KV-pool store.

    For each boundary token (``(position+1) % compress_ratio == 0``), the
    kernel gathers the prior ``(1+overlap)*compress_ratio`` positions of
    kv/score, softmax-reduces over positions, RMSNorms, FP8-quantises the
    nope segment + RoPEs the rope segment, and writes the result into
    ``kv_cache`` at ``kv_slot_mapping[t]``. Non-boundary and unowned CP tokens
    self-skip early via ``kv_slot_mapping[t] < 0``; pass a flat ``[N_tok]``
    slot_mapping.

    When ``may_use_cache=False`` (prefix-free prefill with raw path enabled),
    Triton prunes the entire ``state_cache`` / ``block_table`` lookup path.
    When ``unpad_restore`` is provided on the raw path, raw rows are read
    directly from rank-major ``kv_raw`` / ``score_raw`` via
    ``unpad_restore[flat_idx]`` without an intermediate gather copy.
    """
    N = int(slot_mapping.shape[0])
    if N == 0:
        return
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
        # Triton requires a non-None pointer arg even when BATCHED=False.
        # Pass ``positions`` as a stand-in — the kernel never reads from
        # these tensors when BATCHED=False, but the launcher needs a
        # valid CUDA address to bind.
        seq_start_per_req = positions
        cu_seq_per_req = positions

    if head_dim == 512:
        kernel = _fused_kv_compress_norm_rope_insert_sparse_attn
    else:
        kernel = _fused_kv_compress_norm_rope_insert_indexer_attn

    kernel[(N,)](
        state_cache_arg,
        state_stride0,
        state_stride1,
        token_to_req_indices,
        positions,
        slot_mapping,
        block_table_arg,
        block_table_stride,
        state_block_size,
        # raw path
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
        TILE_T=8,
        num_warps=_fused_num_warps(head_dim, compress_ratio, cfg),
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
