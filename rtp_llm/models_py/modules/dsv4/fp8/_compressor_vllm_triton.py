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

from rtp_llm.models_py.modules.dsv4.fp8._trap_utils import (
    invalid_kv_access_validation_enabled,
    trap_invalid_kv_access_enabled,
    validate_block_table_lookup,
    validate_slot_mapping,
)


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
    block_size,
    num_state_blocks,
    HEAD_SIZE: tl.constexpr,
    TRITON_BLOCK_SIZE: tl.constexpr,
    STATE_WIDTH: tl.constexpr,
    COMPRESS_RATIO: tl.constexpr,
    TRAP_INVALID_KV_ACCESS: tl.constexpr,
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

    kv = tl.load(kv_ptr + token_idx * kv_stride + block, mask=mask)
    tl.store(base_ptr + block, kv, mask=mask)

    position = tl.load(positions_ptr + token_idx)
    ape_row = position % COMPRESS_RATIO
    ape = tl.load(ape_ptr + ape_row * ape_stride + block, mask=mask)
    score = tl.load(score_ptr + token_idx * score_stride + block, mask=mask)
    tl.store(base_ptr + STATE_WIDTH + block, score + ape, mask=mask)


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
):
    token_idx = tl.program_id(0).to(tl.int64)

    position = tl.load(positions_ptr + token_idx).to(tl.int64)
    if (position + 1) % COMPRESS_RATIO != 0:
        return

    req_idx = tl.load(token_to_req_indices_ptr + token_idx).to(tl.int64)

    start = position - (1 + OVERLAP) * COMPRESS_RATIO + 1
    tokens = tl.arange(0, (1 + OVERLAP) * COMPRESS_RATIO)
    pos = start + tokens
    mask_pos = pos >= 0
    head_offset = (tokens >= COMPRESS_RATIO).to(tl.int32) * HEAD_SIZE

    block = tl.arange(0, TRITON_BLOCK_SIZE)
    mask = block < HEAD_SIZE

    # ── Source dispatch (see module docstring) ──
    #   flat_idx in [0, n_raw)  → raw   (covered by this launch's kv_raw)
    #   flat_idx <  0           → cache (prefix-cache hit; resolve via
    #                                    block_table[req_idx])
    # Decode: disable_raw_path → n_raw == 0, every position → cache.
    # BATCHED: per-request raw window — req b's tokens live in
    #   kv_raw[cu_seq_per_req[b] : cu_seq_per_req[b+1]] and cover abs
    #   positions [seq_start_per_req[b], seq_start_per_req[b]+req_n_raw).
    if BATCHED:
        req_seq_start = tl.load(seq_start_per_req_ptr + req_idx)
        req_cu_lo = tl.load(cu_seq_per_req_ptr + req_idx)
        req_cu_hi = tl.load(cu_seq_per_req_ptr + req_idx + 1)
        req_n_raw = req_cu_hi - req_cu_lo
        flat_idx_in_req = pos - req_seq_start
        use_raw = mask_pos & (flat_idx_in_req >= 0) & (flat_idx_in_req < req_n_raw)
        flat_idx = req_cu_lo + flat_idx_in_req
    else:
        flat_idx = pos - seq_start
        use_raw = mask_pos & (flat_idx >= 0) & (flat_idx < n_raw)
    flat_idx_safe = tl.where(use_raw, flat_idx, 0)

    raw_mask = use_raw[:, None] & mask[None, :]
    kv_from_raw = tl.load(
        kv_raw_ptr
        + flat_idx_safe[:, None] * kv_raw_stride
        + head_offset[:, None]
        + block[None, :],
        mask=raw_mask,
        other=0.0,
    )
    score_from_raw = tl.load(
        score_raw_ptr
        + flat_idx_safe[:, None] * score_raw_stride
        + head_offset[:, None]
        + block[None, :],
        mask=raw_mask,
        other=0.0,
    )
    ape_rows = pos % COMPRESS_RATIO
    ape_rows_safe = tl.where(use_raw, ape_rows, 0)
    ape_vals = tl.load(
        ape_ptr
        + ape_rows_safe[:, None] * ape_stride
        + head_offset[:, None]
        + block[None, :],
        mask=raw_mask,
        other=0.0,
    )
    score_from_raw = score_from_raw + ape_vals

    # State-cache path: prefix-cache hit region. The framework supplies sparse
    # absolute logical-block tables. Any non-negative physical block id is
    # readable; negative entries are invalid. Do not clamp here: a
    # block index beyond the table width means the framework failed to grow the
    # table before this launch.
    use_cache = mask_pos & ~use_raw
    block_indices = (pos // block_size) % block_table_stride
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
    ring_mod = STATE_RING_ENTRIES
    block_offsets_raw = pos % ring_mod
    block_offsets = tl.where(mask_pos, block_offsets_raw, 0)

    row_base = (
        state_cache_ptr
        + block_numbers_i64 * state_cache_stride0.to(tl.int64)
        + block_offsets.to(tl.int64) * state_cache_stride1.to(tl.int64)
        + head_offset
    )
    cache_mask = valid_block[:, None] & mask[None, :]
    kv_from_cache = tl.load(
        row_base[:, None] + block[None, :], mask=cache_mask, other=0.0
    )
    score_from_cache = tl.load(
        row_base[:, None] + STATE_WIDTH + block[None, :],
        mask=cache_mask,
        other=0.0,
    )

    kv = tl.where(use_raw[:, None], kv_from_raw, kv_from_cache)
    score = tl.where(use_raw[:, None], score_from_raw, score_from_cache)

    final_valid = use_raw | (use_cache & valid_block)
    combined_mask = final_valid[:, None] & mask[None, :]

    score = tl.where(combined_mask, score, float("-inf"))
    score = tl.softmax(score, dim=0)
    kv = tl.where(combined_mask, kv, 0.0)
    compressed_kv = tl.sum(kv * score, axis=0)

    rms_w = tl.load(rms_norm_weight_ptr + block, mask=mask, other=0.0)
    variance = tl.sum(compressed_kv * compressed_kv, axis=0) / HEAD_SIZE
    rrms = tl.rsqrt(variance + rms_norm_eps)
    normed = compressed_kv * rrms * rms_w

    kv_slot_idx = tl.load(kv_slot_mapping_ptr + token_idx)
    if kv_slot_idx < 0:
        return
    kv_block_idx = kv_slot_idx // kv_cache_block_size
    kv_pos_in_block = kv_slot_idx % kv_cache_block_size
    if kv_block_idx < 0:
        _trap_invalid_kv_access(TRAP_INVALID_KV_ACCESS)
    if kv_block_idx >= NUM_KV_BLOCKS:
        _trap_invalid_kv_access(TRAP_INVALID_KV_ACCESS)

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
):
    token_idx = tl.program_id(0).to(tl.int64)

    position = tl.load(positions_ptr + token_idx).to(tl.int64)
    if (position + 1) % COMPRESS_RATIO != 0:
        return

    req_idx = tl.load(token_to_req_indices_ptr + token_idx).to(tl.int64)

    start = position - (1 + OVERLAP) * COMPRESS_RATIO + 1
    tokens = tl.arange(0, (1 + OVERLAP) * COMPRESS_RATIO)
    pos = start + tokens
    mask_pos = pos >= 0
    head_offset = (tokens >= COMPRESS_RATIO).to(tl.int32) * HEAD_SIZE

    block = tl.arange(0, TRITON_BLOCK_SIZE)
    mask = block < HEAD_SIZE

    # ── Source dispatch (see module docstring) ──
    #   flat_idx in [0, n_raw)  → raw   (covered by this launch's kv_raw)
    #   flat_idx <  0           → cache (prefix-cache hit; resolve via
    #                                    block_table[req_idx])
    # Decode: disable_raw_path → n_raw == 0, every position → cache.
    if BATCHED:
        req_seq_start = tl.load(seq_start_per_req_ptr + req_idx)
        req_cu_lo = tl.load(cu_seq_per_req_ptr + req_idx)
        req_cu_hi = tl.load(cu_seq_per_req_ptr + req_idx + 1)
        req_n_raw = req_cu_hi - req_cu_lo
        flat_idx_in_req = pos - req_seq_start
        use_raw = mask_pos & (flat_idx_in_req >= 0) & (flat_idx_in_req < req_n_raw)
        flat_idx = req_cu_lo + flat_idx_in_req
    else:
        flat_idx = pos - seq_start
        use_raw = mask_pos & (flat_idx >= 0) & (flat_idx < n_raw)
    flat_idx_safe = tl.where(use_raw, flat_idx, 0)

    raw_mask = use_raw[:, None] & mask[None, :]
    kv_from_raw = tl.load(
        kv_raw_ptr
        + flat_idx_safe[:, None] * kv_raw_stride
        + head_offset[:, None]
        + block[None, :],
        mask=raw_mask,
        other=0.0,
    )
    score_from_raw = tl.load(
        score_raw_ptr
        + flat_idx_safe[:, None] * score_raw_stride
        + head_offset[:, None]
        + block[None, :],
        mask=raw_mask,
        other=0.0,
    )
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
    )
    score_from_raw = score_from_raw + ape_vals

    # State-cache path: prefix-cache hit region. The framework supplies sparse
    # absolute logical-block tables. Any non-negative physical block id is
    # readable; negative entries are invalid. Do not clamp here: a
    # block index beyond the table width means the framework failed to grow the
    # table before this launch.
    use_cache = mask_pos & ~use_raw
    block_indices = (pos // block_size) % block_table_stride
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
    ring_mod = STATE_RING_ENTRIES
    block_offsets_raw = pos % ring_mod
    block_offsets = tl.where(mask_pos, block_offsets_raw, 0)

    row_base = (
        state_cache_ptr
        + block_numbers_i64 * state_cache_stride0.to(tl.int64)
        + block_offsets.to(tl.int64) * state_cache_stride1.to(tl.int64)
        + head_offset
    )
    cache_mask = valid_block[:, None] & mask[None, :]
    kv_from_cache = tl.load(
        row_base[:, None] + block[None, :], mask=cache_mask, other=0.0
    )
    score_from_cache = tl.load(
        row_base[:, None] + STATE_WIDTH + block[None, :],
        mask=cache_mask,
        other=0.0,
    )

    # ── Combine: pick raw or cache per position ──
    kv = tl.where(use_raw[:, None], kv_from_raw, kv_from_cache)
    score = tl.where(use_raw[:, None], score_from_raw, score_from_cache)

    # Final mask: position must contribute (either source produced data)
    final_valid = use_raw | (use_cache & valid_block)
    combined_mask = final_valid[:, None] & mask[None, :]

    score = tl.where(combined_mask, score, float("-inf"))
    score = tl.softmax(score, dim=0)

    kv = tl.where(combined_mask, kv, 0.0)

    compressed_kv = tl.sum(kv * score, axis=0)

    rms_w = tl.load(rms_norm_weight_ptr + block, mask=mask, other=0.0)
    variance = tl.sum(compressed_kv * compressed_kv, axis=0) / HEAD_SIZE
    rrms = tl.rsqrt(variance + rms_norm_eps)
    normed = compressed_kv * rrms * rms_w

    kv_slot_idx = tl.load(kv_slot_mapping_ptr + token_idx)
    if kv_slot_idx < 0:
        return
    kv_block_idx = kv_slot_idx // kv_cache_block_size
    kv_pos_in_block = kv_slot_idx % kv_cache_block_size
    if kv_block_idx < 0:
        _trap_invalid_kv_access(TRAP_INVALID_KV_ACCESS)
    if kv_block_idx >= NUM_KV_BLOCKS:
        _trap_invalid_kv_access(TRAP_INVALID_KV_ACCESS)

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
    128: dict(quant_block=128, token_stride=128, scale_dim=4, num_warps=1),
}


def _fused_num_warps(head_dim: int, compress_ratio: int, cfg: dict) -> int:
    """Pick num_warps for the boundary fused kernel.

    Why: the head_dim=512 kernel materialises a ``[(1+overlap)*ratio, 512]``
    fp32 register tile per boundary program for both the gather and the
    softmax reduction. CSA (ratio=4, overlap=1) -> ``[8, 512]`` = 16 KB
    fits comfortably in 4 warps (128 threads, 32 fp32 per thread). HCA
    (ratio=128, overlap=0) -> ``[128, 512]`` = 256 KB blows past the
    SM100 per-thread 256-register limit at 4 warps (~512 fp32/thread)
    and spills into local memory -- the decode timeline showed 20 HCA
    boundary programs at ~650 us each (all of step 0's Top1 13 ms).
    Bumping to 16 warps brings per-thread tile slice down to
    ``128 * 512 / (16*32) = 128`` fp32 ~= 64 registers, well under the
    limit, and also speeds the cross-warp softmax reduce over dim=0.
    """
    if head_dim == 512 and compress_ratio >= 64:
        return 16
    return cfg["num_warps"]


def _validate_fused_state_block_table(
    *,
    site: str,
    block_table: torch.Tensor,
    token_to_req_indices: torch.Tensor,
    positions: torch.Tensor,
    state_block_size: int,
    num_state_blocks: int,
    compress_ratio: int,
    overlap: bool,
    seq_start: int,
    n_raw: int,
    batched: bool,
    seq_start_per_req: torch.Tensor,
    cu_seq_per_req: torch.Tensor,
) -> None:
    if not invalid_kv_access_validation_enabled():
        return

    positions_i64 = positions.detach().reshape(-1).to(torch.int64)
    req_i64 = token_to_req_indices.detach().reshape(-1).to(torch.int64)
    boundary = (positions_i64 + 1) % int(compress_ratio) == 0
    if not bool(boundary.any().item()):
        return

    boundary_pos = positions_i64[boundary]
    boundary_req = req_i64[boundary]
    token_count = (1 + int(overlap)) * int(compress_ratio)
    tokens = torch.arange(token_count, device=positions.device, dtype=torch.int64)
    gathered_pos = boundary_pos[:, None] - token_count + 1 + tokens[None, :]
    mask_pos = gathered_pos >= 0

    if batched:
        rows = int(seq_start_per_req.shape[0])
        safe_req = boundary_req.clamp(0, max(rows - 1, 0))
        req_seq_start = seq_start_per_req.to(torch.int64)[safe_req]
        req_cu_lo = cu_seq_per_req.to(torch.int64)[safe_req]
        req_cu_hi = cu_seq_per_req.to(torch.int64)[safe_req + 1]
        flat_idx_in_req = gathered_pos - req_seq_start[:, None]
        req_n_raw = req_cu_hi - req_cu_lo
        use_raw = (
            mask_pos & (flat_idx_in_req >= 0) & (flat_idx_in_req < req_n_raw[:, None])
        )
    else:
        flat_idx = gathered_pos - int(seq_start)
        use_raw = mask_pos & (flat_idx >= 0) & (flat_idx < int(n_raw))

    use_cache = mask_pos & ~use_raw
    block_table_stride = int(block_table.shape[1])
    block_indices = (gathered_pos // int(state_block_size)) % block_table_stride
    validate_block_table_lookup(
        site,
        block_table,
        boundary_req[:, None].expand_as(block_indices),
        block_indices,
        use_cache,
        num_blocks=int(num_state_blocks),
    )


def run_save_partial_states(
    kv: torch.Tensor,  # [N, coff*head_dim] bf16/fp32, row-strided OK
    score: torch.Tensor,  # [N, coff*head_dim] bf16/fp32, row-strided OK
    ape: torch.Tensor,  # [compress_ratio, coff*head_dim] fp32
    positions: torch.Tensor,  # [N] int64
    state_cache: torch.Tensor,  # [num_blocks, block_size, 2*coff*head_dim] fp32
    slot_mapping: torch.Tensor,  # [N] int64; -1 = skip
    compress_ratio: int,
) -> None:
    """Per-token write of (kv | score+ape) into the fp32 state cache."""
    N = int(slot_mapping.shape[0])
    if N == 0:
        return
    head_size = int(kv.shape[-1])
    state_width = int(state_cache.shape[-1] // 2)
    block_size = int(state_cache.shape[1])
    validate_slot_mapping(
        "compressor.save_partial_states.state_slot_mapping",
        slot_mapping,
        block_size=block_size,
        num_blocks=int(state_cache.shape[0]),
        negative_mode="skip_any",
    )
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
        block_size,
        num_state_blocks=int(state_cache.shape[0]),
        HEAD_SIZE=head_size,
        TRITON_BLOCK_SIZE=triton.next_power_of_2(head_size),
        STATE_WIDTH=state_width,
        COMPRESS_RATIO=compress_ratio,
        TRAP_INVALID_KV_ACCESS=trap_invalid_kv_access_enabled(),
    )


def run_fused_compress_kv_write(
    state_cache: torch.Tensor,  # [num_blocks, block_size, 2*coff*head_dim] fp32
    token_to_req_indices: torch.Tensor,  # [N] int32
    positions: torch.Tensor,  # [N] int64
    slot_mapping: torch.Tensor,  # [N] int64 — state-pool slot per token
    block_table: torch.Tensor,  # [B, max_state_blocks] int — state-pool block table
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
    seq_start_per_req: Optional[torch.Tensor] = None,  # [B] int32/int64
    cu_seq_per_req: Optional[torch.Tensor] = None,  # [B+1] int32/int64
    state_tokens_per_block: int,
) -> None:
    """Boundary-token compress→norm→rope→fp8 quant→KV-pool store.

    For each boundary token (``(position+1) % compress_ratio == 0``), the
    kernel gathers the prior ``(1+overlap)*compress_ratio`` positions of
    kv/score, softmax-reduces over positions, RMSNorms, FP8-quantises the
    nope segment + RoPEs the rope segment, and writes the result into
    ``kv_cache`` at ``kv_slot_mapping[t]``. Non-boundary tokens self-skip
    (early-exit); pass a flat ``[N_tok]`` slot_mapping.

    Source dispatch for each gathered position is described in the module
    docstring. The wrapper materialises ``n_raw`` from ``kv_raw.shape[0]``
    (or 0 if ``disable_raw_path``) and forwards everything else verbatim.
    """
    N = int(slot_mapping.shape[0])
    if N == 0:
        return
    cfg = _FUSED_CONSTEXPR_BY_HEAD_DIM.get(head_dim)
    if cfg is None:
        raise ValueError(f"Unsupported head_dim {head_dim} for fused compressor write")

    state_width = int(state_cache.shape[-1] // 2)
    state_ring_entries = int(state_cache.shape[1])
    state_block_size = state_tokens_per_block
    kv_block_size = int(kv_cache.shape[1])
    kv_block_stride = int(kv_cache.stride(0))
    n_raw = 0 if disable_raw_path else int(kv_raw.shape[0])

    batched = (
        not disable_raw_path
        and seq_start_per_req is not None
        and cu_seq_per_req is not None
    )
    if batched:
        # Match the ``int32`` dtype the kernel expects (positions/req_idx
        # are int32 throughout the wrapper) so the per-request loads stay
        # in-register without an implicit promote.
        seq_start_per_req = seq_start_per_req.to(torch.int32).contiguous()
        cu_seq_per_req = cu_seq_per_req.to(torch.int32).contiguous()
    else:
        # Triton requires a non-None pointer arg even when BATCHED=False.
        # Pass ``positions`` as a stand-in — the kernel never reads from
        # these tensors when BATCHED=False, but the launcher needs a
        # valid CUDA address to bind.
        seq_start_per_req = positions
        cu_seq_per_req = positions

    validate_slot_mapping(
        "compressor.fused_compress.kv_slot_mapping",
        kv_slot_mapping,
        block_size=kv_block_size,
        num_blocks=int(kv_cache.shape[0]),
        negative_mode="skip_any",
    )
    _validate_fused_state_block_table(
        site="compressor.fused_compress.state_block_table",
        block_table=block_table,
        token_to_req_indices=token_to_req_indices,
        positions=positions,
        state_block_size=state_block_size,
        num_state_blocks=int(state_cache.shape[0]),
        compress_ratio=compress_ratio,
        overlap=overlap,
        seq_start=seq_start,
        n_raw=n_raw,
        batched=batched,
        seq_start_per_req=seq_start_per_req,
        cu_seq_per_req=cu_seq_per_req,
    )

    if head_dim == 512:
        kernel = _fused_kv_compress_norm_rope_insert_sparse_attn
    else:
        kernel = _fused_kv_compress_norm_rope_insert_indexer_attn

    kernel[(N,)](
        state_cache,
        state_cache.stride(0),
        state_cache.stride(1),
        token_to_req_indices,
        positions,
        slot_mapping,
        block_table,
        block_table.stride(0),
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
        NUM_STATE_BLOCKS=int(state_cache.shape[0]),
        NUM_KV_BLOCKS=int(kv_cache.shape[0]),
        BATCHED=batched,
        TRAP_INVALID_KV_ACCESS=trap_invalid_kv_access_enabled(),
        STATE_RING_ENTRIES=state_ring_entries,
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
