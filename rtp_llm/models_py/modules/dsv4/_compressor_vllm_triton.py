"""DSV4 CompressorFP8 Triton kernels (ported verbatim from vLLM).

Ports three kernels used by the per-token state pool layout (post commit
``e76867719`` — state pools now use ``entries_per_block=256``):

  * ``_save_partial_states_kernel`` — per-token write of (kv | score+ape)
    into the fp32 state cache. Source: vLLM ``deepseek_compressor.py``.
  * ``_fused_kv_compress_norm_rope_insert_sparse_attn`` — head=512 fused
    boundary writer (compress → RMSNorm → FP8 quant nope → RoPE bf16 →
    584B KV slot store). Source: vLLM ``fused_compress_quant_cache.py``.
  * ``_fused_kv_compress_norm_rope_insert_indexer_attn`` — head=128 fused
    boundary writer (compress → RMSNorm → RoPE → FP8 quant whole token →
    132B KV slot store). Source: vLLM ``fused_compress_quant_cache.py``.

Both fused writers self-skip non-boundary tokens (early-exit when
``(position+1) % COMPRESS_RATIO != 0``), so the caller passes a flat
``[N_tok]`` slot_mapping and lets the kernel decide.

The two thin wrappers ``run_save_partial_states`` / ``run_fused_compress_kv_write``
hide the constexpr table per ``head_dim``.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import triton
import triton.language as tl


# =============================================================================
# Per-token state-cache writer (fp32). ape is added to score in-kernel.
# =============================================================================
@triton.jit
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
    HEAD_SIZE: tl.constexpr,
    TRITON_BLOCK_SIZE: tl.constexpr,
    STATE_WIDTH: tl.constexpr,
    COMPRESS_RATIO: tl.constexpr,
):
    token_idx = tl.program_id(0)
    slot_id = tl.load(slot_mapping_ptr + token_idx)
    if slot_id < 0:
        return

    block_idx = slot_id // block_size
    pos_in_block = slot_id % block_size
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
@triton.jit
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
    # Raw kv/score (current batch) + ape + batch context — see indexer
    # kernel comments for rationale (cyclic state_cache cannot retain
    # earlier-batch data after later writes overwrite it).
    kv_raw_ptr,
    kv_raw_stride,
    score_raw_ptr,
    score_raw_stride,
    ape_ptr,
    ape_stride,
    seq_start,
    n_batch,
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
    KV_BLOCK_STRIDE: tl.constexpr,
    CACHE_WINDOW: tl.constexpr,
):
    token_idx = tl.program_id(0)

    position = tl.load(positions_ptr + token_idx)
    if (position + 1) % COMPRESS_RATIO != 0:
        return

    # Note: we deliberately do NOT early-return on ``slot_mapping[token]<0``.
    # When a boundary's state slot has been evicted (cyclic ring overwrote
    # it within this same launch) the raw path provides the data — the
    # ``kv_slot_idx < 0`` check at the bottom is the real gate for whether
    # this token should write to the KV pool.

    req_idx = tl.load(token_to_req_indices_ptr + token_idx)

    start = position - (1 + OVERLAP) * COMPRESS_RATIO + 1
    tokens = tl.arange(0, (1 + OVERLAP) * COMPRESS_RATIO)
    pos = start + tokens
    mask_pos = pos >= 0
    head_offset = (tokens >= COMPRESS_RATIO).to(tl.int32) * HEAD_SIZE

    block = tl.arange(0, TRITON_BLOCK_SIZE)
    mask = block < HEAD_SIZE

    # ── Source dispatch: raw for the prefix evicted from cache ──
    # State pool is fixed_blocks_per_req page-aligned blocks per request
    # (NOT a cyclic ring). Cache retains the LAST ``cache_window`` positions
    # = (page_size + (seq_len-1) % page_size + 1) when seq_len > page_size,
    # else seq_len. Earlier in-batch positions were evicted when their
    # logical block rolled out and must come from raw tensors.
    # Out-of-batch (previous-batch overlap) reads from cache too.
    flat_idx = pos - seq_start
    in_batch = mask_pos & (flat_idx >= 0) & (flat_idx < n_batch)
    use_raw = in_batch & (flat_idx < n_batch - CACHE_WINDOW)
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

    # State-cache path. block_table is sized to cover all logical blocks of
    # the request; only the last fixed_blocks_per_req entries are valid
    # (others are -1 sentinels for evicted blocks past the start, or 0 for
    # padding past the current end). The valid_block check filters both.
    use_cache = mask_pos & ~use_raw
    block_indices = pos // block_size
    block_indices_safe = tl.where(use_cache, block_indices, 0)
    block_numbers = tl.load(
        block_table_ptr + req_idx * block_table_stride + block_indices_safe,
        mask=use_cache,
        other=0,
    )
    valid_block = block_numbers > 0
    block_numbers_i64 = tl.where(valid_block, block_numbers, 0).to(tl.int64)
    block_offsets_raw = pos % block_size
    block_offsets = tl.where(mask_pos, block_offsets_raw, 0)

    row_base = (
        state_cache_ptr
        + block_numbers_i64 * state_cache_stride0
        + block_offsets * state_cache_stride1
        + head_offset
    )
    cache_mask = (use_cache & valid_block)[:, None] & mask[None, :]
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

    cache_block_ptr = k_cache_ptr + kv_block_idx.to(tl.int64) * KV_BLOCK_STRIDE
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
@triton.jit
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
    # Raw kv/score (current batch) + ape + batch context. Used for any
    # overlap-window position whose absolute ``pos`` falls inside the
    # current batch (``seq_start <= pos < seq_start + n_batch``); we read
    # the per-token contribution directly from these tensors instead of
    # the cyclic state_cache, which only retains the latest ~512 tokens
    # per request and would have been overwritten by later writes in
    # this same launch.
    kv_raw_ptr,
    kv_raw_stride,
    score_raw_ptr,
    score_raw_stride,
    ape_ptr,
    ape_stride,
    seq_start,
    n_batch,
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
    KV_BLOCK_STRIDE: tl.constexpr,
    CACHE_WINDOW: tl.constexpr,
):
    token_idx = tl.program_id(0)

    position = tl.load(positions_ptr + token_idx)
    if (position + 1) % COMPRESS_RATIO != 0:
        return

    # Note: we deliberately do NOT early-return on ``slot_mapping[token]<0``.
    # When a boundary's state slot has been evicted (cyclic ring overwrote
    # it within this same launch) the raw path provides the data — the
    # ``kv_slot_idx < 0`` check at the bottom is the real gate for whether
    # this token should write to the KV pool.

    req_idx = tl.load(token_to_req_indices_ptr + token_idx)

    start = position - (1 + OVERLAP) * COMPRESS_RATIO + 1
    tokens = tl.arange(0, (1 + OVERLAP) * COMPRESS_RATIO)
    pos = start + tokens
    mask_pos = pos >= 0
    head_offset = (tokens >= COMPRESS_RATIO).to(tl.int32) * HEAD_SIZE

    block = tl.arange(0, TRITON_BLOCK_SIZE)
    mask = block < HEAD_SIZE

    # ── Source dispatch: raw for the prefix evicted from cache ──
    # State pool: page-aligned, fixed_blocks_per_req physical blocks per
    # request (NOT a cyclic 512 ring). Cache retains the LAST cache_window
    # positions = (page_size + (seq_len-1) % page_size + 1) when seq_len >
    # page_size, else seq_len. Earlier in-batch positions whose logical
    # block has rolled out must come from raw kv_flat/score_flat.
    flat_idx = pos - seq_start
    in_batch = mask_pos & (flat_idx >= 0) & (flat_idx < n_batch)
    use_raw = in_batch & (flat_idx < n_batch - CACHE_WINDOW)
    flat_idx_safe = tl.where(use_raw, flat_idx, 0)

    # Raw path: kv_raw[flat_idx, head_offset:head_offset+HEAD_SIZE]
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
    # Add ape (state_cache stores score+ape; raw path adds it here)
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

    # State-cache path. block_table is sized to cover all logical blocks of
    # the request; only the last fixed_blocks_per_req entries are valid
    # (others are -1 sentinels for evicted blocks past the start, or 0 for
    # padding past the current end). The valid_block check filters both.
    use_cache = mask_pos & ~use_raw
    block_indices = pos // block_size
    block_indices_safe = tl.where(use_cache, block_indices, 0)
    block_numbers = tl.load(
        block_table_ptr + req_idx * block_table_stride + block_indices_safe,
        mask=use_cache,
        other=0,
    )
    valid_block = block_numbers > 0
    block_numbers_i64 = tl.where(valid_block, block_numbers, 0).to(tl.int64)
    block_offsets_raw = pos % block_size
    block_offsets = tl.where(mask_pos, block_offsets_raw, 0)

    row_base = (
        state_cache_ptr
        + block_numbers_i64 * state_cache_stride0
        + block_offsets * state_cache_stride1
        + head_offset
    )
    cache_mask = (use_cache & valid_block)[:, None] & mask[None, :]
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

    cache_block_ptr = k_cache_ptr + kv_block_idx.to(tl.int64) * KV_BLOCK_STRIDE
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


def run_save_partial_states(
    kv: torch.Tensor,  # [N, coff*head_dim] fp32 contiguous
    score: torch.Tensor,  # [N, coff*head_dim] fp32 contiguous
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
        HEAD_SIZE=head_size,
        TRITON_BLOCK_SIZE=triton.next_power_of_2(head_size),
        STATE_WIDTH=state_width,
        COMPRESS_RATIO=compress_ratio,
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
    # Raw current-batch tensors + ape + batch context. The state pool is a
    # cyclic buffer of only ~512 slots per request (2 fixed blocks * 256
    # entries), so within a single launch later writes overwrite earlier
    # tokens' state. The kernel reads these raw tensors directly for any
    # overlap-window position whose absolute pos falls inside the current
    # batch [seq_start, seq_start + n_batch).
    kv_raw: torch.Tensor,  # [N_batch, (1+overlap)*head_dim] fp32 contiguous
    score_raw: torch.Tensor,  # [N_batch, (1+overlap)*head_dim] fp32 contiguous
    ape: torch.Tensor,  # [compress_ratio, (1+overlap)*head_dim] fp32
    seq_start: int,  # absolute position of kv_raw[0] (== sp_int)
    disable_raw_path: bool = False,  # decode path: skip raw, read cache only
    *,
    head_dim: int,
    rope_head_dim: int,
    compress_ratio: int,
    overlap: bool,
    fixed_blocks_per_req: int = 2,
) -> None:
    """Boundary-token compress→norm→rope→fp8 quant→KV-pool store."""
    N = int(slot_mapping.shape[0])
    if N == 0:
        return
    cfg = _FUSED_CONSTEXPR_BY_HEAD_DIM.get(head_dim)
    if cfg is None:
        raise ValueError(f"Unsupported head_dim {head_dim} for fused compressor write")

    state_width = int(state_cache.shape[-1] // 2)
    state_block_size = int(state_cache.shape[1])
    kv_block_size = int(kv_cache.shape[1])
    kv_block_stride = int(kv_cache.stride(0))
    n_batch = 0 if disable_raw_path else int(kv_raw.shape[0])

    # Cache window per request: the framework keeps the last
    # ``fixed_blocks_per_req`` page-aligned state blocks. The most recent
    # block is partially filled with ``(seq_len-1) % page_size + 1`` valid
    # entries; older blocks are full. So the window is:
    #   seq_len <= page_size              -> seq_len
    #   page_size < seq_len <= 2*page_size -> page_size + (seq_len-1)%page_size + 1
    #   ...
    # (We support fixed_blocks_per_req == 2 as the only configured DSV4 case;
    # the formula generalises to ``min(seq_len, (k-1)*page_size + tail)``.)
    seq_len = int(seq_start) + n_batch
    if seq_len <= 0:
        cache_window = 0
    elif seq_len <= state_block_size:
        cache_window = seq_len
    else:
        tail = ((seq_len - 1) % state_block_size) + 1
        cache_window = (fixed_blocks_per_req - 1) * state_block_size + tail
        cache_window = min(cache_window, seq_len)

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
        n_batch,
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
        CACHE_WINDOW=cache_window,
        num_warps=cfg["num_warps"],
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
