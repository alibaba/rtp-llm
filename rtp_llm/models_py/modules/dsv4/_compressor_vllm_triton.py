"""DSV4 compressor Triton kernels — vLLM per-token state-pool flow, BF16
KV-pool writeback (no FP8 quantization).

Direct adaptation of the FP8 path in
``new_ft/RTP-LLM/.../dsv4/fp8/_compressor_vllm_triton.py``:

  * ``_save_partial_states_kernel`` — per-token write of (kv | score+ape)
    into the FP32 state cache (identical to the FP8 path).
  * ``_fused_kv_compress_norm_rope_insert_bf16`` — boundary fused writer
    (compress → softmax → RMSNorm → RoPE → BF16 KV-pool store). Single
    kernel; works for both ``head_dim==512`` (CSA / HCA, NoPE+RoPE) and
    ``head_dim==128`` (indexer compressor, NoPE+RoPE) because the BF16
    layout is uniform.

The boundary writer self-skips non-boundary tokens (early-exit when
``(position+1) % COMPRESS_RATIO != 0``), so the caller passes a flat
``[N_tok]`` slot_mapping and lets the kernel decide.

Source dispatch (see :func:`run_fused_compress_kv_write_bf16` for the
full reasoning) pins ``CACHE_WINDOW=0``, so the boundary writer reads
the entire in-batch range from raw and only consults the state pool for
(a) continuation-prefill prefix overlap and (b) decode. Prefill
correctness therefore does not depend on state-pool capacity at all.
"""

from __future__ import annotations

import os
from typing import Optional

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
# BF16 boundary writer: works for both head_dim=512 and head_dim=128.
# Mirrors the FP8 kernel's control flow but emits a single bf16 head_dim
# slot at the end (no FP8 quant, no UE8M0 scales).
# =============================================================================
@triton.jit
def _fused_kv_compress_norm_rope_insert_bf16(
    state_cache_ptr,
    state_cache_stride0,
    state_cache_stride1,
    token_to_req_indices_ptr,
    positions_ptr,
    slot_mapping_ptr,
    block_table_ptr,
    block_table_stride,
    block_size,
    # Raw current-batch tensors + ape + batch context. With the wrapper
    # forcing ``CACHE_WINDOW=0``, the kernel reads from these raw tensors
    # for *every* overlap-window position whose absolute pos falls inside
    # ``[seq_start, seq_start + n_batch)``; only positions before
    # ``seq_start`` (continuation-prefill prefix overlap or decode) hit
    # the state cache.
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
    k_cache_ptr,  # bf16 [total_slots, head_dim] flat, slot-addressed
    kv_slot_mapping_ptr,
    boundary_token_indices_ptr,
    HEAD_SIZE: tl.constexpr,
    TRITON_BLOCK_SIZE: tl.constexpr,
    STATE_WIDTH: tl.constexpr,
    COMPRESS_RATIO: tl.constexpr,
    OVERLAP: tl.constexpr,
    ROPE_HEAD_DIM: tl.constexpr,
    CACHE_WINDOW: tl.constexpr,
    COMPACT_BOUNDARY: tl.constexpr,
):
    pid = tl.program_id(0)
    if COMPACT_BOUNDARY:
        token_idx = tl.load(boundary_token_indices_ptr + pid)
    else:
        token_idx = pid

    position = tl.load(positions_ptr + token_idx)
    if not COMPACT_BOUNDARY:
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

    # ── Source dispatch ──
    # With ``CACHE_WINDOW=0`` (wrapper default) ``use_raw == in_batch``,
    # so all in-batch positions read from the raw tensors and only
    # ``pos < seq_start`` falls through to the cache path below.
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

    # State-cache path. The state pool is cyclic with fixed_blocks_per_req
    # physical blocks per request, matching the Python-side state slot mapping.
    use_cache = mask_pos & ~use_raw
    block_indices = (pos // block_size) % block_table_stride
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

    NOPE_HEAD_DIM: tl.constexpr = HEAD_SIZE - ROPE_HEAD_DIM
    HALF_ROPE: tl.constexpr = ROPE_HEAD_DIM // 2
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
    # For nope pairs (is_rope_pair=False): cos=1, sin=0 → pair passes through
    # unchanged, so a single ``result``-wide store correctly reproduces both
    # the un-rotated nope half and the rotated rope half of ``normed``.
    cos_v = tl.load(cache_base + cs_idx, mask=is_rope_pair, other=1.0)
    sin_v = tl.load(cache_base + HALF_ROPE + cs_idx, mask=is_rope_pair, other=0.0)

    new_even = even * cos_v - odd * sin_v
    new_odd = odd * cos_v + even * sin_v
    result = tl.interleave(new_even, new_odd)

    out_ptr = k_cache_ptr + kv_slot_idx.to(tl.int64) * HEAD_SIZE
    tl.store(out_ptr + block, result.to(tl.bfloat16), mask=mask)


@triton.jit
def _fused_kv_compress_norm_rope_insert_bf16_ratio128_tile(
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
    n_batch,
    kv_slot_mapping_ptr,
    boundary_token_indices_ptr,
    compressed_tmp_ptr,
    partial_sumsq_ptr,
    HEAD_SIZE: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    boundary_pid = tl.program_id(0)
    tile_pid = tl.program_id(1)
    token_idx = tl.load(boundary_token_indices_ptr + boundary_pid)
    kv_slot_idx = tl.load(kv_slot_mapping_ptr + token_idx)
    if kv_slot_idx < 0:
        return

    position = tl.load(positions_ptr + token_idx)
    req_idx = tl.load(token_to_req_indices_ptr + token_idx)

    tokens = tl.arange(0, 128)
    pos = position - 127 + tokens
    mask_pos = pos >= 0

    head = tile_pid * BLOCK_H + tl.arange(0, BLOCK_H)
    head_mask = head < HEAD_SIZE

    flat_idx = pos - seq_start
    in_batch = mask_pos & (flat_idx >= 0) & (flat_idx < n_batch)
    flat_idx_safe = tl.where(in_batch, flat_idx, 0)
    raw_mask = in_batch[:, None] & head_mask[None, :]

    kv_from_raw = tl.load(
        kv_raw_ptr + flat_idx_safe[:, None] * kv_raw_stride + head[None, :],
        mask=raw_mask,
        other=0.0,
    )
    score_from_raw = tl.load(
        score_raw_ptr + flat_idx_safe[:, None] * score_raw_stride + head[None, :],
        mask=raw_mask,
        other=0.0,
    )
    ape_rows = pos % 128
    ape_rows_safe = tl.where(in_batch, ape_rows, 0)
    ape_vals = tl.load(
        ape_ptr + ape_rows_safe[:, None] * ape_stride + head[None, :],
        mask=raw_mask,
        other=0.0,
    )
    score_from_raw = score_from_raw + ape_vals

    use_cache = mask_pos & ~in_batch
    block_indices = (pos // block_size) % block_table_stride
    block_indices_safe = tl.where(use_cache, block_indices, 0)
    block_numbers = tl.load(
        block_table_ptr + req_idx * block_table_stride + block_indices_safe,
        mask=use_cache,
        other=0,
    )
    valid_block = block_numbers > 0
    block_numbers_i64 = tl.where(valid_block, block_numbers, 0).to(tl.int64)
    block_offsets = tl.where(mask_pos, pos % block_size, 0)
    row_base = (
        state_cache_ptr
        + block_numbers_i64 * state_cache_stride0
        + block_offsets * state_cache_stride1
    )
    cache_mask = (use_cache & valid_block)[:, None] & head_mask[None, :]
    kv_from_cache = tl.load(
        row_base[:, None] + head[None, :], mask=cache_mask, other=0.0
    )
    score_from_cache = tl.load(
        row_base[:, None] + HEAD_SIZE + head[None, :],
        mask=cache_mask,
        other=0.0,
    )

    kv = tl.where(in_batch[:, None], kv_from_raw, kv_from_cache)
    score = tl.where(in_batch[:, None], score_from_raw, score_from_cache)
    final_valid = in_batch | (use_cache & valid_block)
    combined_mask = final_valid[:, None] & head_mask[None, :]

    score = tl.where(combined_mask, score, float("-inf"))
    score = tl.softmax(score, dim=0)
    kv = tl.where(combined_mask, kv, 0.0)
    compressed = tl.sum(kv * score, axis=0)

    tl.store(
        compressed_tmp_ptr + boundary_pid * HEAD_SIZE + head,
        compressed,
        mask=head_mask,
    )
    sumsq = tl.sum(compressed * compressed, axis=0)
    tl.store(partial_sumsq_ptr + boundary_pid * (HEAD_SIZE // BLOCK_H) + tile_pid, sumsq)


@triton.jit
def _fused_kv_compress_norm_rope_insert_bf16_ratio128_finalize(
    positions_ptr,
    rms_norm_weight_ptr,
    rms_norm_eps,
    cos_sin_cache_ptr,
    cos_sin_stride,
    k_cache_ptr,
    kv_slot_mapping_ptr,
    boundary_token_indices_ptr,
    compressed_tmp_ptr,
    partial_sumsq_ptr,
    HEAD_SIZE: tl.constexpr,
    TRITON_BLOCK_SIZE: tl.constexpr,
    ROPE_HEAD_DIM: tl.constexpr,
    NUM_TILES: tl.constexpr,
):
    boundary_pid = tl.program_id(0)
    token_idx = tl.load(boundary_token_indices_ptr + boundary_pid)
    kv_slot_idx = tl.load(kv_slot_mapping_ptr + token_idx)
    if kv_slot_idx < 0:
        return

    tile = tl.arange(0, NUM_TILES)
    partial = tl.load(partial_sumsq_ptr + boundary_pid * NUM_TILES + tile)
    variance = tl.sum(partial, axis=0) / HEAD_SIZE
    rrms = tl.rsqrt(variance + rms_norm_eps)

    block = tl.arange(0, TRITON_BLOCK_SIZE)
    mask = block < HEAD_SIZE
    compressed = tl.load(
        compressed_tmp_ptr + boundary_pid * HEAD_SIZE + block, mask=mask, other=0.0
    )
    rms_w = tl.load(rms_norm_weight_ptr + block, mask=mask, other=0.0)
    normed = compressed * rrms * rms_w

    NOPE_HEAD_DIM: tl.constexpr = HEAD_SIZE - ROPE_HEAD_DIM
    HALF_ROPE: tl.constexpr = ROPE_HEAD_DIM // 2
    NUM_PAIRS: tl.constexpr = TRITON_BLOCK_SIZE // 2
    NOPE_PAIRS: tl.constexpr = NOPE_HEAD_DIM // 2

    pair_2d = tl.reshape(normed, (NUM_PAIRS, 2))
    even, odd = tl.split(pair_2d)
    pair_idx = tl.arange(0, NUM_PAIRS)
    rope_pair_local = pair_idx - NOPE_PAIRS
    is_rope_pair = rope_pair_local >= 0
    cs_idx = tl.maximum(rope_pair_local, 0)

    position = tl.load(positions_ptr + token_idx)
    compressed_pos = (position // 128) * 128
    cache_base = cos_sin_cache_ptr + compressed_pos * cos_sin_stride
    cos_v = tl.load(cache_base + cs_idx, mask=is_rope_pair, other=1.0)
    sin_v = tl.load(cache_base + HALF_ROPE + cs_idx, mask=is_rope_pair, other=0.0)

    new_even = even * cos_v - odd * sin_v
    new_odd = odd * cos_v + even * sin_v
    result = tl.interleave(new_even, new_odd)

    out_ptr = k_cache_ptr + kv_slot_idx.to(tl.int64) * HEAD_SIZE
    tl.store(out_ptr + block, result.to(tl.bfloat16), mask=mask)


# =============================================================================
# Python wrappers
# =============================================================================
def _env_flag(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.lower() in ("1", "true", "yes", "on")


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _should_use_compact_boundary(
    n_tokens: int,
    n_boundaries: int,
    *,
    head_dim: int,
    compress_ratio: int,
) -> bool:
    """Heuristic from local CUDA13 microbenchmarks.

    Compact launch removes boundary self-skip blocks, but the extra index load
    is not free. Keep the dense launch for measured small/head128 cases where
    it is faster, and return early before this helper when there are no
    boundaries.
    """
    if n_boundaries <= 0:
        return False
    if compress_ratio == 128:
        return head_dim == 512
    if compress_ratio == 4:
        if head_dim == 128:
            return n_tokens >= 4096
        return n_tokens >= 1024
    return n_boundaries < n_tokens


def _should_use_ratio128_tiled(
    n_boundaries: int,
    *,
    head_dim: int,
    rope_head_dim: int,
    compress_ratio: int,
    overlap: bool,
) -> bool:
    if _env_flag("DSV4_KV_COMPRESS_RATIO128_TILED_DISABLE", False):
        return False
    if _env_flag("DSV4_KV_COMPRESS_RATIO128_HEAD512_TILED_DISABLE", False):
        return False
    if (
        head_dim not in (128, 512)
        or rope_head_dim != 64
        or compress_ratio != 128
        or overlap
        or n_boundaries <= 0
    ):
        return False
    min_boundaries = _env_int(
        "DSV4_KV_COMPRESS_RATIO128_HEAD512_TILED_MIN",
        _env_int("DSV4_KV_COMPRESS_RATIO128_TILED_MIN", 1),
    )
    return n_boundaries >= min_boundaries


def _select_kv_write_dispatch(
    n_tokens: int,
    n_boundaries: int,
    *,
    has_boundary_indices: bool,
    head_dim: int,
    rope_head_dim: int,
    compress_ratio: int,
    overlap: bool,
) -> str:
    if n_tokens == 0:
        return "skip"
    if has_boundary_indices and n_boundaries == 0:
        return "skip"
    if has_boundary_indices and _should_use_ratio128_tiled(
        n_boundaries,
        head_dim=head_dim,
        rope_head_dim=rope_head_dim,
        compress_ratio=compress_ratio,
        overlap=overlap,
    ):
        return "ratio128_tiled"
    if has_boundary_indices and _should_use_compact_boundary(
        n_tokens,
        n_boundaries,
        head_dim=head_dim,
        compress_ratio=compress_ratio,
    ):
        return "compact_boundary"
    return "dense"


def run_save_partial_states(
    kv: torch.Tensor,  # [N, coff*head_dim] fp32 contiguous
    score: torch.Tensor,  # [N, coff*head_dim] fp32 contiguous
    ape: torch.Tensor,  # [compress_ratio, coff*head_dim] fp32
    positions: torch.Tensor,  # [N] int64
    state_cache: torch.Tensor,  # [num_blocks, state_eb, 2*coff*head_dim] fp32
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


def _run_ratio128_tiled_bf16(
    state_cache: torch.Tensor,
    token_to_req_indices: torch.Tensor,
    positions: torch.Tensor,
    block_table: torch.Tensor,
    rms_norm_weight: torch.Tensor,
    rms_norm_eps: float,
    cos_sin_cache: torch.Tensor,
    kv_cache: torch.Tensor,
    kv_slot_mapping: torch.Tensor,
    kv_raw: torch.Tensor,
    score_raw: torch.Tensor,
    ape: torch.Tensor,
    seq_start: int,
    n_batch: int,
    boundary_token_indices: torch.Tensor,
    *,
    head_dim: int,
    rope_head_dim: int,
) -> None:
    boundary_count = int(boundary_token_indices.shape[0])
    if boundary_count == 0:
        return
    default_block_h = 32 if head_dim == 512 and boundary_count >= 512 else 64
    block_h = _env_int("DSV4_KV_COMPRESS_RATIO128_TILE_HEAD_BLOCK", default_block_h)
    if block_h not in (16, 32, 64, 128) or head_dim % block_h != 0:
        block_h = 64
    num_tiles = head_dim // block_h
    compressed_tmp = torch.empty(
        (boundary_count, head_dim), device=kv_cache.device, dtype=torch.float32
    )
    partial_sumsq = torch.empty(
        (boundary_count, num_tiles), device=kv_cache.device, dtype=torch.float32
    )

    _fused_kv_compress_norm_rope_insert_bf16_ratio128_tile[
        (boundary_count, num_tiles)
    ](
        state_cache,
        state_cache.stride(0),
        state_cache.stride(1),
        token_to_req_indices,
        positions,
        block_table,
        block_table.stride(0),
        int(state_cache.shape[1]),
        kv_raw,
        kv_raw.stride(0),
        score_raw,
        score_raw.stride(0),
        ape,
        ape.stride(0),
        seq_start,
        n_batch,
        kv_slot_mapping,
        boundary_token_indices,
        compressed_tmp,
        partial_sumsq,
        HEAD_SIZE=head_dim,
        BLOCK_H=block_h,
        num_warps=4,
    )
    _fused_kv_compress_norm_rope_insert_bf16_ratio128_finalize[
        (boundary_count,)
    ](
        positions,
        rms_norm_weight,
        rms_norm_eps,
        cos_sin_cache,
        cos_sin_cache.stride(0),
        kv_cache,
        kv_slot_mapping,
        boundary_token_indices,
        compressed_tmp,
        partial_sumsq,
        HEAD_SIZE=head_dim,
        TRITON_BLOCK_SIZE=triton.next_power_of_2(head_dim),
        ROPE_HEAD_DIM=rope_head_dim,
        NUM_TILES=num_tiles,
        num_warps=4,
    )


def run_fused_compress_kv_write_bf16(
    state_cache: torch.Tensor,  # [num_blocks, state_eb, 2*coff*head_dim] fp32
    token_to_req_indices: torch.Tensor,  # [N] int32
    positions: torch.Tensor,  # [N] int64
    slot_mapping: torch.Tensor,  # [N] int64 — state-pool slot per token
    block_table: torch.Tensor,  # [B, max_state_blocks] int — state-pool block table
    rms_norm_weight: torch.Tensor,  # [head_dim] bf16
    rms_norm_eps: float,
    cos_sin_cache: torch.Tensor,  # [max_pos, rope_head_dim] fp32
    kv_cache: torch.Tensor,  # [total_slots, head_dim] bf16 — flat slot-addressed
    kv_slot_mapping: torch.Tensor,  # [N] int64 — KV-pool slot per token (-1 if non-boundary)
    kv_raw: torch.Tensor,  # [N_batch, (1+overlap)*head_dim] fp32 contiguous
    score_raw: torch.Tensor,  # [N_batch, (1+overlap)*head_dim] fp32 contiguous
    ape: torch.Tensor,  # [compress_ratio, (1+overlap)*head_dim] fp32
    seq_start: int,  # absolute position of kv_raw[0]
    disable_raw_path: bool = False,
    boundary_token_indices: Optional[torch.Tensor] = None,
    *,
    head_dim: int,
    rope_head_dim: int,
    compress_ratio: int,
    overlap: bool,
) -> None:
    """Boundary-token compress→norm→rope→BF16 KV-pool store.

    Source dispatch policy (set ``CACHE_WINDOW=0`` always):
      * **Prefill / in-batch token** (``flat_idx in [0, n_batch)``) → raw
        path reads directly from ``kv_raw`` / ``score_raw``. State pool is
        never consulted, so its sizing / cyclic-overwrite behaviour does
        not affect prefill correctness.
      * **Continuation prefill, prefix overlap** (``pos < seq_start``)
        → cache path, reads from ``state_cache`` (populated by a previous
        prefill via :func:`run_save_partial_states`).
      * **Decode** (``disable_raw_path=True`` → ``n_batch=0``) → all
        overlap-window reads go through the cache path.

    The vLLM source kernel uses a non-zero ``CACHE_WINDOW`` so the back of
    the in-batch range falls into the (cache-friendly) state pool read; it
    requires a large per-request state pool (256 entries × 2 blocks = 512
    slots) to remain correct when batches exceed the cyclic capacity. We
    skip that micro-optimization: raw and cache reads are bit-equivalent,
    and forcing raw for the entire in-batch range eliminates state-pool
    sizing as a correctness constraint."""
    N = int(slot_mapping.shape[0])
    if N == 0:
        return
    if head_dim not in (128, 512):
        raise ValueError(
            f"BF16 vLLM compressor expects head_dim in {{128, 512}}; got {head_dim}"
        )

    state_width = int(state_cache.shape[-1] // 2)
    state_block_size = int(state_cache.shape[1])
    n_batch = 0 if disable_raw_path else int(kv_raw.shape[0])
    cache_window = 0

    compact_boundary = boundary_token_indices is not None
    if compact_boundary:
        assert boundary_token_indices is not None
        boundary_token_indices = boundary_token_indices.contiguous()
        boundary_count = int(boundary_token_indices.shape[0])
        if boundary_count == 0:
            return
        dispatch = _select_kv_write_dispatch(
            N,
            boundary_count,
            has_boundary_indices=True,
            head_dim=head_dim,
            rope_head_dim=rope_head_dim,
            compress_ratio=compress_ratio,
            overlap=overlap,
        )
        if dispatch == "ratio128_tiled":
            _run_ratio128_tiled_bf16(
                state_cache,
                token_to_req_indices,
                positions,
                block_table,
                rms_norm_weight,
                rms_norm_eps,
                cos_sin_cache,
                kv_cache,
                kv_slot_mapping,
                kv_raw,
                score_raw,
                ape,
                seq_start,
                n_batch,
                boundary_token_indices,
                head_dim=head_dim,
                rope_head_dim=rope_head_dim,
            )
            return
        compact_boundary = dispatch == "compact_boundary"
        grid_n = boundary_count if compact_boundary else N
        if not compact_boundary:
            boundary_token_indices = positions
    else:
        boundary_token_indices = positions
        grid_n = N

    num_warps = 4 if head_dim == 512 else 1

    _fused_kv_compress_norm_rope_insert_bf16[(grid_n,)](
        state_cache,
        state_cache.stride(0),
        state_cache.stride(1),
        token_to_req_indices,
        positions,
        slot_mapping,
        block_table,
        block_table.stride(0),
        state_block_size,
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
        boundary_token_indices,
        HEAD_SIZE=head_dim,
        TRITON_BLOCK_SIZE=triton.next_power_of_2(head_dim),
        STATE_WIDTH=state_width,
        COMPRESS_RATIO=compress_ratio,
        OVERLAP=overlap,
        ROPE_HEAD_DIM=rope_head_dim,
        CACHE_WINDOW=cache_window,
        COMPACT_BOUNDARY=compact_boundary,
        num_warps=num_warps,
    )


def build_cos_sin_cache(freqs_cis: torch.Tensor) -> "tuple[torch.Tensor, int]":
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
