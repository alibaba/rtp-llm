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

Both fused writers self-skip non-boundary tokens (early-exit when
``(position+1) % COMPRESS_RATIO != 0``), so the caller passes a flat
``[N_tok]`` slot_mapping and lets the kernel decide.

NOTE on state-pool sizing: this kernel mirrors vLLM's "per-token state
pool" design, which in the source's C++ config uses
``entries_per_block=256, fixed_blocks_per_req=2`` (= 512 slots per
request). The current project's C++ config uses much smaller per-block
counts (CSA / INDEXER state ``entries_per_block=8``, HCA state
``entries_per_block=128``) — for short prefills + decode this still works
because the boundary kernel falls back to the raw ``kv_flat`` /
``score_flat`` for in-batch positions whose state has been cyclically
overwritten. For long prefills past the cyclic capacity, out-of-batch
positions will be wrong. Bump the state-pool ``entries_per_block`` in
``DSV4ConfigCreator.cc`` to match the source if you need long-prefill
correctness.
"""

from __future__ import annotations

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
    # Raw current-batch tensors + ape + batch context. The state pool
    # is a cyclic buffer (fixed_blocks_per_req * entries_per_block slots
    # per request), so within a single launch later writes overwrite
    # earlier tokens' state. The kernel reads these raw tensors directly
    # for any overlap-window position whose absolute pos falls inside
    # ``[seq_start, seq_start + n_batch)`` AND whose state is no longer
    # in the cyclic cache window.
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
    HEAD_SIZE: tl.constexpr,
    TRITON_BLOCK_SIZE: tl.constexpr,
    STATE_WIDTH: tl.constexpr,
    COMPRESS_RATIO: tl.constexpr,
    OVERLAP: tl.constexpr,
    ROPE_HEAD_DIM: tl.constexpr,
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

    # State-cache path. block_table covers all logical blocks of the
    # request; cyclic blocks past fixed_blocks_per_req hold sentinel ids
    # (==0). The valid_block check filters those.
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


# =============================================================================
# Python wrappers
# =============================================================================
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
    *,
    head_dim: int,
    rope_head_dim: int,
    compress_ratio: int,
    overlap: bool,
    fixed_blocks_per_req: int = 2,
) -> None:
    """Boundary-token compress→norm→rope→BF16 KV-pool store."""
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

    # Cache window per request: the framework keeps the last
    # ``fixed_blocks_per_req`` page-aligned state blocks. The most recent
    # block is partially filled with ``(seq_len-1) % page_size + 1`` valid
    # entries; older blocks are full. So the window is:
    #   seq_len <= page_size                 -> seq_len
    #   page_size < seq_len <= 2*page_size   -> page_size + (seq_len-1)%page_size + 1
    #   ...
    seq_len = int(seq_start) + n_batch
    if seq_len <= 0:
        cache_window = 0
    elif seq_len <= state_block_size:
        cache_window = seq_len
    else:
        tail = ((seq_len - 1) % state_block_size) + 1
        cache_window = (fixed_blocks_per_req - 1) * state_block_size + tail
        cache_window = min(cache_window, seq_len)

    num_warps = 4 if head_dim == 512 else 1

    _fused_kv_compress_norm_rope_insert_bf16[(N,)](
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
        HEAD_SIZE=head_dim,
        TRITON_BLOCK_SIZE=triton.next_power_of_2(head_dim),
        STATE_WIDTH=state_width,
        COMPRESS_RATIO=compress_ratio,
        OVERLAP=overlap,
        ROPE_HEAD_DIM=rope_head_dim,
        CACHE_WINDOW=cache_window,
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
