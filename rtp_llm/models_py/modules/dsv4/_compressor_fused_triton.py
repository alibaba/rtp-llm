"""DSV4 fused Compressor kernel — pool + RMSNorm + RoPE + FP8 quant + cache write.

Port of vLLM's ``_fused_kv_compress_norm_rope_insert_indexer_attn``
(``vllm/v1/attention/ops/deepseek_v4_ops/fused_compress_quant_cache.py``)
adapted to RTP-LLM conventions.

The kernel collapses what was previously a 4-launch chain on the indexer's
nested compressor decode/prefill hot path:

    1. ``v4_compressor_pool``        — softmax(score) * kv → reduce over G
    2. ``rtp_llm_ops.rmsnorm``       — RMSNorm with bf16 weight
    3. ``apply_rotary_emb``           — partial RoPE on trailing rope_head_dim
    4. ``quantize_indexer_k``        — per-token FP8 (e4m3) absmax/448 quant
                                        + write into per-block grouped cache

into a single Triton launch (one program per token).

Differences from vLLM:

  * State is fed in already-staged form ``[B, G, D]`` (G = 2*ratio for CSA,
    G = ratio for HCA), not as a paged ``state_cache`` indexed via
    ``token_to_req_indices`` + ``positions`` + ``block_table``.  The
    compressor's ``_bind_state_from_pool`` gathers the state ahead of time;
    the kernel just consumes those tiles.
  * Slot-mapping only — drop the ``(position+1) % COMPRESS_RATIO`` boundary
    check; the caller bakes that into ``slot_mapping`` (PAD_ID = -1) the
    same way ``get_compressed_slot_mapping`` does in vLLM.
  * RoPE is supplied as separate ``rope_cos[B, rope_head_dim/2]`` and
    ``rope_sin[B, rope_head_dim/2]`` fp32 tensors (one row per token; caller
    gathers from its own ``freqs_cis`` per token).  Avoids encoding any
    particular layout convention inside the kernel.
  * FP8 cache layout matches RTP-LLM's ``_indexer_fp8_quant_triton``:
    per-block bytes ``[0, block_size*128)`` = K (token-major), bytes
    ``[block_size*128, +block_size*4)`` = fp32 scales (one per token).

Algorithm per token (slot >= 0):

    score = softmax(score, dim=G)                     # [G, D] fp32
    pooled = sum(kv * score, axis=G)                  # [D] fp32
    rrms   = 1 / sqrt(mean(pooled^2) + eps)
    normed = pooled * rrms * weight                   # bf16 weight
    rope on normed[..., -rope_head_dim:] in fp32
    bf16-roundtrip → absmax → fp32 scale = absmax/448
    fp8 store + scale store
"""

from __future__ import annotations

from typing import Optional

import torch
import triton
import triton.language as tl

# Kept compatible with _indexer_fp8_quant_triton.
INDEXER_HEAD_DIM = 128
INDEXER_ENTRY_BYTES = 132
FP8_E4M3_MAX = 448.0


@triton.jit
def _v4_compressor_fused_kernel(
    # ── input state (already gathered, [B, G, D] fp32) ──
    kv_ptr,  # fp32
    score_ptr,  # fp32
    kv_b_stride,
    kv_g_stride,
    score_b_stride,
    score_g_stride,
    # ── per-token slot/cos/sin ──
    slot_mapping_ptr,  # int64; -1 = skip
    rope_cos_ptr,  # fp32 [B, rope_head_dim/2]
    rope_sin_ptr,  # fp32 [B, rope_head_dim/2]
    rope_cs_b_stride,
    # ── RMSNorm weight ──
    rms_norm_weight_ptr,  # bf16 [HEAD_SIZE]
    rms_norm_eps,
    # ── output FP8 cache ──
    cache_ptr,  # uint8 [num_blocks, block_size, ENTRY_BYTES] flat
    cache_block_size,  # tokens per block
    cache_stride_b,  # bytes per block = block_size * ENTRY_BYTES
    # ── geometry / constexprs ──
    HEAD_SIZE: tl.constexpr,
    G: tl.constexpr,
    BLOCK_D: tl.constexpr,  # power-of-two ≥ HEAD_SIZE
    ROPE_HEAD_DIM: tl.constexpr,
    OVERLAP: tl.constexpr,  # 0 / 1
    SPLIT_D: tl.constexpr,  # = HEAD_SIZE when OVERLAP else 0
    FP8_MAX: tl.constexpr,
    ENTRY_BYTES: tl.constexpr,  # 132
):
    """One program per token. Mirrors vLLM indexer fused kernel (head=128,
    single quant block) but reads pre-gathered state."""
    pid = tl.program_id(0).to(tl.int64)

    slot = tl.load(slot_mapping_ptr + pid).to(tl.int64)
    if slot < 0:
        return

    g_off = tl.arange(0, G)
    d_off = tl.arange(0, BLOCK_D)
    d_mask = d_off < HEAD_SIZE

    # OVERLAP: kv/score laid out as [B, 2r, 2d]; first ratio rows take the
    # lower-half of the d axis, the rest take the upper half (matches the
    # post-cat view ``cat([:, :r, :d], [:, r:, d:])`` without copying).
    if OVERLAP:
        ratio = G // 2
        upper = (g_off >= ratio).to(tl.int64)
        d_idx = d_off[None, :] + upper[:, None] * SPLIT_D
    else:
        d_idx = d_off[None, :] + (g_off[:, None] - g_off[:, None])

    base_kv = kv_ptr + pid * kv_b_stride
    base_sc = score_ptr + pid * score_b_stride

    score_ptrs = base_sc + g_off[:, None] * score_g_stride + d_idx
    score = tl.load(score_ptrs, mask=d_mask[None, :], other=float("-inf")).to(
        tl.float32
    )
    score_max = tl.max(score, axis=0)
    score = tl.exp(score - score_max[None, :])
    score_sum = tl.sum(score, axis=0)
    score = score / score_sum[None, :]

    kv_ptrs = base_kv + g_off[:, None] * kv_g_stride + d_idx
    kv = tl.load(kv_ptrs, mask=d_mask[None, :], other=0.0).to(tl.float32)

    pooled = tl.sum(kv * score, axis=0)  # [BLOCK_D] fp32

    # ── RMSNorm (fp32 throughout, bf16 weight cast to fp32) ──
    rms_w = tl.load(rms_norm_weight_ptr + d_off, mask=d_mask, other=0.0).to(tl.float32)
    pooled_sq = tl.where(d_mask, pooled * pooled, 0.0)
    variance = tl.sum(pooled_sq, axis=0) / HEAD_SIZE
    rrms = tl.rsqrt(variance + rms_norm_eps)
    normed = pooled * rrms * rms_w

    # ── Partial RoPE on the trailing rope_head_dim ──
    NOPE_HEAD_DIM: tl.constexpr = HEAD_SIZE - ROPE_HEAD_DIM
    HALF_ROPE: tl.constexpr = ROPE_HEAD_DIM // 2
    NUM_PAIRS: tl.constexpr = BLOCK_D // 2
    NOPE_PAIRS: tl.constexpr = NOPE_HEAD_DIM // 2

    normed_2d = tl.reshape(normed, (NUM_PAIRS, 2))
    even, odd = tl.split(normed_2d)  # [NUM_PAIRS] fp32 each

    pair_idx = tl.arange(0, NUM_PAIRS)
    rope_pair_local = pair_idx - NOPE_PAIRS
    is_rope_pair = (rope_pair_local >= 0) & (rope_pair_local < HALF_ROPE)
    cs_idx = tl.maximum(tl.minimum(rope_pair_local, HALF_ROPE - 1), 0)

    cs_base = pid * rope_cs_b_stride
    cos_v = tl.load(rope_cos_ptr + cs_base + cs_idx, mask=is_rope_pair, other=1.0).to(
        tl.float32
    )
    sin_v = tl.load(rope_sin_ptr + cs_base + cs_idx, mask=is_rope_pair, other=0.0).to(
        tl.float32
    )

    new_even = tl.where(is_rope_pair, even * cos_v - odd * sin_v, even)
    new_odd = tl.where(is_rope_pair, odd * cos_v + even * sin_v, odd)
    rotated = tl.interleave(new_even, new_odd)  # [BLOCK_D] fp32

    # ── FP8 UE8M0 quant (single block, head_dim==QUANT_BLOCK). Matches
    # vLLM ``_fused_kv_compress_norm_rope_insert_indexer_attn``
    # (vendored at _vllm_ref/fused_compress_quant_cache.py:368-391):
    #   absmax    = max(|rotated_bf|, 1e-4)
    #   exponent  = ceil(log2(absmax / FP8_MAX))
    #   inv_scale = 2^(-exponent)               # divide by power-of-2
    #   scale     = 2^exponent                  # stored as fp32 scale
    # DeepGEMM ``fp8_paged_mqa_logits`` consumes the fp32 scale; UE8M0
    # snapping makes scale storage hardware-friendly and matches vLLM's
    # canonical layout. bf16 round-trip on the input mirrors downstream
    # numerics (pool+norm+rope are otherwise fp32 throughout).
    INV_FP8_MAX: tl.constexpr = 1.0 / FP8_MAX
    rotated_bf = rotated.to(tl.bfloat16).to(tl.float32)
    abs_v = tl.where(d_mask, tl.abs(rotated_bf), 0.0)
    absmax = tl.max(abs_v, axis=0)
    absmax = tl.maximum(absmax, 1e-4)
    raw_scale = absmax * INV_FP8_MAX
    exponent = tl.ceil(tl.log2(raw_scale))
    inv_scale = tl.exp2(-exponent)
    scale = tl.exp2(exponent)
    x_scaled = rotated_bf * inv_scale
    x_clamped = tl.minimum(tl.maximum(x_scaled, -FP8_MAX), FP8_MAX)
    q_fp8 = x_clamped.to(tl.float8e4nv)
    q_u8 = q_fp8.to(tl.uint8, bitcast=True)

    # ── Write FP8 K + fp32 scale into the per-block grouped cache layout ──
    block_idx = slot // cache_block_size
    block_off = slot % cache_block_size
    block_base = cache_ptr + block_idx * cache_stride_b
    k_dst = (block_base + block_off * HEAD_SIZE + d_off).to(tl.pointer_type(tl.uint8))
    tl.store(k_dst, q_u8, mask=d_mask)

    scale_region = block_base + cache_block_size * HEAD_SIZE
    scale_dst = (scale_region + block_off * 4).to(tl.pointer_type(tl.float32))
    tl.store(scale_dst, scale)


def v4_compressor_fused(
    kv_state: torch.Tensor,  # [B, 2r, 2d] (overlap) or [B, r, d] (non-overlap), fp32
    score_state: torch.Tensor,  # same shape as kv_state, fp32
    slot_mapping: torch.Tensor,  # [B] int64; -1 = skip
    norm_weight: torch.Tensor,  # [head_dim] bf16
    rope_cos: torch.Tensor,  # [B, rope_head_dim//2] fp32
    rope_sin: torch.Tensor,  # [B, rope_head_dim//2] fp32
    out_kv_cache_packed: torch.Tensor,  # [num_blocks, block_size, ENTRY_BYTES] uint8
    *,
    overlap: bool,
    head_dim: int = INDEXER_HEAD_DIM,
    rope_head_dim: int,
    norm_eps: float = 1e-6,
) -> None:
    """Fused {pool → RMSNorm → RoPE → FP8 quant → cache scatter}.

    In-place writes ``out_kv_cache_packed``; no return value.  Skips tokens
    with ``slot_mapping[t] < 0``.

    Currently restricted to ``head_dim == 128`` (indexer compressor).  The
    quant step relies on ``QUANT_BLOCK == HEAD_SIZE`` (single-block flat
    reduction); the kernel itself generalises but the cache layout written
    here only matches the indexer's per-block grouped FP8 region.
    """
    assert (
        kv_state.dim() == 3 and score_state.shape == kv_state.shape
    ), f"kv/score must be [B, G, D] (got {tuple(kv_state.shape)})"
    assert kv_state.dtype == torch.float32 and score_state.dtype == torch.float32
    assert kv_state.is_contiguous() and score_state.is_contiguous()
    B, G, D_in = kv_state.shape
    if overlap:
        assert G % 2 == 0, "overlap requires G = 2*ratio"
        assert (
            D_in == 2 * head_dim
        ), f"overlap raw-state D_in must be 2*head_dim ({head_dim}); got {D_in}"
        SPLIT_D = head_dim
    else:
        assert D_in == head_dim, f"non-overlap D_in must == head_dim; got {D_in}"
        SPLIT_D = 0
    assert (
        head_dim == INDEXER_HEAD_DIM
    ), f"v4_compressor_fused currently locked to head_dim==128 (indexer); got {head_dim}"
    assert G & (G - 1) == 0 and G <= 256, f"G must be power-of-2 ≤256, got {G}"

    assert slot_mapping.dim() == 1 and slot_mapping.shape[0] == B
    if slot_mapping.dtype != torch.int64:
        slot_mapping = slot_mapping.to(torch.int64)
    slot_mapping = slot_mapping.contiguous()

    assert norm_weight.dim() == 1 and norm_weight.shape[0] == head_dim
    assert norm_weight.dtype == torch.bfloat16

    assert rope_cos.shape == (B, rope_head_dim // 2)
    assert rope_sin.shape == rope_cos.shape
    assert rope_cos.dtype == torch.float32 and rope_sin.dtype == torch.float32
    rope_cos = rope_cos.contiguous()
    rope_sin = rope_sin.contiguous()

    assert (
        out_kv_cache_packed.dim() == 3
        and out_kv_cache_packed.shape[-1] == INDEXER_ENTRY_BYTES
        and out_kv_cache_packed.dtype == torch.uint8
    ), (
        f"cache must be [num_blocks, block_size, {INDEXER_ENTRY_BYTES}] uint8; "
        f"got {tuple(out_kv_cache_packed.shape)}/{out_kv_cache_packed.dtype}"
    )
    if B == 0:
        return

    cache_block_size = out_kv_cache_packed.shape[1]
    cache_stride_b = cache_block_size * INDEXER_ENTRY_BYTES
    BLOCK_D = triton.next_power_of_2(head_dim)

    _v4_compressor_fused_kernel[(B,)](
        kv_state,
        score_state,
        kv_state.stride(0),
        kv_state.stride(1),
        score_state.stride(0),
        score_state.stride(1),
        slot_mapping,
        rope_cos,
        rope_sin,
        rope_cos.stride(0),
        norm_weight,
        norm_eps,
        out_kv_cache_packed,
        cache_block_size,
        cache_stride_b,
        HEAD_SIZE=head_dim,
        G=G,
        BLOCK_D=BLOCK_D,
        ROPE_HEAD_DIM=rope_head_dim,
        OVERLAP=int(overlap),
        SPLIT_D=SPLIT_D,
        FP8_MAX=FP8_E4M3_MAX,
        ENTRY_BYTES=INDEXER_ENTRY_BYTES,
        num_warps=4,
    )


def freqs_cis_to_cos_sin(
    freqs_cis_per_b: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert ``freqs_cis [..., k] complex64`` into ``(cos, sin)`` fp32 pair.

    Compressor / Indexer hold ``freqs_cis`` as a complex tensor where
    ``apply_rotary_emb`` does ``view_as_complex`` on consecutive
    ``(even, odd)`` pairs and multiplies; that is identical to the GPT-J
    interleaved-pair formula the fused kernel applies (``new_even =
    even*cos - odd*sin``, ``new_odd = even*sin + odd*cos``).

    So we just split into ``real = cos`` and ``imag = sin``.
    """
    assert freqs_cis_per_b.dtype == torch.complex64
    cos = freqs_cis_per_b.real.contiguous().to(torch.float32)
    sin = freqs_cis_per_b.imag.contiguous().to(torch.float32)
    return cos, sin
