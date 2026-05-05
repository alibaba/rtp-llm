"""DSV4 fused Compressor kernel for the **CSA / HCA KV pool** (head_dim=512).

Sister of ``_compressor_fused_triton.py`` (which targets the indexer's 132 B
``head_dim==128`` cache). This kernel writes the canonical
``fp8_model1_mla`` layout consumed by FlashMLA's
``flash_mla_sparse_fwd`` FP8 path; see ``cpp/cache/DSV4CacheConfig.h:78-91``.

Per-block byte layout (``cache_block_size`` tokens per block):

  bytes [0          : bs * 576)  -- token data, 576 B/token =
                                     448 fp8_e4m3 NoPE + 64 bf16 RoPE (128 B)
  bytes [bs * 576   : bs * 576 + bs * 8)
                                  -- per-token UE8M0 scales, 8 B/token
                                     (7 real, one per QUANT_BLOCK=64; 1 pad)
  trailing pad to satisfy ``DSV4PoolSpec::padded_block_size_bytes`` (576 B
                                     alignment for FlashMLA TMA path).

Per-token "entry" averages 576 + 8 = **584 B** which is the
``KV_ENTRY_BYTES_FP8`` constant in ``DSV4CacheConfig.h`` — but the data
and scale regions are **per-block striped**, NOT per-slot interleaved.
The matching reader is ``_swa_fp8_dequant_triton.py`` (miji's port of
vLLM's ``dequantize_and_gather_k_cache``); both compute scale offsets
as ``cache_block_size * 576 + pos_in_block * 8``.

Sources / references:

  * **vLLM ``quantize_and_insert_k_cache``** —
    ``vllm/v1/attention/ops/deepseek_v4_ops/cache_utils.py``
    (functions ``quantize_and_insert_k_cache`` /
    ``quantize_and_insert_k_kernel``). Drop-in algorithm reference for
    the bf16 → UE8M0 FP8 quant + striped-layout cache write. Our kernel
    fuses the upstream pool / RMSNorm / RoPE compute that vLLM keeps in
    a separate ``_fused_kv_compress_norm_rope_insert_sparse_attn``
    launch (same file's sibling).
  * **vLLM ``_fused_kv_compress_norm_rope_insert_sparse_attn``** —
    ``vllm/v1/attention/ops/deepseek_v4_ops/fused_compress_quant_cache.py``
    (lines 31-214). Algorithm reference for the upstream
    softmax / pool / RMSNorm / RoPE chain.
  * **Sister kernel ``_compressor_fused_triton.py``** in this repo —
    same adaptation pattern (pre-gathered ``[B, G, D_in]`` state instead
    of vLLM's per-token block-table walk).
  * **Reader ``_swa_fp8_dequant_triton.py``** in this repo — canonical
    striped-layout read pointer arithmetic matches this writer.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

# Canonical layout constants (must match DSV4CacheConfig.h:78-91 and the
# ``quantize_and_insert_k_cache`` reference). The "584" value is the
# **average** per-token byte cost (576 data + 8 scale), striped per-block.
KV_HEAD_DIM = 512
KV_NOPE_DIM = 448
KV_ROPE_DIM = 64  # rope_head_dim per fp8_model1_mla
KV_QUANT_BLOCK = 64  # one UE8M0 scale per 64-elem fp8 group
KV_N_NOPE_BLOCKS = KV_NOPE_DIM // KV_QUANT_BLOCK  # 7 real scales
KV_SCALES_PER_TOKEN = 8  # 7 real + 1 pad (matches vLLM SCALE_DIM=8)
KV_TOKEN_DATA_SIZE = KV_NOPE_DIM + 2 * KV_ROPE_DIM  # 448 + 128 = 576
KV_ENTRY_BYTES = KV_TOKEN_DATA_SIZE + KV_SCALES_PER_TOKEN  # 584 average / token
FP8_E4M3_MAX = 448.0


@triton.jit
def _v4_compressor_kv_fused_kernel(
    # ── input state (already gathered, [B, G, D_in] fp32) ──
    kv_ptr,  # fp32
    score_ptr,  # fp32
    kv_b_stride,
    kv_g_stride,
    score_b_stride,
    score_g_stride,
    # ── per-token slot / cos / sin ──
    slot_mapping_ptr,  # int64; -1 = skip
    rope_cos_ptr,  # fp32 [B, ROPE_HEAD_DIM/2]
    rope_sin_ptr,  # fp32 [B, ROPE_HEAD_DIM/2]
    rope_cs_b_stride,
    # ── RMSNorm ──
    rms_norm_weight_ptr,  # bf16 [HEAD_SIZE]
    rms_norm_eps,
    # ── output FP8 cache (canonical striped layout) ──
    cache_ptr,  # uint8 base
    cache_block_size,  # tokens per block
    cache_block_stride_bytes,  # bytes per block (TMA-padded; pass stride(0))
    # ── geometry ──
    HEAD_SIZE: tl.constexpr,  # 512
    G: tl.constexpr,
    BLOCK_D: tl.constexpr,  # next_pow2(HEAD_SIZE) = 512
    ROPE_HEAD_DIM: tl.constexpr,  # 64
    NOPE_HEAD_DIM: tl.constexpr,  # 448
    QUANT_BLOCK: tl.constexpr,  # 64
    N_NOPE_BLOCKS: tl.constexpr,  # 7
    SCALES_PER_TOKEN: tl.constexpr,  # 8 (= vLLM SCALE_DIM)
    TOKEN_DATA_SIZE: tl.constexpr,  # 576 (= 448 fp8 + 128 bf16)
    OVERLAP: tl.constexpr,  # 0 / 1
    SPLIT_D: tl.constexpr,  # = HEAD_SIZE when OVERLAP else 0
    FP8_MAX: tl.constexpr,  # 448.0
):
    """One program per compressed token. Striped per-block layout follows
    ``vllm/v1/attention/ops/deepseek_v4_ops/cache_utils.py::quantize_and_insert_k_kernel``
    (bytes ``[0, bs*576)`` for data, ``[bs*576, +bs*8)`` for scales).

    Reads pre-gathered ``[B, G, D_in]`` state — adaptation pattern from
    the sister ``_compressor_fused_triton.py`` so we don't repeat vLLM's
    per-token block-table walk for the upstream state read."""
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

    # ── Softmax over G — same pattern as vLLM sparse-attn kernel
    #     (fused_compress_quant_cache.py:115-127). ──
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

    # ── RMSNorm (fp32, bf16 weight cast to fp32) ──
    #     Same as vLLM sparse-attn kernel lines 131-135.
    rms_w = tl.load(rms_norm_weight_ptr + d_off, mask=d_mask, other=0.0).to(tl.float32)
    pooled_sq = tl.where(d_mask, pooled * pooled, 0.0)
    variance = tl.sum(pooled_sq, axis=0) / HEAD_SIZE
    rrms = tl.rsqrt(variance + rms_norm_eps)
    normed = pooled * rrms * rms_w

    # ── Cache pointers (striped layout — see ``quantize_and_insert_k_kernel``
    #     in vllm/.../cache_utils.py:62-80) ──
    block_idx = slot // cache_block_size
    block_off = slot % cache_block_size
    block_base = cache_ptr + block_idx * cache_block_stride_bytes
    # Token data is contiguous at the start of the block: bs * 576 bytes.
    token_data_ptr = block_base + block_off * TOKEN_DATA_SIZE
    # Scales follow ALL token data: + cache_block_size * 576 + pos * 8.
    token_scale_ptr = (
        block_base + cache_block_size * TOKEN_DATA_SIZE + block_off * SCALES_PER_TOKEN
    )

    # ── NoPE FP8-UE8M0 quant (per QUANT_BLOCK=64 elements) ──
    #     bf16 round-trip on the input matches vLLM sparse-attn reference
    #     (fused_compress_quant_cache.py:155-187). Note: vLLM's
    #     ``quantize_and_insert_k_kernel`` (the standalone variant we're
    #     borrowing the layout from) loops with ``tl.static_range`` and
    #     stores per-block; we keep the flat 2D reshape from the
    #     sparse-attn fused variant since we already have the full
    #     [BLOCK_D] vector in registers post-RMSNorm.
    NUM_PAIRS: tl.constexpr = BLOCK_D // 2
    NOPE_PAIRS: tl.constexpr = NOPE_HEAD_DIM // 2
    HALF_ROPE: tl.constexpr = ROPE_HEAD_DIM // 2
    INV_FP8_MAX: tl.constexpr = 1.0 / FP8_MAX

    quant_input = normed.to(tl.bfloat16).to(tl.float32)
    # Reshape *all* of [BLOCK_D] (= HEAD_SIZE = 512) into 8 blocks of 64.
    # The trailing block (pair_idx >= NOPE_PAIRS) is the rope half — its
    # quantized output is unused; we only store the first N_NOPE_BLOCKS
    # of fp8 bytes and the first N_NOPE_BLOCKS of UE8M0 scales.
    N_TOTAL_BLOCKS: tl.constexpr = BLOCK_D // QUANT_BLOCK  # 8
    quant_2d = tl.reshape(quant_input, (N_TOTAL_BLOCKS, QUANT_BLOCK))
    abs_2d = tl.abs(quant_2d)
    block_absmax = tl.max(abs_2d, axis=1)  # [N_TOTAL_BLOCKS] fp32
    block_absmax = tl.maximum(block_absmax, 1e-4)

    raw_scales = block_absmax * INV_FP8_MAX
    exponents = tl.ceil(tl.log2(raw_scales))
    inv_scales = tl.exp2(-exponents)
    inv_scales_col = tl.reshape(inv_scales, (N_TOTAL_BLOCKS, 1))
    x_scaled = quant_2d * inv_scales_col
    x_clamped = tl.minimum(tl.maximum(x_scaled, -FP8_MAX), FP8_MAX)
    x_fp8 = x_clamped.to(tl.float8e4nv)
    x_uint8 = x_fp8.to(tl.uint8, bitcast=True)
    x_uint8_flat = tl.reshape(x_uint8, (BLOCK_D,))

    # NoPE bytes: store first NOPE_HEAD_DIM bytes of the slot's data region.
    nope_mask = d_mask & (d_off < NOPE_HEAD_DIM)
    fp8_dst = (token_data_ptr + d_off).to(tl.pointer_type(tl.uint8))
    tl.store(fp8_dst, x_uint8_flat, mask=nope_mask)

    # UE8M0 scales — store first N_NOPE_BLOCKS bytes; trailing pad byte
    # zeroed by pool init (matches vLLM ``quantize_and_insert_k_kernel``
    # cache_utils.py:127-128 which explicitly stores 0 there).
    scale_idx = tl.arange(0, N_TOTAL_BLOCKS)
    encoded = exponents + 127.0
    encoded = tl.minimum(tl.maximum(encoded, 0.0), 255.0)
    scale_dst = (token_scale_ptr + scale_idx).to(tl.pointer_type(tl.uint8))
    tl.store(scale_dst, encoded.to(tl.uint8), mask=scale_idx < N_NOPE_BLOCKS)

    # ── RoPE on the trailing rope_head_dim, stored as bf16 ──
    #     vLLM sparse-attn lines 189-214; same GPT-J interleaved-pair form.
    pair_2d = tl.reshape(normed, (NUM_PAIRS, 2))
    even, odd = tl.split(pair_2d)  # each [NUM_PAIRS] fp32

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

    # bf16 RoPE store at byte offset NOPE_HEAD_DIM (= 448) inside the slot's
    # data region (128 bytes covering 64 elems).
    bf16_ptr = (token_data_ptr + NOPE_HEAD_DIM).to(tl.pointer_type(tl.bfloat16))
    rope_local = d_off - NOPE_HEAD_DIM
    is_rope = (d_off >= NOPE_HEAD_DIM) & d_mask
    tl.store(bf16_ptr + rope_local, rotated.to(tl.bfloat16), mask=is_rope)


def v4_compressor_kv_fused(
    kv_state: torch.Tensor,  # [B, G, D_in] fp32
    score_state: torch.Tensor,  # same shape, fp32
    slot_mapping: torch.Tensor,  # [B] int64; -1 = skip
    norm_weight: torch.Tensor,  # [HEAD_SIZE] bf16
    rope_cos: torch.Tensor,  # [B, ROPE_HEAD_DIM//2] fp32
    rope_sin: torch.Tensor,  # [B, ROPE_HEAD_DIM//2] fp32
    out_kv_cache_packed: torch.Tensor,  # uint8 [num_blocks, block_size, 584]
    cache_block_stride_bytes: int,  # pool_view.stride(0); TMA-padded
    *,
    overlap: bool,
    head_dim: int = KV_HEAD_DIM,
    rope_head_dim: int = KV_ROPE_DIM,
    norm_eps: float = 1e-6,
) -> None:
    """Fused {pool → RMSNorm → RoPE → FP8 quant → cache scatter} for the
    canonical CSA/HCA KV-cache layout (see module docstring).

    Skips tokens where ``slot_mapping[t] < 0``. In-place writes
    ``out_kv_cache_packed``; no return.

    Layout / quant scheme is byte-compatible with vLLM's
    ``quantize_and_insert_k_cache``
    (``vllm/v1/attention/ops/deepseek_v4_ops/cache_utils.py``); see the
    module docstring for the full source list.

    Currently locked to ``head_dim == 512`` and ``rope_head_dim == 64``
    (the only fp8_model1_mla shape).
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
        head_dim == KV_HEAD_DIM
    ), f"v4_compressor_kv_fused locked to head_dim==512 (CSA/HCA); got {head_dim}"
    assert (
        rope_head_dim == KV_ROPE_DIM
    ), f"v4_compressor_kv_fused locked to rope_head_dim==64; got {rope_head_dim}"
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
        and out_kv_cache_packed.shape[-1] == KV_ENTRY_BYTES
        and out_kv_cache_packed.dtype == torch.uint8
    ), (
        f"cache must be [num_blocks, block_size, {KV_ENTRY_BYTES}] uint8; "
        f"got {tuple(out_kv_cache_packed.shape)}/{out_kv_cache_packed.dtype}"
    )
    assert cache_block_stride_bytes > 0
    if B == 0:
        return

    cache_block_size = out_kv_cache_packed.shape[1]
    BLOCK_D = triton.next_power_of_2(head_dim)

    _v4_compressor_kv_fused_kernel[(B,)](
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
        cache_block_stride_bytes,
        HEAD_SIZE=head_dim,
        G=G,
        BLOCK_D=BLOCK_D,
        ROPE_HEAD_DIM=rope_head_dim,
        NOPE_HEAD_DIM=head_dim - rope_head_dim,
        QUANT_BLOCK=KV_QUANT_BLOCK,
        N_NOPE_BLOCKS=(head_dim - rope_head_dim) // KV_QUANT_BLOCK,
        SCALES_PER_TOKEN=KV_SCALES_PER_TOKEN,
        TOKEN_DATA_SIZE=KV_TOKEN_DATA_SIZE,
        OVERLAP=int(overlap),
        SPLIT_D=SPLIT_D,
        FP8_MAX=FP8_E4M3_MAX,
        num_warps=4,
    )
