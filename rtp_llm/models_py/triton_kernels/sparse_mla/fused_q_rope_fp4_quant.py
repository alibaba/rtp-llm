"""Fused RoPE + Hadamard + FP4 (e2m1, UE8M0 per-group) Q quantization.

FP4 sibling of ``fused_q_rope_quant.py`` for the GLM5 indexer FP4 path.

Single Triton kernel that:
1. Applies RoPE on the first ROT_DIM dims (NeOX or GPT-J interleaved)
2. Passes through NoPE dims unchanged
3. Applies N-point Walsh-Hadamard transform on the full HEAD_DIM vector
4. Quantizes to FP4 e2m1 with per-32-element UE8M0 scales, packing two FP4
   nibbles per byte and packing four UE8M0 exponent bytes per int32 (matches
   ``deep_gemm.utils.per_token_cast_to_fp4(use_ue8m0=True, gran_k=32,
   use_packed_ue8m0=True)``)

Output shapes match ``IndexerOp._fp4_quant_q``:
  q_fp4:   int8   [num_tokens, num_heads, HEAD_DIM // 2]
  q_scale: int32  [num_tokens, num_heads]   (1 int32 = 4 packed UE8M0 bytes)

Hadamard reuses the same butterfly scratch buffer as the FP8 path
(``LOG2_HEAD_DIM`` stages, ``HEAD_DIM`` float32s per program).
"""

import math

import torch
import triton
import triton.language as tl

_FP4_GRAN_K = 32
_FP4_SCALES_PACK_FACTOR = 4


@triton.jit
def _fused_q_rope_hadamard_fp4_quant_kernel(
    pos_ptr,
    q_ptr,
    q_stride0,
    q_stride1,
    cos_sin_cache_ptr,
    cos_sin_cache_stride,
    HALF_ROT_DIM: tl.constexpr,
    q_fp4_ptr,
    q_fp4_stride0,
    q_fp4_stride1,
    q_scale_ptr,
    q_scale_stride0,
    scratch_ptr,
    HEAD_DIM: tl.constexpr,
    IS_NEOX_STYLE: tl.constexpr,
    LOG2_HEAD_DIM: tl.constexpr,
    GRAN_K: tl.constexpr,
    SCALES_PACK: tl.constexpr,
):
    """Grid: [num_tokens, num_heads]; one program per (token, head).

    Pipeline:
      Load → RoPE → Hadamard (LOG2_HEAD_DIM butterfly stages) → FP4 + UE8M0.

    Matches ``deep_gemm.utils.per_token_cast_to_fp4`` byte-for-byte at the
    cache slot: code table {0, ±.5, ±1, ±1.5, ±2, ±3, ±4, ±6} with sign
    bit at position 3, low nibble = even index / high nibble = odd index.
    """
    ROT_DIM: tl.constexpr = 2 * HALF_ROT_DIM
    NOPE_DIM: tl.constexpr = HEAD_DIM - ROT_DIM
    N_QUANT_GROUPS: tl.constexpr = HEAD_DIM // GRAN_K
    HALF_DIM: tl.constexpr = HEAD_DIM // 2
    tl.static_assert(NOPE_DIM >= 0)
    tl.static_assert(HEAD_DIM % GRAN_K == 0)
    tl.static_assert(N_QUANT_GROUPS == SCALES_PACK)  # int32 packing assumes 4 groups

    tok_idx = tl.program_id(0).to(tl.int64)
    head_idx = tl.program_id(1)
    num_heads = tl.num_programs(1)

    pos = tl.load(pos_ptr + tok_idx)
    base_ptr = q_ptr + tok_idx * q_stride0 + head_idx * q_stride1

    work_id = tok_idx * num_heads + head_idx
    scratch_base = scratch_ptr + work_id * HEAD_DIM

    # ---- Step 1: RoPE on first ROT_DIM dims → scratch ----
    half_offset = tl.arange(0, HALF_ROT_DIM)
    cos = tl.load(cos_sin_cache_ptr + pos * cos_sin_cache_stride + half_offset).to(
        tl.float32
    )
    sin = tl.load(
        cos_sin_cache_ptr + pos * cos_sin_cache_stride + half_offset + HALF_ROT_DIM
    ).to(tl.float32)

    if IS_NEOX_STYLE:
        x_first = tl.load(base_ptr + half_offset).to(tl.float32)
        x_second = tl.load(base_ptr + half_offset + HALF_ROT_DIM).to(tl.float32)
        r_first = x_first * cos - x_second * sin
        r_second = x_second * cos + x_first * sin
        r_first = r_first.to(tl.bfloat16).to(tl.float32)
        r_second = r_second.to(tl.bfloat16).to(tl.float32)
        tl.store(scratch_base + half_offset, r_first)
        tl.store(scratch_base + half_offset + HALF_ROT_DIM, r_second)
    else:
        x_even = tl.load(base_ptr + half_offset * 2).to(tl.float32)
        x_odd = tl.load(base_ptr + half_offset * 2 + 1).to(tl.float32)
        r_first = x_even * cos - x_odd * sin
        r_second = x_odd * cos + x_even * sin
        r_first = r_first.to(tl.bfloat16).to(tl.float32)
        r_second = r_second.to(tl.bfloat16).to(tl.float32)
        tl.store(scratch_base + half_offset * 2, r_first)
        tl.store(scratch_base + half_offset * 2 + 1, r_second)

    # ---- Step 2: NoPE dims pass-through → scratch ----
    if NOPE_DIM > 0:
        nope_offset = tl.arange(0, NOPE_DIM)
        x_nope = tl.load(base_ptr + ROT_DIM + nope_offset).to(tl.float32)
        tl.store(scratch_base + ROT_DIM + nope_offset, x_nope)

    # ---- Step 3: Walsh-Hadamard butterfly ----
    idx = tl.arange(0, HEAD_DIM)
    tl.debug_barrier()
    for s_log in range(LOG2_HEAD_DIM):
        stride = 1 << s_log
        is_upper = (idx & stride) != 0
        partner_idx = idx ^ stride
        self_val = tl.load(scratch_base + idx)
        partner_val = tl.load(scratch_base + partner_idx)
        result = tl.where(is_upper, partner_val - self_val, self_val + partner_val)
        tl.store(scratch_base + idx, result)
        tl.debug_barrier()

    # ---- Step 4: per-group UE8M0 scale ----
    # Compute one UE8M0 byte per 32-elem group via fp32 bit math (matches
    # deep_gemm.utils.ceil_to_ue8m0). For HD=128, GRAN_K=32: 4 groups.
    group_idx = idx // GRAN_K  # [HEAD_DIM]
    # bf16 roundtrip after the * D^-0.5 normalization mirrors the production
    # path that feeds ``deep_gemm.per_token_cast_to_fp4`` a bf16 tensor.
    # Without it, fp32 values lying just past an FP4 midpoint (e.g. 0.7499)
    # would round one nibble differently than the deep_gemm reference does.
    data = (
        (tl.load(scratch_base + idx) * (HEAD_DIM**-0.5)).to(tl.bfloat16).to(tl.float32)
    )
    abs_data = tl.abs(data)

    # amax per group — broadcast each group's elements to that group's amax.
    # Use a one-hot mask trick: for each group g, take max over abs_data
    # where group_idx == g, store in a length-N_QUANT_GROUPS register vector.
    g_offsets = tl.arange(0, N_QUANT_GROUPS)  # [4]
    # mask: [HEAD_DIM, N_QUANT_GROUPS]
    mask_2d = group_idx[:, None] == g_offsets[None, :]
    masked = tl.where(mask_2d, abs_data[:, None], 0.0)
    amax = tl.max(masked, axis=0)  # [N_QUANT_GROUPS]
    amax = tl.maximum(amax, 6.0 * (2.0**-126))

    # UE8M0 scale = 2^ceil(log2(amax / 6.0)). Mantissa-aware ceil matches
    # ceil_to_ue8m0(): exp_up = ((bits>>23)&0xFF) + (mantissa != 0).
    sf = amax / 6.0
    sf_bits = sf.to(tl.int32, bitcast=True)
    exp = (sf_bits >> 23) & 0xFF
    mant = sf_bits & 0x7FFFFF
    exp = exp + (mant != 0).to(tl.int32)
    exp = tl.minimum(tl.maximum(exp, 1), 254)  # [N_QUANT_GROUPS] int32 in [1, 254]

    # Broadcast each group's 1/scale to its elements. scale = 2^(exp - 127),
    # so 1/scale = 2^(127 - exp); use exp2 for clarity over bit-shifting.
    inv_scale_per_group = tl.exp2(127.0 - exp.to(tl.float32))
    # Gather per-element inv_scale: pick from inv_scale_per_group[group_idx[i]]
    inv_scale_2d = tl.where(mask_2d, inv_scale_per_group[None, :], 0.0)
    inv_scale = tl.sum(inv_scale_2d, axis=1)  # [HEAD_DIM]
    scaled = data * inv_scale  # [HEAD_DIM] fp32

    # ---- FP4 e2m1 quantize (matches _quantize_to_fp4_e2m1) ----
    # boundaries (positive): 0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0.
    # Strict ``>`` so midpoints fall into the lower level — matches
    # ``torch.bucketize(ax, boundaries, right=False)`` used by deep_gemm.
    ax = tl.minimum(tl.abs(scaled), 6.0)
    code = (ax > 0.25).to(tl.int32)
    code = code + (ax > 0.75).to(tl.int32)
    code = code + (ax > 1.25).to(tl.int32)
    code = code + (ax > 1.75).to(tl.int32)
    code = code + (ax > 2.5).to(tl.int32)
    code = code + (ax > 3.5).to(tl.int32)
    code = code + (ax > 5.0).to(tl.int32)
    sign = ((scaled < 0) & (code != 0)).to(tl.int32)
    code = code | (sign << 3)  # 4-bit e2m1 code in low nibble
    code = code & 0x0F  # mask to nibble

    # ---- pack 2 codes/byte: low nibble = even index, high nibble = odd ----
    # Use one-hot pair selection (parity-mask) to gather even/odd halves.
    pair_idx = tl.arange(0, HALF_DIM)
    even_pos = pair_idx * 2  # [HALF_DIM]
    odd_pos = pair_idx * 2 + 1
    # Same parity-mask trick as amax-per-group above.
    even_mask = idx[:, None] == even_pos[None, :]  # [HEAD_DIM, HALF_DIM]
    odd_mask = idx[:, None] == odd_pos[None, :]
    even_codes = tl.sum(tl.where(even_mask, code[:, None], 0), axis=0)  # [HALF_DIM]
    odd_codes = tl.sum(tl.where(odd_mask, code[:, None], 0), axis=0)
    packed = ((even_codes & 0x0F) | ((odd_codes & 0x0F) << 4)).to(tl.uint8)

    fp4_base = q_fp4_ptr + tok_idx * q_fp4_stride0 + head_idx * q_fp4_stride1
    tl.store(fp4_base + pair_idx, packed)

    # ---- pack 4 UE8M0 bytes (one per group) into one int32 ----
    # exp[g] is the UE8M0 byte for group g; pack as
    #   int32 = exp[0] | (exp[1]<<8) | (exp[2]<<16) | (exp[3]<<24)
    shifts = (g_offsets * 8).to(tl.int32)  # [N_QUANT_GROUPS]
    packed_scale = tl.sum((exp & 0xFF) << shifts, axis=0)  # scalar int32
    tl.store(q_scale_ptr + tok_idx * q_scale_stride0 + head_idx, packed_scale)


def fused_q_rope_fp4_quant(
    q: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    index_n_heads: int,
    index_head_dim: int,
    rope_head_dim: int,
    is_neox_style: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fused RoPE + Hadamard + FP4 quant for indexer Q (Blackwell only).

    Args:
        q: [num_tokens, index_n_heads, index_head_dim] bf16/fp16
        positions: [num_tokens] int64
        cos_sin_cache: [max_pos, rope_head_dim]  (RoPE cos/sin lookup)
        index_n_heads: number of indexer heads
        index_head_dim: indexer head dim (must be power of 2; v1 supports 128)
        rope_head_dim: total rotation dim
        is_neox_style: True for NeOX-style RoPE

    Returns:
        (q_fp4, q_scale) matching ``IndexerOp._fp4_quant_q`` output:
            q_fp4:   int8  [num_tokens, index_n_heads, index_head_dim // 2]
            q_scale: int32 [num_tokens, index_n_heads]  (4 packed UE8M0)
    """
    assert (
        index_head_dim % _FP4_GRAN_K == 0
    ), f"index_head_dim={index_head_dim} must be a multiple of gran_k={_FP4_GRAN_K}"
    n_groups = index_head_dim // _FP4_GRAN_K
    assert n_groups == _FP4_SCALES_PACK_FACTOR, (
        f"v1 supports exactly {_FP4_SCALES_PACK_FACTOR} groups per token "
        f"(1 int32 = {_FP4_SCALES_PACK_FACTOR} packed UE8M0); got {n_groups} "
        f"(head_dim={index_head_dim})"
    )
    num_tokens = q.shape[0]
    half_rot_dim = rope_head_dim // 2
    log2_head_dim = int(math.log2(index_head_dim))
    assert 2**log2_head_dim == index_head_dim, "index_head_dim must be a power of 2"

    q_fp4 = torch.empty(
        (num_tokens, index_n_heads, index_head_dim // 2),
        dtype=torch.int8,
        device=q.device,
    )
    q_scale = torch.empty(
        (num_tokens, index_n_heads),
        dtype=torch.int32,
        device=q.device,
    )
    scratch = torch.empty(
        num_tokens * index_n_heads * index_head_dim,
        dtype=torch.float32,
        device=q.device,
    )

    if num_tokens > 0:
        grid = (num_tokens, index_n_heads)
        _fused_q_rope_hadamard_fp4_quant_kernel[grid](
            positions,
            q,
            q.stride(0),
            q.stride(1),
            cos_sin_cache,
            cos_sin_cache.stride(0),
            half_rot_dim,
            q_fp4,
            q_fp4.stride(0),
            q_fp4.stride(1),
            q_scale,
            q_scale.stride(0),
            scratch,
            index_head_dim,
            is_neox_style,
            log2_head_dim,
            _FP4_GRAN_K,
            _FP4_SCALES_PACK_FACTOR,
            num_warps=1,
        )

    return q_fp4, q_scale


# ====================================================================
#  Fused QK: K(RoPE+Hadamard→bf16) + Q(RoPE+Hadamard+FP4) in one kernel
# ====================================================================


@triton.jit
def _fused_qk_rope_hadamard_fp4_quant_kernel(
    pos_ptr,
    q_ptr,
    q_stride0,
    q_stride1,
    k_ptr,
    k_stride0,
    cos_sin_cache_ptr,
    cos_sin_cache_stride,
    HALF_ROT_DIM: tl.constexpr,
    q_fp4_ptr,
    q_fp4_stride0,
    q_fp4_stride1,
    q_scale_ptr,
    q_scale_stride0,
    k_out_ptr,
    k_out_stride0,
    scratch_ptr,
    HEAD_DIM: tl.constexpr,
    IS_NEOX_STYLE: tl.constexpr,
    LOG2_HEAD_DIM: tl.constexpr,
    GRAN_K: tl.constexpr,
    SCALES_PACK: tl.constexpr,
):
    """Fused QK: K(RoPE+Hadamard→bf16) + Q(RoPE+Hadamard+FP4 e2m1, UE8M0/32).

    Grid: [num_tokens, num_heads]
    head_idx==0 processes K then Q; all other heads process Q only.
    K uses the same per-program scratch buffer as Q — head 0 finishes K
    before Q overwrites scratch, so per-program reuse is race-free.
    """
    ROT_DIM: tl.constexpr = 2 * HALF_ROT_DIM
    NOPE_DIM: tl.constexpr = HEAD_DIM - ROT_DIM
    N_QUANT_GROUPS: tl.constexpr = HEAD_DIM // GRAN_K
    HALF_DIM: tl.constexpr = HEAD_DIM // 2
    tl.static_assert(NOPE_DIM >= 0)
    tl.static_assert(HEAD_DIM % GRAN_K == 0)
    tl.static_assert(N_QUANT_GROUPS == SCALES_PACK)

    tok_idx = tl.program_id(0).to(tl.int64)
    head_idx = tl.program_id(1)
    num_heads = tl.num_programs(1)

    pos = tl.load(pos_ptr + tok_idx)
    work_id = tok_idx * num_heads + head_idx
    scratch_base = scratch_ptr + work_id * HEAD_DIM

    half_offset = tl.arange(0, HALF_ROT_DIM)
    cos = tl.load(cos_sin_cache_ptr + pos * cos_sin_cache_stride + half_offset).to(
        tl.float32
    )
    sin = tl.load(
        cos_sin_cache_ptr + pos * cos_sin_cache_stride + half_offset + HALF_ROT_DIM
    ).to(tl.float32)
    idx = tl.arange(0, HEAD_DIM)

    # ── K path (head 0 only) ── identical to FP8 QK kernel; K stays bf16.
    if head_idx == 0:
        k_base = k_ptr + tok_idx * k_stride0
        if IS_NEOX_STYLE:
            x_first = tl.load(k_base + half_offset).to(tl.float32)
            x_second = tl.load(k_base + half_offset + HALF_ROT_DIM).to(tl.float32)
            r_first = (x_first * cos - x_second * sin).to(tl.bfloat16).to(tl.float32)
            r_second = (x_second * cos + x_first * sin).to(tl.bfloat16).to(tl.float32)
            tl.store(scratch_base + half_offset, r_first)
            tl.store(scratch_base + half_offset + HALF_ROT_DIM, r_second)
        else:
            x_even = tl.load(k_base + half_offset * 2).to(tl.float32)
            x_odd = tl.load(k_base + half_offset * 2 + 1).to(tl.float32)
            r_first = (x_even * cos - x_odd * sin).to(tl.bfloat16).to(tl.float32)
            r_second = (x_odd * cos + x_even * sin).to(tl.bfloat16).to(tl.float32)
            tl.store(scratch_base + half_offset * 2, r_first)
            tl.store(scratch_base + half_offset * 2 + 1, r_second)

        if NOPE_DIM > 0:
            nope_offset = tl.arange(0, NOPE_DIM)
            tl.store(
                scratch_base + ROT_DIM + nope_offset,
                tl.load(k_base + ROT_DIM + nope_offset).to(tl.float32),
            )

        tl.debug_barrier()
        for s_log in range(LOG2_HEAD_DIM):
            stride = 1 << s_log
            is_upper = (idx & stride) != 0
            self_val = tl.load(scratch_base + idx)
            partner_val = tl.load(scratch_base + (idx ^ stride))
            tl.store(
                scratch_base + idx,
                tl.where(is_upper, partner_val - self_val, self_val + partner_val),
            )
            tl.debug_barrier()

        tl.store(
            k_out_ptr + tok_idx * k_out_stride0 + idx,
            (tl.load(scratch_base + idx) * (HEAD_DIM**-0.5)).to(tl.bfloat16),
        )

    # ── Q path (all heads) ── RoPE → Hadamard → FP4 e2m1 + UE8M0/32 scale.
    q_base = q_ptr + tok_idx * q_stride0 + head_idx * q_stride1
    if IS_NEOX_STYLE:
        x_first = tl.load(q_base + half_offset).to(tl.float32)
        x_second = tl.load(q_base + half_offset + HALF_ROT_DIM).to(tl.float32)
        r_first = (x_first * cos - x_second * sin).to(tl.bfloat16).to(tl.float32)
        r_second = (x_second * cos + x_first * sin).to(tl.bfloat16).to(tl.float32)
        tl.store(scratch_base + half_offset, r_first)
        tl.store(scratch_base + half_offset + HALF_ROT_DIM, r_second)
    else:
        x_even = tl.load(q_base + half_offset * 2).to(tl.float32)
        x_odd = tl.load(q_base + half_offset * 2 + 1).to(tl.float32)
        r_first = (x_even * cos - x_odd * sin).to(tl.bfloat16).to(tl.float32)
        r_second = (x_odd * cos + x_even * sin).to(tl.bfloat16).to(tl.float32)
        tl.store(scratch_base + half_offset * 2, r_first)
        tl.store(scratch_base + half_offset * 2 + 1, r_second)

    if NOPE_DIM > 0:
        nope_offset = tl.arange(0, NOPE_DIM)
        tl.store(
            scratch_base + ROT_DIM + nope_offset,
            tl.load(q_base + ROT_DIM + nope_offset).to(tl.float32),
        )

    tl.debug_barrier()
    for s_log in range(LOG2_HEAD_DIM):
        stride = 1 << s_log
        is_upper = (idx & stride) != 0
        self_val = tl.load(scratch_base + idx)
        partner_val = tl.load(scratch_base + (idx ^ stride))
        tl.store(
            scratch_base + idx,
            tl.where(is_upper, partner_val - self_val, self_val + partner_val),
        )
        tl.debug_barrier()

    # ---- per-group UE8M0 scale + FP4 e2m1 quant (mirrors Q-only kernel) ----
    group_idx = idx // GRAN_K
    data = (
        (tl.load(scratch_base + idx) * (HEAD_DIM**-0.5)).to(tl.bfloat16).to(tl.float32)
    )
    abs_data = tl.abs(data)

    g_offsets = tl.arange(0, N_QUANT_GROUPS)
    mask_2d = group_idx[:, None] == g_offsets[None, :]
    masked = tl.where(mask_2d, abs_data[:, None], 0.0)
    amax = tl.max(masked, axis=0)
    amax = tl.maximum(amax, 6.0 * (2.0**-126))

    sf = amax / 6.0
    sf_bits = sf.to(tl.int32, bitcast=True)
    exp = (sf_bits >> 23) & 0xFF
    mant = sf_bits & 0x7FFFFF
    exp = exp + (mant != 0).to(tl.int32)
    exp = tl.minimum(tl.maximum(exp, 1), 254)

    inv_scale_per_group = tl.exp2(127.0 - exp.to(tl.float32))
    inv_scale_2d = tl.where(mask_2d, inv_scale_per_group[None, :], 0.0)
    inv_scale = tl.sum(inv_scale_2d, axis=1)
    scaled = data * inv_scale

    ax = tl.minimum(tl.abs(scaled), 6.0)
    code = (ax > 0.25).to(tl.int32)
    code = code + (ax > 0.75).to(tl.int32)
    code = code + (ax > 1.25).to(tl.int32)
    code = code + (ax > 1.75).to(tl.int32)
    code = code + (ax > 2.5).to(tl.int32)
    code = code + (ax > 3.5).to(tl.int32)
    code = code + (ax > 5.0).to(tl.int32)
    sign = ((scaled < 0) & (code != 0)).to(tl.int32)
    code = code | (sign << 3)
    code = code & 0x0F

    pair_idx = tl.arange(0, HALF_DIM)
    even_pos = pair_idx * 2
    odd_pos = pair_idx * 2 + 1
    even_mask = idx[:, None] == even_pos[None, :]
    odd_mask = idx[:, None] == odd_pos[None, :]
    even_codes = tl.sum(tl.where(even_mask, code[:, None], 0), axis=0)
    odd_codes = tl.sum(tl.where(odd_mask, code[:, None], 0), axis=0)
    packed = ((even_codes & 0x0F) | ((odd_codes & 0x0F) << 4)).to(tl.uint8)

    fp4_base = q_fp4_ptr + tok_idx * q_fp4_stride0 + head_idx * q_fp4_stride1
    tl.store(fp4_base + pair_idx, packed)

    shifts = (g_offsets * 8).to(tl.int32)
    packed_scale = tl.sum((exp & 0xFF) << shifts, axis=0)
    tl.store(q_scale_ptr + tok_idx * q_scale_stride0 + head_idx, packed_scale)


def fused_qk_rope_fp4_quant(
    q: torch.Tensor,
    k: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    index_n_heads: int,
    index_head_dim: int,
    rope_head_dim: int,
    is_neox_style: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fused QK: K(RoPE+Hadamard→bf16) + Q(RoPE+Hadamard+FP4) in one launch.

    FP4 sibling of ``fused_qk_rope_quant``. K output is bit-equivalent to
    ``apply_rope_and_rotate_k(k, positions)``; Q output is bit-equivalent
    to ``fused_q_rope_fp4_quant(q, positions, ...)`` on the same inputs
    — both are pinned by ``test_fused_qk_rope_fp4_quant.py``.

    Args:
        q: [num_tokens, index_n_heads, index_head_dim] bf16
        k: [num_tokens, index_head_dim] bf16 (after k_norm)
        positions: [num_tokens] int64
        cos_sin_cache: [max_pos, rope_head_dim]
        index_n_heads, index_head_dim, rope_head_dim, is_neox_style: as
            in the Q-only entry.

    Returns:
        (q_fp4, q_scale, k_out):
            q_fp4:   int8  [num_tokens, index_n_heads, index_head_dim // 2]
            q_scale: int32 [num_tokens, index_n_heads]
            k_out:   bf16  [num_tokens, index_head_dim]
    """
    assert (
        index_head_dim % _FP4_GRAN_K == 0
    ), f"index_head_dim={index_head_dim} must be a multiple of gran_k={_FP4_GRAN_K}"
    n_groups = index_head_dim // _FP4_GRAN_K
    assert n_groups == _FP4_SCALES_PACK_FACTOR, (
        f"v1 supports exactly {_FP4_SCALES_PACK_FACTOR} groups per token "
        f"(1 int32 = {_FP4_SCALES_PACK_FACTOR} packed UE8M0); got {n_groups} "
        f"(head_dim={index_head_dim})"
    )
    num_tokens = q.shape[0]
    half_rot_dim = rope_head_dim // 2
    log2_head_dim = int(math.log2(index_head_dim))
    assert 2**log2_head_dim == index_head_dim, "index_head_dim must be a power of 2"

    q_fp4 = torch.empty(
        (num_tokens, index_n_heads, index_head_dim // 2),
        dtype=torch.int8,
        device=q.device,
    )
    q_scale = torch.empty(
        (num_tokens, index_n_heads),
        dtype=torch.int32,
        device=q.device,
    )
    k_out = torch.empty(
        (num_tokens, index_head_dim),
        dtype=torch.bfloat16,
        device=q.device,
    )
    scratch = torch.empty(
        num_tokens * index_n_heads * index_head_dim,
        dtype=torch.float32,
        device=q.device,
    )

    if num_tokens > 0:
        grid = (num_tokens, index_n_heads)
        _fused_qk_rope_hadamard_fp4_quant_kernel[grid](
            positions,
            q,
            q.stride(0),
            q.stride(1),
            k,
            k.stride(0),
            cos_sin_cache,
            cos_sin_cache.stride(0),
            half_rot_dim,
            q_fp4,
            q_fp4.stride(0),
            q_fp4.stride(1),
            q_scale,
            q_scale.stride(0),
            k_out,
            k_out.stride(0),
            scratch,
            index_head_dim,
            is_neox_style,
            log2_head_dim,
            _FP4_GRAN_K,
            _FP4_SCALES_PACK_FACTOR,
            num_warps=1,
        )

    return q_fp4, q_scale, k_out
