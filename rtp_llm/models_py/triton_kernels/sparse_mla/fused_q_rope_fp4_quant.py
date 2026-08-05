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

Output shapes match ``per_token_group_quant_fp4``:
  q_fp4:   int8   [num_tokens, num_heads, HEAD_DIM // 2]
  q_scale: int32  [num_tokens, num_heads]   (1 int32 = 4 packed UE8M0 bytes)
"""

import math

import torch
import triton
import triton.language as tl

_FP4_GRAN_K = 32
_FP4_SCALES_PACK_FACTOR = 4


@triton.jit
def _butterfly_step_128(
    x,
    N_GROUPS: tl.constexpr,
    STRIDE: tl.constexpr,
):
    """One in-register Hadamard stage for a 128-element row."""
    x4 = tl.reshape(x, [N_GROUPS, 2, STRIDE])
    x4 = tl.permute(x4, (0, 2, 1))
    a, b = tl.split(x4)
    x4 = tl.join(a + b, a - b)
    x4 = tl.permute(x4, (0, 2, 1))
    return tl.reshape(x4, [128])


@triton.jit
def _hadamard_128_inline(x):
    """Seven-stage 128-point Walsh-Hadamard transform in registers."""
    x = _butterfly_step_128(x, 64, 1)
    x = _butterfly_step_128(x, 32, 2)
    x = _butterfly_step_128(x, 16, 4)
    x = _butterfly_step_128(x, 8, 8)
    x = _butterfly_step_128(x, 4, 16)
    x = _butterfly_step_128(x, 2, 32)
    x = _butterfly_step_128(x, 1, 64)
    return x


@triton.jit
def _fp4_quant_pack_from_values(
    values,
    q_fp4_base,
    q_scale_ptr,
    q_scale_off,
    HEAD_DIM: tl.constexpr,
    GRAN_K: tl.constexpr,
    N_QUANT_GROUPS: tl.constexpr,
):
    """FP4 e2m1 + packed UE8M0 directly from an in-register Hadamard row."""
    PAIRS_PER_GROUP: tl.constexpr = GRAN_K // 2
    data = (values * (HEAD_DIM**-0.5)).to(tl.bfloat16).to(tl.float32)
    grouped = tl.reshape(data, [N_QUANT_GROUPS, GRAN_K])
    amax = tl.maximum(
        tl.max(tl.abs(grouped), axis=1),
        6.0 * (2.0**-126),
    )

    sf = amax / 6.0
    sf_bits = sf.to(tl.int32, bitcast=True)
    exp = (sf_bits >> 23) & 0xFF
    exp = exp + ((sf_bits & 0x7FFFFF) != 0).to(tl.int32)
    exp = tl.minimum(tl.maximum(exp, 1), 254)

    scaled = grouped * tl.exp2(127.0 - exp[:, None].to(tl.float32))
    ax = tl.minimum(tl.abs(scaled), 6.0)
    code = (ax > 0.25).to(tl.int32)
    code += (ax > 0.75).to(tl.int32)
    code += (ax > 1.25).to(tl.int32)
    code += (ax > 1.75).to(tl.int32)
    code += (ax > 2.5).to(tl.int32)
    code += (ax > 3.5).to(tl.int32)
    code += (ax > 5.0).to(tl.int32)
    sign = ((scaled < 0) & (code != 0)).to(tl.int32)
    code = (code | (sign << 3)) & 0x0F

    code_pairs = tl.reshape(code, [N_QUANT_GROUPS, PAIRS_PER_GROUP, 2])
    even_code, odd_code = tl.split(code_pairs)
    packed = (even_code & 0x0F) | ((odd_code & 0x0F) << 4)
    packed_offs = tl.arange(0, HEAD_DIM // 2)
    tl.store(
        q_fp4_base + packed_offs,
        tl.reshape(packed, [HEAD_DIM // 2]).to(tl.uint8),
    )

    group_idx = tl.arange(0, N_QUANT_GROUPS)
    packed_scale = tl.sum((exp & 0xFF) << (group_idx * 8), axis=0)
    tl.store(q_scale_ptr + q_scale_off, packed_scale)


@triton.jit
def _fp4_quant_pack_from_hadamard_scratch(
    scratch_base,
    q_fp4_base,
    q_scale_ptr,
    q_scale_off,
    HEAD_DIM: tl.constexpr,
    GRAN_K: tl.constexpr,
    N_QUANT_GROUPS: tl.constexpr,
):
    """FP4 e2m1 + packed UE8M0 from Hadamard scratch (one token-head)."""
    PAIRS_PER_GROUP: tl.constexpr = GRAN_K // 2
    norm = HEAD_DIM**-0.5
    packed_scale = tl.zeros([], dtype=tl.int32)
    col_sel = tl.arange(0, 2)

    for g in tl.static_range(N_QUANT_GROUPS):
        offs = tl.arange(0, GRAN_K)
        data = (
            (tl.load(scratch_base + g * GRAN_K + offs) * norm)
            .to(tl.bfloat16)
            .to(tl.float32)
        )
        amax = tl.maximum(tl.max(tl.abs(data)), 6.0 * (2.0**-126))

        sf = amax / 6.0
        sf_bits = sf.to(tl.int32, bitcast=True)
        exp = (sf_bits >> 23) & 0xFF
        exp = exp + ((sf_bits & 0x7FFFFF) != 0).to(tl.int32)
        exp = tl.minimum(tl.maximum(exp, 1), 254)

        inv_scale = tl.exp2(127.0 - exp.to(tl.float32))
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
        code = (code | (sign << 3)) & 0x0F

        codes_2d = tl.reshape(code, [PAIRS_PER_GROUP, 2])
        even_codes = tl.max(tl.where(col_sel[None, :] == 0, codes_2d, 0), axis=1)
        odd_codes = tl.max(tl.where(col_sel[None, :] == 1, codes_2d, 0), axis=1)
        packed = (even_codes & 0x0F) | ((odd_codes & 0x0F) << 4)
        tl.store(
            q_fp4_base + g * PAIRS_PER_GROUP + tl.arange(0, PAIRS_PER_GROUP),
            packed.to(tl.uint8),
        )
        packed_scale = packed_scale | ((exp & 0xFF) << (g * 8))

    tl.store(q_scale_ptr + q_scale_off, packed_scale)


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
    N_QUANT_GROUPS: tl.constexpr,
):
    """Grid: [num_tokens, num_heads]; RoPE → Hadamard → FP4 + UE8M0."""
    ROT_DIM: tl.constexpr = 2 * HALF_ROT_DIM
    NOPE_DIM: tl.constexpr = HEAD_DIM - ROT_DIM
    tl.static_assert(NOPE_DIM >= 0)
    tl.static_assert(HEAD_DIM % GRAN_K == 0)
    tl.static_assert(N_QUANT_GROUPS == 4)

    tok_idx = tl.program_id(0).to(tl.int64)
    head_idx = tl.program_id(1)
    num_heads = tl.num_programs(1)

    pos = tl.load(pos_ptr + tok_idx)
    base_ptr = q_ptr + tok_idx * q_stride0 + head_idx * q_stride1
    work_id = tok_idx * num_heads + head_idx
    scratch_base = scratch_ptr + work_id * HEAD_DIM

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

    if NOPE_DIM > 0:
        nope_offset = tl.arange(0, NOPE_DIM)
        x_nope = tl.load(base_ptr + ROT_DIM + nope_offset).to(tl.float32)
        tl.store(scratch_base + ROT_DIM + nope_offset, x_nope)

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

    fp4_base = q_fp4_ptr + tok_idx * q_fp4_stride0 + head_idx * q_fp4_stride1
    _fp4_quant_pack_from_hadamard_scratch(
        scratch_base,
        fp4_base,
        q_scale_ptr,
        tok_idx * q_scale_stride0 + head_idx,
        HEAD_DIM,
        GRAN_K,
        N_QUANT_GROUPS,
    )


def fused_q_rope_fp4_quant(
    q: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    index_n_heads: int,
    index_head_dim: int,
    rope_head_dim: int,
    is_neox_style: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fused RoPE + Hadamard + FP4 quant for indexer Q (Blackwell only)."""
    assert index_head_dim % _FP4_GRAN_K == 0
    n_groups = index_head_dim // _FP4_GRAN_K
    assert n_groups == _FP4_SCALES_PACK_FACTOR
    num_tokens = q.shape[0]
    half_rot_dim = rope_head_dim // 2
    log2_head_dim = int(math.log2(index_head_dim))
    assert 2**log2_head_dim == index_head_dim

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
            n_groups,
            num_warps=1,
        )

    return q_fp4, q_scale


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
    HEAD_DIM: tl.constexpr,
    IS_NEOX_STYLE: tl.constexpr,
    GRAN_K: tl.constexpr,
    N_QUANT_GROUPS: tl.constexpr,
):
    """Fused QK: K(RoPE+Hadamard→bf16) + Q(RoPE+Hadamard+FP4 e2m1, UE8M0/32)."""
    ROT_DIM: tl.constexpr = 2 * HALF_ROT_DIM
    NOPE_DIM: tl.constexpr = HEAD_DIM - ROT_DIM
    tl.static_assert(NOPE_DIM >= 0)
    tl.static_assert(HEAD_DIM % GRAN_K == 0)
    tl.static_assert(HEAD_DIM == 128)
    tl.static_assert(N_QUANT_GROUPS == 4)

    tok_idx = tl.program_id(0).to(tl.int64)
    head_idx = tl.program_id(1)

    pos = tl.load(pos_ptr + tok_idx)
    idx = tl.arange(0, HEAD_DIM)

    if head_idx == 0:
        k_base = k_ptr + tok_idx * k_stride0
        k_full = tl.load(k_base + idx).to(tl.float32)
        if IS_NEOX_STYLE:
            partner_idx = idx ^ HALF_ROT_DIM
            k_partner = tl.load(k_base + partner_idx).to(tl.float32)
            rope_idx = idx & (HALF_ROT_DIM - 1)
            sign = tl.where(idx < HALF_ROT_DIM, -1.0, 1.0)
        else:
            k_pairs = tl.reshape(k_full, [HEAD_DIM // 2, 2])
            k_even, k_odd = tl.split(k_pairs)
            k_partner = tl.reshape(tl.join(k_odd, k_even), [HEAD_DIM])
            rope_idx = (idx >> 1) & (HALF_ROT_DIM - 1)
            sign = tl.where((idx & 1) == 0, -1.0, 1.0)

        cos = tl.load(cos_sin_cache_ptr + pos * cos_sin_cache_stride + rope_idx).to(
            tl.float32
        )
        sin = tl.load(
            cos_sin_cache_ptr + pos * cos_sin_cache_stride + HALF_ROT_DIM + rope_idx
        ).to(tl.float32)
        k_rope = (
            tl.where(
                idx < ROT_DIM,
                k_full * cos + sign * k_partner * sin,
                k_full,
            )
            .to(tl.bfloat16)
            .to(tl.float32)
        )
        k_had = _hadamard_128_inline(k_rope)

        tl.store(
            k_out_ptr + tok_idx * k_out_stride0 + idx,
            (k_had * (HEAD_DIM**-0.5)).to(tl.bfloat16),
        )

    q_base = q_ptr + tok_idx * q_stride0 + head_idx * q_stride1
    q_full = tl.load(q_base + idx).to(tl.float32)
    if IS_NEOX_STYLE:
        partner_idx = idx ^ HALF_ROT_DIM
        q_partner = tl.load(q_base + partner_idx).to(tl.float32)
        rope_idx = idx & (HALF_ROT_DIM - 1)
        sign = tl.where(idx < HALF_ROT_DIM, -1.0, 1.0)
    else:
        q_pairs = tl.reshape(q_full, [HEAD_DIM // 2, 2])
        q_even, q_odd = tl.split(q_pairs)
        q_partner = tl.reshape(tl.join(q_odd, q_even), [HEAD_DIM])
        rope_idx = (idx >> 1) & (HALF_ROT_DIM - 1)
        sign = tl.where((idx & 1) == 0, -1.0, 1.0)

    cos = tl.load(cos_sin_cache_ptr + pos * cos_sin_cache_stride + rope_idx).to(
        tl.float32
    )
    sin = tl.load(
        cos_sin_cache_ptr + pos * cos_sin_cache_stride + HALF_ROT_DIM + rope_idx
    ).to(tl.float32)
    q_rope = (
        tl.where(
            idx < ROT_DIM,
            q_full * cos + sign * q_partner * sin,
            q_full,
        )
        .to(tl.bfloat16)
        .to(tl.float32)
    )
    q_had = _hadamard_128_inline(q_rope)

    fp4_base = q_fp4_ptr + tok_idx * q_fp4_stride0 + head_idx * q_fp4_stride1
    _fp4_quant_pack_from_values(
        q_had,
        fp4_base,
        q_scale_ptr,
        tok_idx * q_scale_stride0 + head_idx,
        HEAD_DIM,
        GRAN_K,
        N_QUANT_GROUPS,
    )


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
    """Fused QK: K(RoPE+Hadamard→bf16) + Q(RoPE+Hadamard+FP4) in one launch."""
    assert index_head_dim % _FP4_GRAN_K == 0
    n_groups = index_head_dim // _FP4_GRAN_K
    assert n_groups == _FP4_SCALES_PACK_FACTOR
    num_tokens = q.shape[0]
    half_rot_dim = rope_head_dim // 2
    log2_head_dim = int(math.log2(index_head_dim))
    assert 2**log2_head_dim == index_head_dim

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
            index_head_dim,
            is_neox_style,
            _FP4_GRAN_K,
            n_groups,
            num_warps=1,
        )

    return q_fp4, q_scale, k_out
