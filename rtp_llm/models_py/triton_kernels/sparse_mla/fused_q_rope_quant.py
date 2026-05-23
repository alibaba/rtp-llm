"""Fused RoPE + Hadamard + FP8 Quantization for Indexer Q.

Single Triton kernel that:
1. Applies RoPE on the first ROT_DIM dims (NeOX or GPT-J interleaved)
2. Passes through NoPE dims unchanged
3. Applies N-point Walsh-Hadamard transform on the full HEAD_DIM vector
4. Quantizes to FP8 (e4m3fn) with ue8m0 per-token-per-head scale

The Hadamard transform uses 7 butterfly stages (for HEAD_DIM=128) through
a per-program scratch buffer in global memory (512 bytes/program, fits in
L1 cache).  Algorithm adapted from SGLang's fused_q_indexer_rope_hadamard_quant.
"""

import math

import torch
import triton
import triton.language as tl


@triton.jit
def _fused_q_rope_hadamard_quant_kernel(
    pos_ptr,
    q_ptr,
    q_stride0,
    q_stride1,
    cos_sin_cache_ptr,
    cos_sin_cache_stride,
    HALF_ROT_DIM: tl.constexpr,
    q_fp8_ptr,
    q_fp8_stride0,
    q_fp8_stride1,
    q_scale_ptr,
    q_scale_stride0,
    scratch_ptr,
    HEAD_DIM: tl.constexpr,
    IS_NEOX_STYLE: tl.constexpr,
    LOG2_HEAD_DIM: tl.constexpr,
):
    """Fused RoPE + Hadamard + ue8m0 FP8 quant for indexer Q.

    Grid: [num_tokens, num_heads]
    Each program processes one (token, head) pair.

    Pipeline: Load → RoPE → Hadamard (7 butterfly stages) → FP8 quant
    """
    ROT_DIM: tl.constexpr = 2 * HALF_ROT_DIM
    NOPE_DIM: tl.constexpr = HEAD_DIM - ROT_DIM
    tl.static_assert(NOPE_DIM >= 0)

    tok_idx = tl.program_id(0)
    head_idx = tl.program_id(1)
    num_heads = tl.num_programs(1)

    pos = tl.load(pos_ptr + tok_idx)
    base_ptr = q_ptr + tok_idx * q_stride0 + head_idx * q_stride1

    # Per-program scratch area (HEAD_DIM float32s) for Hadamard butterfly
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
        # bf16 roundtrip for numerical parity with unfused path
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

    # ---- Step 2: NoPE dims (pass-through) → scratch ----
    if NOPE_DIM > 0:
        nope_offset = tl.arange(0, NOPE_DIM)
        x_nope = tl.load(base_ptr + ROT_DIM + nope_offset).to(tl.float32)
        tl.store(scratch_base + ROT_DIM + nope_offset, x_nope)

    # ---- Step 3: Walsh-Hadamard Transform (LOG2_HEAD_DIM butterfly stages) ----
    # Each stage: partner = idx XOR stride
    #   lower half (bit not set): result = self + partner
    #   upper half (bit set):     result = partner - self
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

    data = tl.load(scratch_base + idx) * (HEAD_DIM**-0.5)

    # ---- Step 4: ue8m0 FP8 quantization ----
    amax = tl.max(tl.abs(data))
    q_scale_val = tl.maximum(amax, 1e-4) / 448.0
    q_scale_val = tl.math.exp2(tl.math.ceil(tl.math.log2(q_scale_val)))

    fp8_base = q_fp8_ptr + tok_idx * q_fp8_stride0 + head_idx * q_fp8_stride1
    tl.store(fp8_base + idx, (data / q_scale_val).to(tl.float8e4nv))

    # Store ue8m0 scale directly as float32 via bitcast
    log2_scale = tl.math.ceil(tl.math.log2(tl.maximum(amax, 1e-4) / 448.0)) + 127.0
    log2_scale = tl.minimum(tl.maximum(log2_scale, 0.0), 255.0)
    scale_i32 = log2_scale.to(tl.int32) << 23
    scale_f32 = scale_i32.to(tl.float32, bitcast=True)
    tl.store(q_scale_ptr + tok_idx * q_scale_stride0 + head_idx, scale_f32)


def fused_q_rope_quant(
    q: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    index_n_heads: int,
    index_head_dim: int,
    rope_head_dim: int,
    is_neox_style: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fused RoPE + Hadamard + FP8 quantization for indexer Q.

    Args:
        q: Query tensor [num_tokens, index_n_heads, index_head_dim] (bf16)
        positions: Position IDs [num_tokens]
        cos_sin_cache: Precomputed cos/sin cache [max_pos, rot_dim]
        index_n_heads: Number of indexer heads
        index_head_dim: Dimension of indexer heads (must be power of 2)
        rope_head_dim: Total rotation dimension (first dims affected by RoPE)
        is_neox_style: True for NeOX-style RoPE, False for GPT-J interleaved

    Returns:
        (q_fp8, q_scale):
            q_fp8: [num_tokens, index_n_heads, index_head_dim] float8_e4m3fn
            q_scale: [num_tokens, index_n_heads, 1] float32 (unpacked ue8m0)
    """
    num_tokens = q.shape[0]
    half_rot_dim = rope_head_dim // 2
    log2_head_dim = int(math.log2(index_head_dim))
    assert 2**log2_head_dim == index_head_dim, "head_dim must be power of 2"

    q_fp8 = torch.empty(
        (num_tokens, index_n_heads, index_head_dim),
        dtype=torch.float8_e4m3fn,
        device=q.device,
    )
    q_scale = torch.empty(
        (num_tokens, index_n_heads),
        dtype=torch.float32,
        device=q.device,
    )
    # Scratch for Hadamard butterfly (HEAD_DIM float32s per program instance)
    scratch = torch.empty(
        num_tokens * index_n_heads * index_head_dim,
        dtype=torch.float32,
        device=q.device,
    )

    if num_tokens > 0:
        grid = (num_tokens, index_n_heads)
        _fused_q_rope_hadamard_quant_kernel[grid](
            positions,
            q,
            q.stride(0),
            q.stride(1),
            cos_sin_cache,
            cos_sin_cache.stride(0),
            half_rot_dim,
            q_fp8,
            q_fp8.stride(0),
            q_fp8.stride(1),
            q_scale,
            q_scale.stride(0),
            scratch,
            index_head_dim,
            is_neox_style,
            log2_head_dim,
            num_warps=1,
        )

    return q_fp8, q_scale.unsqueeze(-1)


# ====================================================================
#  Fused QK: K(RoPE+Hadamard→bf16) + Q(RoPE+Hadamard+FP8) in one kernel
# ====================================================================


@triton.jit
def _fused_qk_rope_hadamard_quant_kernel(
    pos_ptr,
    q_ptr,
    q_stride0,
    q_stride1,
    k_ptr,
    k_stride0,
    cos_sin_cache_ptr,
    cos_sin_cache_stride,
    HALF_ROT_DIM: tl.constexpr,
    q_fp8_ptr,
    q_fp8_stride0,
    q_fp8_stride1,
    q_scale_ptr,
    q_scale_stride0,
    k_out_ptr,
    k_out_stride0,
    scratch_ptr,
    HEAD_DIM: tl.constexpr,
    IS_NEOX_STYLE: tl.constexpr,
    LOG2_HEAD_DIM: tl.constexpr,
):
    """Fused QK: K(RoPE+Hadamard→bf16) + Q(RoPE+Hadamard+ue8m0 FP8).

    Grid: [num_tokens, num_heads]
    head_idx==0 processes K then Q; all other heads process Q only.
    """
    ROT_DIM: tl.constexpr = 2 * HALF_ROT_DIM
    NOPE_DIM: tl.constexpr = HEAD_DIM - ROT_DIM
    tl.static_assert(NOPE_DIM >= 0)

    tok_idx = tl.program_id(0)
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

    # ── K path (head 0 only) ──
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

    # ── Q path (all heads) ──
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

    data = tl.load(scratch_base + idx) * (HEAD_DIM**-0.5)

    # ue8m0 FP8 quantization
    amax = tl.max(tl.abs(data))
    q_scale_val = tl.maximum(amax, 1e-4) / 448.0
    q_scale_val = tl.math.exp2(tl.math.ceil(tl.math.log2(q_scale_val)))

    fp8_base = q_fp8_ptr + tok_idx * q_fp8_stride0 + head_idx * q_fp8_stride1
    tl.store(fp8_base + idx, (data / q_scale_val).to(tl.float8e4nv))

    log2_scale = tl.math.ceil(tl.math.log2(tl.maximum(amax, 1e-4) / 448.0)) + 127.0
    log2_scale = tl.minimum(tl.maximum(log2_scale, 0.0), 255.0)
    scale_i32 = log2_scale.to(tl.int32) << 23
    scale_f32 = scale_i32.to(tl.float32, bitcast=True)
    tl.store(q_scale_ptr + tok_idx * q_scale_stride0 + head_idx, scale_f32)


def fused_qk_rope_quant(
    q: torch.Tensor,
    k: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    index_n_heads: int,
    index_head_dim: int,
    rope_head_dim: int,
    is_neox_style: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fused QK: K(RoPE+Hadamard→bf16) + Q(RoPE+Hadamard+FP8) in one kernel.

    Args:
        q: Query tensor [num_tokens, index_n_heads, index_head_dim] (bf16)
        k: Key tensor [num_tokens, index_head_dim] (bf16, after k_norm)
        positions: Position IDs [num_tokens]
        cos_sin_cache: Precomputed cos/sin cache [max_pos, rot_dim]
        index_n_heads: Number of indexer heads
        index_head_dim: Dimension of indexer heads (must be power of 2)
        rope_head_dim: Total rotation dimension (first dims affected by RoPE)
        is_neox_style: True for NeOX-style RoPE, False for GPT-J interleaved

    Returns:
        (q_fp8, q_scale, k_out):
            q_fp8: [num_tokens, index_n_heads, index_head_dim] float8_e4m3fn
            q_scale: [num_tokens, index_n_heads, 1] float32 (unpacked ue8m0)
            k_out: [num_tokens, index_head_dim] bf16 (RoPE + Hadamard applied)
    """
    num_tokens = q.shape[0]
    half_rot_dim = rope_head_dim // 2
    log2_head_dim = int(math.log2(index_head_dim))
    assert 2**log2_head_dim == index_head_dim, "head_dim must be power of 2"

    q_fp8 = torch.empty(
        (num_tokens, index_n_heads, index_head_dim),
        dtype=torch.float8_e4m3fn,
        device=q.device,
    )
    q_scale = torch.empty(
        (num_tokens, index_n_heads),
        dtype=torch.float32,
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
        _fused_qk_rope_hadamard_quant_kernel[grid](
            positions,
            q,
            q.stride(0),
            q.stride(1),
            k,
            k.stride(0),
            cos_sin_cache,
            cos_sin_cache.stride(0),
            half_rot_dim,
            q_fp8,
            q_fp8.stride(0),
            q_fp8.stride(1),
            q_scale,
            q_scale.stride(0),
            k_out,
            k_out.stride(0),
            scratch,
            index_head_dim,
            is_neox_style,
            log2_head_dim,
            num_warps=1,
        )

    return q_fp8, q_scale.unsqueeze(-1), k_out
