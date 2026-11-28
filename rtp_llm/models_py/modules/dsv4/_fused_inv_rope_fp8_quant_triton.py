"""DeepSeek-V4 fused inverse-RoPE + per-token-group FP8 quant — single Triton kernel.

Replaces the following Python/torch chain on the hot decode path (see
``attention.py:_wo_a_grouped_fp8`` entry):

    apply_rotary_emb_batched(o[..., -rd:], freqs_cis_per_req, inverse=True)
    o_4d = o.reshape(B, S, G, K)
    # …then per-group `per_token_group_quant_fp8(o[g], ...)` inside
    # `_wo_a_grouped_fp8` (G launches).

Merged into one launch that produces the exact FP8 + UE8M0 packed scale
layout `deep_gemm.fp8_einsum("bhr,hdr->bhd", …, recipe=(1, 1, 128))`
expects:

    o_fp8   : [M, G, d]     float8_e4m3fn   (view of [G, M, d] transposed)
    o_scale : [M, G, K/512] int32 UE8M0     stride (1, K/512 * tma_M, tma_M)

Port of ``vllm/v1/attention/ops/deepseek_v4_ops/fused_inv_rope_fp8_quant.py``
supporting both pre-indexed ``freqs_cis_per_b`` tensors and inline table
lookup via ``position_ids`` against a 2D ``[max_pos, rope_dim // 2]`` cache.

Validated by ``test/test_fused_inv_rope_fp8_quant.py`` (≤1 FP8 ULP vs
eager inv-RoPE + per-group quant, and ≤2-ULP end-to-end vs wo_a GEMM).
"""

from __future__ import annotations

import functools

import torch
import triton
import triton.language as tl

try:
    from rtp_llm.models_py.triton_kernels.fast_launch import fast_launch
except ImportError:  # pragma: no cover

    def fast_launch(jit_fn, grid, *args, **kwargs):
        return jit_fn[grid](*args, **kwargs)


_FP8_E4M3_MAX = float(torch.finfo(torch.float8_e4m3fn).max)
_CACHED_CUDA_INV_ROPE_OP = False
_CUDA_INV_ROPE_OP = None


def _get_cuda_inv_rope_fp8_quant_op():
    global _CACHED_CUDA_INV_ROPE_OP, _CUDA_INV_ROPE_OP
    if not _CACHED_CUDA_INV_ROPE_OP:
        try:
            from rtp_llm.ops import rtp_llm_ops

            _CUDA_INV_ROPE_OP = getattr(
                rtp_llm_ops, "dsv4_fused_inv_rope_fp8_quant", None
            )
        except Exception:
            _CUDA_INV_ROPE_OP = None
        _CACHED_CUDA_INV_ROPE_OP = True
    return _CUDA_INV_ROPE_OP


@functools.cache
def _is_blackwell_device_idx(dev_idx: int) -> bool:
    try:
        major, _ = torch.cuda.get_device_capability(dev_idx)
    except Exception:
        return False
    return major >= 10


def _is_blackwell_device(device: torch.device | int | None = None) -> bool:
    if isinstance(device, int):
        return _is_blackwell_device_idx(device)
    if isinstance(device, torch.device):
        if device.index is not None:
            return _is_blackwell_device_idx(device.index)
        return _is_blackwell_device_idx(torch.cuda.current_device())
    if device is None:
        try:
            return _is_blackwell_device_idx(torch.cuda.current_device())
        except Exception:
            return False
    try:
        major, _ = torch.cuda.get_device_capability(device)
    except Exception:
        return False
    return major >= 10


@triton.jit(do_not_specialize=["num_tokens", "scale_stride_k", "max_pos"])
def _fused_inv_rope_fp8_quant_per_head(
    o_ptr,  # [M, H, D] bf16
    cos_ptr,  # [B_or_max, RD_HALF] fp32 (from freqs_cis.real)
    sin_ptr,  # [B_or_max, RD_HALF] fp32 (from freqs_cis.imag)
    pos_ptr,  # [B] int32/int64 or None
    fp8_ptr,  # [G, M_pad, d] fp8 e4m3 (d = heads_per_group * D)
    scale_ptr,  # [G, M_pad, packed_sf_k] int32 UE8M0, TMA-aligned
    num_tokens,
    max_pos,
    q_len_per_b: tl.constexpr,
    heads_per_group: tl.constexpr,
    o_stride_token,
    o_stride_head,
    cos_stride_b,
    sin_stride_b,
    fp8_stride_group,
    fp8_stride_token,
    scale_stride_group,
    scale_stride_k,  # for TMA-aligned layout this is ``tma_aligned_M``
    fp8_max: tl.constexpr,
    eps: tl.constexpr,
    QUANT_GROUP_SIZE: tl.constexpr,  # 128
    CHUNKS_PER_HEAD: tl.constexpr,  # head_dim // 128 (= 4 for dsv4 head_dim=512)
    ROPE_START: tl.constexpr,  # nope_dim % QUANT_GROUP_SIZE
    HALF_ROPE: tl.constexpr,  # rope_head_dim // 2
    HAS_POS: tl.constexpr,
):
    # int64: stride multiply overflows int32 past num_tokens=32768 (IMA).
    pid_token = tl.program_id(0).to(tl.int64)
    pid_gh = tl.program_id(1).to(tl.int64)

    g = pid_gh // heads_per_group
    head_in_group = pid_gh % heads_per_group
    global_head = pid_gh  # flat head index in original o layout [M, H, D]

    # Padding rows in the TMA-aligned scale buffer: fill with zero, skip quant.
    if pid_token >= num_tokens:
        scale_addr = (
            scale_ptr
            + g * scale_stride_group
            + pid_token  # M offset (innermost, stride=1)
            + head_in_group * scale_stride_k
        )
        tl.store(scale_addr, tl.zeros((), dtype=tl.int32))
        return

    input_base = o_ptr + pid_token * o_stride_token + global_head * o_stride_head

    HEAD_DIM: tl.constexpr = CHUNKS_PER_HEAD * QUANT_GROUP_SIZE
    offsets = tl.arange(0, HEAD_DIM)
    x = tl.load(input_base + offsets).to(tl.float32)

    # --- inverse RoPE on last RD columns ---------------------------------
    rope_abs_start: tl.constexpr = (CHUNKS_PER_HEAD - 1) * QUANT_GROUP_SIZE + ROPE_START
    is_rope = offsets >= rope_abs_start
    rope_local = offsets - rope_abs_start

    # Partner load: interleaved (real, imag, real, imag, …) — XOR last bit
    # swaps real↔imag.  Masked to zero outside rope region so non-rope
    # positions pass through unchanged via the ``tl.where`` below.
    x_partner = tl.load(input_base + (offsets ^ 1), mask=is_rope, other=0.0).to(
        tl.float32
    )
    # cs_idx = floor(rope_local / 2).  ``tl.maximum(.., 0)`` protects
    # non-rope lanes (they'd get negative indices and load OOB even
    # behind a mask, which Triton dislikes).
    cs_idx = tl.maximum(rope_local >> 1, 0)
    b_idx = pid_token // q_len_per_b
    if HAS_POS:
        pos_val = tl.load(pos_ptr + b_idx).to(tl.int64)
        freq_row = tl.maximum(tl.minimum(pos_val, max_pos - 1), 0)
    else:
        freq_row = b_idx
    cos_v = tl.load(cos_ptr + freq_row * cos_stride_b + cs_idx, mask=is_rope, other=1.0)
    sin_v = tl.load(sin_ptr + freq_row * sin_stride_b + cs_idx, mask=is_rope, other=0.0)
    # Inverse RoPE == multiply by conj(freqs_cis):
    #   (a + bi) · conj(c + di) = (ac + bd) + (bc − ad) i
    # For our interleaved layout, even offsets hold real (a), odd hold
    # imag (b).  At an even lane we output (a·cos + partner·sin); at an
    # odd lane (b·cos − partner·sin).  Matches ``apply_rotary_emb_batched(…,
    # inverse=True)`` in rope.py.
    x_add = x * cos_v + x_partner * sin_v
    x_sub = x * cos_v - x_partner * sin_v
    is_even = (rope_local & 1) == 0
    rotated = tl.where(is_even, x_add, x_sub)
    x = tl.where(is_rope, rotated, x)

    # --- per-128-column FP8 quant ----------------------------------------
    x_2d = tl.reshape(tl.abs(x), (CHUNKS_PER_HEAD, QUANT_GROUP_SIZE))
    block_absmax = tl.maximum(tl.max(x_2d, axis=1), eps)
    scale_raw = block_absmax * (1.0 / fp8_max)
    scales = tl.math.exp2(tl.ceil(tl.log2(scale_raw)))  # UE8M0-round

    scales_exp = tl.reshape(
        tl.broadcast_to(
            tl.reshape(scales, (CHUNKS_PER_HEAD, 1)),
            (CHUNKS_PER_HEAD, QUANT_GROUP_SIZE),
        ),
        (HEAD_DIM,),
    )
    x_quant = tl.clamp(x / scales_exp, -fp8_max, fp8_max).to(tl.float8e4nv)

    fp8_base = (
        fp8_ptr
        + g * fp8_stride_group
        + pid_token * fp8_stride_token
        + head_in_group * HEAD_DIM
    )
    tl.store(fp8_base + offsets, x_quant)

    # --- pack CHUNKS_PER_HEAD UE8M0 bytes into 1 int32 -------------------
    # fp32 bits [30:23] are the biased exponent (same 8-bit wire format
    # as UE8M0).  Since ``scales`` is already power-of-two via
    # exp2(ceil(log2(.))) above, mantissa is zero and ``bits >> 23 & 0xFF``
    # is the exact UE8M0 byte.  Requires CHUNKS_PER_HEAD ≤ 4 (≤32 bits
    # of shift total).
    block_offsets = tl.arange(0, CHUNKS_PER_HEAD)
    scale_bits = scales.to(tl.int32, bitcast=True)
    ue8m0_bytes = (scale_bits >> 23) & 0xFF
    packed_val = tl.sum(ue8m0_bytes << (block_offsets * 8))
    scale_addr = (
        scale_ptr
        + g * scale_stride_group
        + pid_token  # M offset
        + head_in_group * scale_stride_k
    )
    tl.store(scale_addr, packed_val)


@triton.jit(do_not_specialize=["num_tokens", "scale_stride_k", "max_pos"])
def _fused_inv_rope_fp8_quant_group_heads(
    o_ptr,  # [M, H, D] bf16
    freqs_ri_ptr,  # torch.view_as_real(freqs), float32 [B_or_max, RD_HALF, 2]
    pos_ptr,  # [B] int32/int64 or None
    fp8_ptr,  # [G, M_pad, d] fp8 e4m3
    scale_ptr,  # [G, M_pad, packed_sf_k] int32 UE8M0
    num_tokens,
    max_pos,
    q_len_per_b: tl.constexpr,
    heads_per_group: tl.constexpr,
    o_stride_token,
    o_stride_head,
    freqs_stride_b,
    freqs_stride_k,
    fp8_stride_group,
    fp8_stride_token,
    scale_stride_group,
    scale_stride_k,
    fp8_max: tl.constexpr,
    eps: tl.constexpr,
    QUANT_GROUP_SIZE: tl.constexpr,
    CHUNKS_PER_HEAD: tl.constexpr,
    ROPE_START: tl.constexpr,
    ROPE_DIM: tl.constexpr,
    BLOCK_ROPE: tl.constexpr,
    BLOCK_TAIL_NOPE: tl.constexpr,
    HEADS_PER_CTA: tl.constexpr,
    HAS_POS: tl.constexpr,
):
    pid_token = tl.program_id(0).to(tl.int64)
    g = tl.program_id(1).to(tl.int64)
    head_tile = tl.program_id(2).to(tl.int64)
    head_base = head_tile * HEADS_PER_CTA
    h_idx = head_base + tl.arange(0, HEADS_PER_CTA)

    scale_tok_base = scale_ptr + g * scale_stride_group + pid_token
    if pid_token >= num_tokens:
        tl.store(
            scale_tok_base + h_idx * scale_stride_k,
            tl.zeros((HEADS_PER_CTA,), dtype=tl.int32),
        )
        return

    HEAD_DIM: tl.constexpr = CHUNKS_PER_HEAD * QUANT_GROUP_SIZE
    o_tok_base = (
        o_ptr + pid_token * o_stride_token + g * heads_per_group * o_stride_head
    )
    o_heads_base = o_tok_base + h_idx[:, None] * o_stride_head
    fp8_tok_base = fp8_ptr + g * fp8_stride_group + pid_token * fp8_stride_token
    fp8_heads_base = fp8_tok_base + h_idx[:, None] * HEAD_DIM

    # Pure NoPE chunks [0 .. (CHUNKS_PER_HEAD - 1) * QUANT_GROUP_SIZE):
    # no tl.split/tl.interleave and bounded register footprint per 128-col tile.
    q_off = tl.arange(0, QUANT_GROUP_SIZE)
    packed_val = tl.zeros((HEADS_PER_CTA,), dtype=tl.int32)
    for chunk_idx in tl.static_range(CHUNKS_PER_HEAD - 1):
        col_base = chunk_idx * QUANT_GROUP_SIZE
        xc = tl.load(o_heads_base + (col_base + q_off)[None, :]).to(tl.float32)
        block_absmax = tl.maximum(tl.max(tl.abs(xc), axis=1), eps)
        scale_raw = block_absmax * (1.0 / fp8_max)
        scale_raw_bits = scale_raw.to(tl.int32, bitcast=True)
        exp = ((scale_raw_bits >> 23) & 0xFF) + ((scale_raw_bits & 0x7FFFFF) != 0)
        exp = tl.minimum(tl.maximum(exp, 1), 254)
        # Exact power-of-two reciprocal for UE8M0 scale in [1, 254];
        # exp == 254 (2^+127) maps to subnormal 2^-127 (0x00400000).
        inv_bits = tl.where(exp == 254, 1 << 22, (254 - exp) << 23)
        inv_scale = inv_bits.to(tl.float32, bitcast=True)
        xc_quant = tl.clamp(xc * inv_scale[:, None], -fp8_max, fp8_max).to(
            tl.float8e4nv
        )
        tl.store(fp8_heads_base + (col_base + q_off)[None, :], xc_quant)
        packed_val = packed_val | (exp << (chunk_idx * 8))

    # Last 128-col quant chunk: single coalesced 128-col load, in-register
    # inverse-RoPE on [ROPE_START, QUANT_GROUP_SIZE), single absmax + store.
    b_idx = pid_token // q_len_per_b
    if HAS_POS:
        pos_val = tl.load(pos_ptr + b_idx).to(tl.int64)
        freq_row = tl.maximum(tl.minimum(pos_val, max_pos - 1), 0)
    else:
        freq_row = b_idx

    HALF_CHUNK: tl.constexpr = QUANT_GROUP_SIZE // 2
    NOPE_PAIRS: tl.constexpr = ROPE_START // 2
    HALF_ROPE: tl.constexpr = ROPE_DIM // 2
    pair_off = tl.arange(0, HALF_CHUNK)
    rope_pair = pair_off - NOPE_PAIRS
    is_rope_pair = (rope_pair >= 0) & (rope_pair < HALF_ROPE)
    cs_idx = tl.where(is_rope_pair, rope_pair, 0)
    freq_base = freqs_ri_ptr + freq_row * freqs_stride_b + cs_idx * freqs_stride_k
    cos_v = tl.load(freq_base, mask=is_rope_pair, other=1.0)
    sin_v = tl.load(freq_base + 1, mask=is_rope_pair, other=0.0)

    last_col_base: tl.constexpr = (CHUNKS_PER_HEAD - 1) * QUANT_GROUP_SIZE
    x_last = tl.load(o_heads_base + (last_col_base + q_off)[None, :]).to(tl.float32)

    x_even, x_odd = tl.split(tl.reshape(x_last, (HEADS_PER_CTA, HALF_CHUNK, 2)))
    rot_even = x_even * cos_v[None, :] + x_odd * sin_v[None, :]
    rot_odd = x_odd * cos_v[None, :] - x_even * sin_v[None, :]
    new_even = tl.where(is_rope_pair[None, :], rot_even, x_even)
    new_odd = tl.where(is_rope_pair[None, :], rot_odd, x_odd)
    x_last_rot = tl.interleave(new_even, new_odd)

    absmax_last = tl.maximum(tl.max(tl.abs(x_last_rot), axis=1), eps)
    scale_raw_last = absmax_last * (1.0 / fp8_max)
    scale_raw_bits_last = scale_raw_last.to(tl.int32, bitcast=True)
    exp_last = ((scale_raw_bits_last >> 23) & 0xFF) + (
        (scale_raw_bits_last & 0x7FFFFF) != 0
    )
    exp_last = tl.minimum(tl.maximum(exp_last, 1), 254)
    inv_bits_last = tl.where(exp_last == 254, 1 << 22, (254 - exp_last) << 23)
    inv_scale_last = inv_bits_last.to(tl.float32, bitcast=True)

    last_quant = tl.clamp(
        x_last_rot * inv_scale_last[:, None], -fp8_max, fp8_max
    ).to(tl.float8e4nv)
    tl.store(fp8_heads_base + (last_col_base + q_off)[None, :], last_quant)

    packed_val = packed_val | (exp_last << ((CHUNKS_PER_HEAD - 1) * 8))
    tl.store(scale_tok_base + h_idx * scale_stride_k, packed_val)


def _alloc_output_buffers(
    M: int,
    n_groups: int,
    d_per_group: int,
    packed_sf_k: int,
    device: torch.device,
    fp8_buf: torch.Tensor | None = None,
    scale_buf: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    tma_M = ((M + 3) // 4) * 4
    if fp8_buf is None:
        fp8_work = torch.empty(
            (n_groups, M, d_per_group), dtype=torch.float8_e4m3fn, device=device
        )
    else:
        assert fp8_buf.dtype == torch.float8_e4m3fn
        assert fp8_buf.shape[0] == n_groups and fp8_buf.shape[1] >= M
        assert fp8_buf.shape[2] == d_per_group
        fp8_work = fp8_buf[:, :M, :]

    if scale_buf is None:
        scale_work = torch.empty(
            n_groups * packed_sf_k * tma_M, dtype=torch.int32, device=device
        ).as_strided(
            (n_groups, M, packed_sf_k),
            (packed_sf_k * tma_M, 1, tma_M),
        )
    else:
        assert scale_buf.dtype == torch.int32
        assert scale_buf.shape[0] == n_groups and scale_buf.shape[1] >= M
        assert scale_buf.shape[2] == packed_sf_k
        scale_work = scale_buf[:, :M, :]
    return fp8_work, scale_work


def fused_inv_rope_fp8_quant(
    o: torch.Tensor,  # [B, S, H, head_dim] or [M, H, head_dim] bf16
    freqs_cis_per_b: torch.Tensor,  # [B or M, RD/2] or [max_pos, RD/2] complex64
    n_groups: int,
    heads_per_group: int,
    nope_dim: int,
    rope_head_dim: int,
    quant_group_size: int = 128,
    eps: float = 1e-10,
    fp8_buf: torch.Tensor | None = None,
    scale_buf: torch.Tensor | None = None,
    impl: str | None = None,
    heads_per_cta: int | None = None,
    position_ids: torch.Tensor | None = None,
):
    """Fused inverse-RoPE + block-scaled FP8 quant for wo_a input.

    Args:
        o: Attention output ``[B, S, H, head_dim]`` or flattened
           ``[M, H, head_dim]`` bf16.  ``H == n_groups * heads_per_group``.
           ``head_dim == nope_dim + rope_head_dim`` and must be a multiple
           of ``quant_group_size``.
        freqs_cis_per_b: Per-request ``[B, rope_head_dim // 2]``,
           per-token ``[M, rope_head_dim // 2]``, or full RoPE table
           ``[max_pos, rope_head_dim // 2]`` when ``position_ids`` is given.
        n_groups, heads_per_group: wo_a grouping.
        nope_dim, rope_head_dim: per-head NoPE / RoPE split.
        quant_group_size: FP8 quant block size along K (fixed at 128 for V4).
        eps: Numerical epsilon used inside absmax-scale computation.
        heads_per_cta: Optimized Triton path head grouping.  Valid values are
           1, 2, 4, and 8; ``None`` picks the hardware default (8 on
           Blackwell, 2 elsewhere, falling back to 1 when not divisible).
        position_ids: Optional ``[B]`` or ``[M]`` int32/int64 position indices
           for inline lookup into a 2D ``freqs_cis_per_b`` table.

    Returns:
        (o_fp8, o_scale) where:
            o_fp8   shape ``[M, G, d]``     float8_e4m3fn
            o_scale shape ``[M, G, K/512]`` int32, UE8M0 packed,
                    stride ``(1, K/512 * tma_M, tma_M)``.
        Both are drop-in inputs for
        ``deep_gemm.fp8_einsum("bhr,hdr->bhd", (o_fp8, o_scale),
        (wo_a_fp8, wo_a_scale), out, recipe=(1, 1, 128))``.
    """
    assert o.is_cuda and o.dtype == torch.bfloat16
    assert freqs_cis_per_b.is_cuda and freqs_cis_per_b.dtype == torch.complex64
    assert o.is_contiguous()
    assert freqs_cis_per_b.is_contiguous()

    # Normalize input to [M, H, D]
    if o.dim() == 4:
        B, S, H, D = o.shape
        M = B * S
        o_flat = o.view(M, H, D)
    else:
        assert o.dim() == 3
        M, H, D = o.shape
        B = M
        S = 1
        o_flat = o

    if position_ids is not None:
        assert freqs_cis_per_b.dim() == 2
        pos_flat = position_ids.reshape(-1)
        if pos_flat.device != o.device:
            pos_flat = pos_flat.to(device=o.device)
        if pos_flat.dtype not in (torch.int32, torch.int64):
            pos_flat = pos_flat.to(dtype=torch.int32)
        if not pos_flat.is_contiguous():
            pos_flat = pos_flat.contiguous()
        pos_rows = int(pos_flat.shape[0])
        if pos_rows == M:
            q_len_per_b = 1
        elif o.dim() == 4 and pos_rows == B:
            q_len_per_b = S
        elif pos_rows > 0 and M % pos_rows == 0:
            q_len_per_b = M // pos_rows
        else:
            raise AssertionError(f"M={M} incompatible with pos_rows={pos_rows}")
        max_pos = int(freqs_cis_per_b.shape[0])
        has_pos = True
    else:
        pos_flat = None
        max_pos = 1
        has_pos = False
        if o.dim() == 4:
            freq_rows = int(freqs_cis_per_b.shape[0])
            if freq_rows == M:
                q_len_per_b = 1
            elif freq_rows == B:
                q_len_per_b = S
            else:
                raise AssertionError(
                    f"freq rows must be B={B} or M={M}; got {freq_rows}"
                )
        else:
            freq_B = freqs_cis_per_b.shape[0]
            assert M % freq_B == 0, f"M={M} not divisible by B={freq_B}"
            q_len_per_b = M // freq_B

    assert H == n_groups * heads_per_group
    assert D == nope_dim + rope_head_dim
    assert D % quant_group_size == 0
    assert nope_dim % quant_group_size == (quant_group_size - rope_head_dim), (
        f"RoPE must end the last {quant_group_size}-column quant block: "
        f"nope_dim={nope_dim}, rope_head_dim={rope_head_dim}, "
        f"quant_group_size={quant_group_size}"
    )
    assert rope_head_dim % 2 == 0
    assert freqs_cis_per_b.shape[-1] == rope_head_dim // 2

    chunks_per_head = D // quant_group_size
    assert chunks_per_head <= 4, (
        f"kernel packs CHUNKS_PER_HEAD UE8M0 bytes into 1 int32; need "
        f"chunks_per_head<=4, got {chunks_per_head}"
    )

    d_per_group = heads_per_group * D
    num_scale_blocks = d_per_group // quant_group_size
    assert num_scale_blocks % chunks_per_head == 0, (
        f"packed_sf_k expects heads_per_group alignment: "
        f"num_scale_blocks={num_scale_blocks}, chunks_per_head={chunks_per_head}"
    )
    packed_sf_k = num_scale_blocks // chunks_per_head  # = heads_per_group

    fp8_max = _FP8_E4M3_MAX

    fp8_work, scale_work = _alloc_output_buffers(
        M, n_groups, d_per_group, packed_sf_k, o.device, fp8_buf, scale_buf
    )
    if M == 0:
        return fp8_work.transpose(0, 1), scale_work.transpose(0, 1)
    tma_M = scale_work.stride(2)
    scale_zero_rows = (
        min(int(scale_buf.shape[1]), int(tma_M))
        if scale_buf is not None
        else int(tma_M)
    )

    cuda_op = _get_cuda_inv_rope_fp8_quant_op()
    can_use_cuda = (
        cuda_op is not None
        and D == 512
        and nope_dim == 448
        and rope_head_dim == 64
        and quant_group_size == 128
        and o_flat.stride(2) == 1
        and (o_flat.data_ptr() & 15) == 0
        and (o_flat.stride(0) & 7) == 0
        and (o_flat.stride(1) & 7) == 0
        and fp8_work.stride(2) == 1
        and (fp8_work.data_ptr() & 7) == 0
        and (fp8_work.stride(0) & 7) == 0
        and (fp8_work.stride(1) & 7) == 0
        and scale_work.stride(1) == 1
        and (freqs_cis_per_b.data_ptr() & 15) == 0
    )

    selected_impl = (
        impl.lower() if impl is not None else ("cuda" if can_use_cuda else "optimized")
    )
    if selected_impl == "cuda":
        if not can_use_cuda:
            raise ValueError(
                "impl='cuda' requested but CUDA kernel/constraints unavailable for this tensor shape"
            )
        freqs_ri = torch.view_as_real(freqs_cis_per_b)
        cuda_op(
            o_flat,
            freqs_ri,
            pos_flat,
            fp8_work,
            scale_work,
            q_len_per_b,
            n_groups,
            heads_per_group,
            nope_dim,
            rope_head_dim,
            quant_group_size,
            fp8_max,
            eps,
            scale_zero_rows,
        )
    elif selected_impl == "legacy":
        # complex64 .real / .imag are stride-2 views; .contiguous() is a real
        # layout copy. Keep this path only as a correctness/perf baseline.
        cos = freqs_cis_per_b.real.contiguous()
        sin = freqs_cis_per_b.imag.contiguous()
        grid = (tma_M, n_groups * heads_per_group)
        fast_launch(
            _fused_inv_rope_fp8_quant_per_head,
            grid,
            o_flat,
            cos,
            sin,
            pos_flat,
            fp8_work,
            scale_work,
            M,
            max_pos,
            q_len_per_b=q_len_per_b,
            heads_per_group=heads_per_group,
            o_stride_token=o_flat.stride(0),
            o_stride_head=o_flat.stride(1),
            cos_stride_b=cos.stride(0),
            sin_stride_b=sin.stride(0),
            fp8_stride_group=fp8_work.stride(0),
            fp8_stride_token=fp8_work.stride(1),
            scale_stride_group=scale_work.stride(0),
            scale_stride_k=scale_work.stride(2),
            fp8_max=fp8_max,
            eps=eps,
            QUANT_GROUP_SIZE=quant_group_size,
            CHUNKS_PER_HEAD=chunks_per_head,
            ROPE_START=nope_dim % quant_group_size,
            HALF_ROPE=rope_head_dim // 2,
            HAS_POS=has_pos,
            num_warps=1,
            num_stages=1,
        )
    elif selected_impl == "optimized":
        if heads_per_cta is not None:
            selected_heads_per_cta = heads_per_cta
        else:
            selected_heads_per_cta = 8 if _is_blackwell_device(o.device) else 2
        if selected_heads_per_cta not in (1, 2, 4, 8):
            raise ValueError(
                f"invalid heads_per_cta={selected_heads_per_cta}; "
                "expected 1, 2, 4, or 8"
            )
        if heads_per_group % selected_heads_per_cta != 0:
            selected_heads_per_cta = 1
        selected_num_warps = min(8, max(2, (selected_heads_per_cta * D) // 512))
        freqs_ri = torch.view_as_real(freqs_cis_per_b)
        rope_start = nope_dim % quant_group_size
        block_rope = triton.next_power_of_2(max(rope_head_dim, 2))
        block_tail_nope = triton.next_power_of_2(max(rope_start, 1))
        grid = (tma_M, n_groups, heads_per_group // selected_heads_per_cta)
        fast_launch(
            _fused_inv_rope_fp8_quant_group_heads,
            grid,
            o_flat,
            freqs_ri,
            pos_flat,
            fp8_work,
            scale_work,
            M,
            max_pos,
            q_len_per_b=q_len_per_b,
            heads_per_group=heads_per_group,
            o_stride_token=o_flat.stride(0),
            o_stride_head=o_flat.stride(1),
            freqs_stride_b=freqs_ri.stride(0),
            freqs_stride_k=freqs_ri.stride(1),
            fp8_stride_group=fp8_work.stride(0),
            fp8_stride_token=fp8_work.stride(1),
            scale_stride_group=scale_work.stride(0),
            scale_stride_k=scale_work.stride(2),
            fp8_max=fp8_max,
            eps=eps,
            QUANT_GROUP_SIZE=quant_group_size,
            CHUNKS_PER_HEAD=chunks_per_head,
            ROPE_START=rope_start,
            ROPE_DIM=rope_head_dim,
            BLOCK_ROPE=block_rope,
            BLOCK_TAIL_NOPE=block_tail_nope,
            HEADS_PER_CTA=selected_heads_per_cta,
            HAS_POS=has_pos,
            num_warps=selected_num_warps,
            num_stages=1,
        )
    else:
        raise ValueError(
            f"invalid impl={selected_impl!r}; " "expected legacy|optimized|cuda"
        )
    # Transpose (0, 1) so the consumer sees [M, G, …]:
    #   fp8  : [M, G, d]          stride (d, M*d, 1)   — contiguous-like on inner
    #   scale: [M, G, packed_sf_k] stride (1, packed_sf_k * tma_M, tma_M)
    return fp8_work.transpose(0, 1), scale_work.transpose(0, 1)


def fused_inv_rope_fp8_quant_legacy(*args, **kwargs):
    kwargs["impl"] = "legacy"
    return fused_inv_rope_fp8_quant(*args, **kwargs)


def fused_inv_rope_fp8_quant_optimized(*args, **kwargs):
    kwargs["impl"] = "optimized"
    return fused_inv_rope_fp8_quant(*args, **kwargs)
