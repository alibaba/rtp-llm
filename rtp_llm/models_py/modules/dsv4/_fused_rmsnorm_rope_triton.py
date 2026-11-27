"""Fused RMSNorm + partial RoPE Triton kernel for DeepSeek-V4 decode.

Audit doc §7.4 P0 (row 1): replaces the 2-launch torch RMSNorm + eager
``apply_rotary_emb_batched`` sequence on the Q/KV decode/prefill path
with a single Triton launch. Correctness and perf (1.25-1.75x vs the
eager baseline across T in {64,128,256,4096,65536}) are validated in
``rtp_llm/models_py/modules/dsv4/test/test_fused_rmsnorm_rope.py`` —
that UT is the source of truth for both kernel and wrapper.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Candidate fused kernel — RMSNorm over last D + partial RoPE on last RD dims.
# ---------------------------------------------------------------------------
@triton.jit(do_not_specialize=["max_pos"])
def _fused_rmsnorm_rope_kernel(
    x_ptr,
    w_ptr,
    freqs_ri_ptr,
    pos_ptr,
    out_ptr,
    x_stride_n,
    out_stride_n,
    freq_stride_n: tl.constexpr,
    freqs_stride_b,
    freqs_stride_k,
    max_pos,
    D: tl.constexpr,
    RD: tl.constexpr,
    RD_HALF: tl.constexpr,
    NOPE_OFFSET: tl.constexpr,
    INVERSE: tl.constexpr,
    EPS: tl.constexpr,
    HAS_WEIGHT: tl.constexpr,
    HAS_POS: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_RD: tl.constexpr,
    BLOCK_RD_HALF: tl.constexpr,
):
    # Cast pid to int64 BEFORE the row-stride multiply: the prefill Q
    # site flattens [B, S, H, D] to [B*S*H, D], so N can exceed 2^22 and
    # ``pid * x_stride_n`` silently overflows int32 (e.g. S=65600 H=64
    # D=512: max pid * stride = 4198399 * 512 = 2.15B > INT32_MAX →
    # CUDA_ERROR_ILLEGAL_ADDRESS).  Same pitfall as v4_rmsnorm.
    pid = tl.program_id(0).to(tl.int64)
    # Tokens per freq slot — decouples the kernel from decode/prefill layout:
    #   decode Q  [B, 1, H, D] + freqs [B, RD/2]     -> freq_stride_n = H
    #   decode KV [B, 1,    D] + freqs [B, RD/2]     -> freq_stride_n = 1
    #   prefill Q [1, S, H, D] + freqs [S, RD/2]     -> freq_stride_n = H
    #   prefill KV[1, S,    D] + freqs [S, RD/2]     -> freq_stride_n = 1
    freq_idx = pid // freq_stride_n

    d_off = tl.arange(0, BLOCK_D)
    d_mask = d_off < D

    x_row = x_ptr + pid * x_stride_n
    x = tl.load(x_row + d_off, mask=d_mask, other=0.0).to(tl.float32)

    var = tl.sum(x * x, axis=0) / D
    inv = tl.rsqrt(var + EPS)
    y = x * inv
    if HAS_WEIGHT:
        w = tl.load(w_ptr + d_off, mask=d_mask, other=0.0).to(tl.float32)
        y = y * w

    out_row = out_ptr + pid * out_stride_n
    if RD == 0:
        tl.store(out_row + d_off, y, mask=d_mask)
    else:
        # Store NoPE prefix directly so the full [BLOCK_D] register live range
        # ends before RoPE loads/rotates the trailing [NOPE_OFFSET, D) slice.
        # RoPE columns remain untouched in x_row even when out aliases x.
        if NOPE_OFFSET > 0:
            nope_mask = d_mask & (d_off < NOPE_OFFSET)
            tl.store(out_row + d_off, y, mask=nope_mask)

        if HAS_POS:
            pos_val = tl.load(pos_ptr + freq_idx).to(tl.int64)
            row_idx = tl.maximum(tl.minimum(pos_val, max_pos - 1), 0)
        else:
            row_idx = freq_idx

        rope_off = tl.arange(0, BLOCK_RD)
        rope_mask = rope_off < RD
        x_rope = tl.load(
            x_row + NOPE_OFFSET + rope_off, mask=rope_mask, other=0.0
        ).to(tl.float32)
        y_rope = x_rope * inv
        if HAS_WEIGHT:
            w_rope = tl.load(
                w_ptr + NOPE_OFFSET + rope_off, mask=rope_mask, other=0.0
            ).to(tl.float32)
            y_rope = y_rope * w_rope

        pair_off = tl.arange(0, BLOCK_RD_HALF)
        pair_mask = pair_off < RD_HALF
        freq_base = (
            freqs_ri_ptr + row_idx * freqs_stride_b + pair_off * freqs_stride_k
        )
        cos = tl.load(freq_base, mask=pair_mask, other=1.0)
        sin = tl.load(freq_base + 1, mask=pair_mask, other=0.0)
        if INVERSE:
            sin = -sin

        y_even, y_odd = tl.split(tl.reshape(y_rope, (BLOCK_RD_HALF, 2)))
        rot_even = y_even * cos - y_odd * sin
        rot_odd = y_even * sin + y_odd * cos
        out_rope = tl.interleave(rot_even, rot_odd)
        tl.store(out_row + NOPE_OFFSET + rope_off, out_rope, mask=rope_mask)


@triton.jit(do_not_specialize=["max_pos"])
def _fused_rmsnorm_rope_group_heads_kernel(
    x_ptr,
    w_ptr,
    freqs_ri_ptr,
    pos_ptr,
    out_ptr,
    x_stride_n,
    out_stride_n,
    freq_stride_n: tl.constexpr,
    freqs_stride_b,
    freqs_stride_k,
    max_pos,
    D: tl.constexpr,
    RD: tl.constexpr,
    RD_HALF: tl.constexpr,
    NOPE_OFFSET: tl.constexpr,
    INVERSE: tl.constexpr,
    EPS: tl.constexpr,
    HAS_WEIGHT: tl.constexpr,
    HAS_POS: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_RD: tl.constexpr,
    BLOCK_RD_HALF: tl.constexpr,
    GROUP_HEADS: tl.constexpr,
):
    freq_idx = tl.program_id(0).to(tl.int64)
    head_tile = tl.program_id(1).to(tl.int64)
    row_start = freq_idx * freq_stride_n + head_tile * GROUP_HEADS

    h_off = tl.arange(0, GROUP_HEADS)
    d_off = tl.arange(0, BLOCK_D)
    rows = row_start + h_off
    x_rows = x_ptr + rows[:, None] * x_stride_n
    out_rows = out_ptr + rows[:, None] * out_stride_n
    d_mask = d_off < D
    x = tl.load(x_rows + d_off[None, :], mask=d_mask[None, :], other=0.0).to(tl.float32)

    var = tl.sum(x * x, axis=1) / D
    inv = tl.rsqrt(var + EPS)
    y = x * inv[:, None]
    if HAS_WEIGHT:
        w = tl.load(w_ptr + d_off, mask=d_mask, other=0.0).to(tl.float32)
        y = y * w[None, :]

    if RD == 0:
        tl.store(out_rows + d_off[None, :], y, mask=d_mask[None, :])
    else:
        # Pure scaled-RMSNorm store for NoPE columns [0, NOPE_OFFSET); skip
        # tl.split/tl.interleave and cos/sin math on all non-RoPE columns.
        if NOPE_OFFSET > 0:
            nope_mask = d_mask & (d_off < NOPE_OFFSET)
            tl.store(out_rows + d_off[None, :], y, mask=nope_mask[None, :])

        if HAS_POS:
            pos_val = tl.load(pos_ptr + freq_idx).to(tl.int64)
            row_idx = tl.maximum(tl.minimum(pos_val, max_pos - 1), 0)
        else:
            row_idx = freq_idx

        pair_off = tl.arange(0, BLOCK_RD_HALF)
        pair_mask = pair_off < RD_HALF
        freq_base = (
            freqs_ri_ptr + row_idx * freqs_stride_b + pair_off * freqs_stride_k
        )
        cos = tl.load(freq_base, mask=pair_mask, other=1.0)
        sin = tl.load(freq_base + 1, mask=pair_mask, other=0.0)
        if INVERSE:
            sin = -sin

        rope_off = tl.arange(0, BLOCK_RD)
        rope_mask = rope_off < RD
        x_rope = tl.load(
            x_rows + (NOPE_OFFSET + rope_off)[None, :],
            mask=rope_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        y_rope = x_rope * inv[:, None]
        if HAS_WEIGHT:
            w_rope = tl.load(
                w_ptr + NOPE_OFFSET + rope_off, mask=rope_mask, other=0.0
            ).to(tl.float32)
            y_rope = y_rope * w_rope[None, :]

        y_even, y_odd = tl.split(
            tl.reshape(y_rope, (GROUP_HEADS, BLOCK_RD_HALF, 2))
        )
        rot_even = y_even * cos[None, :] - y_odd * sin[None, :]
        rot_odd = y_even * sin[None, :] + y_odd * cos[None, :]
        out_rope = tl.interleave(rot_even, rot_odd)
        tl.store(
            out_rows + (NOPE_OFFSET + rope_off)[None, :],
            out_rope,
            mask=rope_mask[None, :],
        )


def fused_rmsnorm_rope(
    x: torch.Tensor,
    weight: torch.Tensor | None,
    freqs_cis: torch.Tensor,
    rope_head_dim: int,
    *,
    eps: float = 1e-6,
    inverse: bool = False,
    out: torch.Tensor | None = None,
    inplace: bool = False,
    group_heads: int | None = None,
    position_ids: torch.Tensor | None = None,
) -> torch.Tensor:
    """Fused RMSNorm-over-last-dim + partial RoPE on the final ``rope_head_dim`` cols.

    ``x``          may be ``[B, S, D]`` (single-head, e.g. KV) or
                   ``[B, S, H, D]`` (multi-head, e.g. Q).
    ``freqs_cis``  complex64 with any leading shape whose total element count
                   N_freq divides the number of tokens in ``x`` (when
                   ``position_ids is None``), or the full 2D position table
                   ``[max_pos, RD/2]`` when ``position_ids`` is given. Supports:
                     - decode per-request:  ``[B, RD/2]``
                     - prefill per-position:``[S, RD/2]`` or ``[B, S, RD/2]``
                     - batched decode:      ``[B, 1, RD/2]``
                     - inline table lookup: ``[max_pos, RD/2]`` + ``position_ids``
                   Mapping: each freq slot covers ``N_tokens // N_freq``
                   consecutive tokens in the ravelled x layout.

    Returns a tensor of the same shape & dtype as ``x``.  By default a new
    output tensor is allocated.  ``out`` can provide a preallocated contiguous
    output buffer, ``out`` may alias ``x``, and ``inplace=True`` writes back
    into ``x``.  ``group_heads`` groups Q-style rows sharing the same frequency
    slot; valid values are 1, 2, 4, and 8.
    """
    assert x.is_cuda
    assert x.dtype in (torch.bfloat16, torch.float16, torch.float32)
    assert x.is_contiguous()
    assert freqs_cis.is_contiguous()
    assert not (out is not None and inplace), "out and inplace are mutually exclusive"
    orig_shape = x.shape
    D = orig_shape[-1]
    RD = rope_head_dim
    assert RD % 2 == 0 and RD <= D
    x_flat = x.view(-1, D)
    N = x_flat.shape[0]
    if inplace:
        out_flat = x_flat
    elif out is not None:
        assert out.shape == x.shape
        assert out.dtype == x.dtype and out.is_cuda and out.is_contiguous()
        out_flat = out.view(-1, D)
    else:
        out_flat = torch.empty_like(x_flat)
    if N == 0:
        return out_flat.view(*orig_shape)

    if weight is not None:
        assert weight.shape == (D,) and weight.is_contiguous()
        assert weight.is_cuda and weight.dtype in (
            torch.bfloat16,
            torch.float16,
            torch.float32,
        )
        w = weight
        has_weight = True
    else:
        w = torch.empty(1, dtype=torch.float32, device=x.device)
        has_weight = False

    if position_ids is not None:
        assert freqs_cis.dim() == 2
        pos_flat = position_ids.reshape(-1)
        if pos_flat.device != x.device:
            pos_flat = pos_flat.to(device=x.device)
        if pos_flat.dtype not in (torch.int32, torch.int64):
            pos_flat = pos_flat.to(dtype=torch.int32)
        if not pos_flat.is_contiguous():
            pos_flat = pos_flat.contiguous()
        N_freq = pos_flat.shape[0]
        assert N % N_freq == 0, f"N_tokens={N} not divisible by pos_len={N_freq}"
        freq_stride_n = N // N_freq
        freqs_ri = torch.view_as_real(freqs_cis)
        max_pos = freqs_ri.shape[0]
        assert freqs_ri.shape[1:] == (RD // 2, 2)
        has_pos = True
    else:
        pos_flat = None
        freqs_flat = freqs_cis.view(-1, freqs_cis.shape[-1])
        N_freq = freqs_flat.shape[0]
        assert N % N_freq == 0, f"N_tokens={N} not divisible by N_freq={N_freq}"
        freq_stride_n = N // N_freq
        freqs_ri = torch.view_as_real(freqs_flat)
        max_pos = 1
        assert freqs_ri.shape == (N_freq, RD // 2, 2)
        has_pos = False

    BLOCK_D = triton.next_power_of_2(D)
    assert BLOCK_D <= 4096
    BLOCK_RD = triton.next_power_of_2(max(RD, 2))
    BLOCK_RD_HALF = BLOCK_RD // 2

    if group_heads is not None:
        selected_group_heads = group_heads
    else:
        selected_group_heads = 1
        if D <= 512 and not inplace:
            for gh in (8, 4, 2):
                if (
                    gh * D <= 2048
                    and freq_stride_n % gh == 0
                    and (freq_stride_n // gh) <= 65535
                ):
                    selected_group_heads = gh
                    break
    if selected_group_heads not in (1, 2, 4, 8):
        raise ValueError(
            f"invalid group_heads={selected_group_heads}; " "expected 1, 2, 4, or 8"
        )
    can_group_heads = (
        selected_group_heads in (2, 4, 8)
        and not inplace
        and freq_stride_n % selected_group_heads == 0
        and (freq_stride_n // selected_group_heads) <= 65535
        and D <= 512
    )
    if can_group_heads:
        grid = (N_freq, freq_stride_n // selected_group_heads)
        group_num_warps = (
            8 if (BLOCK_D > 512 or selected_group_heads * BLOCK_D >= 4096) else 4
        )
        _fused_rmsnorm_rope_group_heads_kernel[grid](
            x_flat,
            w,
            freqs_ri,
            pos_flat,
            out_flat,
            x_flat.stride(0),
            out_flat.stride(0),
            freq_stride_n=freq_stride_n,
            freqs_stride_b=freqs_ri.stride(0),
            freqs_stride_k=freqs_ri.stride(1),
            max_pos=max_pos,
            D=D,
            RD=RD,
            RD_HALF=RD // 2,
            NOPE_OFFSET=D - RD,
            INVERSE=inverse,
            EPS=eps,
            HAS_WEIGHT=has_weight,
            HAS_POS=has_pos,
            BLOCK_D=BLOCK_D,
            BLOCK_RD=BLOCK_RD,
            BLOCK_RD_HALF=BLOCK_RD_HALF,
            GROUP_HEADS=selected_group_heads,
            num_warps=group_num_warps,
            num_stages=1,
        )
    else:
        _fused_rmsnorm_rope_kernel[(N,)](
            x_flat,
            w,
            freqs_ri,
            pos_flat,
            out_flat,
            x_flat.stride(0),
            out_flat.stride(0),
            freq_stride_n=freq_stride_n,
            freqs_stride_b=freqs_ri.stride(0),
            freqs_stride_k=freqs_ri.stride(1),
            max_pos=max_pos,
            D=D,
            RD=RD,
            RD_HALF=RD // 2,
            NOPE_OFFSET=D - RD,
            INVERSE=inverse,
            EPS=eps,
            HAS_WEIGHT=has_weight,
            HAS_POS=has_pos,
            BLOCK_D=BLOCK_D,
            BLOCK_RD=BLOCK_RD,
            BLOCK_RD_HALF=BLOCK_RD_HALF,
            num_warps=4 if BLOCK_D <= 512 else 8,
            num_stages=1,
        )
    return out_flat.view(*orig_shape)
