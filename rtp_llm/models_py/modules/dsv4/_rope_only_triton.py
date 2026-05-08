"""RoPE-only Triton kernel for DeepSeek-V4 indexer prefill.

This replaces the eager ``apply_rotary_emb`` path for tensors that have
already been normalized/projected and only need in-place rotary embedding on
their RoPE tail.  Unlike ``_fused_rmsnorm_rope_triton``, this kernel does not
compute RMSNorm.
"""

from __future__ import annotations

import os

import torch
import triton
import triton.language as tl


_DEFAULT_GROUP_HEADS = 8


@triton.jit
def _rope_only_kernel(
    x_ptr,
    freqs_ri_ptr,
    row_stride,
    freq_stride_n: tl.constexpr,
    freqs_stride_b,
    freqs_stride_k,
    RD: tl.constexpr,
    RD_HALF: tl.constexpr,
    INVERSE: tl.constexpr,
    BLOCK_RD_HALF: tl.constexpr,
):
    pid = tl.program_id(0).to(tl.int64)
    freq_idx = pid // freq_stride_n
    pair_off = tl.arange(0, BLOCK_RD_HALF)
    mask = pair_off < RD_HALF

    row = x_ptr + pid * row_stride
    real_off = 2 * pair_off
    imag_off = real_off + 1
    real = tl.load(row + real_off, mask=mask, other=0.0).to(tl.float32)
    imag = tl.load(row + imag_off, mask=mask, other=0.0).to(tl.float32)

    freq_base = freqs_ri_ptr + freq_idx * freqs_stride_b + pair_off * freqs_stride_k
    cos = tl.load(freq_base, mask=mask, other=1.0)
    sin = tl.load(freq_base + 1, mask=mask, other=0.0)
    if INVERSE:
        sin = -sin

    new_real = real * cos - imag * sin
    new_imag = real * sin + imag * cos
    tl.store(row + real_off, new_real, mask=mask)
    tl.store(row + imag_off, new_imag, mask=mask)


@triton.jit
def _rope_only_group_heads_kernel(
    x_ptr,
    freqs_ri_ptr,
    row_stride,
    freq_stride_n: tl.constexpr,
    freqs_stride_b,
    freqs_stride_k,
    RD: tl.constexpr,
    RD_HALF: tl.constexpr,
    INVERSE: tl.constexpr,
    BLOCK_RD_HALF: tl.constexpr,
    GROUP_HEADS: tl.constexpr,
):
    freq_idx = tl.program_id(0).to(tl.int64)
    head_tile = tl.program_id(1).to(tl.int64)
    row_start = freq_idx * freq_stride_n + head_tile * GROUP_HEADS

    h_off = tl.arange(0, GROUP_HEADS)
    pair_off = tl.arange(0, BLOCK_RD_HALF)
    mask = pair_off < RD_HALF

    rows = x_ptr + (row_start + h_off)[:, None] * row_stride
    real_off = 2 * pair_off
    imag_off = real_off + 1
    real = tl.load(rows + real_off[None, :], mask=mask[None, :], other=0.0).to(tl.float32)
    imag = tl.load(rows + imag_off[None, :], mask=mask[None, :], other=0.0).to(tl.float32)

    freq_base = freqs_ri_ptr + freq_idx * freqs_stride_b + pair_off * freqs_stride_k
    cos = tl.load(freq_base, mask=mask, other=1.0)
    sin = tl.load(freq_base + 1, mask=mask, other=0.0)
    if INVERSE:
        sin = -sin

    new_real = real * cos[None, :] - imag * sin[None, :]
    new_imag = real * sin[None, :] + imag * cos[None, :]
    tl.store(rows + real_off[None, :], new_real, mask=mask[None, :])
    tl.store(rows + imag_off[None, :], new_imag, mask=mask[None, :])


def rope_only_inplace(
    x: torch.Tensor,
    freqs_cis: torch.Tensor,
    *,
    inverse: bool = False,
    group_heads: int | None = None,
) -> torch.Tensor:
    """Apply RoPE in-place to ``x``.

    ``x`` is the RoPE tail view, typically ``q[..., -rd:]``.  It may be a
    non-contiguous tail view from a larger head dimension, but the final
    dimension must have stride 1 and all flattened rows must have a constant
    stride.
    """
    if x.numel() == 0 or freqs_cis.numel() == 0:
        return x
    assert x.is_cuda
    assert x.dtype in (torch.bfloat16, torch.float16, torch.float32)
    assert x.dim() >= 2
    assert x.stride(-1) == 1

    RD = x.shape[-1]
    assert RD % 2 == 0
    N = x.numel() // RD
    row_stride = x.stride(-2)

    if not freqs_cis.is_contiguous():
        freqs_cis = freqs_cis.contiguous()
    freqs_flat = freqs_cis.view(-1, freqs_cis.shape[-1])
    N_freq = freqs_flat.shape[0]
    assert N % N_freq == 0, f"N_tokens={N} not divisible by N_freq={N_freq}"
    assert freqs_flat.shape[-1] == RD // 2
    freq_stride_n = N // N_freq
    freqs_ri = torch.view_as_real(freqs_flat)

    block_half = triton.next_power_of_2(RD // 2)
    selected_group_heads = (
        group_heads
        if group_heads is not None
        else int(os.environ.get("DSV4_ROPE_ONLY_GROUP_HEADS", str(_DEFAULT_GROUP_HEADS)))
    )
    can_group_heads = (
        selected_group_heads in (2, 4, 8)
        and freq_stride_n % selected_group_heads == 0
        and RD <= 128
    )
    if can_group_heads:
        grid = (N_freq, freq_stride_n // selected_group_heads)
        _rope_only_group_heads_kernel[grid](
            x,
            freqs_ri,
            row_stride,
            freq_stride_n=freq_stride_n,
            freqs_stride_b=freqs_ri.stride(0),
            freqs_stride_k=freqs_ri.stride(1),
            RD=RD,
            RD_HALF=RD // 2,
            INVERSE=inverse,
            BLOCK_RD_HALF=block_half,
            GROUP_HEADS=selected_group_heads,
            num_warps=4,
        )
    else:
        _rope_only_kernel[(N,)](
            x,
            freqs_ri,
            row_stride,
            freq_stride_n=freq_stride_n,
            freqs_stride_b=freqs_ri.stride(0),
            freqs_stride_k=freqs_ri.stride(1),
            RD=RD,
            RD_HALF=RD // 2,
            INVERSE=inverse,
            BLOCK_RD_HALF=block_half,
            num_warps=4,
        )
    return x
