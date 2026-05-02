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
@triton.jit
def _fused_rmsnorm_rope_kernel(
    x_ptr,
    w_ptr,
    cos_ptr,
    sin_ptr,
    out_ptr,
    x_stride_n,
    out_stride_n,
    freq_stride_n: tl.constexpr,
    D: tl.constexpr,
    RD: tl.constexpr,
    RD_HALF: tl.constexpr,
    NOPE_OFFSET: tl.constexpr,
    INVERSE: tl.constexpr,
    EPS: tl.constexpr,
    HAS_WEIGHT: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid = tl.program_id(0)
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
    nope_mask = d_mask & (d_off < NOPE_OFFSET)
    tl.store(out_row + d_off, y, mask=nope_mask)

    pair_off = tl.arange(0, RD_HALF)
    real_off = NOPE_OFFSET + 2 * pair_off
    imag_off = real_off + 1

    real = tl.load(x_row + real_off).to(tl.float32) * inv
    imag = tl.load(x_row + imag_off).to(tl.float32) * inv
    if HAS_WEIGHT:
        w_real = tl.load(w_ptr + real_off).to(tl.float32)
        w_imag = tl.load(w_ptr + imag_off).to(tl.float32)
        real = real * w_real
        imag = imag * w_imag

    cos = tl.load(cos_ptr + freq_idx * RD_HALF + pair_off)
    sin = tl.load(sin_ptr + freq_idx * RD_HALF + pair_off)
    if INVERSE:
        sin = -sin

    new_real = real * cos - imag * sin
    new_imag = real * sin + imag * cos

    tl.store(out_row + real_off, new_real)
    tl.store(out_row + imag_off, new_imag)


def fused_rmsnorm_rope(
    x: torch.Tensor,
    weight: torch.Tensor | None,
    freqs_cis: torch.Tensor,
    rope_head_dim: int,
    *,
    eps: float = 1e-6,
    inverse: bool = False,
) -> torch.Tensor:
    """Fused RMSNorm-over-last-dim + partial RoPE on the final ``rope_head_dim`` cols.

    ``x``          may be ``[B, S, D]`` (single-head, e.g. KV) or
                   ``[B, S, H, D]`` (multi-head, e.g. Q).
    ``freqs_cis``  complex64 with any leading shape whose total element count
                   N_freq divides the number of tokens in ``x``.  Supports:
                     - decode per-request:  ``[B, RD/2]``
                     - prefill per-position:``[S, RD/2]`` or ``[B, S, RD/2]``
                     - batched decode:      ``[B, 1, RD/2]``
                   Mapping: each freq slot covers ``N_tokens // N_freq``
                   consecutive tokens in the ravelled x layout.

    Returns a new tensor of the same shape & dtype as ``x``.
    """
    assert x.is_cuda
    assert x.dtype in (torch.bfloat16, torch.float16, torch.float32)
    assert x.is_contiguous()
    assert freqs_cis.is_contiguous()
    orig_shape = x.shape
    D = orig_shape[-1]
    RD = rope_head_dim
    assert RD % 2 == 0 and RD <= D
    x_flat = x.view(-1, D)
    out = torch.empty_like(x_flat)
    N = x_flat.shape[0]
    if N == 0:
        return out.view(*orig_shape)

    if weight is not None:
        assert weight.shape == (D,) and weight.is_contiguous()
        w = weight.to(torch.float32)
        has_weight = True
    else:
        w = torch.empty(1, dtype=torch.float32, device=x.device)
        has_weight = False

    freqs_flat = freqs_cis.view(-1, freqs_cis.shape[-1])
    N_freq = freqs_flat.shape[0]
    assert N % N_freq == 0, f"N_tokens={N} not divisible by N_freq={N_freq}"
    freq_stride_n = N // N_freq
    # complex64 .real / .imag are stride-2 views; .contiguous() is a real
    # layout copy. Kernel hardcodes cos/sin outer stride = RD/2.
    cos = freqs_flat.real.contiguous()
    sin = freqs_flat.imag.contiguous()
    assert cos.shape == (N_freq, RD // 2)

    BLOCK_D = triton.next_power_of_2(D)
    assert BLOCK_D <= 4096

    _fused_rmsnorm_rope_kernel[(N,)](
        x_flat,
        w,
        cos,
        sin,
        out,
        x_flat.stride(0),
        out.stride(0),
        freq_stride_n=freq_stride_n,
        D=D,
        RD=RD,
        RD_HALF=RD // 2,
        NOPE_OFFSET=D - RD,
        INVERSE=inverse,
        EPS=eps,
        HAS_WEIGHT=has_weight,
        BLOCK_D=BLOCK_D,
        num_warps=4 if BLOCK_D <= 512 else 8,
    )
    return out.view(*orig_shape)
