"""Fused RMSNorm + partial RoPE Triton kernel for DeepSeek-V4 decode.

Audit doc §7.4 P0 (row 1): replaces the 2-launch torch RMSNorm + eager
``apply_rotary_emb_batched`` sequence on the Q/KV decode/prefill path
with a single Triton launch. Correctness and perf (1.25-1.75x vs the
eager baseline across T in {64,128,256,4096,65536}) are validated in
``rtp_llm/models_py/modules/dsv4/test/test_fused_rmsnorm_rope.py`` —
that UT is the source of truth for both kernel and wrapper.
"""

from __future__ import annotations

import os

import torch
import triton
import triton.language as tl


_BLACKWELL_GROUP_HEADS8_MIN_FREQ = 65536


def _is_blackwell_device(device: torch.device | int | None = None) -> bool:
    try:
        major, _ = torch.cuda.get_device_capability(device)
    except Exception:
        return False
    return major >= 10


def _is_fake_or_meta_tensor(x: torch.Tensor) -> bool:
    if x.is_meta:
        return True
    try:
        from torch._subclasses.fake_tensor import FakeTensor

        return isinstance(x, FakeTensor)
    except Exception:
        return False


def _is_torch_compiling() -> bool:
    is_compiling = getattr(getattr(torch, "compiler", None), "is_compiling", None)
    return bool(is_compiling is not None and is_compiling())


# ---------------------------------------------------------------------------
# Candidate fused kernel — RMSNorm over last D + partial RoPE on last RD dims.
# ---------------------------------------------------------------------------
@triton.jit
def _fused_rmsnorm_rope_kernel(
    x_ptr,
    w_ptr,
    freqs_ri_ptr,
    out_ptr,
    x_stride_n,
    out_stride_n,
    freq_stride_n: tl.constexpr,
    freqs_stride_b,
    freqs_stride_k,
    D: tl.constexpr,
    RD: tl.constexpr,
    RD_HALF: tl.constexpr,
    NOPE_OFFSET: tl.constexpr,
    INVERSE: tl.constexpr,
    EPS: tl.constexpr,
    HAS_WEIGHT: tl.constexpr,
    BLOCK_D: tl.constexpr,
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

    freq_base = freqs_ri_ptr + freq_idx * freqs_stride_b + pair_off * freqs_stride_k
    cos = tl.load(freq_base)
    sin = tl.load(freq_base + 1)
    if INVERSE:
        sin = -sin

    new_real = real * cos - imag * sin
    new_imag = real * sin + imag * cos

    tl.store(out_row + real_off, new_real)
    tl.store(out_row + imag_off, new_imag)


@triton.jit
def _fused_rmsnorm_rope_group_heads_kernel(
    x_ptr,
    w_ptr,
    freqs_ri_ptr,
    out_ptr,
    x_stride_n,
    out_stride_n,
    freq_stride_n: tl.constexpr,
    freqs_stride_b,
    freqs_stride_k,
    D: tl.constexpr,
    RD: tl.constexpr,
    RD_HALF: tl.constexpr,
    NOPE_OFFSET: tl.constexpr,
    INVERSE: tl.constexpr,
    EPS: tl.constexpr,
    HAS_WEIGHT: tl.constexpr,
    BLOCK_D: tl.constexpr,
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
    x = tl.load(x_rows + d_off[None, :], mask=d_mask[None, :], other=0.0).to(
        tl.float32
    )

    var = tl.sum(x * x, axis=1) / D
    inv = tl.rsqrt(var + EPS)
    y = x * inv[:, None]
    if HAS_WEIGHT:
        w = tl.load(w_ptr + d_off, mask=d_off < D, other=0.0).to(tl.float32)
        y = y * w[None, :]

    nope_mask = d_mask & (d_off < NOPE_OFFSET)
    tl.store(out_rows + d_off[None, :], y, mask=nope_mask[None, :])

    pair_off = tl.arange(0, RD_HALF)
    real_off = NOPE_OFFSET + 2 * pair_off
    imag_off = real_off + 1

    real = tl.load(x_rows + real_off[None, :]).to(tl.float32) * inv[:, None]
    imag = tl.load(x_rows + imag_off[None, :]).to(tl.float32) * inv[:, None]
    if HAS_WEIGHT:
        w_real = tl.load(w_ptr + real_off).to(tl.float32)
        w_imag = tl.load(w_ptr + imag_off).to(tl.float32)
        real = real * w_real[None, :]
        imag = imag * w_imag[None, :]

    freq_base = freqs_ri_ptr + freq_idx * freqs_stride_b + pair_off * freqs_stride_k
    cos = tl.load(freq_base)
    sin = tl.load(freq_base + 1)
    if INVERSE:
        sin = -sin

    new_real = real * cos[None, :] - imag * sin[None, :]
    new_imag = real * sin[None, :] + imag * cos[None, :]
    tl.store(out_rows + real_off[None, :], new_real)
    tl.store(out_rows + imag_off[None, :], new_imag)


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

    Returns a tensor of the same shape & dtype as ``x``.  By default a new
    output tensor is allocated.  ``out`` can provide a preallocated contiguous
    output buffer, and ``inplace=True`` writes back into ``x``.  ``group_heads``
    groups Q-style rows sharing the same frequency slot; valid values are
    1, 2, 4, and 8.
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
    if _is_fake_or_meta_tensor(x) or _is_fake_or_meta_tensor(freqs_cis):
        return out_flat.view(*orig_shape)
    if N == 0:
        return out_flat.view(*orig_shape)

    if weight is not None:
        assert weight.shape == (D,) and weight.is_contiguous()
        assert weight.is_cuda and weight.dtype in (torch.bfloat16, torch.float16, torch.float32)
        w = weight
        has_weight = True
    else:
        w = torch.empty(1, dtype=torch.float32, device=x.device)
        has_weight = False

    freqs_flat = freqs_cis.view(-1, freqs_cis.shape[-1])
    N_freq = freqs_flat.shape[0]
    assert N % N_freq == 0, f"N_tokens={N} not divisible by N_freq={N_freq}"
    freq_stride_n = N // N_freq
    if not freqs_flat.is_complex():
        if _is_torch_compiling():
            return out_flat.view(*orig_shape)
        raise TypeError(f"freqs_cis must be complex, got {freqs_cis.dtype}")
    freqs_ri = torch.view_as_real(freqs_flat)
    assert freqs_ri.shape == (N_freq, RD // 2, 2)

    BLOCK_D = triton.next_power_of_2(D)
    assert BLOCK_D <= 4096

    env_group_heads = os.environ.get("DSV4_RMSNORM_ROPE_GROUP_HEADS")
    if group_heads is not None:
        selected_group_heads = group_heads
    elif env_group_heads is not None:
        selected_group_heads = int(env_group_heads)
    else:
        selected_group_heads = 1
        if _is_blackwell_device(x.device):
            group_heads8_min_freq = int(
                os.environ.get(
                    "DSV4_RMSNORM_ROPE_GROUP_HEADS8_MIN_FREQ",
                    str(_BLACKWELL_GROUP_HEADS8_MIN_FREQ),
                )
            )
            selected_group_heads = 8 if N_freq >= group_heads8_min_freq else 4
    if selected_group_heads not in (1, 2, 4, 8):
        raise ValueError(
            f"invalid DSV4_RMSNORM_ROPE_GROUP_HEADS={selected_group_heads}; "
            "expected 1, 2, 4, or 8"
        )
    can_group_heads = (
        selected_group_heads in (2, 4, 8)
        and not inplace
        and freq_stride_n % selected_group_heads == 0
        and D <= 512
    )
    if can_group_heads:
        grid = (N_freq, freq_stride_n // selected_group_heads)
        _fused_rmsnorm_rope_group_heads_kernel[grid](
            x_flat,
            w,
            freqs_ri,
            out_flat,
            x_flat.stride(0),
            out_flat.stride(0),
            freq_stride_n=freq_stride_n,
            freqs_stride_b=freqs_ri.stride(0),
            freqs_stride_k=freqs_ri.stride(1),
            D=D,
            RD=RD,
            RD_HALF=RD // 2,
            NOPE_OFFSET=D - RD,
            INVERSE=inverse,
            EPS=eps,
            HAS_WEIGHT=has_weight,
            BLOCK_D=BLOCK_D,
            GROUP_HEADS=selected_group_heads,
            num_warps=4 if BLOCK_D <= 512 else 8,
        )
    else:
        _fused_rmsnorm_rope_kernel[(N,)](
            x_flat,
            w,
            freqs_ri,
            out_flat,
            x_flat.stride(0),
            out_flat.stride(0),
            freq_stride_n=freq_stride_n,
            freqs_stride_b=freqs_ri.stride(0),
            freqs_stride_k=freqs_ri.stride(1),
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
    return out_flat.view(*orig_shape)
