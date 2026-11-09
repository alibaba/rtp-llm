"""Standalone Triton kernels for the V4.1 PPU mHC op.

# Adapted from sglang-for-sail python/sglang/kernels/ops/layernorm/mhc.py
# (512b03d), Apache-2.0.

Copied kernels (numerics identical, dispatch plumbing removed):
* ``_hc_split_sinkhorn_triton`` (+ its kernel) -- the pre/post/comb split with
  the reference sinkhorn eps placement;
* ``hc_mix_stats`` (+ partial/reduce kernels) -- the batch-invariant
  ``F.linear(x.float(), fn) * rsqrt(mean(x^2) + eps)`` projection;
* ``mhc_combine`` (+ ``_mhc_combine_kernel``) -- the FP32 readout collapse
  ``y[m, h] = sum_k pre[m, k] * x_flat[m, k*H + h]``.

``mhc_post`` is a fresh Triton port of sglang's TileLang
``mhc_post_tilelang`` (TileLang is not available on PPU); it computes
``R'[t, i] = post[t, i] * x[t] + sum_j comb[t, j, i] * R[t, j]`` with FP32
accumulation, matching ``mhc_post_tilelang``'s arithmetic order.

Deliberate deviations from the sglang sources (all numerically neutral):
* sglang.srt platform/env dispatch (get_platform, is_hip, envs) removed --
  ``ppu/mhc.py`` selects this path at call time;
* dot precision for the projection defaults to ``"ieee"`` (FP32-exact, matching
  the torch authority path). sglang used ``"tf32x3"`` on SM90/Blackwell for
  speed; flip ``_HC_MIX_DOT_PRECISION`` if the PPU compiler prefers it;
* row-offset pointer arithmetic widened to int64 so >2^31-element prefill
  tensors cannot silently wrap;
* ``hc_split_sinkhorn`` consumes the flat 2-D ``[n, mix_width]`` layout and
  keeps FP32 outputs (the caller owns dtype policy).

Importing this module requires ``triton``; import it lazily (see
``rtp_llm.models_py.modules.dsv41.ppu.mhc._triton_kernels``).
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

__all__ = [
    "hc_mix_stats",
    "hc_split_sinkhorn",
    "mhc_combine",
    "mhc_post",
]


# ---------------------------------------------------------------------------
# Split + sinkhorn (copied)
# ---------------------------------------------------------------------------


@triton.jit
def _hc_split_sinkhorn_triton_kernel(
    mixes_ptr,
    hc_scale_ptr,
    hc_base_ptr,
    pre_ptr,
    post_ptr,
    comb_ptr,
    n,
    HC: tl.constexpr,
    MIX_HC: tl.constexpr,
    SINKHORN_ITERS: tl.constexpr,
    EPS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Triton port of the TileLang hc_split_sinkhorn kernel (one program per row)."""
    row = tl.program_id(0)
    if row >= n:
        return

    scale0 = tl.load(hc_scale_ptr + 0)
    scale1 = tl.load(hc_scale_ptr + 1)
    scale2 = tl.load(hc_scale_ptr + 2)

    j = tl.arange(0, BLOCK)
    jmask = j < HC

    # pre = sigmoid(mixes[:HC] * scale0 + base[:HC]) + eps
    base_pre = tl.load(hc_base_ptr + j, mask=jmask, other=0.0)
    mix_pre = tl.load(mixes_ptr + row * MIX_HC + j, mask=jmask, other=0.0)
    pre = tl.sigmoid(mix_pre * scale0 + base_pre) + EPS
    tl.store(pre_ptr + row * HC + j, pre, mask=jmask)

    # post = 2 * sigmoid(mixes[HC:2*HC] * scale1 + base[HC:2*HC])
    base_post = tl.load(hc_base_ptr + HC + j, mask=jmask, other=0.0)
    mix_post = tl.load(mixes_ptr + row * MIX_HC + HC + j, mask=jmask, other=0.0)
    post = 2.0 * tl.sigmoid(mix_post * scale1 + base_post)
    tl.store(post_ptr + row * HC + j, post, mask=jmask)

    # comb[j, k] = mixes[2*HC + j*HC + k] * scale2 + base[2*HC + j*HC + k]
    jj = j[:, None]
    kk = j[None, :]
    mmask = (jj < HC) & (kk < HC)
    coff = 2 * HC + jj * HC + kk
    base_c = tl.load(hc_base_ptr + coff, mask=mmask, other=0.0)
    mix_c = tl.load(mixes_ptr + row * MIX_HC + coff, mask=mmask, other=0.0)
    comb = mix_c * scale2 + base_c

    # Initial row softmax (numerically stabilized) then column normalize.
    comb_masked = tl.where(mmask, comb, float("-inf"))
    row_max = tl.max(comb_masked, axis=1)
    comb = tl.exp(comb - row_max[:, None])
    comb = tl.where(mmask, comb, 0.0)
    row_sum = tl.sum(comb, axis=1)
    comb = comb / row_sum[:, None] + EPS
    comb = tl.where(mmask, comb, 0.0)
    col_sum = tl.sum(comb, axis=0)
    comb = comb / (col_sum[None, :] + EPS)
    comb = tl.where(mmask, comb, 0.0)

    for _ in tl.static_range(SINKHORN_ITERS - 1):
        row_sum = tl.sum(comb, axis=1)
        comb = comb / (row_sum[:, None] + EPS)
        comb = tl.where(mmask, comb, 0.0)
        col_sum = tl.sum(comb, axis=0)
        comb = comb / (col_sum[None, :] + EPS)
        comb = tl.where(mmask, comb, 0.0)

    tl.store(comb_ptr + row * HC * HC + jj * HC + kk, comb, mask=mmask)


def hc_split_sinkhorn(
    mixes: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    hc_mult: int = 4,
    sinkhorn_iters: int = 20,
    eps: float = 1e-6,
):
    """Split ``mixes [n, (2+hc)*hc]`` into (pre, post, comb), all FP32.

    Adapted from sglang's ``_hc_split_sinkhorn_triton`` for the flat 2-D layout.
    """
    n, mix_hc = mixes.shape
    hc = hc_mult
    if n == 0:
        return (
            mixes.new_empty(n, hc, dtype=torch.float32),
            mixes.new_empty(n, hc, dtype=torch.float32),
            mixes.new_empty(n, hc, hc, dtype=torch.float32),
        )
    if mix_hc != (2 + hc) * hc:
        raise ValueError(
            f"mixes width {mix_hc} != (2+hc)*hc = {(2 + hc) * hc} for hc_mult={hc}"
        )

    flat = mixes.reshape(n, mix_hc).float().contiguous()
    scale = hc_scale.float().reshape(3).contiguous()
    base = hc_base.float().reshape(mix_hc).contiguous()

    pre = mixes.new_empty(n, hc, dtype=torch.float32)
    post = mixes.new_empty(n, hc, dtype=torch.float32)
    comb = mixes.new_empty(n, hc, hc, dtype=torch.float32)

    _hc_split_sinkhorn_triton_kernel[(n,)](
        flat,
        scale,
        base,
        pre,
        post,
        comb,
        n,
        HC=hc,
        MIX_HC=mix_hc,
        SINKHORN_ITERS=sinkhorn_iters,
        EPS=eps,
        BLOCK=triton.next_power_of_2(hc),
    )
    return pre, post, comb


# ---------------------------------------------------------------------------
# Mix-stats projection (copied)
# ---------------------------------------------------------------------------


@triton.jit
def _hc_mix_stats_partial_kernel(
    x_ptr,
    w_ptr,
    part_mix_ptr,
    part_sq_ptr,
    M,
    K,
    x_stride_m,
    w_stride_n,
    MIX: tl.constexpr,
    MIX_PAD: tl.constexpr,
    NUM_SLICES: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
    DOT_PRECISION: tl.constexpr,
):
    """Mixing dot products and row sum of squares over one K slice."""
    pid_m = tl.program_id(0)
    pid_s = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_m64 = offs_m.to(tl.int64)
    offs_n = tl.arange(0, MIX_PAD)
    mask_m = offs_m < M
    mask_n = offs_n < MIX
    k_per_slice = K // NUM_SLICES
    k_start = pid_s * k_per_slice
    acc = tl.zeros([BLOCK_M, MIX_PAD], dtype=tl.float32)
    sq = tl.zeros([BLOCK_M], dtype=tl.float32)
    for kb in range(0, k_per_slice, BLOCK_K):
        offs_k = k_start + kb + tl.arange(0, BLOCK_K)
        mask_k = offs_k < k_start + k_per_slice
        x_tile = tl.load(
            x_ptr + offs_m64[:, None] * x_stride_m + offs_k[None, :],
            mask=mask_m[:, None] & mask_k[None, :],
            other=0.0,
        ).to(tl.float32)
        w_tile = tl.load(
            w_ptr + offs_n[None, :] * w_stride_n + offs_k[:, None],
            mask=mask_n[None, :] & mask_k[:, None],
            other=0.0,
        ).to(tl.float32)
        acc += tl.dot(x_tile, w_tile, input_precision=DOT_PRECISION)
        sq += tl.sum(x_tile * x_tile, axis=1)
    part_off = pid_s.to(tl.int64) * M + offs_m64
    tl.store(
        part_mix_ptr + part_off[:, None] * MIX + offs_n[None, :],
        acc,
        mask=mask_m[:, None] & mask_n[None, :],
    )
    tl.store(part_sq_ptr + part_off, sq, mask=mask_m)


@triton.jit
def _hc_mix_stats_reduce_kernel(
    part_mix_ptr,
    part_sq_ptr,
    mixes_ptr,
    M,
    inv_k,
    eps,
    MIX: tl.constexpr,
    MIX_PAD: tl.constexpr,
    NUM_SLICES: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    """Sum the NUM_SLICES partials in slice order and apply the rms scaling."""
    pid_m = tl.program_id(0)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_m64 = offs_m.to(tl.int64)
    offs_n = tl.arange(0, MIX_PAD)
    mask_m = offs_m < M
    mask_n = offs_n < MIX
    acc = tl.zeros([BLOCK_M, MIX_PAD], dtype=tl.float32)
    sq = tl.zeros([BLOCK_M], dtype=tl.float32)
    # Promote M to int64 WITHOUT ``M.to`` — the P container's Triton binds
    # runtime scalars as values with ``.to``, the D container's binds them
    # as plain python ints in make_ir. ``offs_m64 * 0 + M`` promotes through
    # the tensor operand on both (works for int and tensor-like M).
    m64 = offs_m64 * 0 + M
    for s in tl.static_range(NUM_SLICES):
        part_off = s * m64 + offs_m64
        acc += tl.load(
            part_mix_ptr + part_off[:, None] * MIX + offs_n[None, :],
            mask=mask_m[:, None] & mask_n[None, :],
            other=0.0,
        )
        sq += tl.load(part_sq_ptr + part_off, mask=mask_m, other=0.0)
    rsqrt = 1.0 / tl.sqrt(sq * inv_k + eps)
    tl.store(
        mixes_ptr + offs_m64[:, None] * MIX + offs_n[None, :],
        acc * rsqrt[:, None],
        mask=mask_m[:, None] & mask_n[None, :],
    )


# K slicing, BLOCK_K and dot precision must stay independent of M;
# a row must produce the same bits alone and in any batch routed to this backend.
_HC_MIX_SLICE_CHOICES = (80, 64, 40, 32, 16, 8, 4, 2, 1)
_HC_MIX_BLOCK_M = 32
_HC_MIX_BLOCK_K = 64
_HC_MIX_NUM_WARPS = 4
# sglang used "tf32x3" on SM90/Blackwell (compensated) and "ieee" on HIP; the
# PPU reference stays FP32-exact to match the torch authority path.
_HC_MIX_DOT_PRECISION = "ieee"
_HC_MIX_NUM_STAGES = 1

# BLOCK_M 8/16/32 preserve each row's K reduction order (measured on GB300 in
# sglang); keep below 64, where Triton lowers tf32x3 to plain TF32.
_HC_MIX_BLOCK_M_SMALL = 8
_HC_MIX_BLOCK_M_MID = 16
_HC_MIX_MID_MAX_M = 2048


def _block_m_for(m: int) -> int:
    """Row-tile choices preserve each row's arithmetic and may depend on M."""
    if m <= _HC_MIX_BLOCK_M_SMALL:
        return _HC_MIX_BLOCK_M_SMALL
    if m <= _HC_MIX_MID_MAX_M:
        return _HC_MIX_BLOCK_M_MID
    return _HC_MIX_BLOCK_M


def _num_slices_for(k: int) -> int:
    """Slice count depends only on K, never on batch size M."""
    blocks = k // _HC_MIX_BLOCK_K
    if k % _HC_MIX_BLOCK_K:
        raise ValueError(f"hc_mix_stats requires K % {_HC_MIX_BLOCK_K} == 0; got K={k}")
    for n in _HC_MIX_SLICE_CHOICES:
        if blocks % n == 0:
            return n
    return 1


def hc_mix_stats(x_flat: torch.Tensor, hc_fn: torch.Tensor, eps: float) -> torch.Tensor:
    """Batch-invariant F.linear(x_flat.float(), hc_fn) * rsqrt(mean(x_flat^2) + eps).

    x_flat is [M, K] in any float dtype; hc_fn is [MIX, K] fp32; returns [M, MIX] fp32.
    K slicing and reduction order are independent of M, so each row is bitwise
    identical whether computed alone or in any batch routed to this backend.
    """
    assert x_flat.dim() == 2 and hc_fn.dim() == 2
    assert x_flat.stride(1) == 1 and hc_fn.stride(1) == 1
    assert hc_fn.dtype == torch.float32
    m, k = x_flat.shape
    mix = hc_fn.shape[0]
    assert hc_fn.shape[1] == k
    num_slices = _num_slices_for(k)
    mix_pad = max(16, triton.next_power_of_2(mix))
    part_mix = torch.empty(
        (num_slices, m, mix), dtype=torch.float32, device=x_flat.device
    )
    part_sq = torch.empty((num_slices, m), dtype=torch.float32, device=x_flat.device)
    mixes = torch.empty((m, mix), dtype=torch.float32, device=x_flat.device)
    if m == 0:
        return mixes
    block_m = _block_m_for(m)
    grid_m = triton.cdiv(m, block_m)
    _hc_mix_stats_partial_kernel[(grid_m, num_slices)](
        x_flat,
        hc_fn,
        part_mix,
        part_sq,
        m,
        k,
        x_flat.stride(0),
        hc_fn.stride(0),
        MIX=mix,
        MIX_PAD=mix_pad,
        NUM_SLICES=num_slices,
        BLOCK_M=block_m,
        BLOCK_K=_HC_MIX_BLOCK_K,
        DOT_PRECISION=_HC_MIX_DOT_PRECISION,
        num_warps=_HC_MIX_NUM_WARPS,
        num_stages=_HC_MIX_NUM_STAGES,
    )
    _hc_mix_stats_reduce_kernel[(grid_m,)](
        part_mix,
        part_sq,
        mixes,
        m,
        1.0 / k,
        eps,
        MIX=mix,
        MIX_PAD=mix_pad,
        NUM_SLICES=num_slices,
        BLOCK_M=block_m,
        num_warps=4,
    )
    return mixes


# ---------------------------------------------------------------------------
# Readout collapse (copied)
# ---------------------------------------------------------------------------


@triton.jit
def _mhc_combine_kernel(
    x_ptr,
    pre_ptr,
    y_ptr,
    H,
    x_stride_m,
    pre_stride_m,
    pre_stride_k,
    y_stride_m,
    HC: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_h = tl.program_id(1)
    m64 = pid_m.to(tl.int64)
    offs_h = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    mask = offs_h < H
    acc = tl.zeros([BLOCK_H], dtype=tl.float32)
    for k in tl.static_range(HC):
        pk = tl.load(pre_ptr + m64 * pre_stride_m + k * pre_stride_k).to(tl.float32)
        xv = tl.load(
            x_ptr + m64 * x_stride_m + k * H + offs_h, mask=mask, other=0.0
        ).to(tl.float32)
        acc += pk * xv
    tl.store(y_ptr + m64 * y_stride_m + offs_h, acc, mask=mask)


def mhc_combine(
    x_flat: torch.Tensor, pre: torch.Tensor, hc: int, out_dtype: torch.dtype
) -> torch.Tensor:
    """Fused y[m, h] = sum_k pre[m, k] * x_flat[m, k*H + h] (FP32 accumulate)."""
    assert x_flat.stride(1) == 1
    m = x_flat.shape[0]
    h = x_flat.shape[1] // hc
    y = torch.empty((m, h), dtype=out_dtype, device=x_flat.device)
    if m == 0:
        return y
    block_h = 1024
    _mhc_combine_kernel[(m, triton.cdiv(h, block_h))](
        x_flat,
        pre,
        y,
        h,
        x_flat.stride(0),
        pre.stride(0),
        pre.stride(1),
        y.stride(0),
        HC=hc,
        BLOCK_H=block_h,
    )
    return y


# ---------------------------------------------------------------------------
# Post (fresh Triton port of sglang's mhc_post_tilelang)
# ---------------------------------------------------------------------------


@triton.jit
def _mhc_post_kernel(
    x_ptr,
    shared_ptr,
    r_ptr,
    post_ptr,
    comb_ptr,
    y_ptr,
    H,
    x_stride_m,
    shared_stride_m,
    r_stride_m,
    post_stride_m,
    comb_stride_m,
    y_stride_m,
    HC: tl.constexpr,
    BLOCK_H: tl.constexpr,
    HAS_SHARED: tl.constexpr,
):
    """R'[t, i] = post[t, i] * x[t] + sum_j comb[t, j, i] * R[t, j], FP32 accum."""
    pid_m = tl.program_id(0)
    pid_h = tl.program_id(1)
    m64 = pid_m.to(tl.int64)
    offs_h = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    mask = offs_h < H
    ii = tl.arange(0, HC)

    post_v = tl.load(post_ptr + m64 * post_stride_m + ii).to(tl.float32)
    xv = tl.load(x_ptr + m64 * x_stride_m + offs_h, mask=mask, other=0.0).to(tl.float32)
    if HAS_SHARED:
        shared = tl.load(
            shared_ptr + m64 * shared_stride_m + offs_h, mask=mask, other=0.0
        ).to(tl.float32)
        # Retain the BF16 landing of the standalone routed + shared add.
        xv = (xv + shared).to(tl.bfloat16).to(tl.float32)
    # acc[i, h] starts at post[i] * x[h].
    acc = post_v[:, None] * xv[None, :]

    # Add sum_j comb[j, i] * R[j, h] lane by lane (same order as
    # mhc_post_tilelang's serial(hc) loop).
    for j in tl.static_range(HC):
        cj = tl.load(comb_ptr + m64 * comb_stride_m + j * HC + ii).to(tl.float32)
        rj = tl.load(
            r_ptr + m64 * r_stride_m + j * H + offs_h, mask=mask, other=0.0
        ).to(tl.float32)
        acc += cj[:, None] * rj[None, :]

    y_off = m64 * y_stride_m + ii[:, None] * H + offs_h[None, :]
    tl.store(y_ptr + y_off, acc, mask=mask[None, :])


def mhc_post(
    x: torch.Tensor,
    residual: torch.Tensor,
    post: torch.Tensor,
    comb: torch.Tensor,
    *,
    shared: torch.Tensor | None = None,
) -> torch.Tensor:
    """``R'[t, i] = post[t, i] * x[t] + sum_j comb[t, j, i] * R[t, j]``.

    x: [T, H]; residual: [T, hc, H]; post: [T, hc]; comb: [T, hc, hc].
    Returns [T, hc, H] in the residual's dtype.
    """
    assert x.dim() == 2 and residual.dim() == 3
    assert post.dim() == 2 and comb.dim() == 3
    assert residual.shape[0] == x.shape[0] == post.shape[0] == comb.shape[0]
    hc = residual.shape[1]
    h = residual.shape[2]
    assert post.shape[1] == hc and comb.shape[1] == hc and comb.shape[2] == hc
    assert h == x.shape[1]
    if shared is not None:
        assert shared.shape == x.shape
        assert shared.device == x.device
        assert shared.dtype == x.dtype == torch.bfloat16
        shared = shared.contiguous()

    x = x.contiguous()
    residual = residual.contiguous()
    post = post.contiguous()
    comb = comb.contiguous()
    out = torch.empty_like(residual)
    m = residual.shape[0]
    if m == 0:
        return out
    if hc & (hc - 1):
        raise ValueError(f"mhc_post Triton kernel requires power-of-2 hc; got {hc}")
    block_h = 1024
    _mhc_post_kernel[(m, triton.cdiv(h, block_h))](
        x,
        shared if shared is not None else x,
        residual,
        post,
        comb,
        out,
        h,
        x.stride(0),
        shared.stride(0) if shared is not None else 0,
        residual.stride(0),
        post.stride(0),
        comb.stride(0),
        out.stride(0),
        HC=hc,
        BLOCK_H=block_h,
        HAS_SHARED=shared is not None,
        num_warps=4,
    )
    return out
