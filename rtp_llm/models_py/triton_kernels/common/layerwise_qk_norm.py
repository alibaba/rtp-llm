"""Triton kernels for MiniMax-M2 ``LayerwiseQKRMSNorm``.

RMSNorm is taken over the FULL Q / K dims BEFORE per-head reshape; under TP>1
each rank's local sum_sq must be all-reduced before applying rsqrt. Splitting
the work as sumsq → AR → apply lets us issue a single AR per layer on a
[m, 2] fp32 tensor instead of two AR calls or a per-rank rsqrt with bias.
"""

from typing import Optional

import torch
import triton
import triton.language as tl


@triton.jit
def _rmsnorm_qk_sumsq_kernel(
    qkv_ptr,
    sumsq_ptr,
    stride_qkv_m,
    Q_SIZE: tl.constexpr,
    K_SIZE: tl.constexpr,
    BQ: tl.constexpr,
    BK: tl.constexpr,
):
    # Two passes (Q then K) inside the same kernel because their dims differ
    # (M2: q_local=6144/tp, kv_local=1024/tp). Output cols: 0=Q sumsq, 1=K sumsq.
    m = tl.program_id(0)
    row = qkv_ptr + m * stride_qkv_m

    q_offs = tl.arange(0, BQ)
    q_mask = q_offs < Q_SIZE
    q = tl.load(row + q_offs, mask=q_mask, other=0.0).to(tl.float32)
    q_sumsq = tl.sum(q * q, axis=0)

    k_offs = tl.arange(0, BK)
    k_mask = k_offs < K_SIZE
    k = tl.load(row + Q_SIZE + k_offs, mask=k_mask, other=0.0).to(tl.float32)
    k_sumsq = tl.sum(k * k, axis=0)

    tl.store(sumsq_ptr + m * 2, q_sumsq)
    tl.store(sumsq_ptr + m * 2 + 1, k_sumsq)


@triton.jit
def _rmsnorm_qk_apply_kernel(
    qkv_ptr,
    q_w_ptr,
    k_w_ptr,
    sumsq_ptr,
    eps,
    q_total_inv,
    k_total_inv,
    stride_qkv_m,
    Q_SIZE: tl.constexpr,
    K_SIZE: tl.constexpr,
    BQ: tl.constexpr,
    BK: tl.constexpr,
):
    # ``q_total_inv = 1/q_total_elts`` is precomputed on the host so the kernel
    # does one fmul instead of an fdiv per row.
    m = tl.program_id(0)
    row = qkv_ptr + m * stride_qkv_m

    q_sumsq = tl.load(sumsq_ptr + m * 2)
    k_sumsq = tl.load(sumsq_ptr + m * 2 + 1)
    q_scale = tl.math.rsqrt(q_sumsq * q_total_inv + eps)
    k_scale = tl.math.rsqrt(k_sumsq * k_total_inv + eps)

    q_offs = tl.arange(0, BQ)
    q_mask = q_offs < Q_SIZE
    q = tl.load(row + q_offs, mask=q_mask, other=0.0).to(tl.float32)
    q_w = tl.load(q_w_ptr + q_offs, mask=q_mask, other=0.0).to(tl.float32)
    q_out = q * q_scale * q_w
    tl.store(row + q_offs, q_out.to(qkv_ptr.dtype.element_ty), mask=q_mask)

    k_offs = tl.arange(0, BK)
    k_mask = k_offs < K_SIZE
    k = tl.load(row + Q_SIZE + k_offs, mask=k_mask, other=0.0).to(tl.float32)
    k_w = tl.load(k_w_ptr + k_offs, mask=k_mask, other=0.0).to(tl.float32)
    k_out = k * k_scale * k_w
    tl.store(row + Q_SIZE + k_offs, k_out.to(qkv_ptr.dtype.element_ty), mask=k_mask)


def _pick_num_warps(block: int) -> int:
    # One warp per ~256 elts, capped at 8 — keeps register pressure tractable
    # on H20 sm9x for the M2 ranges (BQ up to 8192, BK up to 1024).
    if block >= 4096:
        return 8
    if block >= 1024:
        return 4
    return 2


def rmsnorm_qk_sumsq(
    qkv: torch.Tensor,
    q_size: int,
    k_size: int,
    sumsq_out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    # Caller is responsible for all-reducing the returned [m, 2] fp32 tensor
    # across the TP group when tp_size > 1, before passing to rmsnorm_qk_apply.
    assert qkv.dim() == 2, f"expected 2D qkv, got {qkv.shape}"
    m = qkv.shape[0]
    if sumsq_out is None:
        sumsq_out = torch.empty((m, 2), dtype=torch.float32, device=qkv.device)
    else:
        assert sumsq_out.shape == (m, 2) and sumsq_out.dtype == torch.float32

    BQ = triton.next_power_of_2(q_size)
    BK = triton.next_power_of_2(k_size)
    num_warps = _pick_num_warps(max(BQ, BK))
    _rmsnorm_qk_sumsq_kernel[(m,)](
        qkv,
        sumsq_out,
        qkv.stride(0),
        Q_SIZE=q_size,
        K_SIZE=k_size,
        BQ=BQ,
        BK=BK,
        num_warps=num_warps,
        num_stages=1,
    )
    return sumsq_out


def rmsnorm_qk_apply(
    qkv: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    sumsq: torch.Tensor,
    q_total_elts: int,
    k_total_elts: int,
    eps: float,
) -> torch.Tensor:
    # Writes Q/K in place; V is left untouched. ``sumsq`` must be the post-AR
    # reduced tensor under TP>1 (see rmsnorm_qk_sumsq).
    assert qkv.dim() == 2, f"expected 2D qkv, got {qkv.shape}"
    m = qkv.shape[0]
    q_size = q_weight.numel()
    k_size = k_weight.numel()
    assert sumsq.shape == (m, 2) and sumsq.dtype == torch.float32

    BQ = triton.next_power_of_2(q_size)
    BK = triton.next_power_of_2(k_size)
    num_warps = _pick_num_warps(max(BQ, BK))
    _rmsnorm_qk_apply_kernel[(m,)](
        qkv,
        q_weight,
        k_weight,
        sumsq,
        eps,
        1.0 / float(q_total_elts),
        1.0 / float(k_total_elts),
        qkv.stride(0),
        Q_SIZE=q_size,
        K_SIZE=k_size,
        BQ=BQ,
        BK=BK,
        num_warps=num_warps,
        num_stages=1,
    )
    return qkv
