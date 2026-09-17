"""CUDA Q/K normalization preserving the original 128-element reduction tree.

Each warp sums one 32-element segment. Combine segment sums as (0+2)+(1+3),
matching the existing one-row CUDA kernel before sqrt and BF16 rounding.
The optional prefill path is limited to the validated SM10x BF16 shape.
"""

from functools import lru_cache

import torch
import triton
import triton.language as tl


@triton.jit
def _segment_scale(x0, x1, x2, x3, eps):
    s0 = tl.sum(x0 * x0, axis=1)
    s1 = tl.sum(x1 * x1, axis=1)
    s2 = tl.sum(x2 * x2, axis=1)
    s3 = tl.sum(x3 * x3, axis=1)
    s = s0 + s2 + (s1 + s3)
    return 1 / tl.sqrt(s + eps)


@triton.jit
def _fused_qk_norm_kernel(
    q,
    k,
    qo,
    ko,
    rows,
    eps,
    H: tl.constexpr,
    QS: tl.constexpr,
    KS: tl.constexpr,
    BT: tl.constexpr,
):
    r = tl.program_id(0) * BT + tl.arange(0, BT)
    c = tl.arange(0, 32)
    mask = r[:, None] < rows
    qi = (r // H)[:, None].to(tl.int64) * QS + (r % H)[:, None] * 128 + c[None, :]
    ki = (r // H)[:, None].to(tl.int64) * KS + (r % H)[:, None] * 128 + c[None, :]
    q0 = tl.load(q + qi, mask, 0).to(tl.float32)
    q1 = tl.load(q + qi + 32, mask, 0).to(tl.float32)
    q2 = tl.load(q + qi + 64, mask, 0).to(tl.float32)
    q3 = tl.load(q + qi + 96, mask, 0).to(tl.float32)
    qr = _segment_scale(q0, q1, q2, q3, eps)
    oi = r[:, None].to(tl.int64) * 128 + c[None, :]
    tl.store(qo + oi, q0 * qr[:, None], mask)
    tl.store(qo + oi + 32, q1 * qr[:, None], mask)
    tl.store(qo + oi + 64, q2 * qr[:, None], mask)
    tl.store(qo + oi + 96, q3 * qr[:, None], mask)
    k0 = tl.load(k + ki, mask, 0).to(tl.float32)
    k1 = tl.load(k + ki + 32, mask, 0).to(tl.float32)
    k2 = tl.load(k + ki + 64, mask, 0).to(tl.float32)
    k3 = tl.load(k + ki + 96, mask, 0).to(tl.float32)
    kr = _segment_scale(k0, k1, k2, k3, eps)
    tl.store(ko + oi, k0 * kr[:, None], mask)
    tl.store(ko + oi + 32, k1 * kr[:, None], mask)
    tl.store(ko + oi + 64, k2 * kr[:, None], mask)
    tl.store(ko + oi + 96, k3 * kr[:, None], mask)


@lru_cache(maxsize=None)
def _supported_device(index):
    return (
        torch.version.hip is None and torch.cuda.get_device_capability(index)[0] == 10
    )


def supports_exact_qk_norm(q, k):
    return (
        q.is_cuda
        and k.device == q.device
        and q.dtype == k.dtype == torch.bfloat16
        and q.shape == k.shape
        and q.ndim == 4
        and q.shape[0] == 1
        and q.shape[1] >= 2048
        and q.shape[2] > 0
        and q.shape[3] == 128
        and q.stride(-1) == k.stride(-1) == 1
        and q.stride(-2) == k.stride(-2) == 128
        and _supported_device(q.device.index)
    )


def fused_l2norm_qk_exact(q, k, *, tile_rows=4):
    if not supports_exact_qk_norm(q, k):
        raise ValueError(
            "Exact CUDA Q/K norm requires the validated SM10x BF16 prefill layout"
        )
    if tile_rows != 4:
        raise ValueError("Only tile_rows=4 preserves the validated reduction layout")
    bt = tile_rows
    (_, t, h, d) = q.shape
    qo = torch.empty(q.shape, device=q.device, dtype=q.dtype)
    ko = torch.empty_like(qo)
    _fused_qk_norm_kernel[(triton.cdiv(t * h, bt),)](
        q,
        k,
        qo,
        ko,
        t * h,
        1e-06,
        h,
        q.stride(1),
        k.stride(1),
        bt,
        num_warps=4,
        num_stages=3,
        enable_fp_fusion=False,
    )
    return (qo, ko)
