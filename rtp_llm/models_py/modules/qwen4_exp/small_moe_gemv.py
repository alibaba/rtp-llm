# SPDX-License-Identifier: Apache-2.0

"""Experimental BF16 small-token MoE for Qwen4 on SM120.

The up GEMV preserves the intermediate BF16 rounding before SiLU.
Down GEMV applies router weights before BF16 rounding; top-k reduction
uses the existing Torch implementation. Routing is read on each replay.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _small_moe_up(
    x,
    w1,
    ids,
    activation,
    H: tl.constexpr,
    I: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
):
    r = tl.program_id(0)
    e = tl.load(ids + r).to(tl.int64)
    n = tl.program_id(1) * BN + tl.arange(0, BN)
    k = tl.arange(0, BK)
    a = tl.load(x + (r // 10) * H + k, mask=k < H, other=0).to(tl.float32)
    v = tl.load(
        w1 + e * (2 * I * H) + n[:, None] * H + k[None, :],
        mask=(n[:, None] < I) & (k[None, :] < H),
        other=0,
    ).to(tl.float32)
    g = tl.load(
        w1 + e * (2 * I * H) + (n[:, None] + I) * H + k[None, :],
        mask=(n[:, None] < I) & (k[None, :] < H),
        other=0,
    ).to(tl.float32)
    value = tl.sum(v * a[None, :], 1).to(tl.bfloat16).to(tl.float32)
    gate = tl.sum(g * a[None, :], 1).to(tl.bfloat16).to(tl.float32)
    y = (gate * tl.sigmoid(gate)) * value
    tl.store(activation + r * I + n, y, mask=n < I)


@triton.jit
def _small_moe_down(
    activation,
    w2,
    ids,
    weights,
    out,
    H: tl.constexpr,
    I: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
):
    r = tl.program_id(0)
    e = tl.load(ids + r).to(tl.int64)
    n = tl.program_id(1) * BN + tl.arange(0, BN)
    k = tl.arange(0, BK)
    a = tl.load(activation + r * I + k, mask=k < I, other=0).to(tl.float32)
    w = tl.load(
        w2 + e * (H * I) + n[:, None] * I + k[None, :],
        mask=(n[:, None] < H) & (k[None, :] < I),
        other=0,
    ).to(tl.float32)
    y = tl.sum(w * a[None, :], 1) * tl.load(weights + r).to(tl.float32)
    tl.store(out + r * H + n, y, mask=n < H)


def is_supported(x, w1, w2, ids, weights):
    tensors = (x, w1, w2, ids, weights)
    if torch.version.hip is not None or not all(
        t.is_cuda and t.is_contiguous() for t in tensors
    ):
        return False
    if any(t.device != x.device for t in tensors) or x.ndim != 2:
        return False
    m = x.shape[0]
    return (
        1 <= m <= 4
        and x.shape == (m, 2560)
        and w1.shape == (512, 160, 2560)
        and w2.shape == (512, 2560, 80)
        and ids.shape == (m, 10)
        and weights.shape == (m, 10)
        and all(t.dtype == torch.bfloat16 for t in (x, w1, w2))
        and ids.dtype == torch.int32
        and weights.dtype == torch.float32
        and torch.cuda.get_device_capability(x.device) == (12, 0)
    )


def small_moe_gemv(x, w1, w2, ids, weights):
    """Return None for unsupported layouts; caller retains the original path."""
    if not is_supported(x, w1, w2, ids, weights):
        return None
    m = x.shape[0]
    activation = torch.empty((m * 10, 80), dtype=x.dtype, device=x.device)
    out = torch.empty((m * 10, 2560), dtype=x.dtype, device=x.device)
    _small_moe_up[(m * 10, 20)](
        x,
        w1,
        ids,
        activation,
        H=2560,
        I=80,
        BN=4,
        BK=4096,
        num_warps=4,
        enable_fp_fusion=False,
    )
    _small_moe_down[(m * 10, 80)](
        activation,
        w2,
        ids,
        weights,
        out,
        H=2560,
        I=80,
        BN=32,
        BK=128,
        num_warps=4,
        enable_fp_fusion=False,
    )
    return out.view(m, 10, 2560).sum(1)
