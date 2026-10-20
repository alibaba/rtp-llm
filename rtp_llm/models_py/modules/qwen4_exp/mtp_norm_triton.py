"""Pinned CUDA mean order for the MTP embedding's 2560-wide RMSNorm."""

import math

import torch
import triton
import triton.language as tl

_TORCH_GIT = "70d99e998b4955e0049d13a98d77ae1b14db1f45"


@triton.jit
def _mtp_embedding_rms_norm(
    X,
    G,
    Y,
    V,
    M: tl.constexpr,
    L: tl.constexpr,
    EPS: tl.constexpr,
    WRITE_VARIANCE: tl.constexpr = False,
):
    row = tl.program_id(0)
    lane = tl.arange(0, L)
    a = tl.full((L,), 0, tl.float32)
    b = tl.full((L,), 0, tl.float32)
    c = tl.full((L,), 0, tl.float32)
    d = tl.full((L,), 0, tl.float32)
    for block in tl.static_range(triton.cdiv(2560, L * 4)):
        col = block * L * 4 + lane * 4
        x0 = tl.load(X + row * 2560 + col, col < 2560, other=0.0).to(tl.float32)
        x1 = tl.load(X + row * 2560 + col + 1, col + 1 < 2560, other=0.0).to(tl.float32)
        x2 = tl.load(X + row * 2560 + col + 2, col + 2 < 2560, other=0.0).to(tl.float32)
        x3 = tl.load(X + row * 2560 + col + 3, col + 3 < 2560, other=0.0).to(tl.float32)
        a += x0 * x0
        b += x1 * x1
        c += x2 * x2
        d += x3 * x3
    sums = ((a + b) + c) + d
    if L > 32:
        sums = tl.sum(sums.reshape(L // 32, 32), 0)
    variance = tl.sum(sums, 0) * (1.0 / 2560.0)
    if WRITE_VARIANCE:
        tl.store(V + row, variance)
    inverse = tl.rsqrt(variance + EPS)
    columns = tl.arange(0, 4096)
    values = tl.load(X + row * 2560 + columns, columns < 2560, other=0.0).to(tl.float32)
    gain = 1.0 + tl.load(G + columns, columns < 2560, other=0.0).to(tl.float32)
    tl.store(Y + row * 2560 + columns, (values * inverse) * gain, columns < 2560)


def try_mtp_embedding_rms_norm(x, gamma, eps):
    if not (
        x.is_cuda
        and torch.version.hip is None
        and torch.version.git_version == _TORCH_GIT
        and not torch.are_deterministic_algorithms_enabled()
        and x.ndim == 2
        and x.shape[1] == 2560
        and 1 <= x.shape[0] <= 32768
        and x.dtype == torch.bfloat16
        and gamma.shape == (2560,)
        and gamma.dtype in (torch.bfloat16, torch.float32)
        and gamma.device == x.device
        and x.is_contiguous()
        and gamma.is_contiguous()
        and not (x.requires_grad or gamma.requires_grad)
        and isinstance(eps, (float, int))
        and math.isfinite(eps)
        and eps >= 0
        and torch.cuda.get_device_capability(x.device) == (12, 0)
    ):
        return None
    rows = x.shape[0]
    # CUDA uses four input elements per lane and up to 512 threads. Output
    # rows select its x/y block dimensions; preserve that reduction order.
    lanes = max(32, 512 // (1 << (rows.bit_length() - 1)))
    output = torch.empty_like(x)
    _mtp_embedding_rms_norm[(rows,)](
        x,
        gamma,
        output,
        output,
        M=rows,
        L=lanes,
        EPS=eps,
        num_warps=4,
        enable_fp_fusion=False,
    )
    return output
