"""Contiguous head-row scheduling with the original FP32 reduction order.

vLLM's GDN norm flattens token/head rows. Preserve that traversal while
retaining RTP's four adjacent values per lane and original 1/sqrt arithmetic.
"""

import torch
import triton
from triton.experimental import gluon
from triton.experimental.gluon import language as gl


@gluon.jit(do_not_specialize=["M", "ZX"])
def _gdn_flat_head_norm(
    X,
    Z,
    W,
    Y,
    M,
    N: gl.constexpr,
    ZX,
    EPS: gl.constexpr,
    GROUPS: gl.constexpr,
):
    layout: gl.constexpr = gl.BlockedLayout([1, 4], [1, 32], [1, 1], [1, 0])
    flat = gl.program_id(0) * 8 + gl.arange(0, 8, layout=gl.SliceLayout(1, layout))
    row = flat // GROUPS
    head = flat % GROUPS
    col = gl.arange(0, 128, layout=gl.SliceLayout(0, layout))
    offset = head[:, None] * 128 + col[None, :]
    valid = flat[:, None] < M * GROUPS
    value = gl.load(X + row[:, None] * N + offset, valid, other=0.0).to(gl.float32)
    variance = gl.sum(value * value, 1) / 128.0
    inverse = 1.0 / gl.sqrt(variance + EPS)
    weight = gl.load(W + col).to(gl.float32)
    gate = gl.load(Z + row[:, None] * ZX + offset, valid, other=0.0).to(gl.float32)
    result = (value * inverse[:, None]) * weight[None, :]
    result *= 1.0 / (1.0 + gl.exp(-gate))
    gl.store(Y + row[:, None] * N + offset, result, valid)


def flat_head_norm(x, gate, weight, eps):
    rows, width = x.shape
    out = torch.empty_like(x)
    _gdn_flat_head_norm[(triton.cdiv(rows * (width // 128), 8),)](
        x,
        gate,
        weight,
        out,
        M=rows,
        N=width,
        ZX=gate.stride(0),
        EPS=eps,
        GROUPS=width // 128,
        num_warps=1,
    )
    return out
