# SPDX-License-Identifier: Apache-2.0
"""Single-launch BF16 M11 gate with the contiguous ATen dot order."""

from triton.experimental import gluon
from triton.experimental.gluon import language as gl


@gluon.jit
def _dot_sum(product, BLOCK_X: gl.constexpr):
    # ATen uses 512 threads, each accumulating two contiguous 8-value vectors.
    packed: gl.constexpr = gl.BlockedLayout([1, 8, 2], [32, 1, 1], [4, 1, 1], [1, 2, 0])
    vector: gl.constexpr = gl.BlockedLayout([1, 8], [32, 1], [4, 1], [1, 0])
    pair = gl.convert_layout(gl.permute(product.reshape(2, 512, 8), (1, 2, 0)), packed)
    first, second = gl.split(pair)
    values = gl.convert_layout(first + second, vector)
    partial = gl.gather(values, gl.full((512, 1), 0, gl.int32, vector), 1)
    for i in gl.static_range(1, 8):
        partial += gl.gather(values, gl.full((512, 1), i, gl.int32, vector), 1)

    if BLOCK_X == 32:
        reduction: gl.constexpr = gl.BlockedLayout([1, 1], [1, 32], [4, 1], [1, 0])
        lanes = gl.convert_layout(partial.reshape(16, 32), reduction)
        xsum = gl.sum(lanes, 1)
    else:
        # Match ATen's cross-warp X reduction before the shuffle within
        # each warp, then reduce the block's Y dimension.
        small_layout: gl.constexpr = gl.BlockedLayout(
            [1, 1, 1], [1, 1, 32], [1, 4, 1], [2, 1, 0]
        )
        lanes = gl.convert_layout(
            partial.reshape(512 // BLOCK_X, BLOCK_X // 32, 32), small_layout
        )
        xsum = gl.sum(gl.sum(lanes, 1), 1)
    return gl.sum(gl.convert_layout(xsum, gl.BlockedLayout([1], [32], [4], [0])), 0)


@gluon.jit
def _rms(x, gamma, eps):
    return (x * gl.rsqrt(gl.sum(x * x, 0) / 8192 + eps) * (1 + gamma)).to(gl.bfloat16)


@gluon.jit
def m11_gate(
    Key,
    Value,
    Hidden,
    GK,
    GQ,
    GC,
    Gated,
    Normed,
    EPS: gl.constexpr,
    BLOCK_X: gl.constexpr = 32,
):
    row = gl.program_id(0).to(gl.int64)
    branch = gl.program_id(1)
    layout: gl.constexpr = gl.BlockedLayout([8], [32], [4], [0])
    col = gl.arange(0, 8192, layout=layout)
    channel = branch * 8192 + col
    offset = row * 32768 + channel
    k = gl.load(Key + offset).to(gl.float32)
    q = gl.load(Hidden + offset).to(gl.float32)
    kn = _rms(k, gl.load(GK + channel).to(gl.float32), EPS).to(gl.float32)
    qn = _rms(q, gl.load(GQ + channel).to(gl.float32), EPS).to(gl.float32)
    product = (kn * qn).to(gl.bfloat16).to(gl.float32)
    dot = _dot_sum(product, BLOCK_X).to(gl.bfloat16).to(gl.float32)
    agreement = (dot / gl.sqrt(8192.0)).to(gl.bfloat16).to(gl.float32)
    floor = gl.full((), 1e-6, gl.bfloat16, layout).to(gl.float32)
    magnitude = (
        gl.sqrt(gl.maximum(gl.abs(agreement), floor)).to(gl.bfloat16).to(gl.float32)
    )
    sign = gl.where(agreement > 0, 1.0, gl.where(agreement < 0, -1.0, 0.0))
    weight = (1 / (1 + gl.exp(-(magnitude * sign)))).to(gl.bfloat16).to(gl.float32)
    v = gl.load(Value + row * 8192 + col).to(gl.float32)
    gated = (weight * v).to(gl.bfloat16)
    normed = _rms(gated.to(gl.float32), gl.load(GC + channel).to(gl.float32), EPS)
    gl.store(Gated + offset, gated)
    gl.store(Normed + offset, normed)
