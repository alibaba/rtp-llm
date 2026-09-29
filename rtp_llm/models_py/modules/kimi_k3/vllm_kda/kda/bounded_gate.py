# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Songlin Yang, Yu Zhang, Zhiyuan Li
# Explicit-layout adaptation of the pinned vLLM K3 bounded gate scan.
# The original FLA-derived gate is also distributed under the MIT license.
# See chunk.py and ../UPSTREAM.json for source provenance.
from triton.experimental import gluon as g
from triton.experimental.gluon import language as gl


@g.jit
def _add(a, b):
    return a + b


@g.jit
def bounded_gate_scan(
    Raw, RawBeta, A, Bias, Out, BetaOut, Cu, Indices, T,
    StrideBetaBatch, StrideBetaToken, StrideBetaHead, LowerBound,
    H: gl.constexpr, S: gl.constexpr, BT: gl.constexpr,
    HAS_BIAS: gl.constexpr, IS_VARLEN: gl.constexpr,
):
    tile, chunk, bh = gl.program_id(0), gl.program_id(1), gl.program_id(2)
    batch, head = bh // H, bh % H
    if IS_VARLEN:
        sequence = gl.load(Indices + chunk * 2).to(gl.int32)
        chunk = gl.load(Indices + chunk * 2 + 1).to(gl.int32)
        bos = gl.load(Cu + sequence).to(gl.int32)
        T = gl.load(Cu + sequence + 1).to(gl.int32) - bos
    else:
        bos = batch * T
    if tile == 0:
        layout_beta: gl.constexpr = gl.BlockedLayout([1], [32], [4], [0])
        row = chunk * BT + gl.arange(0, BT, layout=layout_beta)
        if IS_VARLEN:
            offset = (bos + row) * StrideBetaToken + head * StrideBetaHead
        else:
            offset = batch * StrideBetaBatch + row * StrideBetaToken + head * StrideBetaHead
        value = gl.load(RawBeta + offset, row < T, 0.0).to(gl.float32)
        beta = 1.0 / (1.0 + gl.exp(-value))
        gl.store(BetaOut + (bos + row) * H + head, beta, row < T)
        return
    tile -= 1
    # Match the native Triton 3.7 BS=32 scan layout under RTP's Triton 3.6.
    # Changing this distribution changes the FP32 prefix-sum association.
    width: gl.constexpr = 8 if HAS_BIAS else 4
    layout: gl.constexpr = gl.BlockedLayout([1, width], [width, 32 // width], [4, 1], [1, 0])
    row = chunk * BT + gl.arange(0, BT, layout=gl.SliceLayout(1, layout))
    col = tile * 32 + gl.arange(0, 32, layout=gl.SliceLayout(0, layout))
    offset = (bos + row[:, None]) * H * S + head * S + col[None, :]
    mask = (row[:, None] < T) & (col[None, :] < S)
    value = gl.load(Raw + offset, mask, 0.0).to(gl.float32)
    if HAS_BIAS:
        bias = gl.load(Bias + head * S + col, col < S, 0.0).to(gl.float32)
        value = value + bias[None, :]
    a = gl.exp(gl.load(A + head).to(gl.float32))
    # Preserve reciprocal followed by multiplication, including FP32 rounding.
    gate = LowerBound * (1.0 / (1.0 + gl.exp(-(a * value))))
    cumulative = gl.associative_scan(gate, axis=0, combine_fn=_add) * 1.4426950216
    gl.store(Out + offset, cumulative, mask)
