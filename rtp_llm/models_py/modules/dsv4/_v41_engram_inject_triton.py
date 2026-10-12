# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fused V4.1 Engram gate and residual update.

Adapted via RTP-LLM donor 88993fcb36838def206525a079145a823e879e8a from
vLLM common/engram.py's _fused_engram_post_wkv_kernel, source SHA256
41c5bdf25cf8337088247be768e3d43fa41e6358397800f8b602ebf1ede69480.
RTP uses contiguous local tokens, a mandatory live mask and exact dead-row bypass.
"""

import triton
import triton.language as tl


@triton.jit
def engram_inject_kernel(
    Hidden,
    Projected,
    Q,
    K,
    Mask,
    Output,
    DIM: tl.constexpr,
    HC: tl.constexpr,
    EPS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    token = tl.program_id(0).to(tl.int64)
    head = tl.program_id(1)
    col = tl.arange(0, BLOCK)
    valid = col < DIM
    enabled = tl.load(Mask + token)
    offset = (token * HC + head) * DIM + col
    # One CTA owns a whole lane. All hidden reads precede writes to that lane,
    # so an exact Hidden/Output alias is safe; other input aliases are rejected.
    original = tl.load(Hidden + offset, valid, 0)
    hidden = original.to(tl.float32)
    key = tl.load(
        Projected + token * (HC + 1) * DIM + head * DIM + col, valid & enabled, 0
    ).to(tl.float32)
    q = tl.load(Q + head * DIM + col, valid, 0).to(tl.float32)
    k = tl.load(K + head * DIM + col, valid, 0).to(tl.float32)
    rstd = tl.rsqrt(tl.sum(hidden * hidden, 0) / DIM + EPS)
    rstd *= tl.rsqrt(tl.sum(key * key, 0) / DIM + EPS)
    dot = tl.sum(hidden * (q * k) * key, 0) * rstd * (DIM**-0.5)
    magnitude = tl.sqrt(tl.maximum(tl.abs(dot), 1.0e-6))
    # Bitwise copysign preserves negative zero in both hardware and interpreter.
    bits = magnitude.to(tl.int32, bitcast=True) | (
        dot.to(tl.int32, bitcast=True) & -2147483648
    )
    gate = tl.sigmoid(bits.to(tl.float32, bitcast=True))
    value = tl.load(
        Projected + token * (HC + 1) * DIM + HC * DIM + col, valid & enabled, 0
    ).to(tl.float32)
    result = tl.where(enabled, (hidden + gate * value).to(original.dtype), original)
    tl.store(Output + offset, result, valid)
