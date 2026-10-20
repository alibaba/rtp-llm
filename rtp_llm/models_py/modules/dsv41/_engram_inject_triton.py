# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fused V4.1 Engram gate and residual update.

Adapted from vLLM common/engram.py's _fused_engram_post_wkv_kernel,
source SHA256 41c5bdf25cf8337088247be768e3d43fa41e6358397800f8b602ebf1ede69480.
RTP uses contiguous local tokens and preserves copysign at zero.
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
    HAS_MASK: tl.constexpr,
    EPS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    token = tl.program_id(0).to(tl.int64)
    head = tl.program_id(1)
    col = tl.arange(0, BLOCK)
    live = col < DIM
    offset = (token * HC + head) * DIM + col
    # Each CTA owns one complete (token, head) row. It reads all hidden
    # values before storing that same row; exact Hidden/Output aliasing
    # is safe because no other CTA reads it. Other inputs must not alias.
    hidden = tl.load(Hidden + offset, live, 0).to(tl.float32)
    key = tl.load(Projected + token * (HC + 1) * DIM + head * DIM + col, live, 0).to(
        tl.float32
    )
    q = tl.load(Q + head * DIM + col, live, 0).to(tl.float32)
    k = tl.load(K + head * DIM + col, live, 0).to(tl.float32)
    rstd = tl.rsqrt(tl.sum(hidden * hidden, 0) / DIM + EPS)
    rstd *= tl.rsqrt(tl.sum(key * key, 0) / DIM + EPS)
    dot = tl.sum(hidden * q * k * key, 0) * rstd * (DIM**-0.5)
    gate_input = tl.sqrt(tl.maximum(tl.abs(dot), 1.0e-6))
    gate_input = tl.extra.cuda.libdevice.copysign(gate_input, dot)
    gate = tl.sigmoid(gate_input)
    if HAS_MASK:
        gate = tl.where(tl.load(Mask + token), gate, 0)
    value = tl.load(Projected + token * (HC + 1) * DIM + HC * DIM + col, live, 0).to(
        tl.float32
    )
    tl.store(Output + offset, hidden + gate * value, live)
