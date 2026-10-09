"""Experimental BF16 router probabilities retaining ATen's 128-column topology."""

import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice
from triton.experimental import gluon
from triton.experimental.gluon import language as gl


@gluon.jit
def _warp_xor(value, offset):
    bits = gl.inline_asm_elementwise(
        "shfl.sync.bfly.b32 $0, $1, $2, 31, -1;",
        constraints="=r,r,r",
        args=[value.to(gl.int32, bitcast=True), offset],
        dtype=gl.int32, is_pure=True, pack=1,
    )
    return bits.to(gl.float32, bitcast=True)


@gluon.jit
def _warp_softmax_row(X, Y, row, column, COUNT):
    offset = row.to(gl.int64) * 128 + column
    valid = row < COUNT
    x0 = gl.load(X + offset, valid, other=-float("inf")).to(gl.float32)
    x1 = gl.load(X + offset + 32, valid, other=-float("inf")).to(gl.float32)
    x2 = gl.load(X + offset + 64, valid, other=-float("inf")).to(gl.float32)
    x3 = gl.load(X + offset + 96, valid, other=-float("inf")).to(gl.float32)
    maximum = gl.where(x0 > x1, x0, x1)
    maximum = gl.where(maximum > x2, maximum, x2)
    maximum = gl.where(maximum > x3, maximum, x3)
    for bit in gl.static_range(4, -1, -1):
        other = _warp_xor(maximum, 1 << bit)
        maximum = gl.where(maximum < other, other, maximum)
    e0 = libdevice.exp(x0 - maximum)
    e1 = libdevice.exp(x1 - maximum)
    e2 = libdevice.exp(x2 - maximum)
    e3 = libdevice.exp(x3 - maximum)
    total = e0 + e1
    total = total + e2
    total = total + e3
    for bit in gl.static_range(4, -1, -1):
        total = total + _warp_xor(total, 1 << bit)
    gl.store(Y + offset, gl.div_rn(e0, total).to(gl.bfloat16), valid)
    gl.store(Y + offset + 32, gl.div_rn(e1, total).to(gl.bfloat16), valid)
    gl.store(Y + offset + 64, gl.div_rn(e2, total).to(gl.bfloat16), valid)
    gl.store(Y + offset + 96, gl.div_rn(e3, total).to(gl.bfloat16), valid)


@gluon.jit
def _router_softmax_warp_kernel(
    X, Y, COUNT, WARPS: gl.constexpr, ROWS_PER_WARP: gl.constexpr,
):
    thread = gl.arange(
        0, WARPS * 32,
        layout=gl.BlockedLayout([1], [32], [WARPS], [0]),
    )
    row = gl.program_id(0).to(gl.int64) * WARPS * ROWS_PER_WARP
    row = row + (thread // 32) * ROWS_PER_WARP
    column = thread % 32
    _warp_softmax_row(X, Y, row, column, COUNT)
    if ROWS_PER_WARP == 2:
        _warp_softmax_row(X, Y, row + 1, column, COUNT)


@triton.jit
def _router_softmax_kernel(X, Y, COUNT, ROWS: tl.constexpr):
    row = tl.program_id(0).to(tl.int64) * ROWS + tl.arange(0, ROWS)
    lane = tl.arange(0, 32)
    offset = row[:, None] * 128 + lane[None, :]
    valid = row[:, None] < COUNT
    x0 = tl.load(X + offset, valid, other=-float("inf")).to(tl.float32)
    x1 = tl.load(X + offset + 32, valid, other=-float("inf")).to(tl.float32)
    x2 = tl.load(X + offset + 64, valid, other=-float("inf")).to(tl.float32)
    x3 = tl.load(X + offset + 96, valid, other=-float("inf")).to(tl.float32)
    maximum = tl.where(x0 > x1, x0, x1)
    maximum = tl.where(maximum > x2, maximum, x2)
    maximum = tl.where(maximum > x3, maximum, x3)
    for bit in tl.static_range(4, -1, -1):
        peer = tl.broadcast_to((lane ^ (1 << bit))[None, :], (ROWS, 32))
        other = tl.gather(maximum, peer, axis=1)
        maximum = tl.where(maximum < other, other, maximum)
    e0 = libdevice.exp(x0 - maximum)
    e1 = libdevice.exp(x1 - maximum)
    e2 = libdevice.exp(x2 - maximum)
    e3 = libdevice.exp(x3 - maximum)
    total = e0 + e1
    total = total + e2
    total = total + e3
    for bit in tl.static_range(4, -1, -1):
        peer = tl.broadcast_to((lane ^ (1 << bit))[None, :], (ROWS, 32))
        total = total + tl.gather(total, peer, axis=1)
    tl.store(Y + offset, tl.div_rn(e0, total).to(tl.bfloat16), valid)
    tl.store(Y + offset + 32, tl.div_rn(e1, total).to(tl.bfloat16), valid)
    tl.store(Y + offset + 64, tl.div_rn(e2, total).to(tl.bfloat16), valid)
    tl.store(Y + offset + 96, tl.div_rn(e3, total).to(tl.bfloat16), valid)


def router_probabilities(scores, rows_per_block=8, variant="initial", return_kernel=False):
    if (
        not scores.is_cuda
        or scores.dtype != torch.bfloat16
        or scores.dim() != 2
        or scores.shape[1] != 128
        or not 1 <= scores.shape[0] <= 131072
        or not scores.is_contiguous()
        or rows_per_block not in (4, 8, 16, 32)
        or variant not in ("initial", "warp1", "warp4", "adaptive-warp")
    ):
        return None
    output = torch.empty_like(scores)
    if variant == "initial":
        kernel = _router_softmax_kernel[(triton.cdiv(scores.shape[0], rows_per_block),)](
            scores, output, scores.shape[0], rows_per_block,
            num_warps=4, enable_fp_fusion=False,
        )
    else:
        warps, pair = (1, 1) if variant == "warp1" or (variant == "adaptive-warp" and scores.shape[0] <= 256) else (4, 2)
        kernel = _router_softmax_warp_kernel[(triton.cdiv(scores.shape[0], warps * pair),)](
            scores, output, scores.shape[0], warps, pair,
            num_warps=warps, enable_fp_fusion=False,
        )
    return (output, kernel) if return_kernel else output
