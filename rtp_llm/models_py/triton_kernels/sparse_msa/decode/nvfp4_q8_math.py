"""Shared arithmetic for MiniMax-M3.1 Q8KV4 decode kernels.

The operation order matches the training-compatible MiniMax sparse-attention
reference.  Cache addressing deliberately lives in the caller so RTP keeps
its own page-major ABI instead of inheriting SGLang's slot-major layout.
"""

import math

import numpy as np
import triton
import triton.language as tl

LOG2E_F32 = float(np.float32(math.log2(math.e)))
LN2_F32 = float(np.float32(math.log(2.0)))
POLY_EX2_C1 = 0.695146143436431884765625
POLY_EX2_C2 = 0.227564394474029541015625
POLY_EX2_C3 = 0.077119089663028717041015625
FP32_ROUND_INT = float(2**23 + 2**22)


def softmax_scale_log2(sm_scale: float) -> float:
    return float(np.float32(np.float32(sm_scale) * np.float32(LOG2E_F32)))


@triton.jit
def ex2_ftz(x):
    return tl.inline_asm_elementwise(
        "ex2.approx.ftz.f32 $0, $1;",
        "=f,f",
        [x],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def lg2_ftz(x):
    return tl.inline_asm_elementwise(
        "lg2.approx.ftz.f32 $0, $1;",
        "=f,f",
        [x],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def rcp_ftz(x):
    return tl.inline_asm_elementwise(
        "rcp.approx.ftz.f32 $0, $1;",
        "=f,f",
        [x],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def rcp_rn(x):
    return tl.inline_asm_elementwise(
        "rcp.rn.f32 $0, $1;",
        "=f,f",
        [x],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _add_rm_ftz(x, y):
    return tl.inline_asm_elementwise(
        "add.rm.ftz.f32 $0, $1, $2;",
        "=f,f,f",
        [x, y],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _fmax(x, y):
    return tl.inline_asm_elementwise(
        "max.f32 $0, $1, $2;",
        "=f,f,f",
        [x, y],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def ex2_emulated(
    x, C1: tl.constexpr, C2: tl.constexpr, C3: tl.constexpr, RINT: tl.constexpr
):
    xc = _fmax(x, -127.0)
    rounded = _add_rm_ftz(xc, RINT)
    rounded_base = rounded - RINT
    fraction = tl.fma(rounded_base, -1.0, xc)
    out = tl.fma(C3, fraction, C2)
    out = tl.fma(out, fraction, C1)
    out = tl.fma(out, fraction, 1.0)
    bits = (rounded.to(tl.int32, bitcast=True) << 23) + out.to(tl.int32, bitcast=True)
    return bits.to(tl.float32, bitcast=True)


_E2M1X4_ASM = (
    "{ .reg .b8 b0, b1, b2, b3, s0, s1, s2, s3; "
    ".reg .b16 sf, e0, e1; .reg .b32 sfx2, h0, h1;\n"
    "mov.b32 {s0, s1, s2, s3}, $2;\n"
    "mov.b16 sf, {s0, s0};\n"
    "cvt.rn.f16x2.e4m3x2 sfx2, sf;\n"
    "mov.b32 {b0, b1, b2, b3}, $1;\n"
    "cvt.rn.f16x2.e2m1x2 h0, b%d;\n"
    "cvt.rn.f16x2.e2m1x2 h1, b%d;\n"
    "mul.rn.f16x2 h0, h0, sfx2;\n"
    "mul.rn.f16x2 h1, h1, sfx2;\n"
    "cvt.rn.satfinite.e4m3x2.f16x2 e0, h0;\n"
    "cvt.rn.satfinite.e4m3x2.f16x2 e1, h1;\n"
    "mov.b32 $0, {e0, e1}; }"
)
E2M1X4_ASM_LO = tl.constexpr(_E2M1X4_ASM % (0, 1))
E2M1X4_ASM_HI = tl.constexpr(_E2M1X4_ASM % (2, 3))


@triton.jit
def packed4_to_fp8(packed4, scales4, ASM: tl.constexpr):
    return tl.inline_asm_elementwise(
        ASM,
        "=r,r,r",
        [packed4, scales4],
        dtype=tl.uint8,
        is_pure=True,
        pack=4,
    )


@triton.jit
def tree_sum_128(values, rows: tl.constexpr):
    values = tl.reshape(values, [rows, 16, 8])
    index = tl.arange(0, 16)[None, :, None]
    lanes = tl.zeros([rows, 8], dtype=tl.float32)
    for i in tl.static_range(16):
        lanes += tl.sum(tl.where(index == i, values, 0.0), axis=1)
    lane = tl.arange(0, 8)[None, :]
    s0 = tl.sum(tl.where(lane == 0, lanes, 0.0), axis=1)
    s1 = tl.sum(tl.where(lane == 1, lanes, 0.0), axis=1)
    s2 = tl.sum(tl.where(lane == 2, lanes, 0.0), axis=1)
    s3 = tl.sum(tl.where(lane == 3, lanes, 0.0), axis=1)
    s4 = tl.sum(tl.where(lane == 4, lanes, 0.0), axis=1)
    s5 = tl.sum(tl.where(lane == 5, lanes, 0.0), axis=1)
    s6 = tl.sum(tl.where(lane == 6, lanes, 0.0), axis=1)
    s7 = tl.sum(tl.where(lane == 7, lanes, 0.0), axis=1)
    return ((s0 + s2) + (s4 + s6)) + ((s1 + s3) + (s5 + s7))
