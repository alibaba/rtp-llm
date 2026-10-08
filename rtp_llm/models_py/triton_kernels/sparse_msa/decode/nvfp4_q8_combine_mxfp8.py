"""Fuse Q8KV4 partial reduction and MXFP8 output for the SM103 O-projection.

Preserve the original BF16 rounding and quantization arithmetic without
materializing the intermediate BF16 output. Callers gate the supported geometry.
"""

import torch
import triton
import triton.language as tl

from .nvfp4_q8_math import ex2_ftz, rcp_rn


@triton.jit
def _max_pair_to_f32(a, b, c, d):
    return tl.inline_asm_elementwise(
        """
    { .reg .b32 u,v,w,lo,hi; .reg .f32 f0,f1;
      max.bf16x2 u,$1,$2; max.bf16x2 v,$3,$4; max.bf16x2 w,u,v;
      and.b32 lo,w,0xFFFF; shr.b32 hi,w,16;
      shl.b32 lo,lo,16; shl.b32 hi,hi,16;
      mov.b32 f0,lo; mov.b32 f1,hi; max.f32 $0,f0,f1; }
    """,
        constraints="=f,r,r,r,r",
        args=[a, b, c, d],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _max_f32(a, b):
    return tl.inline_asm_elementwise(
        "max.f32 $0,$1,$2;",
        constraints="=f,f,f",
        args=[a, b],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _mul_f32(a, b):
    return tl.inline_asm_elementwise(
        "mul.rn.f32 $0,$1,$2;",
        constraints="=f,f,f",
        args=[a, b],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _scale_inv(n):
    return tl.inline_asm_elementwise(
        """
    { .reg .pred p_zero,p_mant,p_ovf,p_szero;
      .reg .u32 bits,e,m,bump,s,f; .reg .s32 ne;
      setp.le.f32 p_zero,$2,0f00000000; mov.b32 bits,$2;
      shr.b32 e,bits,23; and.b32 e,e,255; and.b32 m,bits,0x7FFFFF;
      setp.ne.u32 p_mant,m,0; selp.u32 bump,1,0,p_mant; add.u32 s,e,bump;
      setp.gt.u32 p_ovf,s,254; selp.u32 s,254,s,p_ovf;
      selp.u32 $0,0,s,p_zero;
      setp.eq.u32 p_szero,$0,0; sub.s32 ne,254,$0; max.s32 ne,ne,0;
      shl.b32 f,ne,23; mov.b32 $1,f; @p_szero mov.b32 $1,0; }
    """,
        constraints="=r,=f,f",
        args=[n],
        dtype=(tl.uint32, tl.float32),
        is_pure=True,
        pack=1,
    )


@triton.jit
def _fp8_byte(x):
    # Duplicate operands: low result byte equals original scalar conversion.
    return tl.inline_asm_elementwise(
        """
    { .reg .b16 p; .reg .u32 u;
      cvt.rn.satfinite.e4m3x2.f32 p,$1,$1; cvt.u32.u16 u,p; and.b32 $0,u,255; }
    """,
        constraints="=r,f",
        args=[x],
        dtype=tl.uint32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _combine_mxfp8(
    partial_ptr,
    lse_ptr,
    counts_ptr,
    fp8_ptr,
    packed_ptr,
    ALIGNED_M: tl.constexpr,
    LOG2E: tl.constexpr,
    valid_token_mask=None,
    HAS_VALID_TOKEN_MASK: tl.constexpr = False,
    valid_token_mask_stride=1,
):
    batch = tl.program_id(0)
    kv_head = tl.program_id(1)
    group = tl.arange(0, 16)
    dim = tl.arange(0, 128)
    slot = tl.arange(0, 16)
    row = batch * 4 + kv_head
    count = tl.load(counts_ptr + row)
    valid = slot < count
    lse_base = (row * 16 + slot)[:, None] * 16 + group[None, :]
    lse = tl.load(lse_ptr + lse_base, mask=valid[:, None], other=float("-inf"))
    finite = lse != float("-inf")
    has_finite = tl.sum(finite.to(tl.int32), axis=0) > 0
    maximum = tl.max(lse, axis=0)
    maximum_safe = tl.where(maximum == float("-inf"), 0.0, maximum)
    weight = ex2_ftz(tl.fma(lse, LOG2E, -(maximum_safe * LOG2E)[None, :]))
    weight_lanes = tl.reshape(weight, [4, 4, 16])
    lane_index = tl.arange(0, 4)[:, None, None]
    lane_sum = tl.zeros([4, 16], dtype=tl.float32)
    for i in tl.static_range(4):
        lane_sum += tl.sum(tl.where(lane_index == i, weight_lanes, 0.0), axis=0)
    lanes = tl.arange(0, 4)[:, None]
    s0 = tl.sum(tl.where(lanes == 0, lane_sum, 0.0), axis=0)
    s1 = tl.sum(tl.where(lanes == 1, lane_sum, 0.0), axis=0)
    s2 = tl.sum(tl.where(lanes == 2, lane_sum, 0.0), axis=0)
    s3 = tl.sum(tl.where(lanes == 3, lane_sum, 0.0), axis=0)
    denominator = (s0 + s2) + (s1 + s3)
    good = has_finite & (denominator != 0.0) & (denominator == denominator)
    inverse = tl.where(good, rcp_rn(tl.where(good, denominator, 1.0)), 0.0)
    weight *= inverse[None, :]
    accumulator = tl.zeros([16, 128], dtype=tl.float32)
    for i in tl.static_range(16):
        selected_weight = tl.sum(tl.where(slot[:, None] == i, weight, 0.0), axis=0)
        partial = tl.load(
            partial_ptr + ((row * 16 + i) * 16 + group)[:, None] * 128 + dim[None, :],
            mask=(i < count) & (group[:, None] >= 0),
            other=0.0,
        ).to(tl.float32)
        accumulator = tl.where(
            selected_weight[:, None] > 0.0,
            tl.fma(selected_weight[:, None], partial, accumulator),
            accumulator,
        )
    if HAS_VALID_TOKEN_MASK:
        # Clear padding even when the partial reduction produced NaN. Multiplying
        # by zero would retain NaN and would not match torch.where semantics.
        accumulator = tl.where(
            tl.load(valid_token_mask + batch * valid_token_mask_stride),
            accumulator,
            0.0,
        )
    # Rounded BF16 is exactly the intermediate consumed by FlashInfer.
    rounded = accumulator.to(tl.bfloat16)
    pair_bits = tl.reshape(
        rounded.to(tl.uint16, bitcast=True).to(tl.uint32), [16, 4, 4, 4, 2]
    )
    pair_lane = tl.arange(0, 2)[None, None, None, None, :]
    pairs = tl.sum(pair_bits << (pair_lane * 16), axis=4)
    pairs = pairs & 0x7FFF7FFF
    pair_index = tl.arange(0, 4)[None, None, None, :]
    p0 = tl.sum(tl.where(pair_index == 0, pairs, 0), axis=3)
    p1 = tl.sum(tl.where(pair_index == 1, pairs, 0), axis=3)
    p2 = tl.sum(tl.where(pair_index == 2, pairs, 0), axis=3)
    p3 = tl.sum(tl.where(pair_index == 3, pairs, 0), axis=3)
    local = _max_pair_to_f32(p0, p1, p2, p3)
    thread_index = tl.arange(0, 4)[None, None, :]
    # Avoid floating tl.sum for extracting NaNs/signed0: use gather instead.
    t0 = tl.sum(
        tl.where(thread_index == 0, local.to(tl.uint32, bitcast=True), 0), axis=2
    ).to(tl.float32, bitcast=True)
    t1 = tl.sum(
        tl.where(thread_index == 1, local.to(tl.uint32, bitcast=True), 0), axis=2
    ).to(tl.float32, bitcast=True)
    t2 = tl.sum(
        tl.where(thread_index == 2, local.to(tl.uint32, bitcast=True), 0), axis=2
    ).to(tl.float32, bitcast=True)
    t3 = tl.sum(
        tl.where(thread_index == 3, local.to(tl.uint32, bitcast=True), 0), axis=2
    ).to(tl.float32, bitcast=True)
    maximum32 = _max_f32(_max_f32(t0, t1), _max_f32(t2, t3))
    normalized = _mul_f32(maximum32, 0.0022321429569274187)
    scales, inv = _scale_inv(normalized)
    expanded = tl.reshape(rounded.to(tl.float32), [16, 4, 32])
    product = _mul_f32(expanded, inv[:, :, None])
    quant = _fp8_byte(product)
    feature = (kv_head * 16 + group)[:, None] * 128 + dim[None, :]
    tl.store(fp8_ptr + batch * 8192 + feature, tl.reshape(quant, [16, 128]))
    shift = tl.arange(0, 4)[None, :] * 8
    words = tl.sum(scales << shift, axis=1).to(tl.int32)
    tl.store(packed_ptr + batch + (kv_head * 16 + group) * ALIGNED_M, words)
