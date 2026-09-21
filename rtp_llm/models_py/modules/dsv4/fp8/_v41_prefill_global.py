# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compact V4.1 prefill compressor and planar FP4 cache epilogues.

The stage organization follows vLLM models/deepseek_v41/common/ops/
fused_compress_quant_cache.py and indexer_k_store.py. RTP retains compact
closed-pair rows (the index projection must not process twice as many rows),
complex64 GPT-J frequencies and BF16 rounding before either cache quantizer.
CP ownership is supplied as slot mappings, rather than using vLLM's ring.

The caller snapshots/all-reduces predecessor state before these stages. The
compressor only reads that immutable snapshot and the preceding tile's carry;
state stores and publication of the next carry happen afterwards on the same
stream. Neither communication nor carry lifetime is changed by this module.
"""

from __future__ import annotations

import math
import os

import torch
import triton
import triton.language as tl
import triton.language.extra.cuda.libdevice as libdevice

from rtp_llm.models_py.modules.dsv4.fp8._trap_utils import (
    invalid_kv_access_validation_enabled,
    trap_invalid_kv_access_enabled,
)
from rtp_llm.models_py.modules.dsv4.fp8._v41_fp4_triton import (
    _round_e2m1,
    _round_power_of_two_scale,
)


def _enabled(tensor):
    return (
        os.environ.get("DSV41_FUSED_PREFILL_GLOBAL", "1") != "0"
        and tensor.is_cuda
        and torch.version.hip is None
        and torch.cuda.get_device_capability(tensor.device)[0] == 10
        and not invalid_kv_access_validation_enabled()
    )


def _matrix(tensor, rows, cols, dtype, device):
    return (
        isinstance(tensor, torch.Tensor)
        and tensor.device == device
        and tensor.dtype == dtype
        and tensor.shape == (rows, cols)
        and tensor.stride(1) == 1
        and tensor.stride(0) >= cols
        and not (torch.is_grad_enabled() and tensor.requires_grad)
    )


def _vector(tensor, rows, device, dtypes=(torch.int32, torch.int64)):
    return (
        isinstance(tensor, torch.Tensor)
        and tensor.device == device
        and tensor.dtype in dtypes
        and tensor.shape == (rows,)
        and tensor.is_contiguous()
        and not (torch.is_grad_enabled() and tensor.requires_grad)
    )


def _pool(pool, device, width):
    return (
        isinstance(pool, torch.Tensor)
        and pool.device == device
        and pool.dtype == torch.uint8
        and pool.ndim == 3
        and pool.shape[0] > 0
        and pool.shape[1] > 0
        and pool.shape[2] == width
        and pool.stride(2) == 1
        and pool.stride(1) == width
        and pool.stride(0) >= pool.shape[1] * width
    )


def _frequencies(freqs, device):
    return (
        isinstance(freqs, torch.Tensor)
        and freqs.device == device
        and freqs.dtype == torch.complex64
        and freqs.ndim == 2
        and freqs.shape[1] == 32
        and freqs.is_contiguous()
    )


@triton.jit
def _trap():
    tl.inline_asm_elementwise(
        "trap; // dummy $0", "=r", [], dtype=tl.int32, is_pure=False, pack=1
    )


@triton.jit
def _rope_bf16(x, frequencies, position, D: tl.constexpr):
    # Match eager CUDA complex multiplication: imag=fma(imag,cos,real*sin).
    # Keep scalar multiply rounding and signed-zero negation explicitly.
    columns = tl.arange(0, D)
    values = x.to(tl.bfloat16).to(tl.float32)
    partner = tl.gather(values, columns ^ 1, axis=0)
    rotary = columns >= D - 64
    angle = tl.maximum((columns - (D - 64)) // 2, 0)
    cosine = tl.load(frequencies + position * 64 + angle * 2, rotary, other=1)
    sine = tl.load(frequencies + position * 64 + angle * 2 + 1, rotary, other=0)
    product = _mul_rn(partner, sine)
    signed_product = (
        product.to(tl.uint32, bitcast=True)
        ^ tl.where(columns % 2 == 0, 0x80000000, 0).to(tl.uint32)
    ).to(tl.float32, bitcast=True)
    rotated = tl.fma(values, cosine, signed_product)
    return tl.where(rotary, rotated, values).to(tl.bfloat16)


@triton.jit
def _mul_rn(a, b):
    # Prevent ptxas from contracting adjacent packed f32x2 mul/add into an
    # FMA even when Triton's enable_fp_fusion is false (SM103).
    return tl.inline_asm_elementwise(
        "mul.rn.f32 $0, $1, $2;",
        "=f,f,f",
        [a, b],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _add_rn(a, b):
    return tl.inline_asm_elementwise(
        "add.rn.f32 $0, $1, $2;",
        "=f,f,f",
        [a, b],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _torch_mean_square(x, D: tl.constexpr, WIDTH: tl.constexpr):
    # ATen Reduce.cuh vectorized inner reduction: four separate accumulators,
    # each lane visits vector4 rows separated by WIDTH, then combines left to
    # right before descending block/warp reduction. WIDTH depends on M.
    lanes = tl.arange(0, WIDTH)
    a0 = tl.full((WIDTH,), 0, tl.float32)
    a1 = tl.full((WIDTH,), 0, tl.float32)
    a2 = tl.full((WIDTH,), 0, tl.float32)
    a3 = tl.full((WIDTH,), 0, tl.float32)
    for r in tl.static_range(D // (WIDTH * 4)):
        offset = lanes * 4 + r * WIDTH * 4
        v0 = tl.gather(x, offset, axis=0)
        v1 = tl.gather(x, offset + 1, axis=0)
        v2 = tl.gather(x, offset + 2, axis=0)
        v3 = tl.gather(x, offset + 3, axis=0)
        a0 = _add_rn(a0, _mul_rn(v0, v0))
        a1 = _add_rn(a1, _mul_rn(v1, v1))
        a2 = _add_rn(a2, _mul_rn(v2, v2))
        a3 = _add_rn(a3, _mul_rn(v3, v3))
    partial = _add_rn(_add_rn(_add_rn(a0, a1), a2), a3)
    for shift in tl.static_range(0, tl.constexpr(WIDTH.bit_length() - 1)):
        reduce_offset = WIDTH >> (shift + 1)
        partial = _add_rn(partial, tl.gather(partial, lanes ^ reduce_offset, axis=0))
    total = tl.sum(tl.where(lanes == 0, partial, 0), 0)
    return total / D


@triton.jit(do_not_specialize=["POOL_STRIDE", "BLOCKS"])
def _prefill_compress_main_kernel(
    values,
    scores,
    weight,
    positions,
    requests,
    starts,
    previous,
    boundary_indices,
    carry_values,
    carry_scores,
    frequencies,
    pool,
    slots,
    output,
    VALUE_STRIDE: tl.constexpr,
    SCORE_STRIDE: tl.constexpr,
    PREVIOUS_STRIDE: tl.constexpr,
    HAS_CARRY: tl.constexpr,
    RATIO: tl.constexpr,
    EPS: tl.constexpr,
    ENTRIES: tl.constexpr,
    POOL_STRIDE,
    BLOCKS,
    TRAP: tl.constexpr,
    REDUCE_WIDTH: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    token = tl.load(boundary_indices + row).to(tl.int64)
    columns = tl.arange(0, 512)
    current = tl.load(values + token * VALUE_STRIDE + columns)
    position = tl.load(positions + token).to(tl.int64)
    if RATIO == 2:
        request = tl.load(requests + token).to(tl.int64)
        current_score = tl.load(scores + token * SCORE_STRIDE + columns)
        if position == tl.load(starts + request):
            prev = tl.load(previous + request * PREVIOUS_STRIDE + columns)
            prev_score = tl.load(previous + request * PREVIOUS_STRIDE + 512 + columns)
        elif token == 0:
            if HAS_CARRY:
                prev = tl.load(carry_values + columns)
                prev_score = tl.load(carry_scores + columns)
            else:
                # Identical to the original first-tile head fallback. Real
                # request starts above always use the immutable snapshot.
                prev, prev_score = current, current_score
        else:
            prev = tl.load(values + (token - 1) * VALUE_STRIDE + columns)
            prev_score = tl.load(scores + (token - 1) * SCORE_STRIDE + columns)
        maximum = tl.maximum(prev_score, current_score)
        e0 = libdevice.exp(prev_score - maximum)
        e1 = libdevice.exp(current_score - maximum)
        denom = e0 + e1
        pooled = _add_rn(
            _mul_rn(prev, tl.div_rn(e0, denom)), _mul_rn(current, tl.div_rn(e1, denom))
        )
    else:
        pooled = current
    inv = tl.rsqrt(_add_rn(_torch_mean_square(pooled, 512, REDUCE_WIDTH), EPS))
    latent = (pooled * inv * tl.load(weight + columns).to(tl.float32)).to(tl.bfloat16)
    tl.store(output + row * 512 + columns, latent)
    slot = tl.load(slots + row).to(tl.int64)
    if slot >= 0:
        if TRAP:
            if slot >= BLOCKS * ENTRIES:
                _trap()
        rotated = _rope_bf16(latent, frequencies, position // RATIO * RATIO, 512)
        grouped = rotated.to(tl.float32).reshape((32, 16))
        maxima = tl.maximum(tl.max(tl.abs(grouped), 1), 6.0 * (2.0**-9))
        sf = tl.div_rn(maxima, 6.0).to(tl.float8e4nv, fp_downcast_rounding="rtne")
        normalized = tl.minimum(
            tl.maximum(tl.div_rn(grouped, sf.to(tl.float32)[:, None]), -6.0), 6.0
        )
        codes = _round_e2m1(normalized).reshape((256, 2))
        even, odd = tl.split(codes)
        base = pool + (slot // ENTRIES) * POOL_STRIDE
        offset = slot % ENTRIES
        tl.store(base + offset * 256 + tl.arange(0, 256), even | (odd << 4))
        tl.store(
            base + ENTRIES * 256 + offset * 32 + tl.arange(0, 32),
            sf.to(tl.uint8, bitcast=True),
        )


@triton.jit(do_not_specialize=["STATE_ROWS"])
def _prefill_state_store_kernel(
    values,
    scores,
    slots,
    state,
    VALUE_STRIDE: tl.constexpr,
    SCORE_STRIDE: tl.constexpr,
    STATE_STRIDE: tl.constexpr,
    STATE_ROWS,
    TRAP: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    slot = tl.load(slots + row).to(tl.int64)
    if slot >= 0:
        if TRAP:
            if slot >= STATE_ROWS:
                _trap()
        d = tl.arange(0, 512)
        tl.store(
            state + slot * STATE_STRIDE + d, tl.load(values + row * VALUE_STRIDE + d)
        )
        tl.store(
            state + slot * STATE_STRIDE + 512 + d,
            tl.load(scores + row * SCORE_STRIDE + d),
        )


@triton.jit(do_not_specialize=["POOL_STRIDE", "BLOCKS"])
def _prefill_index_store_kernel(
    projected,
    weight,
    positions,
    frequencies,
    pool,
    slots,
    INPUT_STRIDE: tl.constexpr,
    RATIO: tl.constexpr,
    EPS: tl.constexpr,
    ENTRIES: tl.constexpr,
    POOL_STRIDE,
    BLOCKS,
    NATIVE_NORM: tl.constexpr,
    TRAP: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    slot = tl.load(slots + row).to(tl.int64)
    if slot >= 0:
        if TRAP:
            if slot >= BLOCKS * ENTRIES:
                _trap()
        columns = tl.arange(0, 128)
        key = tl.load(projected + row * INPUT_STRIDE + columns).to(tl.float32)
        if NATIVE_NORM:
            # FlashInfer RMSNormKernel<8,bfloat16>: sequential 8-element
            # FMA sum per thread, followed by warp XOR reduction.
            vec = key.reshape((16, 8))
            partial = tl.full((16,), 0, tl.float32)
            for j in tl.static_range(8):
                value = tl.gather(vec, tl.full((16, 1), j, tl.int32), axis=1).reshape(
                    (16,)
                )
                partial = tl.fma(value, value, partial)
            inv = tl.rsqrt(tl.fma(tl.sum(partial, 0), 1.0 / 128, EPS))
        else:
            inv = tl.rsqrt(_add_rn(_torch_mean_square(key, 128, 32), EPS))
        normalized = (key * inv * tl.load(weight + columns).to(tl.float32)).to(
            tl.bfloat16
        )
        position = tl.load(positions + row).to(tl.int64)
        rotated = _rope_bf16(normalized, frequencies, position // RATIO * RATIO, 128)
        grouped = rotated.to(tl.float32).reshape((4, 32))
        maxima = tl.maximum(tl.max(tl.abs(grouped), 1), 6.0 * (2.0**-126))
        sf, exponent = _round_power_of_two_scale(maxima, 1.0 / 6.0)
        normalized = tl.minimum(tl.maximum(tl.div_rn(grouped, sf[:, None]), -6.0), 6.0)
        codes = _round_e2m1(normalized).reshape((64, 2))
        even, odd = tl.split(codes)
        base = pool + (slot // ENTRIES) * POOL_STRIDE
        offset = slot % ENTRIES
        tl.store(base + offset * 64 + tl.arange(0, 64), even | (odd << 4))
        tl.store(base + ENTRIES * 64 + offset * 4 + tl.arange(0, 4), exponent)


def compress_main(
    values,
    scores,
    norm,
    eps,
    positions,
    req_ids,
    starts,
    previous,
    boundary_indices,
    freqs,
    main_pool,
    main_slots,
    ratio,
    carry=None,
):
    """Return compact pre-RoPE BF16 latent and store the owned main-cache rows.

    ``boundary_indices`` indexes the ready tile, not the full request. Its
    length C is known from CPU request lengths; no CUDA nonzero is required.
    ``previous`` is an immutable [requests,1024] FP32 snapshot. ``carry`` is
    the preceding tile's independent pair of [1,512] FP32 tensors.
    All slots are prevalidated CP mappings (-1 means no owned cache row).
    Unsupported layouts return None before mutating a pool; kernel errors
    propagate. The caller retains the original path for warmup without pools.
    """
    if not _enabled(values) or values.ndim != 2:
        return None
    rows, device = values.shape[0], values.device
    count = boundary_indices.numel()
    if not (
        ratio in (1, 2)
        and 0 <= count <= rows
        and math.isfinite(eps)
        and eps > 0
        and _matrix(values, rows, 512, torch.float32, device)
        and _vector(norm, 512, device, (torch.float32, torch.bfloat16))
        and _vector(positions, rows, device)
        and _vector(boundary_indices, count, device)
        and _vector(main_slots, count, device)
        and _frequencies(freqs, device)
        and _pool(main_pool, device, 288)
    ):
        return None
    if ratio == 2 and not (
        _matrix(scores, rows, 512, torch.float32, device)
        and _vector(req_ids, rows, device)
        and isinstance(starts, torch.Tensor)
        and _vector(starts, starts.numel(), device)
        and _matrix(previous, starts.numel(), 1024, torch.float32, device)
        and (
            carry is None
            or (
                len(carry) == 2
                and all(_matrix(t, 1, 512, torch.float32, device) for t in carry)
            )
        )
    ):
        return None
    output = torch.empty((count, 512), device=device, dtype=torch.bfloat16)
    if count == 0:
        return output
    _prefill_compress_main_kernel[(count,)](
        values,
        scores,
        norm,
        positions,
        req_ids,
        starts,
        previous,
        boundary_indices,
        *(carry if carry is not None else (None, None)),
        freqs.view(torch.float32),
        main_pool,
        main_slots,
        output,
        values.stride(0),
        scores.stride(0) if ratio == 2 else 0,
        previous.stride(0) if ratio == 2 else 0,
        carry is not None,
        ratio,
        eps,
        main_pool.shape[1],
        main_pool.stride(0),
        main_pool.shape[0],
        trap_invalid_kv_access_enabled(),
        min(128, 512 // min(1 << (count.bit_length() - 1), 16)),
        num_warps=4,
        enable_fp_fusion=False,
    )
    return output


def store_states(values, scores, state_slots, state_pool):
    """Store only mapped FP32 rows; caller supplies non-aliasing suffix slots."""
    if not _enabled(values) or values.ndim != 2:
        return False
    rows, device = values.shape[0], values.device
    if not (
        _matrix(values, rows, 512, torch.float32, device)
        and _matrix(scores, rows, 512, torch.float32, device)
        and _vector(state_slots, rows, device)
        and isinstance(state_pool, torch.Tensor)
        and state_pool.ndim == 2
        and _matrix(state_pool, state_pool.shape[0], 1024, torch.float32, device)
    ):
        return False
    if rows:
        _prefill_state_store_kernel[(rows,)](
            values,
            scores,
            state_slots,
            state_pool,
            values.stride(0),
            scores.stride(0),
            state_pool.stride(0),
            state_pool.shape[0],
            trap_invalid_kv_access_enabled(),
            num_warps=4,
        )
    return True


def store_index(projected, norm, eps, positions, freqs, index_pool, index_slots, ratio):
    """Normalize compact BF16 index projection and store its planar MXFP4 rows."""
    if not _enabled(projected) or projected.ndim != 2:
        return False
    rows, device = projected.shape[0], projected.device
    if not (
        ratio in (1, 2)
        and math.isfinite(eps)
        and eps > 0
        and _matrix(projected, rows, 128, torch.bfloat16, device)
        and _vector(norm, 128, device, (torch.float32, torch.bfloat16))
        and _vector(positions, rows, device)
        and _vector(index_slots, rows, device)
        and _frequencies(freqs, device)
        and _pool(index_pool, device, 68)
    ):
        return False
    if rows:
        _prefill_index_store_kernel[(rows,)](
            projected,
            norm,
            positions,
            freqs.view(torch.float32),
            index_pool,
            index_slots,
            projected.stride(0),
            ratio,
            eps,
            index_pool.shape[1],
            index_pool.stride(0),
            index_pool.shape[0],
            norm.dtype == torch.bfloat16 and projected.is_contiguous(),
            trap_invalid_kv_access_enabled(),
            num_warps=4,
            enable_fp_fusion=False,
        )
    return True
