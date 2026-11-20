"""V4.1 decode Q RoPE, KV RMSNorm/RoPE, and native 528-byte SWA write."""

from __future__ import annotations

import os
from typing import NamedTuple

import torch
import triton
import triton.language as tl

from rtp_llm.models_py.modules.dsv41.fp8._trap_utils import (
    trap_invalid_kv_access_enabled,
)
from rtp_llm.models_py.modules.dsv41.fp8._v41_sm90_v4_cache import _store_v4_row
from rtp_llm.models_py.modules.dsv41.fp8._v41_sm90_v4_cache import (
    is_supported as v4_swa_is_supported,
)
from rtp_llm.models_py.modules.dsv41.fp8._v41_swa_kv import (
    quantize_swa_row,
)
from rtp_llm.models_py.modules.dsv41.fp8._v41_swa_triton import (
    is_supported as swa_is_supported,
)


class FusedQKVRopeCache(NamedTuple):
    q: torch.Tensor
    kv: torch.Tensor
    freqs_cis: torch.Tensor



@triton.jit
def norm_rope_bf16(
    values, weight, cos, sin, EPS: tl.constexpr, ROUND_NORM: tl.constexpr = True
):
    """Normalize and rotate one row with the selected cache's BF16 boundaries."""
    values = values.to(tl.float32)
    inv = tl.rsqrt(tl.sum(values * values, 0) / 512 + EPS)
    normalized = values * inv * weight.to(tl.float32)
    if ROUND_NORM:
        normalized = normalized.to(tl.bfloat16)
    pair = tl.arange(0, 32)
    real = tl.gather(normalized, 448 + 2 * pair, 0).to(tl.float32)
    imag = tl.gather(normalized, 449 + 2 * pair, 0).to(tl.float32)
    # DI uses an explicit IEEE FMA after materializing the norm to BF16.
    real_out = tl.fma(real, cos, -(imag * sin))
    imag_out = tl.fma(real, sin, imag * cos)
    tail = tl.interleave(real_out, imag_out).to(tl.bfloat16)
    columns = tl.arange(0, 512)
    return tl.where(
        columns < 448,
        normalized.to(tl.bfloat16),
        tl.gather(tail, tl.maximum(columns - 448, 0), 0),
    )


@triton.jit
def _trap():
    tl.inline_asm_elementwise(
        "trap; // dummy $0", "=r", [], dtype=tl.int32, is_pure=False, pack=1
    )


@triton.jit(do_not_specialize=["kv_stride", "block_stride", "num_pages", "num_freqs"])
def _fused_qkv_rope_cache_kernel(
    q,
    q_out,
    kv,
    weight,
    positions,
    freqs,
    pool,
    slots,
    kv_out,
    freqs_out,
    kv_stride,
    block_stride,
    num_pages,
    num_freqs,
    HEADS: tl.constexpr,
    ENTRIES: tl.constexpr,
    EPS: tl.constexpr,
    GATHERED_FREQS: tl.constexpr,
    APPLY_Q_ROPE: tl.constexpr,
    PADDED_HEADS: tl.constexpr,
    TRAP_INVALID_KV_ACCESS: tl.constexpr,
    USE_PDL: tl.constexpr = False,
    KV_READY: tl.constexpr = False,
    V4_CACHE: tl.constexpr = False,
):
    tl.static_assert(not KV_READY or (not APPLY_Q_ROPE and HEADS == PADDED_HEADS))
    if USE_PDL and not KV_READY:
        tl.extra.cuda.gdc_wait()
    row = tl.program_id(0).to(tl.int64)
    tile = tl.program_id(1).to(tl.int64)
    pair = tl.arange(0, 32)
    if GATHERED_FREQS:
        frequency_row = row
    else:
        frequency_row = tl.load(positions + row).to(tl.int64)
        if (frequency_row < 0) | (frequency_row >= num_freqs):
            _trap()
            return
    cos = tl.load(freqs + frequency_row * 64 + 2 * pair)
    sin = tl.load(freqs + frequency_row * 64 + 2 * pair + 1)

    if not APPLY_Q_ROPE and HEADS != PADDED_HEADS and tile < PADDED_HEADS // 2:
        # Copy Q's 16-channel/head-interleaved layout into the native padded
        # layout. This shares the KV insert launch, without expanding Wq GEMM.
        offset = tile * 1024 + tl.arange(0, 1024)
        head = offset // 16 % PADDED_HEADS
        chunk = offset // (16 * PADDED_HEADS)
        source = (chunk * HEADS + head) * 16 + offset % 16
        q_values = tl.load(q + row * HEADS * 512 + source, head < HEADS, other=0)
        tl.store(q_out + row * PADDED_HEADS * 512 + offset, q_values)
    elif APPLY_Q_ROPE and tile < tl.cdiv(HEADS, 8):
        heads = tile * 8 + tl.arange(0, 8)
        q_base = q + (row * HEADS + heads[:, None]) * 512 + 448
        q_real = tl.load(
            q_base + 2 * pair[None, :], heads[:, None] < HEADS, other=0
        ).to(tl.float32)
        q_imag = tl.load(
            q_base + 2 * pair[None, :] + 1, heads[:, None] < HEADS, other=0
        ).to(tl.float32)
        # Pin the original RoPE kernel's FMA operand order at BF16 midpoints.
        q_new_real = tl.fma(q_real, cos[None, :], -(q_imag * sin[None, :]))
        q_new_imag = tl.fma(q_imag, cos[None, :], q_real * sin[None, :])
        tl.store(q_base + 2 * pair[None, :], q_new_real, heads[:, None] < HEADS)
        tl.store(q_base + 2 * pair[None, :] + 1, q_new_imag, heads[:, None] < HEADS)
    else:
        columns = tl.arange(0, 512)
        kv_base = kv + row * kv_stride.to(tl.int64)
        values = tl.load(kv_base + columns).to(tl.float32)
        w = tl.load(weight + columns).to(tl.float32)
        # The V4 cache uses the FP32 RMSNorm-to-RoPE path. V4.1's native
        # cache materializes the norm in BF16 before rotation and quantization.
        rounded = norm_rope_bf16(values, w, cos, sin, EPS, ROUND_NORM=not V4_CACHE)
        if V4_CACHE:
            decoded = rounded
        else:
            payload, encoded, decoded = quantize_swa_row(rounded)
        if USE_PDL and KV_READY:
            # QA supplied raw KV before QB started; keep stores after QB's wait.
            tl.extra.cuda.gdc_wait()
        tl.store(kv_out + row * 512 + columns, decoded)
        if not GATHERED_FREQS:
            tl.store(freqs_out + row * 64 + 2 * pair, cos)
            tl.store(freqs_out + row * 64 + 2 * pair + 1, sin)

        slot = tl.load(slots + row).to(tl.int64)
        if slot < 0:
            return
        page = slot // ENTRIES
        if page >= num_pages:
            if TRAP_INVALID_KV_ACCESS:
                _trap()
            return
        if V4_CACHE:
            _store_v4_row(rounded, pool, slot, block_stride, ENTRIES)
        else:
            position = slot % ENTRIES
            page_base = pool + page * block_stride.to(tl.int64)
            tl.store(
                page_base + position * 528 + columns,
                payload.to(tl.uint8, bitcast=True),
            )
            tl.store(page_base + position * 528 + 512 + tl.arange(0, 16), encoded)
