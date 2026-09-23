"""V4.1 decode Q RoPE, KV RMSNorm/RoPE, and native 528-byte SWA write."""

from __future__ import annotations

import os
from typing import NamedTuple

import torch
import triton
import triton.language as tl

from rtp_llm.models_py.modules.dsv4.fp8._trap_utils import (
    trap_invalid_kv_access_enabled,
)
from rtp_llm.models_py.modules.dsv4.fp8._v41_swa_triton import (
    is_supported as swa_is_supported,
)


class FusedQKVRopeCache(NamedTuple):
    q: torch.Tensor
    kv: torch.Tensor
    freqs_cis: torch.Tensor


@triton.jit
def _trap():
    tl.inline_asm_elementwise(
        "trap; // dummy $0", "=r", [], dtype=tl.int32, is_pure=False, pack=1
    )


@triton.jit(do_not_specialize=["kv_stride", "block_stride", "num_pages", "num_freqs"])
def _fused_qkv_rope_cache_kernel(
    q,
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
    TRAP_INVALID_KV_ACCESS: tl.constexpr,
):
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

    if tile < tl.cdiv(HEADS, 8):
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
        inv = tl.rsqrt(tl.sum(values * values, 0) / 512 + EPS)
        normalized = values * inv * w
        kv_real = tl.load(kv_base + 448 + 2 * pair).to(tl.float32) * inv
        kv_imag = tl.load(kv_base + 448 + 2 * pair + 1).to(tl.float32) * inv
        kv_real = kv_real * tl.load(weight + 448 + 2 * pair).to(tl.float32)
        kv_imag = kv_imag * tl.load(weight + 448 + 2 * pair + 1).to(tl.float32)
        kv_new_real = kv_real * cos - kv_imag * sin
        kv_new_imag = kv_real * sin + kv_imag * cos
        rope = tl.interleave(kv_new_real, kv_new_imag).to(tl.bfloat16)
        # Quantization consumes the BF16 materialization of the old norm/RoPE
        # producer, including its RoPE channels, rather than FP32 intermediates.
        rounded = tl.where(
            columns < 448,
            normalized.to(tl.bfloat16),
            tl.gather(rope, tl.maximum(columns - 448, 0), 0),
        )
        tl.store(kv_out + row * 512 + columns, rounded)
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
        position = slot % ENTRIES
        page_base = pool + page * block_stride.to(tl.int64)
        groups = rounded.to(tl.float32).reshape((16, 32))
        maximum = tl.maximum(tl.max(tl.abs(groups), 1), 1e-4)
        exponent = tl.ceil(tl.log2(maximum * (1.0 / 448.0)))
        scaled = tl.clamp(groups * tl.exp2(-exponent)[:, None], -448.0, 448.0)
        payload = scaled.to(tl.float8e4nv).to(tl.uint8, bitcast=True).reshape((512,))
        tl.store(page_base + position * 512 + columns, payload)
        encoded = tl.clamp(exponent + 127.0, 0.0, 255.0).to(tl.uint8)
        tl.store(page_base + ENTRIES * 512 + position * 16 + tl.arange(0, 16), encoded)


def is_supported(
    q, kv, kv_norm, positions, freqs_table, pool_3d, slots, *, freqs_cis=None
) -> bool:
    if os.environ.get("DSV41_FUSED_QKV_ROPE_CACHE", "1") != "1":
        return False
    tensors = (q, kv, kv_norm, positions, freqs_table, pool_3d, slots)
    if not all(isinstance(value, torch.Tensor) for value in tensors):
        return False
    if (
        not q.is_cuda
        or q.dtype != torch.bfloat16
        or q.ndim != 4
        or q.shape[-1] != 512
        or q.shape[-2] <= 0
        or not q.is_contiguous()
        or any(value.device != q.device for value in tensors[1:])
        or kv.dtype != torch.bfloat16
        or kv.shape != (*q.shape[:2], 512)
        or kv.stride(-1) != 1
        or kv.stride(-2) < 512
        or kv.stride(0) != kv.shape[1] * kv.stride(1)
        or kv_norm.dtype != torch.bfloat16
        or kv_norm.shape != (512,)
        or not kv_norm.is_contiguous()
        or positions.dtype not in (torch.int32, torch.int64)
        or positions.numel() != q.shape[0] * q.shape[1]
        or not positions.is_contiguous()
        or freqs_table.dtype != torch.complex64
        or freqs_table.ndim != 2
        or freqs_table.shape[1] != 32
        or not freqs_table.is_contiguous()
        or slots.numel() != positions.numel()
        or not slots.is_contiguous()
        or not swa_is_supported(pool_3d, slots)
    ):
        return False
    if freqs_cis is not None and (
        not isinstance(freqs_cis, torch.Tensor)
        or freqs_cis.device != q.device
        or freqs_cis.dtype != torch.complex64
        or freqs_cis.shape != (positions.numel(), 32)
        or not freqs_cis.is_contiguous()
    ):
        return False
    return not torch.is_grad_enabled() or not any(
        value.requires_grad for value in tensors
    )


def try_fused_qkv_rope_cache(
    q: torch.Tensor,
    kv: torch.Tensor,
    kv_norm: torch.Tensor,
    positions: torch.Tensor,
    freqs_table: torch.Tensor,
    pool_3d: torch.Tensor,
    slots: torch.Tensor,
    *,
    eps: float = 1e-6,
    freqs_cis: torch.Tensor | None = None,
) -> FusedQKVRopeCache | None:
    """Apply Q RoPE in place and return BF16 KV plus the selected frequencies.

    Only native V4.1 528-byte cache pages are accepted. Physical slots must be
    unique within the launch; any negative slot skips its store. Invalid
    positive slots trigger the existing device trap gate. Unsupported shapes
    return None before modifying inputs, and backend errors propagate.
    """
    if (
        not isinstance(eps, (int, float))
        or eps < 0
        or not is_supported(
            q, kv, kv_norm, positions, freqs_table, pool_3d, slots, freqs_cis=freqs_cis
        )
    ):
        return None
    kv_out = torch.empty(kv.shape, dtype=kv.dtype, device=kv.device)
    gathered_freqs = freqs_cis is not None
    if freqs_cis is None:
        freqs_cis = torch.empty(
            (positions.numel(), 32), dtype=torch.complex64, device=q.device
        )
    if positions.numel():
        _fused_qkv_rope_cache_kernel[
            (positions.numel(), triton.cdiv(q.shape[-2], 8) + 1)
        ](
            q,
            kv,
            kv_norm,
            positions,
            torch.view_as_real(freqs_cis if gathered_freqs else freqs_table),
            pool_3d,
            slots,
            kv_out,
            torch.view_as_real(freqs_cis),
            kv.stride(1),
            pool_3d.stride(0),
            pool_3d.shape[0],
            freqs_table.shape[0],
            HEADS=q.shape[-2],
            ENTRIES=pool_3d.shape[1],
            EPS=eps,
            GATHERED_FREQS=gathered_freqs,
            TRAP_INVALID_KV_ACCESS=trap_invalid_kv_access_enabled(),
            num_warps=4,
        )
    return FusedQKVRopeCache(q, kv_out, freqs_cis)
