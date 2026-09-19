"""V4.1-Flash FP4 KV codec for the GLOBAL (288B) and INDEX_K (68B) pools.

Quantization formulas adapted from DeepSeek-V4.1-Flash inference/kernel.py,
revision 2bc89ac599031fa673cab993f1df02fc4a98c673.
Copyright (c) 2023 DeepSeek (MIT; see the license block below).

The official split is preserved exactly: compressed-KV rows use group-16
E4M3 scales (fp4_max 6.0, amax floor 6*2^-9), indexer keys and queries use
group-32 UE8M0 scales (amax floor 6*2^-126). Pool byte layouts:

  * GLOBAL ``[blocks, entries, 288]`` uint8, row-interleaved: each entry is
    256B of packed e2m1 payload followed by 32 one-byte E4M3 group scales.
  * INDEX_K nominal ``[blocks, entries, 68]`` uint8 view over a per-block
    planar layout: ``entries * 64`` payload bytes followed by ``entries * 4``
    scale bytes (one packed-UE8M0 int32 per entry). This is DeepGEMM's fused
    fp8_fp4 paged cache layout, so the decode/prefill scoring paths can hand
    pool pages to DeepGEMM without an intermediate repack.
"""

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from __future__ import annotations

from typing import Optional

import torch
import triton
import triton.language as tl

FP4_GLOBAL_HEAD_DIM = 512
FP4_GLOBAL_GROUP = 16
FP4_GLOBAL_ENTRY_BYTES = 288  # 256B e2m1 payload + 32B E4M3 scales
FP4_INDEXER_HEAD_DIM = 128
FP4_INDEXER_GROUP = 32
FP4_INDEXER_ENTRY_BYTES = 68  # 64B e2m1 payload + 4B packed UE8M0 scales


@triton.jit
def _round_power_of_two_scale(maximum, INVERSE_MAX: tl.constexpr):
    scaled = maximum * INVERSE_MAX
    bits = scaled.to(tl.uint32, bitcast=True)
    exponent = ((bits >> 23) & 255) + ((bits & 0x7FFFFF) != 0).to(tl.uint32)
    scale = (exponent << 23).to(tl.float32, bitcast=True)
    return scale, exponent.to(tl.uint8)


@triton.jit
def _round_e2m1(values):
    magnitude = tl.abs(values)
    # Alternating strict/non-strict boundaries implement round-to-nearest-even.
    code = (magnitude > 0.25).to(tl.uint8)
    code += (magnitude >= 0.75).to(tl.uint8)
    code += (magnitude > 1.25).to(tl.uint8)
    code += (magnitude >= 1.75).to(tl.uint8)
    code += (magnitude > 2.5).to(tl.uint8)
    code += (magnitude >= 3.5).to(tl.uint8)
    code += (magnitude > 5.0).to(tl.uint8)
    sign = ((values.to(tl.uint32, bitcast=True) >> 28) & 8).to(tl.uint8)
    return code | sign


@triton.jit
def _ue8m0_to_float(byte):
    bits = byte.to(tl.uint32) << 23
    bits = tl.where(byte == 0, 0x00400000, bits)
    bits = tl.where(byte == 255, 0x7FC00000, bits)
    return bits.to(tl.uint32).to(tl.float32, bitcast=True)


@triton.jit
def _e2m1_to_float(code):
    magnitude = code & 7
    normal = tl.exp2((magnitude >> 1).to(tl.float32) - 1.0)
    normal *= 1.0 + (magnitude & 1).to(tl.float32) * 0.5
    value = tl.where(magnitude < 2, magnitude.to(tl.float32) * 0.5, normal)
    return tl.where((code & 8) != 0, -value, value)


@triton.jit
def _pack_e2m1_payload(normalized, DIM: tl.constexpr):
    codes = _round_e2m1(normalized).reshape((DIM // 2, 2))
    even, odd = tl.split(codes)
    return even | (odd << 4)


# ---------------------------------------------------------------------------
# GLOBAL pool: quantize/insert and dequantize (row-interleaved 288B entries)
# ---------------------------------------------------------------------------
@triton.jit(do_not_specialize=["T", "block_stride", "num_cache_blocks"])
def _fp4_global_insert_kernel(
    k_ptr,  # [T, 512] bf16 (post-RoPE compressed keys)
    slot_mapping_ptr,  # [T] int64; -1 = skip
    cache_ptr,  # [num_blocks, block_size, 288] uint8
    T,
    DIM: tl.constexpr,  # 512
    GROUP: tl.constexpr,  # 16
    ROW_BYTES: tl.constexpr,  # 288
    cache_block_size: tl.constexpr,
    block_stride,  # bytes per block (from cache.stride(0))
    num_cache_blocks,
):
    pid = tl.program_id(0).to(tl.int64)
    if pid >= T:
        return
    slot = tl.load(slot_mapping_ptr + pid).to(tl.int64)
    if slot < 0:
        return
    block_idx = slot // cache_block_size
    offset = slot % cache_block_size

    channels = tl.arange(0, DIM)
    x = tl.load(k_ptr + pid * DIM + channels).to(tl.float32)
    x = x.reshape((DIM // GROUP, GROUP))
    maximum = tl.max(tl.abs(x), axis=1)
    # Training's compressed KV: keep even an all-zero group's scale nonzero.
    maximum = tl.maximum(maximum, 6.0 * (2.0**-9))
    scale_fp8 = tl.div_rn(maximum, 6.0).to(
        tl.float8e4nv, fp_downcast_rounding="rtne"
    )
    scale = scale_fp8.to(tl.float32)
    normalized = tl.minimum(tl.maximum(tl.div_rn(x, scale[:, None]), -6.0), 6.0)
    payload = _pack_e2m1_payload(normalized, DIM)

    base = cache_ptr + block_idx.to(tl.int64) * block_stride.to(tl.int64) + offset * ROW_BYTES
    tl.store(base + tl.arange(0, DIM // 2), payload)
    tl.store(
        base + DIM // 2 + tl.arange(0, DIM // GROUP),
        scale_fp8.to(tl.uint8, bitcast=True),
    )


@triton.jit(do_not_specialize=["N", "block_stride", "num_cache_blocks"])
def _fp4_global_dequant_kernel(
    cache_ptr,  # [num_blocks, block_size, 288] uint8
    slot_mapping_ptr,  # [N] int64; <0 = zero-fill
    out_ptr,  # [N, 512] OUT_DTYPE
    N,
    DIM: tl.constexpr,
    GROUP: tl.constexpr,
    ROW_BYTES: tl.constexpr,
    cache_block_size: tl.constexpr,
    block_stride,
    num_cache_blocks,
    OUT_DTYPE: tl.constexpr,
):
    pid = tl.program_id(0).to(tl.int64)
    if pid >= N:
        return
    channels = tl.arange(0, DIM)
    out_row = out_ptr + pid * DIM + channels
    slot = tl.load(slot_mapping_ptr + pid).to(tl.int64)
    if slot < 0:
        tl.store(out_row, tl.zeros((DIM,), dtype=tl.float32).to(OUT_DTYPE))
        return
    block_idx = slot // cache_block_size
    offset = slot % cache_block_size
    base = cache_ptr + block_idx.to(tl.int64) * block_stride.to(tl.int64) + offset * ROW_BYTES
    packed = tl.load(base + channels // 2)
    code = (packed.to(tl.int32) >> ((channels % 2) * 4)) & 15
    scale = tl.load(base + DIM // 2 + channels // GROUP)
    scale_value = scale.to(tl.float8e4nv, bitcast=True).to(tl.float32)
    values = _e2m1_to_float(code) * scale_value
    tl.store(out_row, values.to(OUT_DTYPE))


def quantize_and_insert_k_cache_fp4(
    k: torch.Tensor,  # [T, 512] bf16, contiguous
    k_cache: torch.Tensor,  # [num_blocks, block_size, 288] uint8
    slot_mapping: torch.Tensor,  # [T] int64; -1 = skip
) -> None:
    """Quantize post-RoPE compressed keys to FP4 and write 288B pool entries."""
    if slot_mapping.dtype != torch.int64:
        slot_mapping = slot_mapping.to(torch.int64)
    num_tokens = int(slot_mapping.shape[0])
    if num_tokens == 0:
        return
    block_size = int(k_cache.shape[1])
    _fp4_global_insert_kernel[(num_tokens,)](
        k,
        slot_mapping,
        k_cache,
        num_tokens,
        DIM=FP4_GLOBAL_HEAD_DIM,
        GROUP=FP4_GLOBAL_GROUP,
        ROW_BYTES=FP4_GLOBAL_ENTRY_BYTES,
        cache_block_size=block_size,
        block_stride=int(k_cache.stride(0)),
        num_cache_blocks=int(k_cache.shape[0]),
        num_warps=4,
    )


@triton.jit(do_not_specialize=["N", "block_stride", "num_cache_blocks"])
def _fp4_global_gather_bytes_kernel(
    cache_ptr,  # [num_blocks, block_size, 288] uint8
    slot_mapping_ptr,  # [N] int64; <0 = zero-fill
    out_ptr,  # [N, 288] uint8 (raw pool bytes)
    N,
    ROW_BYTES: tl.constexpr,
    ARANGE: tl.constexpr,  # power-of-2 cover of ROW_BYTES (triton arange)
    cache_block_size: tl.constexpr,
    block_stride,
    num_cache_blocks,
):
    pid = tl.program_id(0).to(tl.int64)
    if pid >= N:
        return
    cols = tl.arange(0, ARANGE)
    mask = cols < ROW_BYTES
    out_row = out_ptr + pid * ROW_BYTES + cols
    slot = tl.load(slot_mapping_ptr + pid).to(tl.int64)
    if slot < 0:
        tl.store(out_row, tl.zeros((ARANGE,), dtype=tl.uint8), mask=mask)
        return
    block_idx = slot // cache_block_size
    offset = slot % cache_block_size
    base = cache_ptr + block_idx.to(tl.int64) * block_stride.to(tl.int64) + offset * ROW_BYTES
    tl.store(out_row, tl.load(base + cols, mask=mask, other=0), mask=mask)


def dequantize_k_cache_slots_fp4(
    pool_3d: torch.Tensor,  # [num_blocks, block_size, 288] uint8
    slot_indices: torch.Tensor,  # [N] int (flat slot ids; <0 = zero-fill)
    *,
    out_dtype: torch.dtype = torch.bfloat16,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Per-slot dequant of 288B FP4 entries to ``[N, 512] out_dtype``."""
    N = int(slot_indices.numel())
    if out is None:
        out = torch.empty((N, FP4_GLOBAL_HEAD_DIM), dtype=out_dtype, device=pool_3d.device)
    if N == 0:
        return out
    slots_i64 = slot_indices.reshape(-1).to(torch.int64).contiguous()
    _OUT_DTYPE = {
        torch.float32: tl.float32,
        torch.bfloat16: tl.bfloat16,
    }[out.dtype]
    _fp4_global_dequant_kernel[(N,)](
        pool_3d,
        slots_i64,
        out,
        N,
        DIM=FP4_GLOBAL_HEAD_DIM,
        GROUP=FP4_GLOBAL_GROUP,
        ROW_BYTES=FP4_GLOBAL_ENTRY_BYTES,
        cache_block_size=int(pool_3d.shape[1]),
        block_stride=int(pool_3d.stride(0)),
        num_cache_blocks=int(pool_3d.shape[0]),
        OUT_DTYPE=_OUT_DTYPE,
        num_warps=4,
    )
    return out


def gather_k_cache_bytes_fp4(
    pool_3d: torch.Tensor,  # [num_blocks, block_size, 288] uint8
    slot_indices: torch.Tensor,  # [N] int (flat slot ids; <0 = zero-fill)
) -> torch.Tensor:
    """Gather raw 288B FP4 pool entries to a contiguous ``[N, 288]`` uint8 tensor.

    Byte-first CP transport: each rank reads only its owned slots (others
    zero-filled) so an all-reduce over the gathered bytes — every byte has
    exactly one owner — reassembles the full entry set at one quarter of the
    BF16 dequantized wire size.
    """
    N = int(slot_indices.numel())
    out = torch.zeros(
        (N, FP4_GLOBAL_ENTRY_BYTES), dtype=torch.uint8, device=pool_3d.device
    )
    if N == 0:
        return out
    slots_i64 = slot_indices.reshape(-1).to(torch.int64).contiguous()
    _fp4_global_gather_bytes_kernel[(N,)](
        pool_3d,
        slots_i64,
        out,
        N,
        ROW_BYTES=FP4_GLOBAL_ENTRY_BYTES,
        ARANGE=triton.next_power_of_2(FP4_GLOBAL_ENTRY_BYTES),
        cache_block_size=int(pool_3d.shape[1]),
        block_stride=int(pool_3d.stride(0)),
        num_cache_blocks=int(pool_3d.shape[0]),
        num_warps=4,
    )
    return out


def dequantize_k_cache_bytes_fp4(
    raw_bytes: torch.Tensor,  # [N, 288] uint8 (row-interleaved entries)
    *,
    out_dtype: torch.dtype = torch.bfloat16,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Dequantize contiguous 288B FP4 entries to ``[N, 512] out_dtype``.

    The receiving side of the byte-first transport: the all-reduced raw bytes
    are viewed as a single-block pool so the existing dequant kernel applies
    unchanged.
    """
    N = int(raw_bytes.shape[0])
    if out is None:
        out = torch.empty((N, FP4_GLOBAL_HEAD_DIM), dtype=out_dtype, device=raw_bytes.device)
    if N == 0:
        return out
    pool_view = raw_bytes.view(1, N, FP4_GLOBAL_ENTRY_BYTES)
    slots = torch.arange(N, dtype=torch.int64, device=raw_bytes.device)
    return dequantize_k_cache_slots_fp4(
        pool_view, slots, out_dtype=out_dtype, out=out
    )


# ---------------------------------------------------------------------------
# INDEX_K pool: quantize/insert, gather (raw DeepGEMM bytes) and dequantize
# (per-block planar layout: payload plane then packed-int32 scale plane)
# ---------------------------------------------------------------------------
@triton.jit(do_not_specialize=["T", "block_stride", "num_cache_blocks"])
def _fp4_indexer_insert_kernel(
    k_ptr,  # [T, 128] bf16 (post-RoPE index keys)
    slot_mapping_ptr,  # [T] int64; -1 = skip
    cache_ptr,  # [num_blocks, block_size, 68] uint8 (planar per block)
    T,
    D: tl.constexpr,  # 128
    GROUP: tl.constexpr,  # 32
    cache_block_size: tl.constexpr,
    block_stride,  # bytes per block (from cache.stride(0))
    num_cache_blocks,
):
    pid = tl.program_id(0).to(tl.int64)
    if pid >= T:
        return
    slot = tl.load(slot_mapping_ptr + pid).to(tl.int64)
    if slot < 0:
        return
    block_idx = slot // cache_block_size
    offset = slot % cache_block_size

    channels = tl.arange(0, D)
    x = tl.load(k_ptr + pid * D + channels).to(tl.float32)
    x = x.reshape((D // GROUP, GROUP))
    maximum = tl.max(tl.abs(x), axis=1)
    maximum = tl.maximum(maximum, 6.0 * (2.0**-126))
    scale, scale_bytes = _round_power_of_two_scale(maximum, 1.0 / 6.0)
    normalized = tl.minimum(tl.maximum(tl.div_rn(x, scale[:, None]), -6.0), 6.0)
    payload = _pack_e2m1_payload(normalized, D)

    block_base = cache_ptr + block_idx.to(tl.int64) * block_stride.to(tl.int64)
    tl.store(block_base + offset * (D // 2) + tl.arange(0, D // 2), payload)
    tl.store(
        block_base + cache_block_size * (D // 2) + offset * (D // GROUP)
        + tl.arange(0, D // GROUP),
        scale_bytes,
    )


@triton.jit(do_not_specialize=["N", "block_stride", "num_cache_blocks"])
def _fp4_indexer_gather_kernel(
    cache_ptr,  # [num_blocks, block_size, 68] uint8 (planar per block)
    slot_mapping_ptr,  # [N] int64; <0 = zero payload/scale
    payload_ptr,  # [N, 64] int8
    sf_ptr,  # [N] int32
    N,
    D: tl.constexpr,
    GROUP: tl.constexpr,
    cache_block_size: tl.constexpr,
    block_stride,
    num_cache_blocks,
):
    pid = tl.program_id(0).to(tl.int64)
    if pid >= N:
        return
    slot = tl.load(slot_mapping_ptr + pid).to(tl.int64)
    if slot < 0:
        tl.store(payload_ptr + pid * (D // 2) + tl.arange(0, D // 2),
                 tl.zeros((D // 2,), dtype=tl.int8))
        tl.store(sf_ptr + pid, 0)
        return
    block_idx = slot // cache_block_size
    offset = slot % cache_block_size
    block_base = cache_ptr + block_idx.to(tl.int64) * block_stride.to(tl.int64)
    payload = tl.load(block_base + offset * (D // 2) + tl.arange(0, D // 2))
    sf = tl.load(
        (block_base + cache_block_size * (D // 2)).to(tl.pointer_type(tl.int32)) + offset
    )
    tl.store(payload_ptr + pid * (D // 2) + tl.arange(0, D // 2), payload.to(tl.int8))
    tl.store(sf_ptr + pid, sf)


@triton.jit(do_not_specialize=["N", "block_stride", "num_cache_blocks"])
def _fp4_indexer_dequant_kernel(
    cache_ptr,  # [num_blocks, block_size, 68] uint8 (planar per block)
    slot_mapping_ptr,  # [N] int64; <0 = zero-fill
    out_ptr,  # [N, 128] OUT_DTYPE
    N,
    D: tl.constexpr,
    GROUP: tl.constexpr,
    cache_block_size: tl.constexpr,
    block_stride,
    num_cache_blocks,
    OUT_DTYPE: tl.constexpr,
):
    pid = tl.program_id(0).to(tl.int64)
    if pid >= N:
        return
    channels = tl.arange(0, D)
    out_row = out_ptr + pid * D + channels
    slot = tl.load(slot_mapping_ptr + pid).to(tl.int64)
    if slot < 0:
        tl.store(out_row, tl.zeros((D,), dtype=tl.float32).to(OUT_DTYPE))
        return
    block_idx = slot // cache_block_size
    offset = slot % cache_block_size
    block_base = cache_ptr + block_idx.to(tl.int64) * block_stride.to(tl.int64)
    payload_base = block_base + offset * (D // 2)
    packed = tl.load(payload_base + channels // 2)
    code = (packed.to(tl.int32) >> ((channels % 2) * 4)) & 15
    # Each channel's group scale byte lives in the block's scale plane.
    scale_base = block_base + cache_block_size * (D // 2) + offset * (D // GROUP)
    scale = tl.load(scale_base + channels // GROUP)
    values = _e2m1_to_float(code) * _ue8m0_to_float(scale)
    tl.store(out_row, values.to(OUT_DTYPE))


def quantize_indexer_k_fp4(
    k_bf16: torch.Tensor,  # [T, 128] bf16, contiguous
    slot_mapping: torch.Tensor,  # [T] int64 (or convertible); -1 = skip
    kv_cache_packed: torch.Tensor,  # [num_blocks, block_size, 68] uint8
) -> None:
    """Quantize index keys to FP4 and write planar 68B-per-entry pool rows."""
    if slot_mapping.dtype != torch.int64:
        slot_mapping = slot_mapping.to(torch.int64)
    slot_mapping = slot_mapping.contiguous()
    T = k_bf16.shape[0]
    if T == 0:
        return
    cache_block_size = kv_cache_packed.shape[1]
    _fp4_indexer_insert_kernel[(T,)](
        k_bf16,
        slot_mapping,
        kv_cache_packed,
        T,
        D=FP4_INDEXER_HEAD_DIM,
        GROUP=FP4_INDEXER_GROUP,
        cache_block_size=cache_block_size,
        block_stride=int(kv_cache_packed.stride(0)),
        num_cache_blocks=int(kv_cache_packed.shape[0]),
        num_warps=4,
    )


def gather_indexer_k_fp4(
    kv_cache_packed: torch.Tensor,  # [num_blocks, block_size, 68] uint8
    slot_mapping: torch.Tensor,  # [N] int64; -1 = zero row
) -> tuple[torch.Tensor, torch.Tensor]:
    """Gather raw FP4 bytes: ``(payload [N, 64] int8, sf [N] int32)``.

    The returned pair is DeepGEMM's MX-mode KV contract for
    ``fp8_fp4_mqa_logits`` / ``fp8_fp4_paged_mqa_logits``.
    """
    if slot_mapping.dtype != torch.int64:
        slot_mapping = slot_mapping.to(torch.int64)
    slot_mapping = slot_mapping.contiguous()
    N = slot_mapping.shape[0]
    device = kv_cache_packed.device
    payload = torch.empty(N, FP4_INDEXER_HEAD_DIM // 2, dtype=torch.int8, device=device)
    sf = torch.empty(N, dtype=torch.int32, device=device)
    if N == 0:
        return payload, sf
    _fp4_indexer_gather_kernel[(N,)](
        kv_cache_packed,
        slot_mapping,
        payload,
        sf,
        N,
        D=FP4_INDEXER_HEAD_DIM,
        GROUP=FP4_INDEXER_GROUP,
        cache_block_size=int(kv_cache_packed.shape[1]),
        block_stride=int(kv_cache_packed.stride(0)),
        num_cache_blocks=int(kv_cache_packed.shape[0]),
        num_warps=4,
    )
    return payload, sf


def dequantize_indexer_k_fp4(
    kv_cache_packed: torch.Tensor,  # [num_blocks, block_size, 68] uint8
    slot_mapping: torch.Tensor,  # [N] int64
    *,
    out_dtype: torch.dtype = torch.float32,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Gather packed FP4 slots and dequantize to ``[N, 128] out_dtype``."""
    if slot_mapping.dtype != torch.int64:
        slot_mapping = slot_mapping.to(torch.int64)
    slot_mapping = slot_mapping.contiguous()
    N = slot_mapping.shape[0]
    if out is None:
        out = torch.empty(N, FP4_INDEXER_HEAD_DIM, dtype=out_dtype, device=kv_cache_packed.device)
    if N == 0:
        return out
    _OUT_DTYPE = {
        torch.float32: tl.float32,
        torch.bfloat16: tl.bfloat16,
        torch.float16: tl.float16,
    }[out.dtype]
    _fp4_indexer_dequant_kernel[(N,)](
        kv_cache_packed,
        slot_mapping,
        out,
        N,
        D=FP4_INDEXER_HEAD_DIM,
        GROUP=FP4_INDEXER_GROUP,
        cache_block_size=int(kv_cache_packed.shape[1]),
        block_stride=int(kv_cache_packed.stride(0)),
        num_cache_blocks=int(kv_cache_packed.shape[0]),
        OUT_DTYPE=_OUT_DTYPE,
        num_warps=4,
    )
    return out


# ---------------------------------------------------------------------------
# Indexer Q (and pool-free K reference): group-32 UE8M0 FP4, DeepGEMM MX form
# ---------------------------------------------------------------------------
@triton.jit(do_not_specialize=["R"])
def _fp4_row_quant_kernel(
    x_ptr,  # [R, 128] bf16
    payload_ptr,  # [R, 64] int8
    sf_ptr,  # [R] int32
    R,
    D: tl.constexpr,  # 128
    GROUP: tl.constexpr,  # 32
):
    pid = tl.program_id(0).to(tl.int64)
    if pid >= R:
        return
    channels = tl.arange(0, D)
    x = tl.load(x_ptr + pid * D + channels).to(tl.float32)
    x = x.reshape((D // GROUP, GROUP))
    maximum = tl.max(tl.abs(x), axis=1)
    maximum = tl.maximum(maximum, 6.0 * (2.0**-126))
    scale, scale_bytes = _round_power_of_two_scale(maximum, 1.0 / 6.0)
    normalized = tl.minimum(tl.maximum(tl.div_rn(x, scale[:, None]), -6.0), 6.0)
    payload = _pack_e2m1_payload(normalized, D)
    tl.store(payload_ptr + pid * (D // 2) + tl.arange(0, D // 2), payload.to(tl.int8))
    sf_base = (sf_ptr + pid).to(tl.pointer_type(tl.uint8))
    tl.store(sf_base + tl.arange(0, D // GROUP), scale_bytes)


def quantize_rows_fp4(
    x: torch.Tensor,  # [..., 128] bf16, contiguous
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize the trailing 128-dim to MX FP4: ``(payload [..., 64] int8, sf [...] int32)``.

    Matches the official indexer-query quantization (group 32, UE8M0 scales
    packed one int32 per row) and DeepGEMM's MX-mode Q contract.
    """
    shape = x.shape
    rows = x.numel() // FP4_INDEXER_HEAD_DIM
    payload = torch.empty(rows, FP4_INDEXER_HEAD_DIM // 2, dtype=torch.int8, device=x.device)
    sf = torch.empty(rows, dtype=torch.int32, device=x.device)
    if rows:
        _fp4_row_quant_kernel[(rows,)](
            x.reshape(rows, FP4_INDEXER_HEAD_DIM),
            payload,
            sf,
            rows,
            D=FP4_INDEXER_HEAD_DIM,
            GROUP=FP4_INDEXER_GROUP,
            num_warps=4,
        )
    return payload.view(*shape[:-1], FP4_INDEXER_HEAD_DIM // 2), sf.view(*shape[:-1])