"""SM90 V4.1 cache adapter for the existing FlashMLA V4/584-byte reader.

Both attention pools use V4 page-planar storage: P rows of 576 payload
bytes followed by P rows of 8 scale bytes. This is NOT 584-byte interleaved
storage. NoPE is E4M3/G64/UE8M0; the last 64 RoPE values remain BF16.
GLOBAL keys retain the model's E2M1/G16/E4M3 fake quantization *before*
the V4 storage quantization. INDEXER_KV is unaffected (68-byte FP4).

The proven V4 writer/dequant/CP kernels are reused. The leaf helpers below
only expose their layout to fused producers and serialize physical slots.
This module contains no attention implementation or fallback.
"""

from __future__ import annotations

from typing import Optional

import torch
import triton
import triton.language as tl

from . import _swa_dequant_triton as _reader
from . import _swa_kv_insert_triton as _writer
from ._v41_fp4_triton import _e2m1_to_float, _round_e2m1

HEAD_DIM = 512
ENTRY_BYTES = 584
TOKEN_DATA_SIZE = 576
SCALE_BYTES_PER_TOKEN = 8


@triton.jit
def _fake_quantize_global_fp4_values(values):
    """Same roundtrip as _fp4_global_insert_kernel + its GLOBAL decoder."""
    groups = values.to(tl.float32).reshape((32, 16))
    maximum = tl.maximum(tl.max(tl.abs(groups), 1), 6.0 * (2.0**-9))
    scales = (
        tl.div_rn(maximum, 6.0)
        .to(tl.float8e4nv, fp_downcast_rounding="rtne")
        .to(tl.float32)
    )
    normalized = tl.minimum(tl.maximum(tl.div_rn(groups, scales[:, None]), -6.0), 6.0)
    codes = _round_e2m1(normalized)
    result = (_e2m1_to_float(codes) * scales[:, None]).to(tl.bfloat16)
    # E2M1 code 8 is -0. Numeric equality cannot distinguish it from +0,
    # but the V4 RoPE tail stores the BF16 bits verbatim. Pin the sign after
    # the last floating-point conversion: inlining the fake-quant path can
    # otherwise simplify the decoder's floating negation of zero away.
    magnitude_bits = result.to(tl.uint16, bitcast=True) & 0x7FFF
    sign_bits = (codes.to(tl.uint16) & 8) << 12
    return (magnitude_bits | sign_bits).to(tl.bfloat16, bitcast=True).reshape((512,))


@triton.jit
def _store_v4_row(values, pool, slot, block_stride, ENTRIES: tl.constexpr):
    """Store a 512-value row; physical slots must be in range or negative."""
    slot64 = tl.cast(slot, tl.int64)
    if slot64 >= 0:
        columns = tl.arange(0, 512)
        groups = values.to(tl.float32).reshape((8, 64))
        maximum = tl.maximum(tl.max(tl.abs(groups), 1), 1.0e-4)
        exponent = tl.ceil(tl.log2(maximum / 448.0))
        scale = tl.exp2(exponent)
        payload = tl.clamp(groups / scale[:, None], -448.0, 448.0)
        payload = payload.to(tl.float8e4nv).to(tl.uint8, bitcast=True).reshape((512,))
        block, position = slot64 // ENTRIES, slot64 % ENTRIES
        # Fused decode specializes this stride, while prefill passes a
        # runtime scalar. tl.cast accepts both; constexpr integers have no .to.
        base = pool + block * tl.cast(block_stride, tl.int64)
        tl.store(base + position * 576 + columns, payload, columns < 448)
        rope_columns = tl.arange(0, 64)
        # Gather the original values, without a reduction that can change -0.
        rope = tl.gather(values, rope_columns + 448, axis=0).to(tl.bfloat16)
        rope_ptr = (base + position * 576 + 448).to(tl.pointer_type(tl.bfloat16))
        tl.store(rope_ptr + rope_columns, rope)
        encoded = tl.clamp(exponent + 127.0, 0.0, 255.0).to(tl.uint8)
        scale_columns = tl.arange(0, 8)
        encoded = tl.where(scale_columns < 7, encoded, 0)
        tl.store(base + ENTRIES * 576 + position * 8 + scale_columns, encoded)


@triton.jit(do_not_specialize=["input_stride"])
def _global_fake_quant_kernel(k, out, input_stride):
    row = tl.program_id(0).to(tl.int64)
    values = tl.load(k + row * input_stride.to(tl.int64) + tl.arange(0, 512))
    tl.store(
        out + row * 512 + tl.arange(0, 512), _fake_quantize_global_fp4_values(values)
    )


@triton.jit(do_not_specialize=["block_stride", "num_blocks"])
def _gather_v4_bytes_kernel(
    pool, slots, out, block_stride, num_blocks, ENTRIES: tl.constexpr
):
    row = tl.program_id(0).to(tl.int64)
    slot = tl.load(slots + row).to(tl.int64)
    valid = (slot >= 0) & (slot < num_blocks.to(tl.int64) * ENTRIES)
    safe = tl.where(valid, slot, 0)
    block, position = safe // ENTRIES, safe % ENTRIES
    columns = tl.arange(0, 1024)
    address = tl.where(
        columns < 576,
        position * 576 + columns,
        ENTRIES * 576 + position * 8 + columns - 576,
    )
    values = tl.load(
        pool + block * block_stride.to(tl.int64) + address,
        valid & (columns < 584),
        other=0,
    )
    tl.store(out + row * 584 + columns, values, columns < 584)


def is_supported(pool_3d: torch.Tensor, slots: torch.Tensor) -> bool:
    return (
        pool_3d.is_cuda
        and pool_3d.dtype == torch.uint8
        and pool_3d.ndim == 3
        and pool_3d.shape[1] > 0
        and pool_3d.shape[2] == ENTRY_BYTES
        and pool_3d.stride(2) == 1
        and pool_3d.stride(1) == ENTRY_BYTES
        and pool_3d.stride(0) >= pool_3d.shape[1] * ENTRY_BYTES
        and slots.device == pool_3d.device
        and slots.dtype in (torch.int32, torch.int64)
    )


def _checked_slots(pool, slots):
    if not is_supported(pool, slots):
        raise ValueError(
            "SM90 V4 cache requires CUDA uint8 [pages,entries,584] and integer slots"
        )
    slots = slots.to(torch.int64).contiguous()
    # Existing V4 writer only skips exactly -1. Normalize arbitrary sentinels
    # and out-of-range indices before handing them to that proven kernel.
    return torch.where(
        (slots >= 0) & (slots < pool.shape[0] * pool.shape[1]), slots, -1
    )


def _check_keys(k, device, rows):
    if (
        k.device != device
        or k.ndim != 2
        or k.shape != (rows, HEAD_DIM)
        or k.dtype not in (torch.bfloat16, torch.float16, torch.float32)
    ):
        raise ValueError(
            "keys must be floating-point [slots.numel(),512] on the cache device"
        )


def quantize_and_insert_swa_k_cache(kv, pool_3d, slots) -> None:
    slots = _checked_slots(pool_3d, slots).reshape(-1)
    _check_keys(kv, pool_3d.device, slots.numel())
    _writer.quantize_and_insert_k_cache(kv.contiguous(), pool_3d, slots)


def fake_quantize_global_fp4(k: torch.Tensor) -> torch.Tensor:
    """Preserve V4.1 compressed-key FP4 semantics independently of storage."""
    _check_keys(k, k.device, k.shape[0])
    if not k.is_cuda:
        raise ValueError("GLOBAL fake quantization requires CUDA")
    out = torch.empty(k.shape, dtype=torch.bfloat16, device=k.device)
    if k.shape[0]:
        _global_fake_quant_kernel[(k.shape[0],)](
            k.contiguous(), out, HEAD_DIM, num_warps=4
        )
    return out


def quantize_and_insert_k_cache(k, k_cache, slot_mapping) -> None:
    """GLOBAL write: E2M1/G16 fake quant, then V4 FP8/BF16 storage.

    Input must be the post-RoPE compressed key, not the pre-RoPE latent.
    SWA callers must use quantize_and_insert_swa_k_cache instead.
    """
    slots = _checked_slots(k_cache, slot_mapping).reshape(-1)
    _check_keys(k, k_cache.device, slots.numel())
    _writer.quantize_and_insert_k_cache(fake_quantize_global_fp4(k), k_cache, slots)


def _output(pool, count, out_dtype, out):
    if out is None:
        out = torch.empty((count, HEAD_DIM), dtype=out_dtype, device=pool.device)
    if (
        out.device != pool.device
        or out.shape != (count, HEAD_DIM)
        or out.dtype not in (torch.bfloat16, torch.float32)
        or out.stride(1) != 1
    ):
        raise ValueError(
            "dequant output must be BF16/FP32 [N,512] with contiguous columns"
        )
    return out


def dequantize_swa_k_cache(pool_3d, slots, *, out_dtype=torch.bfloat16, out=None):
    slots = _checked_slots(pool_3d, slots).reshape(-1)
    out = _output(pool_3d, slots.numel(), out_dtype, out)
    if slots.numel():
        # Launch the existing page-planar reader directly to support the
        # caller's strided destination without an extra allocation/copy.
        _reader._dequantize_slots_kernel[(slots.numel(),)](
            out,
            out.stride(0),
            pool_3d,
            pool_block_stride=pool_3d.stride(0),
            slot_indices_ptr=slots,
            block_size=pool_3d.shape[1],
            fp8_dim=448,
            bf16_dim=64,
            scale_dim=8,
            quant_block=64,
            token_data_size=576,
            num_cache_blocks=pool_3d.shape[0],
            n_quant_blocks=7,
            TRAP_INVALID_KV_ACCESS=_reader.trap_invalid_kv_access_enabled(),
        )
    return out


dequantize_k_cache_slots = dequantize_swa_k_cache


def dequantize_and_gather_k_cache_slots(
    out, k_cache, slot_mapping, gather_lens, offset
):
    slots = _checked_slots(k_cache, slot_mapping)
    if slots.ndim != 2 or out.ndim != 3 or out.shape[0] != slots.shape[0]:
        raise ValueError("gather needs [B,W] slots and [B,M,512] output")
    if offset < 0 or offset + slots.shape[1] > out.shape[1] or out.shape[2] != HEAD_DIM:
        raise ValueError("gather output window is out of range")
    if gather_lens is not None:
        if gather_lens.device != out.device or gather_lens.numel() != slots.shape[0]:
            raise ValueError("gather lengths must have one CUDA value per request")
        gather_lens = gather_lens.reshape(-1).clamp(0, slots.shape[1]).to(torch.int32)
    _reader.dequantize_and_gather_k_cache_slots(
        out, k_cache, slots, gather_lens, offset
    )


def gather_k_cache_bytes(pool_3d, slot_indices):
    """Return [N,584] compact transport rows [576 payload | 8 scales]."""
    slots = _checked_slots(pool_3d, slot_indices).reshape(-1)
    out = torch.empty(
        (slots.numel(), ENTRY_BYTES), dtype=torch.uint8, device=pool_3d.device
    )
    if slots.numel():
        _gather_v4_bytes_kernel[(slots.numel(),)](
            pool_3d,
            slots,
            out,
            pool_3d.stride(0),
            pool_3d.shape[0],
            ENTRIES=pool_3d.shape[1],
            num_warps=4,
        )
    return out


def dequantize_k_cache_bytes(raw_bytes, *, out_dtype=torch.bfloat16, out=None):
    """Read compact transport rows, not a multi-token page-planar view."""
    if (
        not raw_bytes.is_cuda
        or raw_bytes.dtype != torch.uint8
        or raw_bytes.ndim != 2
        or raw_bytes.shape[1] != ENTRY_BYTES
        or not raw_bytes.is_contiguous()
    ):
        raise ValueError("transport bytes must be contiguous CUDA uint8 [N,584]")
    out = _output(raw_bytes, raw_bytes.shape[0], out_dtype, out)
    _reader.dequantize_packed_k_cache_flat(out, raw_bytes)
    return out


def _valid_cp(raw, slots, entries, rank, size, compaction):
    if compaction is None:
        return False
    unique, compact = compaction.unique_blocks, compaction.compact_slots
    return (
        raw.is_cuda
        and raw.dtype == torch.uint8
        and raw.ndim == 2
        and raw.stride(1) == 1
        and entries > 0
        and size > 0
        and 0 <= rank < size
        and raw.shape[1] * size >= entries * ENTRY_BYTES
        and raw.stride(0) >= raw.shape[1]
        and slots.device == unique.device == compact.device == raw.device
        and slots.dtype in (torch.int32, torch.int64)
        and unique.dtype in (torch.int32, torch.int64)
        and unique.ndim == 1
        and compact.dtype in (torch.int32, torch.int64)
        and compact.numel() == slots.numel()
    )


def is_supported_fresh_store(
    k, raw, slots, full_entries_per_block, cp_rank, cp_size, compaction, fresh_slots
):
    if fresh_slots is None or not _valid_cp(
        raw, slots, full_entries_per_block, cp_rank, cp_size, compaction
    ):
        return False
    return (
        k.device == raw.device
        and k.dtype == torch.bfloat16
        and k.ndim == 2
        and k.shape == (slots.numel(), HEAD_DIM)
        and k.stride(1) == 1
        and fresh_slots.device == raw.device
        and fresh_slots.shape == (k.shape[0],)
        and fresh_slots.dtype in (torch.int32, torch.int64)
        and fresh_slots.is_contiguous()
        and all(
            t.untyped_storage().data_ptr() != raw.untyped_storage().data_ptr()
            for t in (
                k,
                slots,
                compaction.unique_blocks,
                compaction.compact_slots,
                fresh_slots,
            )
        )
    )


def quantize_and_insert_k_cache_cp_byte_sliced(
    k,
    k_cache_raw,
    slot_mapping,
    full_entries_per_block,
    cp_rank,
    cp_size,
    compaction,
    *,
    fresh_out=None,
    fresh_slots=None,
):
    """SWA CP write, preserving prior slots and the rank's trailing padding.

    The CP API belongs to SWA; unlike the GLOBAL writer it does NOT fake-quantize
    to FP4. fresh_slots are unique, valid flattened workspace rows from planner.
    """
    if not _valid_cp(
        k_cache_raw, slot_mapping, full_entries_per_block, cp_rank, cp_size, compaction
    ):
        raise ValueError("invalid V4 CP byte-sliced cache contract")
    _check_keys(k, k_cache_raw.device, slot_mapping.numel())
    if (fresh_out is None) != (fresh_slots is None):
        raise ValueError("fresh_out and fresh_slots must be supplied together")
    if fresh_out is not None:
        if (
            not is_supported_fresh_store(
                k,
                k_cache_raw,
                slot_mapping,
                full_entries_per_block,
                cp_rank,
                cp_size,
                compaction,
                fresh_slots,
            )
            or fresh_out.device != k.device
            or fresh_out.dtype != torch.bfloat16
            or not fresh_out.is_contiguous()
            or fresh_out.shape[-1] != HEAD_DIM
            or fresh_out.untyped_storage().data_ptr()
            == k_cache_raw.untyped_storage().data_ptr()
        ):
            raise ValueError("invalid disjoint BF16 fresh workspace")
        fresh_out.view(-1, HEAD_DIM).index_copy_(0, fresh_slots.to(torch.int64), k)
    unique = compaction.unique_blocks.to(torch.int64)
    if slot_mapping.numel() == 0 or unique.numel() == 0:
        return
    local_bytes = k_cache_raw.shape[1]
    full_stride = local_bytes * cp_size
    full_raw = torch.empty(
        (unique.numel(), full_stride), dtype=torch.uint8, device=k.device
    )
    local_slice = full_raw[:, cp_rank * local_bytes : (cp_rank + 1) * local_bytes]
    # Other ranks' bytes are never read or copied out. Keep every unchanged
    # byte of this rank before updating the compact physical slots.
    local_slice.copy_(k_cache_raw.index_select(0, unique))
    view = full_raw.as_strided(
        (unique.numel(), full_entries_per_block, ENTRY_BYTES),
        (full_stride, ENTRY_BYTES, 1),
    )
    quantize_and_insert_swa_k_cache(k, view, compaction.compact_slots)
    k_cache_raw.index_copy_(0, unique, local_slice.contiguous())


def dequantize_and_gather_k_cache_slots_cp_byte_sliced(
    out,
    k_cache_raw,
    slot_mapping,
    gather_lens,
    offset,
    full_entries_per_block,
    cp_rank,
    cp_size,
    compaction,
):
    if not _valid_cp(
        k_cache_raw, slot_mapping, full_entries_per_block, cp_rank, cp_size, compaction
    ):
        raise ValueError("invalid V4 CP byte-sliced cache contract")
    if (
        slot_mapping.ndim != 2
        or out.ndim != 3
        or out.shape[0] != slot_mapping.shape[0]
        or out.shape[-1] != HEAD_DIM
        or offset < 0
        or offset + slot_mapping.shape[1] > out.shape[1]
    ):
        raise ValueError("invalid CP gather output window")
    if gather_lens is not None:
        if (
            gather_lens.device != out.device
            or gather_lens.numel() != slot_mapping.shape[0]
        ):
            raise ValueError("CP gather lengths must have one device value per request")
        gather_lens = (
            gather_lens.reshape(-1).clamp(0, slot_mapping.shape[1]).to(torch.int32)
        )
    if slot_mapping.numel() == 0:
        return
    if compaction.unique_blocks.numel() == 0:
        target = out[:, offset : offset + slot_mapping.shape[1], :]
        if gather_lens is None:
            target.zero_()
        else:
            columns = torch.arange(slot_mapping.shape[1], device=out.device)
            target.copy_(
                torch.where(
                    (columns[None] < gather_lens[:, None])[:, :, None], 0.0, target
                )
            )
        return
    if cp_size == 1:
        view = k_cache_raw.as_strided(
            (k_cache_raw.shape[0], full_entries_per_block, ENTRY_BYTES),
            (k_cache_raw.stride(0), ENTRY_BYTES, 1),
        )
        return dequantize_and_gather_k_cache_slots(
            out, view, slot_mapping, gather_lens, offset
        )
    return _reader.dequantize_and_gather_k_cache_slots_cp_byte_sliced(
        out,
        k_cache_raw,
        slot_mapping,
        gather_lens,
        offset,
        full_entries_per_block,
        cp_rank,
        cp_size,
        compaction,
    )
