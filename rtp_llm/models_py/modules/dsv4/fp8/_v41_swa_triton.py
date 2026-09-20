# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""V4.1 SWA MXFP8 codec, compatible with FlashMLA's ``ModelType::V41``.

Adapted from vLLM's deepseek_v41/common/ops/cache_utils.py. All 512
channels, including RoPE, use group-32 E4M3 with UE8M0 scales. Each page
contains ``entries * 512`` payload bytes followed by ``entries * 16``
scale bytes; its allocation stride may include trailing TMA padding.

Slot ids are physical ``page * entries + offset`` ids. Ring and CP slot
ownership remains the caller's responsibility; masked slots are zero on
read and skipped on write. This format must never be read with V4's 584B
codec, or selected as a fallback for an already allocated 584B cache.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import torch
import triton
import triton.language as tl

if TYPE_CHECKING:
    from rtp_llm.models_py.modules.dsv4.fp8._swa_cp_byte_sliced import (
        CPByteSlicedSlotCompaction,
    )

HEAD_DIM = 512
QUANT_BLOCK = 32
SCALE_BYTES_PER_TOKEN = HEAD_DIM // QUANT_BLOCK
ENTRY_BYTES = HEAD_DIM + SCALE_BYTES_PER_TOKEN


def is_supported(pool_3d: torch.Tensor, slots: torch.Tensor) -> bool:
    """Check the device/layout contract without reading device metadata."""
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


def _check_pool(pool_3d: torch.Tensor, slots: torch.Tensor) -> None:
    if not is_supported(pool_3d, slots):
        raise ValueError(
            "V4.1 SWA requires a CUDA uint8 [pages, entries, 528] pool and integer slots"
        )


@triton.jit(do_not_specialize=["block_stride", "input_stride"])
def _quantize_and_insert_swa_kernel(
    kv,
    pool,
    slots,
    input_stride,
    block_stride,
    ENTRIES: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    slot = tl.load(slots + row).to(tl.int64)
    if slot < 0:
        return
    block = slot // ENTRIES
    position = slot % ENTRIES
    base = pool + block * block_stride.to(tl.int64)
    columns = tl.arange(0, 512)
    values = tl.load(kv + row * input_stride.to(tl.int64) + columns).to(tl.float32)
    groups = values.reshape((16, 32))
    maximum = tl.maximum(tl.max(tl.abs(groups), 1), 1e-4)
    exponent = tl.ceil(tl.log2(maximum * (1.0 / 448.0)))
    scaled = tl.clamp(groups * tl.exp2(-exponent)[:, None], -448.0, 448.0)
    payload = scaled.to(tl.float8e4nv).to(tl.uint8, bitcast=True).reshape((512,))
    tl.store(base + position * 512 + columns, payload)
    encoded = tl.clamp(exponent + 127.0, 0.0, 255.0).to(tl.uint8)
    tl.store(base + ENTRIES * 512 + position * 16 + tl.arange(0, 16), encoded)


@triton.jit
def _load_swa_row(pool, slot, block_stride, ENTRIES: tl.constexpr):
    columns = tl.arange(0, 512)
    slot64 = slot.to(tl.int64)
    block = slot64 // ENTRIES
    position = slot64 % ENTRIES
    base = pool + block * block_stride.to(tl.int64)
    payload = tl.load(base + position * 512 + columns, slot64 >= 0, other=0)
    encoded = tl.load(
        base + ENTRIES * 512 + position * 16 + tl.arange(0, 16),
        slot64 >= 0,
        other=127,
    )
    groups = payload.to(tl.float8e4nv, bitcast=True).to(tl.float32).reshape((16, 32))
    return (groups * tl.exp2(encoded.to(tl.float32) - 127.0)[:, None]).reshape((512,))


@triton.jit(do_not_specialize=["block_stride", "out_stride"])
def _dequantize_swa_kernel(
    pool, slots, out, block_stride, out_stride, ENTRIES: tl.constexpr
):
    row = tl.program_id(0).to(tl.int64)
    slot = tl.load(slots + row)
    values = _load_swa_row(pool, slot, block_stride, ENTRIES)
    tl.store(out + row * out_stride.to(tl.int64) + tl.arange(0, 512), values)


@triton.jit(
    do_not_specialize=[
        "block_stride",
        "slot_stride0",
        "slot_stride1",
        "out_stride0",
        "out_stride1",
        "offset",
    ]
)
def _gather_swa_kernel(
    pool,
    slots,
    lengths,
    out,
    block_stride,
    slot_stride0,
    slot_stride1,
    out_stride0,
    out_stride1,
    offset,
    ENTRIES: tl.constexpr,
):
    batch = tl.program_id(0).to(tl.int64)
    column = tl.program_id(1).to(tl.int64)
    if lengths is not None:
        if column >= tl.load(lengths + batch):
            return
    slot = tl.load(
        slots + batch * slot_stride0.to(tl.int64) + column * slot_stride1.to(tl.int64)
    )
    values = _load_swa_row(pool, slot, block_stride, ENTRIES)
    output = (
        out
        + batch * out_stride0.to(tl.int64)
        + (offset.to(tl.int64) + column) * out_stride1.to(tl.int64)
    )
    tl.store(output + tl.arange(0, 512), values)


def quantize_and_insert_swa_k_cache(
    kv: torch.Tensor, pool_3d: torch.Tensor, slots: torch.Tensor
) -> None:
    """Quantize ``[N, 512]`` keys into the physical slots, skipping ``-1``.

    Valid slots must be in range and unique within a launch. Ring writers
    must apply their normal ownership mask before calling this function.
    """
    _check_pool(pool_3d, slots)
    if (
        kv.device != pool_3d.device
        or kv.ndim != 2
        or kv.shape != (slots.numel(), HEAD_DIM)
        or kv.stride(1) != 1
        or kv.stride(0) < HEAD_DIM
        or kv.dtype not in (torch.bfloat16, torch.float16, torch.float32)
    ):
        raise ValueError(
            "V4.1 SWA keys must be [N, 512] floating rows on the cache device"
        )
    if slots.numel() == 0:
        return
    _quantize_and_insert_swa_kernel[(slots.numel(),)](
        kv,
        pool_3d,
        slots.reshape(-1).contiguous(),
        kv.stride(0),
        pool_3d.stride(0),
        ENTRIES=pool_3d.shape[1],
        num_warps=4,
    )


def dequantize_swa_k_cache(
    pool_3d: torch.Tensor,
    slots: torch.Tensor,
    *,
    out_dtype: torch.dtype = torch.bfloat16,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Gather to ``[slots.numel(), 512]``; masked slots produce zero rows."""
    _check_pool(pool_3d, slots)
    if out is None:
        out = torch.empty(
            (slots.numel(), HEAD_DIM), dtype=out_dtype, device=pool_3d.device
        )
    if (
        out.device != pool_3d.device
        or out.shape != (slots.numel(), HEAD_DIM)
        or out.stride(1) != 1
        or out.stride(0) < HEAD_DIM
        or out.dtype not in (torch.bfloat16, torch.float16, torch.float32)
    ):
        raise ValueError(
            "V4.1 SWA output must be [N, 512] floating rows on the cache device"
        )
    if slots.numel():
        _dequantize_swa_kernel[(slots.numel(),)](
            pool_3d,
            slots.reshape(-1).contiguous(),
            out,
            pool_3d.stride(0),
            out.stride(0),
            ENTRIES=pool_3d.shape[1],
            num_warps=4,
        )
    return out


def dequantize_and_gather_k_cache_slots(
    out: torch.Tensor,
    k_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    gather_lens: Optional[torch.Tensor],
    offset: int,
) -> None:
    """Gather ``[B, W]`` slots into strided ``out[:, offset:offset+W, :]``.

    Rows outside each request's gather length are left untouched. Lengths
    remain on device, including during CUDA graph capture and replay.
    """
    _check_pool(k_cache, slot_mapping)
    if (
        slot_mapping.ndim != 2
        or out.ndim != 3
        or out.shape[0] != slot_mapping.shape[0]
        or out.shape[2] != HEAD_DIM
        or out.stride(2) != 1
        or out.stride(1) < HEAD_DIM
        or out.stride(0) < out.shape[1] * out.stride(1)
        or out.device != k_cache.device
        or out.dtype not in (torch.bfloat16, torch.float16, torch.float32)
        or offset < 0
        or offset + slot_mapping.shape[1] > out.shape[1]
    ):
        raise ValueError("Invalid V4.1 SWA gather output/slot shape or offset")
    if gather_lens is not None and (
        gather_lens.device != out.device
        or gather_lens.dtype not in (torch.int32, torch.int64)
        or gather_lens.shape != (out.shape[0],)
        or not gather_lens.is_contiguous()
    ):
        raise ValueError(
            "SWA gather lengths must be a contiguous device integer [B] tensor"
        )
    if slot_mapping.numel():
        _gather_swa_kernel[tuple(slot_mapping.shape)](
            k_cache,
            slot_mapping,
            gather_lens,
            out,
            k_cache.stride(0),
            slot_mapping.stride(0),
            slot_mapping.stride(1),
            out.stride(0),
            out.stride(1),
            offset,
            ENTRIES=k_cache.shape[1],
            num_warps=4,
        )


def _check_cp_pool(
    raw: torch.Tensor, full_entries_per_block: int, cp_rank: int, cp_size: int
) -> None:
    if (
        not raw.is_cuda
        or raw.dtype != torch.uint8
        or raw.ndim != 2
        or raw.stride(1) != 1
        or full_entries_per_block <= 0
        or cp_size <= 0
        or not 0 <= cp_rank < cp_size
        or raw.shape[1] * cp_size < full_entries_per_block * ENTRY_BYTES
    ):
        raise ValueError("Invalid V4.1 CP SWA byte-sliced pool contract")


def quantize_and_insert_k_cache_cp_byte_sliced(
    k: torch.Tensor,
    k_cache_raw: torch.Tensor,
    slot_mapping: torch.Tensor,
    full_entries_per_block: int,
    cp_rank: int,
    cp_size: int,
    compaction: CPByteSlicedSlotCompaction,
) -> None:
    """Update this rank's byte slice while preserving other slots and padding.

    Only the local slice needs its previous bytes: the encoder computes every
    updated payload/scale byte from ``k`` independently. Bytes in other ranks'
    temporary slices are never copied back, so no all-gather is required.
    """
    _check_cp_pool(k_cache_raw, full_entries_per_block, cp_rank, cp_size)
    if slot_mapping.numel() != compaction.compact_slots.numel():
        raise ValueError("CP SWA compaction and original slots must have equal length")
    unique_blocks = compaction.unique_blocks
    if slot_mapping.numel() == 0 or unique_blocks.numel() == 0:
        return
    local_bytes = k_cache_raw.shape[1]
    full_stride = local_bytes * cp_size
    full_raw = torch.empty(
        (unique_blocks.numel(), full_stride),
        dtype=torch.uint8,
        device=k_cache_raw.device,
    )
    local_slice = full_raw[:, cp_rank * local_bytes : (cp_rank + 1) * local_bytes]
    local_slice.copy_(k_cache_raw.index_select(0, unique_blocks))
    full_view = full_raw.as_strided(
        (unique_blocks.numel(), full_entries_per_block, ENTRY_BYTES),
        (full_stride, ENTRY_BYTES, 1),
    )
    quantize_and_insert_swa_k_cache(k, full_view, compaction.compact_slots)
    k_cache_raw.index_copy_(
        0,
        unique_blocks,
        local_slice.contiguous(),
    )


def dequantize_and_gather_k_cache_slots_cp_byte_sliced(
    out: torch.Tensor,
    k_cache_raw: torch.Tensor,
    slot_mapping: torch.Tensor,
    gather_lens: Optional[torch.Tensor],
    offset: int,
    full_entries_per_block: int,
    cp_rank: int,
    cp_size: int,
    compaction: CPByteSlicedSlotCompaction,
) -> None:
    """All-gather byte slices, restore page planes, then gather on device."""
    _check_cp_pool(k_cache_raw, full_entries_per_block, cp_rank, cp_size)
    if slot_mapping.shape != compaction.compact_slots.shape:
        raise ValueError("CP SWA compaction and original slots must have equal shape")
    if slot_mapping.numel() == 0:
        return
    unique_blocks = compaction.unique_blocks
    local_bytes = k_cache_raw.shape[1]
    if unique_blocks.numel() == 0:
        full_raw = torch.empty(
            (0, local_bytes * cp_size), dtype=torch.uint8, device=k_cache_raw.device
        )
    else:
        if compaction.contiguous_block_start >= 0 and k_cache_raw.is_contiguous():
            local = k_cache_raw.narrow(
                0, compaction.contiguous_block_start, unique_blocks.numel()
            )
        else:
            local = k_cache_raw.index_select(0, unique_blocks).contiguous()
        if cp_size == 1:
            full_raw = local
        else:
            from rtp_llm.models_py.distributed.collective_torch import Group, all_gather

            gathered = all_gather(local, group=Group.TP)
            if gathered.numel() != cp_size * local.numel():
                raise RuntimeError("CP SWA all_gather size does not match cp_size")
            full_raw = (
                gathered.view(cp_size, unique_blocks.numel(), local_bytes)
                .permute(1, 0, 2)
                .reshape(unique_blocks.numel(), cp_size * local_bytes)
                .contiguous()
            )
    full_view = full_raw.as_strided(
        (unique_blocks.numel(), full_entries_per_block, ENTRY_BYTES),
        (full_raw.shape[1], ENTRY_BYTES, 1),
    )
    dequantize_and_gather_k_cache_slots(
        out, full_view, compaction.compact_slots, gather_lens, offset
    )
