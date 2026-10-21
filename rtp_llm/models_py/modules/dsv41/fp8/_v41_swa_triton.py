# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""V4.1 SWA MXFP8 codec, compatible with FlashMLA's ``ModelType::V41``.

Adapted from vLLM's deepseek_v41/common/ops/cache_utils.py. All 512
channels, including RoPE, use group-32 E4M3 with UE8M0 scales. Each token
contains 512 payload bytes immediately followed by 16 scale bytes, matching
FlashMLA. A page allocation may include trailing TMA padding.

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
    from rtp_llm.models_py.modules.dsv41.fp8._swa_cp_byte_sliced import (
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


def _valid_keys(kv, device, rows):
    return (
        kv.device == device
        and kv.ndim == 2
        and kv.shape == (rows, HEAD_DIM)
        and kv.stride(1) == 1
        and kv.stride(0) >= HEAD_DIM
        and kv.dtype in (torch.bfloat16, torch.float16, torch.float32)
    )


@triton.jit(do_not_specialize=["block_stride", "input_stride", "rank", "local_bytes"])
def _quantize_and_insert_swa_kernel(
    kv,
    pool,
    slots,
    input_stride,
    block_stride,
    ENTRIES: tl.constexpr,
    unique_blocks=None,
    rank=0,
    local_bytes=0,
    CP_BYTES: tl.constexpr = False,
    fresh_out=None,
    fresh_slots=None,
    STORE_FRESH: tl.constexpr = False,
):
    row = tl.program_id(0).to(tl.int64)
    columns = tl.arange(0, 512)
    if STORE_FRESH:
        # Fresh attention rows remain valid even when ring ownership skips KV.
        original = tl.load(kv + row * input_stride.to(tl.int64) + columns)
        destination = tl.load(fresh_slots + row).to(tl.int64)
        tl.store(
            fresh_out.to(tl.pointer_type(tl.uint16)) + destination * 512 + columns,
            original.to(tl.uint16, bitcast=True),
        )
    slot = tl.load(slots + row).to(tl.int64)
    if slot < 0:
        return
    block = slot // ENTRIES
    position = slot % ENTRIES
    if CP_BYTES:
        block = tl.load(unique_blocks + block).to(tl.int64)
    base = pool + block * block_stride.to(tl.int64)
    if STORE_FRESH:
        values = original.to(tl.float32)
    else:
        values = tl.load(kv + row * input_stride.to(tl.int64) + columns).to(tl.float32)
    groups = values.reshape((16, 32))
    maximum = tl.maximum(tl.max(tl.abs(groups), 1), 1e-4)
    exponent = tl.ceil(tl.log2(maximum * (1.0 / 448.0)))
    scaled = tl.clamp(groups * tl.exp2(-exponent)[:, None], -448.0, 448.0)
    payload = scaled.to(tl.float8e4nv).to(tl.uint8, bitcast=True).reshape((512,))
    payload_byte = position * 528 + columns
    if CP_BYTES:
        begin = rank.to(tl.int64) * local_bytes
        tl.store(
            base + payload_byte - begin,
            payload,
            (payload_byte >= begin) & (payload_byte < begin + local_bytes),
        )
    else:
        tl.store(base + payload_byte, payload)
    encoded = tl.clamp(exponent + 127.0, 0.0, 255.0).to(tl.uint8)
    scale_byte = position * 528 + 512 + tl.arange(0, 16)
    if CP_BYTES:
        tl.store(
            base + scale_byte - begin,
            encoded,
            (scale_byte >= begin) & (scale_byte < begin + local_bytes),
        )
    else:
        tl.store(base + scale_byte, encoded)


@triton.jit
def _load_swa_row(
    pool,
    slot,
    block_stride,
    ENTRIES: tl.constexpr,
    compact_blocks=0,
    local_bytes=0,
    RANK_MAJOR: tl.constexpr = False,
):
    columns = tl.arange(0, 512)
    slot64 = slot.to(tl.int64)
    block = slot64 // ENTRIES
    position = slot64 % ENTRIES
    payload_byte = position * 528 + columns
    scale_byte = position * 528 + 512 + tl.arange(0, 16)
    if RANK_MAJOR:
        payload_address = (
            (payload_byte // local_bytes) * compact_blocks + block
        ) * local_bytes + payload_byte % local_bytes
        scale_address = (
            (scale_byte // local_bytes) * compact_blocks + block
        ) * local_bytes + scale_byte % local_bytes
    else:
        payload_address = block * block_stride.to(tl.int64) + payload_byte
        scale_address = block * block_stride.to(tl.int64) + scale_byte
    payload = tl.load(pool + payload_address, slot64 >= 0, other=0)
    encoded = tl.load(
        pool + scale_address,
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
        "compact_blocks",
        "local_bytes",
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
    compact_blocks=0,
    local_bytes=0,
    RANK_MAJOR: tl.constexpr = False,
):
    batch = tl.program_id(0).to(tl.int64)
    column = tl.program_id(1).to(tl.int64)
    if lengths is not None:
        if column >= tl.load(lengths + batch):
            return
    slot = tl.load(
        slots + batch * slot_stride0.to(tl.int64) + column * slot_stride1.to(tl.int64)
    )
    values = _load_swa_row(
        pool, slot, block_stride, ENTRIES, compact_blocks, local_bytes, RANK_MAJOR
    )
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
    if out is None:
        out = torch.empty(
            (slots.numel(), HEAD_DIM), dtype=out_dtype, device=pool_3d.device
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


def _gather_swa_rank_major(
    out, gathered, slots, lengths, offset, compact_blocks, local_bytes, entries
):
    """Decode private or all-gathered CP4 bytes; this helper has no collective."""
    if slots.numel():
        _gather_swa_kernel[tuple(slots.shape)](
            gathered,
            slots,
            lengths,
            out,
            local_bytes * (gathered.shape[0] // compact_blocks),
            slots.stride(0),
            slots.stride(1),
            out.stride(0),
            out.stride(1),
            offset,
            ENTRIES=entries,
            compact_blocks=compact_blocks,
            local_bytes=local_bytes,
            RANK_MAJOR=True,
            num_warps=4,
        )


def _cp_addressing_supported(raw, slots, unique, entries, cp_size):
    return (
        cp_size >= 1
        and entries > 0
        and raw.shape[1] * cp_size >= entries * ENTRY_BYTES
        and raw.stride(0) >= raw.shape[1]
        and slots.device == unique.device == raw.device
        and slots.dtype in (torch.int32, torch.int64)
        and slots.is_contiguous()
        and unique.dtype in (torch.int32, torch.int64)
        and unique.ndim == 1
        and unique.is_contiguous()
    )


def is_supported_fresh_store(
    k, raw, slots, full_entries_per_block, cp_rank, cp_size, compaction, fresh_slots
):
    """Qualify a BF16 fresh scatter without reading device slot values.

    Fresh slots must be unique, in-range workspace row indices from the same
    full-token planner as ``k``. Negative cache slots do not suppress scatter.
    """
    if compaction is None or fresh_slots is None:
        return False
    try:
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
            return False
    except (ValueError, AttributeError):
        return False
    return (
        _cp_addressing_supported(
            raw,
            compaction.compact_slots,
            compaction.unique_blocks,
            full_entries_per_block,
            cp_size,
        )
        and _valid_keys(k, raw.device, slots.numel())
        and k.dtype == torch.bfloat16
        and compaction.compact_slots.numel() == k.shape[0]
        and fresh_slots.device == raw.device
        and fresh_slots.dtype in (torch.int32, torch.int64)
        and fresh_slots.shape == (k.shape[0],)
        and fresh_slots.is_contiguous()
        and all(
            t.untyped_storage().data_ptr() != raw.untyped_storage().data_ptr()
            for t in (
                k,
                compaction.compact_slots,
                compaction.unique_blocks,
                fresh_slots,
            )
        )
    )


def quantize_and_insert_k_cache_cp_byte_sliced(
    k: torch.Tensor,
    k_cache_raw: torch.Tensor,
    slot_mapping: torch.Tensor,
    full_entries_per_block: int,
    cp_rank: int,
    cp_size: int,
    compaction: CPByteSlicedSlotCompaction,
    *,
    fresh_out: Optional[torch.Tensor] = None,
    fresh_slots: Optional[torch.Tensor] = None,
) -> None:
    """Update this rank's byte slice while preserving other slots and padding.

    Qualified CP4 layouts store only this rank's bytes directly. Other layouts
    retain the local-slice staging path; no writer all-gather is required.
    Optional fresh output is a disjoint contiguous BF16 [B, M, 512] workspace.
    Only fresh_slots rows are written, with original BF16 bits preserved.
    """
    unique_blocks = compaction.unique_blocks
    if slot_mapping.numel() == 0 or (unique_blocks.numel() == 0 and fresh_out is None):
        return
    local_bytes = k_cache_raw.shape[1]
    if (
        _cp_addressing_supported(
            k_cache_raw,
            compaction.compact_slots,
            unique_blocks,
            full_entries_per_block,
            cp_size,
        )
        and _valid_keys(k, k_cache_raw.device, slot_mapping.numel())
        and all(
            t.untyped_storage().data_ptr() != k_cache_raw.untyped_storage().data_ptr()
            for t in (k, compaction.compact_slots, unique_blocks)
        )
    ):
        _quantize_and_insert_swa_kernel[(slot_mapping.numel(),)](
            k,
            k_cache_raw,
            compaction.compact_slots,
            k.stride(0),
            k_cache_raw.stride(0),
            ENTRIES=full_entries_per_block,
            unique_blocks=unique_blocks,
            rank=cp_rank,
            local_bytes=local_bytes,
            CP_BYTES=True,
            fresh_out=fresh_out,
            fresh_slots=fresh_slots,
            STORE_FRESH=fresh_out is not None,
            num_warps=4,
        )
        return
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


def _try_all_gather_cp4_bytes(local: torch.Tensor) -> Optional[torch.Tensor]:
    """Use an uninitialized receive buffer only for the ordinary NCCL route."""
    if (
        not local.is_cuda
        or local.dtype != torch.uint8
        or local.ndim != 2
        or not local.is_contiguous()
        or local.numel() == 0
        or torch.version.hip is not None
        or torch.cuda.is_current_stream_capturing()
        or not torch.distributed.is_initialized()
    ):
        return None
    from rtp_llm.models_py.distributed import collective_torch as collective

    symm = collective._get_symm_mem().get_symm_mem_communicator()
    if symm is not None and symm.should_torch_symm_mem_allgather(local):
        return None
    group = collective._get_group(collective.Group.TP)
    if torch.distributed.get_backend(group) != "nccl":
        return None
    gathered = torch.empty(
        (torch.distributed.get_world_size(group) * local.shape[0], local.shape[1]),
        device=local.device,
        dtype=local.dtype,
    )
    # The same collective overwrites every receive byte, on the caller's stream.
    torch.distributed.all_gather_into_tensor(gathered, local, group=group)
    return gathered


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
    """All-gather byte slices and decode qualified rank-major storage directly."""
    if slot_mapping.numel() == 0:
        return
    unique_blocks = compaction.unique_blocks
    local_bytes = k_cache_raw.shape[1]
    rank_major = _cp_addressing_supported(
        k_cache_raw,
        compaction.compact_slots,
        unique_blocks,
        full_entries_per_block,
        cp_size,
    )
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

            gathered = _try_all_gather_cp4_bytes(local) if rank_major else None
            if gathered is None:
                gathered = all_gather(local, group=Group.TP)
            if rank_major and gathered.is_contiguous():
                _gather_swa_rank_major(
                    out,
                    gathered,
                    compaction.compact_slots,
                    gather_lens,
                    offset,
                    unique_blocks.numel(),
                    local_bytes,
                    full_entries_per_block,
                )
                return
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
