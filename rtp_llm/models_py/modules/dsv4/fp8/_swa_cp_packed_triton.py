"""Byte-preserving SWA transport for the singleton SM120 CP prefill path."""

from __future__ import annotations

import os
from typing import Optional

import torch
import triton
import triton.language as tl

from rtp_llm.models_py.modules.dsv4.fp8._swa_kv_insert_triton import (
    quantize_and_insert_k_cache,
)
from rtp_llm.models_py.modules.dsv4.fp8._trap_utils import validate_slot_mapping

ENTRY_BYTES = 584
DATA_BYTES = 576


def packed_swa_cp_enabled() -> bool:
    value = os.environ.get("DSV4_CP_SWA_PACKED", "0")
    if value not in ("0", "1"):
        raise ValueError("DSV4_CP_SWA_PACKED must be 0 or 1")
    return value == "1"


@triton.jit
def _insert_packed_rows_kernel(
    rows,
    slots,
    cache,
    block_stride,
    NUM_BLOCKS: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    slot = tl.load(slots + row).to(tl.int64)
    block = slot // PAGE_SIZE
    offset = slot % PAGE_SIZE
    byte = tl.arange(0, BLOCK)
    valid = (slot >= 0) & (block < NUM_BLOCKS) & (byte < 584)
    value = tl.load(rows + row * 584 + byte, mask=valid, other=0)
    dst = tl.where(
        byte < 576,
        offset * 576 + byte,
        PAGE_SIZE * 576 + offset * 8 + byte - 576,
    )
    tl.store(cache + block * block_stride + dst, value, mask=valid)


def insert_packed_rows(
    rows: torch.Tensor, cache: torch.Tensor, slots: torch.Tensor
) -> None:
    """Scatter [N,584] rows into paged data/scales without requantization."""
    assert rows.ndim == 2 and rows.shape[1] == ENTRY_BYTES
    assert rows.dtype == torch.uint8 and rows.is_contiguous()
    assert cache.ndim == 3 and cache.shape[2] == ENTRY_BYTES
    assert cache.dtype == torch.uint8 and cache.stride(1) == ENTRY_BYTES
    assert cache.stride(2) == 1
    assert slots.ndim == 1 and slots.numel() == rows.shape[0]
    assert slots.dtype in (torch.int32, torch.int64) and slots.is_contiguous()
    assert rows.device == cache.device == slots.device and rows.is_cuda
    assert cache.stride(0) >= cache.shape[1] * ENTRY_BYTES
    validate_slot_mapping(
        "swa.cp_packed.insert",
        slots,
        block_size=int(cache.shape[1]),
        num_blocks=int(cache.shape[0]),
        negative_mode="skip_minus_one",
    )
    if rows.shape[0] == 0:
        return
    _insert_packed_rows_kernel[(rows.shape[0],)](
        rows,
        slots,
        cache,
        cache.stride(0),
        NUM_BLOCKS=cache.shape[0],
        PAGE_SIZE=cache.shape[1],
        BLOCK=1024,
        num_warps=4,
    )


def pack_local_kv(kv: torch.Tensor, workspace) -> torch.Tensor:
    """Page size one makes the existing quantizer emit transport rows directly."""
    assert kv.ndim == 2 and kv.shape[1] == 512
    assert kv.dtype == torch.bfloat16 and kv.is_contiguous()
    count = kv.shape[0]
    packed = workspace.sm120_packed_cache(
        "cp_swa_transport", num_tokens=count, page_size=1, entry_bytes=ENTRY_BYTES
    )
    quantize_and_insert_k_cache(
        kv, packed, workspace.sm120_linear_slots("cp_swa_transport", count)
    )
    return packed.view(-1, ENTRY_BYTES)[:count]


def insert_packed_rows_cp_byte_sliced(
    rows: torch.Tensor,
    cache_raw: torch.Tensor,
    *,
    full_entries_per_block: int,
    cp_rank: int,
    cp_size: int,
    compaction,
) -> None:
    """Preserve the existing full-block zero-fill and byte-slice write contract."""
    assert cache_raw.ndim == 2 and cache_raw.dtype == torch.uint8
    assert cache_raw.stride(1) == 1 and 0 <= cp_rank < cp_size
    assert compaction is not None
    blocks = compaction.unique_blocks
    if blocks.numel() == 0:
        return
    local_bytes = int(cache_raw.shape[1])
    full_bytes = local_bytes * cp_size
    assert full_bytes >= full_entries_per_block * ENTRY_BYTES
    full_raw = torch.zeros(
        (blocks.numel(), full_bytes), dtype=torch.uint8, device=rows.device
    )
    view = full_raw.as_strided(
        (blocks.numel(), full_entries_per_block, ENTRY_BYTES),
        (full_bytes, ENTRY_BYTES, 1),
    )
    insert_packed_rows(rows, view, compaction.compact_slots.reshape(-1))
    cache_raw.index_copy_(
        0,
        blocks,
        full_raw[:, cp_rank * local_bytes : (cp_rank + 1) * local_bytes].contiguous(),
    )


def build_attention_cache(
    rows: torch.Tensor,
    workspace,
    *,
    prefix: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Keep cached-prefix arithmetic; install fresh rows using their original bytes."""
    prefix_count = 0 if prefix is None else int(prefix.shape[0])
    count = prefix_count + int(rows.shape[0])
    cache = workspace.sm120_packed_cache(
        "swa", num_tokens=count, page_size=64, entry_bytes=ENTRY_BYTES
    )
    slots = workspace.sm120_linear_slots("swa", count)
    if prefix_count:
        quantize_and_insert_k_cache(prefix.contiguous(), cache, slots[:prefix_count])
    insert_packed_rows(rows, cache, slots[prefix_count:])
    return cache
