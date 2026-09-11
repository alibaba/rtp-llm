"""Lossless selected-block repack from native rows to DeepGEMM planar pages."""

import triton
import triton.language as tl


@triton.jit
def repack_index_blocks(
    source,
    physical_pages,
    candidates,
    visible_lengths,
    destination,
    CANDIDATES: tl.constexpr,
    PAGE_STRIDE: tl.constexpr,
    ENTRIES: tl.constexpr,
):
    row = tl.program_id(0)
    slot = tl.program_id(1)
    block = tl.load(candidates + row * CANDIDATES + slot)
    page = tl.load(physical_pages + row * CANDIDATES + slot)
    visible = tl.load(visible_lengths + row)
    offsets = tl.arange(0, 1024)
    payload = offsets < 512
    token = tl.where(payload, offsets // 64, (offsets - 512) // 4)
    channel = tl.where(payload, offsets % 64, 64 + (offsets - 512) % 4)
    position = block * 8 + token
    address = page.to(tl.int64) * PAGE_STRIDE + (position % ENTRIES) * 68 + channel
    value = tl.load(
        source + address,
        mask=(offsets < 544) & (block >= 0) & (page > 0) & (position < visible),
        other=0,
    )
    tl.store(destination + (row * CANDIDATES + slot) * 1024 + offsets, value)
