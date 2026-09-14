"""Masked raw state movement for framework-owned V4.1 decode pages."""

import triton
import triton.language as tl


@triton.jit
def copy_state_bytes_kernel(
    source,
    destination,
    source_ids,
    destination_ids,
    active_rows,
    status,
    SOURCE_ROWS: tl.constexpr,
    DESTINATION_ROWS: tl.constexpr,
    SOURCE_STRIDE: tl.constexpr,
    DESTINATION_STRIDE: tl.constexpr,
    SOURCE_OFFSET: tl.constexpr,
    DESTINATION_OFFSET: tl.constexpr,
    COPY_BYTES: tl.constexpr,
    WRITE_BYTES: tl.constexpr,
    SOURCE_MIN: tl.constexpr,
    DESTINATION_MIN: tl.constexpr,
    ZERO_INACTIVE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offsets = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    source_id = tl.load(source_ids + row).to(tl.int64)
    destination_id = tl.load(destination_ids + row).to(tl.int64)
    active = tl.load(active_rows + row).to(tl.int1)
    source_valid = (source_id >= SOURCE_MIN) & (source_id < SOURCE_ROWS)
    destination_valid = (destination_id >= DESTINATION_MIN) & (
        destination_id < DESTINATION_ROWS
    )
    valid = active & source_valid & destination_valid
    values = tl.load(
        source + source_id * SOURCE_STRIDE + SOURCE_OFFSET + offsets,
        mask=valid & (offsets < COPY_BYTES),
        other=0,
    )
    write = valid
    if ZERO_INACTIVE:
        write |= ~active & destination_valid
    tl.store(
        destination + destination_id * DESTINATION_STRIDE + DESTINATION_OFFSET + offsets,
        values,
        mask=write & (offsets < WRITE_BYTES),
    )
    if tl.program_id(1) == 0:
        tl.store(status + row, (active & ~valid).to(tl.int32))
