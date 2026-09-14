"""Fused checked boundary for the pinned DeepSelect selection kernel.

The vendor call remains AOT and unchanged. These kernels only prepare its
aligned input and validate/materialize its output, keeping invalid lengths and
NaN rows fail-closed without a chain of torch pointwise/reduction launches.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _sanitize_tiles(
    values,
    value_stride0,
    value_stride1,
    ends,
    scratch,
    scratch_stride0,
    row_flags,
    columns,
    num_tiles,
    HAS_END: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    tile = tl.program_id(1)
    offsets = tile * BLOCK + tl.arange(0, BLOCK)
    end = tl.load(ends + row) if HAS_END else columns
    active = (offsets < columns) & (offsets < end)
    x = tl.load(
        values + row * value_stride0 + offsets.to(tl.int64) * value_stride1,
        mask=offsets < columns,
        other=0.0,
    )
    nan = active & (x != x)
    invalid_end = (end < 0) | (end > columns)
    tl.store(
        scratch + row * scratch_stride0 + offsets,
        tl.where(active & ~nan, x, -float("inf")),
        mask=offsets < columns,
    )
    tl.store(
        row_flags + row * num_tiles + tile,
        tl.max((nan | invalid_end).to(tl.int32), axis=0),
    )


@triton.jit
def _finish_lengths(
    row_flags, ends, safe_ends, statuses, columns, num_tiles,
    HAS_END: tl.constexpr, BLOCK: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    offsets = tl.arange(0, BLOCK)
    flags = tl.load(
        row_flags + row * num_tiles + offsets, mask=offsets < num_tiles, other=0
    )
    bad = tl.max(flags, axis=0) != 0
    end = tl.load(ends + row) if HAS_END else columns
    bad = bad | (end < 0) | (end > columns)
    tl.store(statuses + row, bad.to(tl.int32))
    tl.store(safe_ends + row, tl.where(bad, 0, end))


@triton.jit
def _validate_and_materialize(
    values,
    value_stride0,
    value_stride1,
    safe_ends,
    statuses,
    output_idx,
    output_stride0,
    selected,
    selected_stride0,
    k,
    BLOCK_K: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    offsets = tl.arange(0, BLOCK_K)
    in_output = offsets < k
    indices = tl.load(
        output_idx + row * output_stride0 + offsets, mask=in_output, other=-1
    ).to(tl.int32)
    safe_end = tl.load(safe_ends + row)
    valid = in_output & (indices >= 0) & (indices < safe_end)
    malformed = in_output & ((indices < -1) | (indices >= safe_end))
    sorted_indices = tl.sort(indices, descending=False)
    previous_offsets = tl.maximum(offsets - 1, 0)
    previous = tl.gather(sorted_indices, previous_offsets, axis=0)
    duplicate = (offsets > 0) & (sorted_indices >= 0) & (sorted_indices == previous)
    malformed = malformed | duplicate
    count = tl.sum(valid.to(tl.int32), axis=0)
    malformed = tl.max(malformed.to(tl.int32), axis=0) != 0
    malformed = malformed | (count != tl.minimum(safe_end, k))
    prior_status = tl.load(statuses + row) != 0
    invalid = prior_status | malformed
    out_indices = tl.where(invalid | ~valid, -1, indices)
    tl.store(output_idx + row * output_stride0 + offsets, out_indices, mask=in_output)
    source_indices = tl.where(valid, indices, 0)
    chosen = tl.load(
        values + row * value_stride0 + source_indices.to(tl.int64) * value_stride1,
        mask=valid & ~invalid,
        other=0.0,
    )
    tl.store(
        selected + row * selected_stride0 + offsets,
        tl.where(valid & ~invalid, chosen, -float("inf")),
        mask=in_output,
    )
    tl.store(statuses + row, invalid.to(tl.int32))


def prepare_inputs(values, ends, scratch, k, has_end):
    """Prepare aligned scores and a device-owned status/length boundary."""
    rows, columns = values.shape
    selected = torch.empty((rows, k), dtype=values.dtype, device=values.device)
    num_tiles = triton.cdiv(columns, 1024)
    row_flags = torch.empty((rows, num_tiles), dtype=torch.int32, device=values.device)
    safe_ends = torch.empty_like(ends)
    statuses = torch.empty_like(ends)
    _sanitize_tiles[(rows, num_tiles)](
        values, values.stride(0), values.stride(1), ends, scratch,
        scratch.stride(0), row_flags, columns, num_tiles, BLOCK=1024,
        HAS_END=has_end,
    )
    _finish_lengths[(rows,)](
        row_flags, ends, safe_ends, statuses, columns, num_tiles,
        BLOCK=triton.next_power_of_2(num_tiles),
        HAS_END=has_end,
    )
    return safe_ends, statuses, selected
