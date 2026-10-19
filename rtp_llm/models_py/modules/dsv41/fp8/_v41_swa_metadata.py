"""Fixed-window V4.1 suffix slots, preserving physical block-table aliases."""

import numpy as np
import torch
import triton
import triton.language as tl

from ._swa_cp_byte_sliced import CPByteSlicedSlotCompaction


def _upload(values, device, dtype):
    # PyTorch's pinned allocator records the async copy before releasing storage.
    return torch.tensor(values, dtype=dtype, pin_memory=True).to(
        device, non_blocking=True
    )


@triton.jit(
    do_not_specialize=["columns", "max_rows", "entries", "span", "ring"],
    do_not_specialize_on_alignment=["table", "prefixes", "lengths", "offsets"],
)
def _planned_slots(
    table,
    compact_table,
    prefixes,
    lengths,
    offsets,
    slots,
    compact,
    columns,
    max_rows,
    entries,
    span,
    ring,
    READ: tl.constexpr,
    BLOCK: tl.constexpr,
):
    req = tl.program_id(0)
    row = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    prefix = tl.load(prefixes + req).to(tl.int64)
    length = tl.load(lengths + req).to(tl.int64)
    if READ:
        count = tl.minimum(prefix, 127)
        pos = prefix - count + row
        keep = row < count
        address = req * max_rows + row
        store = row < max_rows
    else:
        count = length
        pos = prefix + row
        end = tl.minimum(((pos // span) + 1) * span, prefix + length)
        keep = (row < count) & (pos + ring >= end)
        address = tl.load(offsets + req).to(tl.int64) + row
        store = row < count
    column = pos // span
    valid = keep & (pos >= 0) & (column >= 0) & (column < columns)
    block = tl.load(table + req * columns + column, mask=valid, other=0).to(tl.int64)
    compact_block = tl.load(
        compact_table + req * columns + column, mask=valid, other=-1
    ).to(tl.int64)
    valid = valid & (block > 0)
    within = pos % ring
    tl.store(slots + address, tl.where(valid, block * entries + within, -1), mask=store)
    tl.store(
        compact + address,
        tl.where(valid, compact_block * entries + within, -1),
        mask=store,
    )


@triton.jit(
    do_not_specialize=["rows", "batch", "columns", "entries", "span"],
    do_not_specialize_on_alignment=[
        "table",
        "prefixes",
        "lengths",
        "requests",
        "positions",
    ],
)
def _planned_position_slots(
    table,
    compact_table,
    prefixes,
    lengths,
    requests,
    positions,
    slots,
    compact,
    rows,
    batch,
    columns,
    entries,
    span,
    BLOCK: tl.constexpr,
):
    """Draft CP rows retain the gathered rank order, including padding."""
    row = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    request = tl.load(requests + row, mask=row < rows, other=-1).to(tl.int64)
    position = tl.load(positions + row, mask=row < rows, other=-1).to(tl.int64)
    valid = (row < rows) & (request >= 0) & (request < batch)
    prefix = tl.load(prefixes + request, mask=valid, other=0).to(tl.int64)
    length = tl.load(lengths + request, mask=valid, other=0).to(tl.int64)
    seq_end = prefix + length
    column = position // span
    block_end = tl.minimum((column + 1) * span, seq_end)
    valid = (
        valid
        & (position >= prefix)
        & (position < seq_end)
        & (column >= 0)
        & (column < columns)
        & (position + entries >= block_end)
    )
    address = request * columns + column
    block = tl.load(table + address, mask=valid, other=0).to(tl.int64)
    compact_block = tl.load(compact_table + address, mask=valid, other=-1).to(tl.int64)
    valid = valid & (block > 0) & (compact_block >= 0)
    within = position % entries
    tl.store(slots + row, tl.where(valid, block * entries + within, -1), row < rows)
    tl.store(
        compact + row,
        tl.where(valid, compact_block * entries + within, -1),
        row < rows,
    )


def _host_compact_table(
    host_table, prefixes, lengths, columns, span, read, *, pin_memory=True
):
    """Build alias IDs from CPU page intervals without per-request Torch ops."""
    if not lengths:
        return [], torch.empty((0, columns), dtype=torch.int32, pin_memory=pin_memory)
    host_rows = host_table.resolve_neg().numpy()
    prefix = np.asarray(prefixes)
    length = np.asarray(lengths)
    starts = np.maximum(prefix - 127, 0) if read else prefix
    ends = prefix if read else prefix + length
    first = starts // span
    stop = np.minimum((ends + span - 1) // span, columns)
    lo, hi = int(first.min()), min(int(stop.max()), host_rows.shape[1])
    block_columns = np.arange(lo, hi)[None, :]
    blocks = host_rows[:, lo:hi]
    active = (
        (block_columns >= first[:, None])
        & (block_columns < stop[:, None])
        & (blocks > 0)
    )
    if not read:
        active &= length[:, None] > 0
    unique, inverse = np.unique(blocks[active], return_inverse=True)
    mapping = torch.empty(
        (len(lengths), columns), dtype=torch.int32, pin_memory=pin_memory
    )
    host_mapping = mapping.numpy()
    host_mapping.fill(-1)
    host_mapping[:, lo:hi][active] = inverse
    return unique.tolist(), mapping


def try_host_slot_metadata(
    table,
    host_table,
    cp,
    *,
    entries,
    span,
    num_blocks,
    read=False,
    request_ids=None,
    positions=None,
):
    """Use the native CPU mirror to fix output sizes and alias dedup before launch.

    The caller binds host/device tables from the same native request descriptor.
    SWA uses one kernel block per physical block, so
    BlockIds::updateKernelSlotAt preserves the physical IDs. The common native
    descriptor pads physical and kernel tables independently across all groups;
    their widths can differ because other groups expand physical blocks.
    Unsupported metadata retains the existing GPU validation/compaction path.
    """
    prefixes = getattr(cp, "prefix_lengths_host", None)
    lengths = getattr(cp, "input_lengths_global_host", None)
    if (
        not table.is_cuda
        or table.dtype != torch.int32
        or not table.is_contiguous()
        or not isinstance(host_table, torch.Tensor)
        or host_table.device.type != "cpu"
        or host_table.dtype not in (torch.int32, torch.int64)
        or table.ndim != 2
        or host_table.ndim != 2
        or host_table.shape[0] != table.shape[0]
        or prefixes is None
        or lengths is None
        or len(prefixes) != table.shape[0]
        or len(lengths) != len(prefixes)
        or not lengths
        or any(p < 0 or n < 0 for p, n in zip(prefixes, lengths))
        or entries <= 0
        or span <= 0
        or num_blocks <= 0
        or getattr(cp, "cp_size", 0) < 1
        or not getattr(cp, "kv_cache_sharded", False)
    ):
        return None
    positional = request_ids is not None or positions is not None
    if positional and (
        read
        or any(
            not isinstance(t, torch.Tensor)
            or t.device != table.device
            or t.ndim != 1
            or not t.is_contiguous()
            or t.dtype not in (torch.int32, torch.int64)
            for t in (request_ids, positions)
        )
        or request_ids.shape != positions.shape
    ):
        return None
    if torch.cuda.is_current_stream_capturing():
        return None
    for name, count in (
        ("prefix_lengths", len(lengths)),
        ("input_lengths_global", len(lengths)),
        ("cu_seqlens_global", len(lengths) + 1),
    ):
        tensor = getattr(cp, name, None)
        if (
            not isinstance(tensor, torch.Tensor)
            or tensor.device != table.device
            or tensor.ndim != 1
            or tensor.numel() != count
            or not tensor.is_contiguous()
            or tensor.dtype not in (torch.int32, torch.int64)
        ):
            return None
    # One CPU NumPy view avoids a Torch select/slice/tolist dispatch chain for
    # every request. Physical/kernel table widths remain independently padded.
    unique, mapping = _host_compact_table(
        host_table, prefixes, lengths, table.shape[1], span, read
    )
    gather = tuple(min(p, 127) for p in prefixes) if read else ()
    max_rows = max(gather) if read else max(lengths)
    shape = (
        request_ids.shape
        if positional
        else (len(lengths), max_rows) if read else (sum(lengths),)
    )
    slots = torch.empty(shape, device=table.device, dtype=torch.long)
    compact = torch.empty_like(slots)
    unique_device = _upload(unique, table.device, torch.long)
    if positional and slots.numel():
        lookup = mapping.to(table.device, non_blocking=True)
        _planned_position_slots[(triton.cdiv(slots.numel(), 256),)](
            table,
            lookup,
            cp.prefix_lengths,
            cp.input_lengths_global,
            request_ids,
            positions,
            slots,
            compact,
            slots.numel(),
            len(lengths),
            table.shape[1],
            entries,
            span,
            BLOCK=256,
            num_warps=4,
        )
    elif max_rows and not positional:
        lookup = mapping.to(table.device, non_blocking=True)
        _planned_slots[(len(lengths), triton.cdiv(max_rows, 256))](
            table,
            lookup,
            cp.prefix_lengths,
            cp.input_lengths_global,
            cp.cu_seqlens_global,
            slots,
            compact,
            table.shape[1],
            max_rows,
            entries,
            span,
            entries,
            READ=read,
            BLOCK=256,
            num_warps=4,
        )
    contiguous = (
        unique[0]
        if read and unique and unique[-1] - unique[0] + 1 == len(unique)
        else -1
    )
    return slots, CPByteSlicedSlotCompaction(unique_device, compact, gather, contiguous)


@triton.jit(
    do_not_specialize=["table_stride", "columns", "width", "entries", "span", "ring"]
)
def _suffix_slots(
    table,
    seq,
    lengths,
    out,
    table_stride,
    columns,
    width,
    entries,
    span,
    ring,
    BLOCK: tl.constexpr,
):
    req = tl.program_id(0)
    col = tl.arange(0, BLOCK)
    length = tl.load(lengths + req).to(tl.int64)
    end = tl.load(seq + req).to(tl.int64)
    pos = end - length + col
    logical = pos // span
    valid = (
        (col < width)
        & (col < length)
        & (pos >= 0)
        & (logical >= 0)
        & (logical < columns)
    )
    physical = tl.load(table + req * table_stride + logical, mask=valid, other=0).to(
        tl.int64
    )
    slots = physical * entries + pos % ring
    tl.store(
        out + req * width + col,
        tl.where(valid & (physical > 0), slots, -1),
        mask=col < width,
    )


def try_suffix_slots(table, seq, lengths, *, max_gather, entries, span, ring):
    batch = seq.numel()
    if (
        not table.is_cuda
        or table.ndim != 2
        or table.stride(1) != 1
        or table.dtype not in (torch.int32, torch.int64)
        or seq.device != table.device
        or lengths.device != table.device
        or seq.dtype not in (torch.int32, torch.int64)
        or lengths.dtype not in (torch.int32, torch.int64)
        or seq.ndim != 1
        or lengths.ndim != 1
        or not seq.is_contiguous()
        or not lengths.is_contiguous()
        or lengths.numel() != batch
        or table.shape[0] < batch
        or not 0 <= max_gather <= 127
        or min(entries, span, ring) <= 0
    ):
        return None
    output = torch.empty((batch, max_gather), device=table.device, dtype=torch.long)
    if batch and max_gather:
        _suffix_slots[(batch,)](
            table,
            seq,
            lengths,
            output,
            table.stride(0),
            table.shape[1],
            max_gather,
            entries,
            span,
            ring,
            BLOCK=128,
            num_warps=4,
        )
    return output
