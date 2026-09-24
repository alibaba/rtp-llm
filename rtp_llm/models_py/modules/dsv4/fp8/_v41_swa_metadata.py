"""Fixed-window V4.1 suffix slots, preserving physical block-table aliases."""

import torch
import triton
import triton.language as tl

from ._swa_cp_byte_sliced import CPByteSlicedSlotCompaction


def _upload(values, device, dtype):
    # PyTorch's pinned allocator records the async copy before releasing storage.
    return torch.tensor(values, dtype=dtype, pin_memory=True).to(
        device, non_blocking=True
    )


@triton.jit(do_not_specialize=["columns", "max_rows", "entries", "span", "ring"])
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


def try_host_slot_metadata(
    table, host_table, cp, *, entries, span, num_blocks, read=False
):
    """Use the native CPU mirror to fix output sizes and alias dedup before launch.

    The caller binds host/device tables from the same native request descriptor.
    Equal shapes reject native kernel-block expansion; with one kernel block per
    physical block, BlockIds::updateKernelSlotAt preserves the physical IDs.
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
        or host_table.shape != table.shape
        or prefixes is None
        or lengths is None
        or len(prefixes) != table.shape[0]
        or len(lengths) != len(prefixes)
        or not lengths
        or any(p < 0 or n <= 0 for p, n in zip(prefixes, lengths))
        or entries <= 0
        or span <= 0
        or num_blocks <= 0
        or getattr(cp, "cp_size", 0) != 4
        or not getattr(cp, "kv_cache_sharded", False)
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
    active = []
    for req, (prefix, length) in enumerate(zip(prefixes, lengths)):
        start, end = (
            (max(0, prefix - 127), prefix) if read else (prefix, prefix + length)
        )
        first = start // span
        stop = min((end + span - 1) // span, table.shape[1])
        # Long reused prefixes can leave hundreds of irrelevant padded columns.
        # Inspect only the request interval; the host tensor never touches CUDA.
        for column, block in enumerate(host_table[req, first:stop].tolist(), first):
            block = int(block)
            if block > 0:
                if block >= num_blocks:
                    raise ValueError("SWA host block id exceeds physical pool")
                active.append((req, column, block))
    unique = sorted({block for _, _, block in active})
    indices = {block: i for i, block in enumerate(unique)}
    mapping = torch.full(host_table.shape, -1, dtype=torch.int32, pin_memory=True)
    if active:
        flat_rows = torch.tensor(
            [req * table.shape[1] + column for req, column, _ in active]
        )
        mapping.view(-1)[flat_rows] = torch.tensor(
            [indices[block] for _, _, block in active], dtype=torch.int32
        )
    gather = tuple(min(p, 127) for p in prefixes) if read else ()
    max_rows = max(gather) if read else max(lengths)
    shape = (len(lengths), max_rows) if read else (sum(lengths),)
    slots = torch.empty(shape, device=table.device, dtype=torch.long)
    compact = torch.empty_like(slots)
    unique_device = _upload(unique, table.device, torch.long)
    if max_rows:
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
