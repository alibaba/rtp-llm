"""Address-only fusions for the native DSpARK Verify journal.

The journal owns one raw generation per query row. These kernels retain the
native allocation check and the original request/generation selection without
changing cache bytes, codecs, or publication. Unsupported layouts use the
caller's existing implementation.
"""

from functools import lru_cache

import torch


def _raw_supported(raw, table, span, entries, ratio, full, pool_blocks):
    return (
        isinstance(raw, torch.Tensor)
        and isinstance(table, torch.Tensor)
        and raw.is_cuda
        and raw.ndim >= 1
        and raw.shape[0] > 0
        and raw.dtype in (torch.int32, torch.int64)
        and raw.is_contiguous()
        and table.device == raw.device
        and table.ndim == 2
        and table.shape[0] >= raw.shape[0]
        and table.shape[1] > 0
        and table.dtype in (torch.int32, torch.int64)
        and min(table.stride()) > 0
        and type(span) is int and span > 0
        and type(entries) is int and entries > 0
        and type(ratio) is int and ratio in (1, 2)
        and type(full) is bool
        and type(pool_blocks) is int and pool_blocks > 0
    )


@lru_cache(maxsize=1)
def _position_kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def prepare_verify_positions(
        positions, valid, entry_positions, commit, current_raw, previous_raw,
        ROWS: tl.constexpr, BLOCK: tl.constexpr,
    ):
        row = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        position = tl.load(positions + row, row < ROWS, other=0)
        active = tl.load(valid + row, row < ROWS, other=False)
        # Keep arithmetic in the input integer dtype, including its wraparound
        # at the representable minimum. Allocation validation remains later.
        entry = position - 1
        publish = active & ((position & 1) == 1)
        tl.store(entry_positions + row, entry, row < ROWS)
        tl.store(commit + row, publish, row < ROWS)
        tl.store(current_raw + row, tl.where(active, position, -1), row < ROWS)
        tl.store(previous_raw + row, tl.where(publish, entry, -1), row < ROWS)

    return triton, prepare_verify_positions


def try_prepare_verify_positions(positions, valid, *, ratio=2):
    """Produce ratio-2 Verify addressing metadata in one fixed-output launch.

    Returns ``(entry_positions, commit, current_raw, previous_raw)``. The last
    tensor serves both previous-CSA reads and global/indexer publication. Every
    output owns its storage, with integer semantics identical to the original
    expressions; in particular this function does not clamp negative inputs.
    """
    if not (
        ratio == 2 and type(ratio) is int
        and isinstance(positions, torch.Tensor)
        and isinstance(valid, torch.Tensor)
        and positions.is_cuda
        and positions.ndim == 1
        and positions.dtype in (torch.int32, torch.int64)
        and positions.is_contiguous()
        and valid.device == positions.device
        and valid.shape == positions.shape
        and valid.dtype == torch.bool
        and valid.is_contiguous()
    ):
        return None
    entry_positions = torch.empty_like(positions)
    commit = torch.empty_like(valid)
    current_raw, previous_raw = torch.empty_like(positions), torch.empty_like(positions)
    if positions.numel():
        triton, kernel = _position_kernel()
        kernel[(triton.cdiv(positions.numel(), 128),)](
            positions, valid, entry_positions, commit, current_raw, previous_raw,
            positions.numel(), 128, num_warps=4,
        )
    return entry_positions, commit, current_raw, previous_raw


@lru_cache(maxsize=1)
def _kernels():
    import triton
    import triton.language as tl

    @triton.jit
    def native_slot(
        raw, row, table, COLUMNS: tl.constexpr,
        TABLE_ROW: tl.constexpr, TABLE_COL: tl.constexpr,
        SPAN: tl.constexpr, ENTRIES: tl.constexpr, RATIO: tl.constexpr,
        FULL: tl.constexpr, POOL_BLOCKS: tl.constexpr,
    ):
        logical = raw // SPAN
        present = (raw >= 0) & (logical < COLUMNS)
        block = tl.load(
            table + row * TABLE_ROW + logical * TABLE_COL, present, other=0
        ).to(tl.int64)
        if FULL:
            offset = (raw % SPAN) // RATIO
        else:
            offset = raw % ENTRIES
        live = present & (block > 0) & (block < POOL_BLOCKS) & (offset < ENTRIES)
        return tl.where(live, block * ENTRIES + offset, -1)

    @triton.jit
    def journal_slots(
        raw, table, output, COUNT: tl.constexpr, PER_ROW: tl.constexpr,
        COLUMNS: tl.constexpr, TABLE_ROW: tl.constexpr, TABLE_COL: tl.constexpr,
        SPAN: tl.constexpr, ENTRIES: tl.constexpr, RATIO: tl.constexpr,
        FULL: tl.constexpr, POOL_BLOCKS: tl.constexpr, BLOCK: tl.constexpr,
    ):
        index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        position = tl.load(raw + index, index < COUNT, other=-1).to(tl.int64)
        slot = native_slot(
            position, index // PER_ROW, table, COLUMNS, TABLE_ROW, TABLE_COL,
            SPAN, ENTRIES, RATIO, FULL, POOL_BLOCKS,
        )
        tl.store(output + index, tl.where(slot >= 0, position, -1), index < COUNT)

    @triton.jit
    def read_journal_rows(
        raw, table, pool, payloads, generations,
        requests, starts, prefixes, valid, output,
        ROWS: tl.constexpr, PER_ROW: tl.constexpr, REQUESTS: tl.constexpr,
        WIDTH: tl.constexpr, COLUMNS: tl.constexpr,
        TABLE_ROW: tl.constexpr, TABLE_COL: tl.constexpr,
        POOL_ROW: tl.constexpr, PAYLOAD_ROW: tl.constexpr,
        SPAN: tl.constexpr, ENTRIES: tl.constexpr, RATIO: tl.constexpr,
        FULL: tl.constexpr, PLANAR: tl.constexpr, FLOAT: tl.constexpr,
        POOL_BLOCKS: tl.constexpr, LAG: tl.constexpr, BLOCK: tl.constexpr,
    ):
        index = tl.program_id(0).to(tl.int64)
        row = index // PER_ROW
        column = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
        position = tl.load(raw + index).to(tl.int64)
        slot = native_slot(
            position, row, table, COLUMNS, TABLE_ROW, TABLE_COL,
            SPAN, ENTRIES, RATIO, FULL, POOL_BLOCKS,
        )
        block = slot // ENTRIES
        offset = slot % ENTRIES
        if PLANAR:
            physical = tl.where(
                column < 64, offset * 64 + column,
                ENTRIES * 64 + offset * 4 + column - 64,
            )
        else:
            physical = offset * WIDTH + column
        native = tl.load(
            pool + block * POOL_ROW + physical,
            (column < WIDTH) & (slot >= 0), other=0,
        )
        request = tl.load(requests + row).to(tl.int64)
        request_live = (request >= 0) & (request < REQUESTS)
        start = tl.load(starts + request, request_live, other=0).to(tl.int64)
        prefix = tl.load(prefixes + request, request_live, other=0).to(tl.int64)
        relative = position + LAG - prefix
        candidate = tl.minimum(tl.maximum(start + relative, 0), ROWS - 1)
        live = (
            request_live & (position >= 0) & (relative >= 0)
            & tl.load(valid + candidate)
            & (tl.load(requests + candidate).to(tl.int64) == request)
            & (tl.load(generations + candidate).to(tl.int64) == position)
        )
        tentative = tl.load(
            payloads + candidate * PAYLOAD_ROW + column,
            (column < WIDTH) & live, other=0,
        )
        if FLOAT:
            # Select bits, including NaN payloads and signed zero.
            bits = tl.where(
                live, tentative.to(tl.uint32, bitcast=True),
                native.to(tl.uint32, bitcast=True),
            )
            value = bits.to(tl.float32, bitcast=True)
        else:
            value = tl.where(live, tentative, native)
        tl.store(output + index * WIDTH + column, value, column < WIDTH)

    return triton, journal_slots, read_journal_rows


def try_journal_slots(raw, table, *, span, entries, ratio, full, pool_blocks):
    """Return allocated raw generations, preserving the caller's integer dtype."""
    if not _raw_supported(raw, table, span, entries, ratio, full, pool_blocks):
        return None
    result = torch.empty_like(raw)
    if raw.numel():
        triton, kernel, _ = _kernels()
        kernel[(triton.cdiv(raw.numel(), 128),)](
            raw, table, result, raw.numel(), raw.numel() // raw.shape[0],
            table.shape[1], *table.stride(), span, entries, ratio, full,
            pool_blocks, 128, num_warps=4,
        )
    return result


def try_read_journal_rows(
    raw, *, row_tables, pool, payloads, generations, requests, starts,
    prefixes, valid, span, entries, ratio, full, planar, width,
):
    """Fuse native packed read with the original journal-generation override.

    ``requests`` must contain the already validated request IDs produced by the
    Verify row mapper. Neither storage nor metadata is modified. In particular,
    a matching journal generation overrides a missing native page exactly as in
    VerifyGraphJournal.read_rows; allocation filtering belongs to ``slots``.
    """
    if not isinstance(pool, torch.Tensor) or pool.ndim != 2 or not _raw_supported(
        raw, row_tables, span, entries, ratio, full, pool.shape[0]
    ):
        return None
    if not all(isinstance(t, torch.Tensor) for t in (
        payloads, generations, requests, starts, prefixes, valid,
    )):
        return None
    rows = raw.shape[0]
    if not (
        type(width) is int and width > 0
        and type(planar) is bool
        and (not planar or width == 68)
        and pool.ndim == payloads.ndim == 2
        and pool.dtype in (torch.uint8, torch.float32)
        and payloads.dtype == pool.dtype
        and payloads.shape == (rows, width)
        and pool.stride(1) == payloads.stride(1) == 1
        and pool.stride(0) >= entries * width
        and payloads.stride(0) >= width
        and generations.shape == requests.shape == valid.shape == (rows,)
        and requests.dtype in (torch.int32, torch.int64)
        and generations.dtype in (torch.int32, torch.int64)
        and valid.dtype == torch.bool
        and starts.ndim == prefixes.ndim == 1
        and starts.numel() == prefixes.numel() > 0
        and starts.dtype in (torch.int32, torch.int64)
        and prefixes.dtype in (torch.int32, torch.int64)
        and all(t.device == raw.device and t.is_contiguous()
                for t in (requests, generations, valid, starts, prefixes))
        and pool.device == payloads.device == raw.device
        and not (pool.requires_grad or payloads.requires_grad)
    ):
        return None
    output = torch.empty((*raw.shape, width), dtype=pool.dtype, device=raw.device)
    if raw.numel():
        triton, _, kernel = _kernels()
        kernel[(raw.numel(), triton.cdiv(width, 128))](
            raw, row_tables, pool, payloads, generations, requests, starts,
            prefixes, valid, output, rows, raw.numel() // rows, starts.numel(),
            width, row_tables.shape[1], *row_tables.stride(), pool.stride(0),
            payloads.stride(0), span, entries, ratio, full, planar,
            pool.dtype == torch.float32, pool.shape[0], int(full and ratio == 2),
            128, num_warps=4,
        )
    return output


@lru_cache(maxsize=1)
def _indexer_entry_kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def read_verify_indexer_entries(
        logical_entries, positions, query_valid, table, pool,
        payloads, generations, requests, starts, prefixes, valid,
        output, effective_valid,
        ROWS: tl.constexpr, WIDTH: tl.constexpr,
        ENTRY_ROW: tl.constexpr, ENTRY_COL: tl.constexpr,
        TABLE_COLUMNS: tl.constexpr, TABLE_ROW: tl.constexpr, TABLE_COL: tl.constexpr,
        POOL_ROW: tl.constexpr, PAYLOAD_ROW: tl.constexpr,
        REQUESTS: tl.constexpr, SPAN: tl.constexpr,
        PHYSICAL_ENTRIES: tl.constexpr, POOL_BLOCKS: tl.constexpr,
        RATIO: tl.constexpr, BLOCK: tl.constexpr,
    ):
        index = tl.program_id(0).to(tl.int64)
        row, entry_column = index // WIDTH, index % WIDTH
        column = tl.arange(0, BLOCK)
        entry = tl.load(logical_entries + row * ENTRY_ROW + entry_column * ENTRY_COL)
        position = tl.load(positions + row)
        end = position + 1
        # Arithmetic right-shift is floor division by two, including negative
        # values and the original int64 wraparound of position + 1.
        visible = end if RATIO == 1 else end >> 1
        requested = tl.load(query_valid + row) & (entry < visible)
        raw = tl.where(requested, entry * RATIO, -1)
        logical_block = raw // SPAN
        present = (raw >= 0) & (logical_block < TABLE_COLUMNS)
        physical_block = tl.load(
            table + row * TABLE_ROW + logical_block * TABLE_COL,
            present, other=0,
        ).to(tl.int64)
        offset = (raw % SPAN) // RATIO
        allocated = (
            present & (physical_block > 0) & (physical_block < POOL_BLOCKS)
            & (offset < PHYSICAL_ENTRIES)
        )
        # Preserve the public slots -> read_rows sequence. A journal entry
        # cannot bypass a failed native allocation check in this producer.
        journal_raw = tl.where(allocated, raw, -1)
        request = tl.load(requests + row).to(tl.int64)
        request_live = (request >= 0) & (request < REQUESTS)
        start = tl.load(starts + request, request_live, other=0).to(tl.int64)
        prefix = tl.load(prefixes + request, request_live, other=0).to(tl.int64)
        relative = journal_raw + (RATIO - 1) - prefix
        candidate = tl.minimum(tl.maximum(start + relative, 0), ROWS - 1)
        tentative = (
            request_live & (journal_raw >= 0) & (relative >= 0)
            & tl.load(valid + candidate)
            & (tl.load(requests + candidate).to(tl.int64) == request)
            & (tl.load(generations + candidate).to(tl.int64) == journal_raw)
        )
        physical = tl.where(
            column < 64, offset * 64 + column,
            PHYSICAL_ENTRIES * 64 + offset * 4 + column - 64,
        )
        committed_value = tl.load(
            pool + physical_block * POOL_ROW + physical,
            (column < 68) & allocated & ~tentative, other=0,
        )
        tentative_value = tl.load(
            payloads + candidate * PAYLOAD_ROW + column,
            (column < 68) & tentative, other=0,
        )
        tl.store(
            output + index * 68 + column,
            tl.where(tentative, tentative_value, committed_value), column < 68,
        )
        tl.store(effective_valid + index, requested & (journal_raw >= 0))

    return triton, read_verify_indexer_entries


def try_read_verify_indexer_entries(
    entries, positions, valid_mask, *, row_tables, pool, payloads, generations,
    requests, starts, prefixes, valid, span, physical_entries, ratio,
):
    """Read Verify's explicit logical Indexer entries without Torch metadata.

    Returns contiguous ``(packed68, effective_valid)``. Logical entries may be
    one-dimensional, a single broadcast row, or one row per query, including
    expanded/noncontiguous nonnegative strides. The original native allocation
    check precedes journal selection; no floating-point operation is changed.
    """
    if not all(isinstance(t, torch.Tensor) for t in (
        entries, positions, valid_mask, row_tables, pool, payloads, generations,
        requests, starts, prefixes, valid,
    )):
        return None
    if not (
        positions.is_cuda and positions.ndim == 1 and positions.numel() > 0
        and positions.dtype == torch.int64 and positions.is_contiguous()
        and valid_mask.dtype == torch.bool and valid_mask.shape == positions.shape
        and valid_mask.is_contiguous() and valid_mask.device == positions.device
        and entries.layout == torch.strided and entries.ndim in (1, 2)
        and entries.dtype == torch.int64 and entries.device == positions.device
        and 0 <= entries.shape[-1] <= 4 and min(entries.stride()) >= 0
        and (entries.ndim == 1 or entries.shape[0] in (1, positions.numel()))
        and pool.ndim == 2 and pool.shape[0] > 0
        and _raw_supported(
            positions, row_tables, span, physical_entries, ratio, True, pool.shape[0]
        )
    ):
        return None
    rows, width = positions.numel(), entries.shape[-1]
    if not (
        pool.dtype == payloads.dtype == torch.uint8
        and payloads.shape == (rows, 68)
        and pool.stride(1) == payloads.stride(1) == 1
        and pool.stride(0) >= physical_entries * 68 and payloads.stride(0) >= 68
        and requests.shape == generations.shape == valid.shape == (rows,)
        and requests.dtype in (torch.int32, torch.int64)
        and generations.dtype in (torch.int32, torch.int64)
        and valid.dtype == torch.bool
        and starts.ndim == prefixes.ndim == 1
        and starts.numel() == prefixes.numel() > 0
        and starts.dtype in (torch.int32, torch.int64)
        and prefixes.dtype in (torch.int32, torch.int64)
        and pool.device == payloads.device == positions.device
        and all(t.device == positions.device and t.is_contiguous()
                for t in (requests, generations, valid, starts, prefixes))
    ):
        return None
    output = torch.empty((rows, width, 68), dtype=torch.uint8, device=positions.device)
    effective_valid = torch.empty((rows, width), dtype=torch.bool, device=positions.device)
    if width:
        row_stride = entries.stride(0) if entries.ndim == 2 and entries.shape[0] != 1 else 0
        _, kernel = _indexer_entry_kernel()
        kernel[(rows * width,)](
            entries, positions, valid_mask, row_tables, pool, payloads, generations,
            requests, starts, prefixes, valid, output, effective_valid,
            rows, width, row_stride, entries.stride(-1), row_tables.shape[1],
            *row_tables.stride(), pool.stride(0), payloads.stride(0), starts.numel(),
            span, physical_entries, pool.shape[0], ratio, 128, num_warps=4,
        )
    return output, effective_valid
