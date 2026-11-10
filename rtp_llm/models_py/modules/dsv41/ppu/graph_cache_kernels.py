"""Fixed-shape device addressing for the native V4.1 packed cache pools.

The Triton launch geometry depends only on captured tensor shapes. Negative
slots are padding and never read or write the reserved native block zero.
The CPU implementations are format oracles for contract tests.
"""

from __future__ import annotations

import weakref
from contextlib import contextmanager
from functools import lru_cache
from threading import Lock
from types import SimpleNamespace

import torch


@lru_cache(maxsize=1)
def _verify_table_check():
    import triton
    import triton.language as tl

    @triton.jit
    def verify_table_tile_mismatch(
        captured, live, tile, BATCH: tl.constexpr, COLUMNS: tl.constexpr,
        CAPTURED_STRIDE_ROW: tl.constexpr, CAPTURED_STRIDE_COL: tl.constexpr,
        LIVE_STRIDE_ROW: tl.constexpr, LIVE_STRIDE_COL: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        index = tile * BLOCK + tl.arange(0, BLOCK)
        row, column = index // COLUMNS, index % COLUMNS
        valid = row < BATCH
        original = tl.load(
            captured + row * CAPTURED_STRIDE_ROW + column * CAPTURED_STRIDE_COL,
            valid, other=0,
        )
        current = tl.load(
            live + row * LIVE_STRIDE_ROW + column * LIVE_STRIDE_COL,
            valid, other=0,
        )
        return tl.max((original != current).to(tl.int32), axis=0)

    @triton.jit
    def verify_table_mismatch(
        captured, live, flags, BATCH: tl.constexpr, COLUMNS: tl.constexpr,
        CAPTURED_STRIDE_ROW: tl.constexpr, CAPTURED_STRIDE_COL: tl.constexpr,
        LIVE_STRIDE_ROW: tl.constexpr, LIVE_STRIDE_COL: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        tile = tl.program_id(0)
        mismatch = verify_table_tile_mismatch(
            captured, live, tile, BATCH, COLUMNS, CAPTURED_STRIDE_ROW,
            CAPTURED_STRIDE_COL, LIVE_STRIDE_ROW, LIVE_STRIDE_COL, BLOCK,
        )
        tl.store(flags + tile, mismatch)

    @triton.jit
    def verify_tables_mismatch(
        captured, live, flags, BATCH: tl.constexpr, BLOCK: tl.constexpr,
        GEOMETRY: tl.constexpr, OFFSETS: tl.constexpr,
    ):
        tile = tl.program_id(0)
        for table in tl.static_range(len(GEOMETRY)):
            if tile >= OFFSETS[table] and tile < OFFSETS[table + 1]:
                mismatch = verify_table_tile_mismatch(
                    captured[table], live[table], tile - OFFSETS[table], BATCH,
                    GEOMETRY[table][0], GEOMETRY[table][1], GEOMETRY[table][2],
                    GEOMETRY[table][3], GEOMETRY[table][4], BLOCK,
                )
                tl.store(flags + tile, mismatch)

    return verify_table_mismatch, verify_tables_mismatch, triton.cdiv


class _PendingVerifyTableCheck:
    def __init__(self, event, flags, host_flags, checks, counts):
        self._event = event
        self._flags = flags
        self._host_flags = host_flags
        self._checks = checks
        self._counts = counts

    def validate(self):
        self._event.synchronize()
        flags = self._host_flags.tolist()
        start = 0
        for (key, _, _), count in zip(self._checks, self._counts):
            if any(flags[start:start + count]):
                raise RuntimeError(f"verify journal native block table changed for {key}")
            start += count


def try_enqueue_verify_table_check(checks, batch):
    """Compare exact page tables and queue one pinned host receipt."""
    if not checks or batch < 1:
        return None
    device = checks[0][1].device
    if device.type != "cuda" or any(
        a.ndim != 2 or b.ndim != 2 or a.device != device or b.device != device
        or a.dtype != b.dtype or a.dtype not in (torch.int32, torch.int64)
        or min(a.size(0), b.size(0)) < batch or not 0 < b.size(1) <= a.size(1)
        for _, a, b in checks
    ):
        return None
    try:
        kernel, _, cdiv = _verify_table_check()
    except ImportError:
        return None
    block = 256
    counts = [cdiv(batch * live.size(1), block) for _, _, live in checks]
    with torch.cuda.device(device):
        if torch.cuda.is_current_stream_capturing():
            return None
        try:
            host_flags = torch.empty(sum(counts), dtype=torch.int32, device="cpu", pin_memory=True)
        except RuntimeError:
            return None
        flags = torch.empty_like(host_flags, device=device)
        stream = torch.cuda.current_stream(device)
        start = 0
        for (_, captured, live), count in zip(checks, counts):
            kernel[(count,)](
                captured, live, flags[start:start + count], batch, live.size(1),
                *captured.stride(), *live.stride(), block, num_warps=4,
            )
            # Aborting a deferred transaction may release the Python owners
            # before the queued comparisons have consumed their inputs.
            captured.record_stream(stream)
            live.record_stream(stream)
            start += count
        host_flags.copy_(flags, non_blocking=True)
        event = torch.cuda.Event()
        event.record(stream)
    return _PendingVerifyTableCheck(event, flags, host_flags, checks, counts)


@lru_cache(maxsize=1)
def _kernels():
    import triton
    import triton.language as tl

    @triton.jit
    def physical_slot(
        position, table, request, active,
        table_columns: tl.constexpr, table_stride_row: tl.constexpr,
        table_stride_column: tl.constexpr, span: tl.constexpr,
        entries: tl.constexpr, ratio: tl.constexpr, full: tl.constexpr,
        pool_blocks: tl.constexpr,
    ):
        logical_block = position // span
        present = active & (position >= 0) & (logical_block < table_columns)
        block = tl.load(
            table + request * table_stride_row + logical_block * table_stride_column,
            present,
            other=0,
        ).to(tl.int64)
        if full:
            offset = (position % span) // ratio
        else:
            offset = position % entries
        valid = present & (block > 0) & (block < pool_blocks) & (offset < entries)
        return tl.where(valid, block * entries + offset, -1)

    @triton.jit
    def position_slots(
        positions,
        table,
        output,
        count: tl.constexpr,
        columns_per_request: tl.constexpr,
        table_rows: tl.constexpr,
        table_columns: tl.constexpr,
        table_stride_row: tl.constexpr,
        table_stride_column: tl.constexpr,
        span: tl.constexpr,
        entries: tl.constexpr,
        ratio: tl.constexpr,
        full: tl.constexpr,
        pool_blocks: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        position = tl.load(positions + index, index < count, other=-1).to(tl.int64)
        request = index // columns_per_request
        slot = physical_slot(
            position, table, request, (index < count) & (request < table_rows),
            table_columns, table_stride_row, table_stride_column, span,
            entries, ratio, full, pool_blocks,
        )
        tl.store(output + index, slot, index < count)

    @triton.jit
    def verify_publication_plan(
        raw, requests, prefixes, ends, valid, table, output,
        request_count, table_rows, table_columns, table_stride_row, table_stride_column,
        span, entries, ratio, full, trim_csa, pool_blocks,
        count: tl.constexpr, BLOCK: tl.constexpr, REQUEST_TABLE: tl.constexpr,
    ):
        row = tl.arange(0, BLOCK)
        position = tl.load(raw + row, row < count, other=-1)
        request = tl.load(requests + row, row < count, other=0)
        bound = (request >= 0) & (request < request_count)
        base = tl.load(prefixes + request, (row < count) & bound, other=0)
        end = tl.load(ends + request, (row < count) & bound, other=0)
        keep = bound & (position >= 0) & tl.load(valid + row, row < count, other=False)
        if full:
            # Prefix lengths are nonnegative. Explicit floor also matches the
            # torch interval if a padding row carries a negative prefix.
            base = (base - tl.where(base < 0, ratio - 1, 0)) // ratio
            end = (end - tl.where(end < 0, ratio - 1, 0)) // ratio
            keep = keep & (position // ratio >= base) & (position // ratio < end)
        else:
            keep = keep & (position >= base) & (position < end)
            if trim_csa:
                keep = keep & (position >= end - entries)
        table_row = request if REQUEST_TABLE else row
        logical = position // span
        present = keep & (table_row < table_rows) & (logical < table_columns)
        physical = tl.load(
            table + table_row * table_stride_row + logical * table_stride_column,
            present, other=0,
        )
        offset = tl.where(full, (position % span) // ratio, position % entries)
        exists = present & (physical > 0) & (physical < pool_blocks) & (offset < entries)
        slot = tl.where(exists, physical.to(tl.int64) * entries + offset, -1)
        # Stable sort in the reference keeps the last row in request-major
        # order for a physical slot. Bounded Verify rows need no sort.
        later = (slot[:, None] == slot[None, :]) & (row[:, None] < row[None, :])
        shadowed = tl.max(later.to(tl.int32), axis=1) != 0
        tl.store(output + row, tl.where(shadowed, -1, slot), row < count)
        tl.store(output + count, tl.sum(keep.to(tl.int32), axis=0))
        invalid = (keep & (slot < 0)) | ((row < count) & ~bound)
        tl.store(output + count + 1, tl.max(invalid.to(tl.int32), axis=0))

    @triton.jit
    def plan_verify_publication(
        raw, requests, prefixes, ends, valid, table, output,
        count: tl.constexpr, request_count: tl.constexpr, table_columns: tl.constexpr,
        table_stride_row: tl.constexpr, table_stride_column: tl.constexpr,
        span: tl.constexpr, entries: tl.constexpr, ratio: tl.constexpr,
        full: tl.constexpr, trim_csa: tl.constexpr, pool_blocks: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        output += tl.program_id(0) * (count + 2)
        verify_publication_plan(
            raw, requests, prefixes, ends, valid, table, output, request_count, count,
            table_columns, table_stride_row, table_stride_column,
            span, entries, ratio, full, trim_csa, pool_blocks, count, BLOCK, False,
        )

    @triton.jit
    def plan_verify_publication_pool(
        descriptors, requests, prefixes, ends, valid, output, pool,
        REQUESTS: tl.constexpr, ROWS: tl.constexpr, FIELDS: tl.constexpr,
    ):
        desc = descriptors + pool * FIELDS
        raw = tl.load(desc).to(tl.pointer_type(tl.int64))
        table = tl.load(desc + 1).to(tl.pointer_type(tl.int32))
        table_rows = tl.load(desc + 2).to(tl.int32)
        table_columns = tl.load(desc + 3).to(tl.int32)
        table_stride_row = tl.load(desc + 4)
        table_stride_column = tl.load(desc + 5)
        span = tl.load(desc + 6).to(tl.int32)
        entries = tl.load(desc + 7).to(tl.int32)
        ratio = tl.load(desc + 8).to(tl.int32)
        full = tl.load(desc + 9) != 0
        trim_csa = tl.load(desc + 10) != 0
        blocks = tl.load(desc + 11).to(tl.int32)
        verify_publication_plan(
            raw, requests, prefixes, ends, valid, table, output,
            REQUESTS, table_rows, table_columns, table_stride_row, table_stride_column,
            span, entries, ratio, full, trim_csa, blocks, ROWS, ROWS, True,
        )

    @triton.jit
    def plan_verify_publication_batch(
        descriptors, requests, prefixes, ends, valid, output,
        REQUESTS: tl.constexpr, ROWS: tl.constexpr, FIELDS: tl.constexpr,
    ):
        pool = tl.program_id(0)
        plan_verify_publication_pool(
            descriptors, requests, prefixes, ends, valid, output + pool * (ROWS + 2),
            pool, REQUESTS, ROWS, FIELDS,
        )

    @triton.jit
    def plan_verify_publication_pair(
        target_desc, target_requests, target_prefixes, target_valid,
        draft_desc, draft_requests, draft_prefixes, draft_valid, ends, output,
        TARGET_POOLS: tl.constexpr, BATCH: tl.constexpr, ROWS: tl.constexpr, FIELDS: tl.constexpr,
    ):
        pool = tl.program_id(0)
        output += pool * (ROWS + 2)
        if pool < TARGET_POOLS:
            plan_verify_publication_pool(
                target_desc, target_requests, target_prefixes, ends, target_valid,
                output, pool, BATCH, ROWS, FIELDS,
            )
        else:
            plan_verify_publication_pool(
                draft_desc, draft_requests, draft_prefixes, ends, draft_valid,
                output, pool - TARGET_POOLS, BATCH, ROWS, FIELDS,
            )

    @triton.jit(do_not_specialize=["packed_counts", "expected0", "expected1", "expected2"])
    def prepare_verify_pair(
        supplied, target, draft, lengths, output, packed_counts,
        expected0, expected1, expected2,
        BATCH: tl.constexpr, WIDTH: tl.constexpr,
    ):
        row = tl.arange(0, BATCH)
        prefix = tl.load(supplied + row).to(tl.int64)
        target_prefix = tl.load(target + row).to(tl.int64)
        draft_prefix = tl.load(draft + row).to(tl.int64)
        draft_length = tl.load(lengths + row)
        packed = tl.where(row < 3, expected0, tl.where(row < 6, expected1, expected2)).to(tl.int64)
        expected = (packed >> (21 * (row % 3))) & ((1 << 21) - 1)
        target_bad = tl.max(((prefix != target_prefix) | (prefix != expected)).to(tl.int32), 0)
        draft_bad = tl.max(((prefix != draft_prefix) | (draft_length != WIDTH)).to(tl.int32), 0)
        counts = (packed_counts >> (3 * row)) & 7
        tl.store(output + row, prefix + counts)
        tl.store(output + BATCH, tl.where(target_bad != 0, 1, tl.where(draft_bad != 0, 2, 0)))

    @triton.jit
    def validate_verify_pair(
        layout, target_plans, draft_plans, flags, status,
        BATCH: tl.constexpr, ROWS: tl.constexpr, FLAGS: tl.constexpr,
        TARGET_POOLS: tl.constexpr, DRAFT_POOLS: tl.constexpr, BLOCK: tl.constexpr,
    ):
        lane = tl.arange(0, BLOCK)
        layout_status = tl.load(layout + BATCH).to(tl.int32)
        table_mismatch = tl.load(flags + lane, lane < FLAGS, other=0)
        first = tl.min(tl.where(table_mismatch != 0, lane, FLAGS), 0)
        offset = lane * (ROWS + 2) + ROWS + 1
        target_bad = tl.max(tl.load(target_plans + offset, lane < TARGET_POOLS, other=0), 0)
        draft_bad = tl.max(tl.load(draft_plans + offset, lane < DRAFT_POOLS, other=0), 0)
        slot_status = tl.where(target_bad != 0, 3, tl.where(draft_bad != 0, 4, 0))
        checked = tl.where(first < FLAGS, first + 5, slot_status)
        tl.store(status, tl.where(layout_status != 0, layout_status, checked))

    @triton.jit
    def copy_verify_publication_row(
        desc, slot, backup, status, row,
        BLOCK: tl.constexpr, WRITE: tl.constexpr, RESTORE: tl.constexpr, CHECKED: tl.constexpr,
    ):
        ready = slot >= 0
        if CHECKED:
            ready = ready & (tl.load(status) == 0)
        if ready:
            native = tl.load(desc).to(tl.pointer_type(tl.int32))
            payload = tl.load(desc + 1).to(tl.pointer_type(tl.int32))
            entries = tl.load(desc + 2).to(tl.int32)
            stride = tl.load(desc + 3)
            width = tl.load(desc + 4).to(tl.int32)
            planar = tl.load(desc + 5) != 0
            backup_start = tl.load(desc + 6)
            column = tl.arange(0, BLOCK)
            block, offset = slot // entries, slot % entries
            physical = tl.where(
                planar,
                tl.where(column < 16, offset * 16 + column, entries * 16 + offset),
                offset * width + column,
            )
            native = native + block * stride + physical
            saved = backup + backup_start + row * width + column
            if WRITE:
                source = saved if RESTORE else payload + row * width + column
                value = tl.load(source, column < width, other=0)
                tl.store(native, value, column < width)
            else:
                value = tl.load(native, column < width, other=0)
                tl.store(saved, value, column < width)

    @triton.jit
    def copy_verify_publication_batch(
        descriptors, plans, backup, status, ROWS: tl.constexpr, FIELDS: tl.constexpr,
        BLOCK: tl.constexpr, WRITE: tl.constexpr, RESTORE: tl.constexpr, CHECKED: tl.constexpr,
    ):
        pool, row = tl.program_id(0), tl.program_id(1)
        slot = tl.load(plans + pool * (ROWS + 2) + row)
        copy_verify_publication_row(
            descriptors + pool * FIELDS, slot, backup, status, row,
            BLOCK, WRITE, RESTORE, CHECKED,
        )

    @triton.jit
    def copy_verify_publication_pair(
        target_desc, draft_desc, plans, backup, status,
        TARGET_POOLS: tl.constexpr, TARGET_WORDS: tl.constexpr,
        ROWS: tl.constexpr, FIELDS: tl.constexpr,
        TARGET_BLOCK: tl.constexpr, DRAFT_BLOCK: tl.constexpr, WRITE: tl.constexpr,
    ):
        pool, row = tl.program_id(0), tl.program_id(1)
        slot = tl.load(plans + pool * (ROWS + 2) + row)
        if pool < TARGET_POOLS:
            copy_verify_publication_row(
                target_desc + pool * FIELDS, slot, backup, status, row,
                TARGET_BLOCK, WRITE, False, True,
            )
        else:
            copy_verify_publication_row(
                draft_desc + (pool - TARGET_POOLS) * FIELDS, slot, backup + TARGET_WORDS, status,
                row, DRAFT_BLOCK, WRITE, False, True,
            )

    @triton.jit
    def packed_rows(
        pool,
        slots,
        rows,
        count: tl.constexpr,
        width: tl.constexpr,
        entries: tl.constexpr,
        pool_blocks: tl.constexpr,
        pool_stride: tl.constexpr,
        planar: tl.constexpr,
        write: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        row = tl.program_id(0)
        column = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
        slot = tl.load(slots + row).to(tl.int64)
        block, offset = slot // entries, slot % entries
        if planar:
            physical_column = tl.where(
                column < 64,
                offset * 64 + column,
                entries * 64 + offset * 4 + column - 64,
            )
        else:
            physical_column = offset * width + column
        valid = (slot >= 0) & (block > 0) & (block < pool_blocks)
        pool_ptr = pool + block * pool_stride + physical_column
        rows_ptr = rows + row * width + column
        if write:
            value = tl.load(rows_ptr, column < width, other=0)
            tl.store(pool_ptr, value, valid & (column < width))
        else:
            value = tl.load(pool_ptr, valid & (column < width), other=0)
            tl.store(rows_ptr, value, column < width)

    @triton.jit
    def store_cp_prefill_rows(
        pool, table, rows, start, first,
        POOL_STRIDE: tl.constexpr, TABLE_STRIDE: tl.constexpr, WIDTH: tl.constexpr,
        ENTRIES: tl.constexpr, PAGE: tl.constexpr, OWNER: tl.constexpr,
        CP: tl.constexpr, RANK: tl.constexpr, PLANAR: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        owned = first + tl.program_id(0).to(tl.int64)
        logical = (owned // OWNER * CP + RANK) * OWNER + owned % OWNER
        block = tl.load(table + (owned // PAGE) * TABLE_STRIDE).to(tl.int64)
        offset = owned % PAGE
        column = tl.arange(0, BLOCK)
        if PLANAR:
            target = tl.where(column < 64, offset * 64 + column,
                              ENTRIES * 64 + offset * 4 + column - 64)
        else:
            target = offset * WIDTH + column
        value = tl.load(rows + (logical - start) * WIDTH + column, column < WIDTH, other=0)
        tl.store(pool + block * POOL_STRIDE + target, value, column < WIDTH)

    @triton.jit
    def store_prefill_csa(pool, plan, rows, scores, WIDTH: tl.constexpr, SEPARATE: tl.constexpr):
        item = tl.program_id(0).to(tl.int64)
        target = tl.load(plan + 2 * item)
        source = tl.load(plan + 2 * item + 1)
        if SEPARATE:
            column = tl.arange(0, WIDTH // 2)
            value = tl.load(rows + source * (WIDTH // 2) + column)
            score = tl.load(scores + source * (WIDTH // 2) + column)
            tl.store(pool + target + column, value)
            tl.store(pool + target + WIDTH // 2 + column, score)
        else:
            column = tl.arange(0, WIDTH)
            value = tl.load(rows + source * WIDTH + column)
            tl.store(pool + target + column, value)

    @triton.jit
    def assemble_cp_prefill_indices(
        metadata, positions, selected, indices, lengths, position_stride,
        selected_row_stride, selected_col_stride,
        TOPK: tl.constexpr, RATIO: tl.constexpr, WINDOW: tl.constexpr,
        WIDTH: tl.constexpr, BLOCK: tl.constexpr,
    ):
        row = tl.program_id(0).to(tl.int64)
        column = tl.arange(0, BLOCK)
        position = tl.load(positions + row * position_stride)
        swa_base = tl.load(metadata + row * 4 + 2)
        swa_first = tl.load(metadata + row * 4 + 3)
        swa_position = position - WINDOW + 1 + column - TOPK
        swa_live = (column >= TOPK) & (column < TOPK + WINDOW) & (swa_position >= swa_first)
        values = tl.where(swa_live, swa_base + swa_position - swa_first, -1)
        if TOPK:
            global_base = tl.load(metadata + row * 4)
            count = tl.load(metadata + row * 4 + 1)
            entry = tl.load(selected + row * selected_row_stride + column * selected_col_stride,
                            column < TOPK, other=-1)
            global_live = (column < TOPK) & (column < (position + 1) // RATIO) & (entry >= 0) & (entry < count)
            values = tl.where(global_live, global_base + entry, values)
        values = values.to(tl.int32)
        live = values >= 0
        rank = tl.cumsum(live.to(tl.int32))
        length = tl.sum(live.to(tl.int32), 0)
        destination = tl.where(live, rank - 1, length + column - rank)
        tl.store(indices + row * WIDTH + destination, values, column < TOPK + WINDOW)
        tl.store(indices + row * WIDTH + column, -1, (column >= TOPK + WINDOW) & (column < WIDTH))
        tl.store(lengths + row, length)

    return SimpleNamespace(
        triton=triton, position_slots=position_slots, packed_rows=packed_rows,
        plan_verify_publication=plan_verify_publication,
        plan_verify_publication_batch=plan_verify_publication_batch,
        plan_verify_publication_pair=plan_verify_publication_pair,
        prepare_verify_pair=prepare_verify_pair,
        validate_verify_pair=validate_verify_pair,
        copy_verify_publication_batch=copy_verify_publication_batch,
        copy_verify_publication_pair=copy_verify_publication_pair,
        store_cp_prefill_rows=store_cp_prefill_rows, store_prefill_csa=store_prefill_csa,
        assemble_cp_prefill_indices=assemble_cp_prefill_indices,
    )


def _owned_before(end, span, owned_start, owned_count):
    cycles, tail = divmod(end, span)
    return cycles * owned_count + min(max(tail - owned_start, 0), owned_count)


def try_assemble_cp_prefill_indices(metadata, positions, selected, ratio, window, width):
    """Assemble and compact the stable global-then-SWA sparse row in one launch."""
    rows = positions.numel()
    if not (
        positions.is_cuda and positions.ndim == 1 and positions.dtype == torch.int64
        and metadata.shape == (rows, 4) and metadata.is_contiguous()
        and metadata.dtype == torch.int64 and metadata.device == positions.device
    ):
        return None
    topk = 0
    selected_strides = (0, 0)
    if selected is not None:
        if not (selected.ndim == 2 and selected.shape[0] == rows and ratio > 0
                and selected.device == positions.device and selected.dtype in (torch.int32, torch.int64)):
            return None
        topk = selected.shape[1]
        selected_strides = selected.stride()
    if not 0 < window <= topk + window <= width:
        return None
    indices = torch.empty((rows, 1, width), dtype=torch.int32, device=positions.device)
    lengths = torch.empty(rows, dtype=torch.int32, device=positions.device)
    if rows:
        kernels = _kernels()
        kernels.assemble_cp_prefill_indices[(rows,)](
            metadata, positions, selected, indices, lengths, positions.stride(0),
            *selected_strides, topk, ratio, window, width, kernels.triton.next_power_of_2(width),
        )
    return indices, lengths


def try_write_cp_prefill_rows(pool, placement, host_table, request, start, rows, *, planar):
    """Store locally owned entries directly into the scheduler's kernel pages."""
    ratio = placement.ratio
    if start % ratio or pool.span % ratio:
        return None
    start //= ratio
    page = pool.span // ratio
    owner = placement.physical_span // ratio
    if page > pool.entries or owner % page:
        return None
    cycle = owner * placement.size
    rank_start = owner * placement.rank
    first = _owned_before(start, cycle, rank_start, owner)
    stop = _owned_before(start + rows.shape[0], cycle, rank_start, owner)
    if first == stop:
        return 0
    page_end = (stop - 1) // page + 1
    if page_end > host_table.shape[1]:
        return None
    blocks = host_table[request, first // page:page_end].tolist()
    if len(set(blocks)) != len(blocks) or any(not 0 <= block < pool.tensor.shape[0] for block in blocks):
        return None
    kernels = _kernels()
    kernels.store_cp_prefill_rows[(stop - first,)](
        pool.tensor, pool.table[request], rows, start, first, pool.tensor.stride(0),
        pool.table.stride(1), rows.shape[1], pool.entries, page, owner, placement.size,
        placement.rank, planar, kernels.triton.next_power_of_2(rows.shape[1]),
    )
    return stop - first


def try_write_prefill_csa(pool, placement, host_table, request, start, length, rows, *, scores=None):
    """Deduplicate latest raw states by physical address on the host."""
    end = start + rows.shape[0]
    last_page = (end - 1) // pool.span + 1
    if end > length or last_page > host_table.shape[1]:
        return None
    entries = pool.entries
    owned_start = placement.rank * entries
    cycle = placement.full_entries
    first_page = start // pool.span
    width = rows.shape[1] if scores is None else 2 * rows.shape[1]
    plan = {}
    for page, block in enumerate(host_table[request, first_page:last_page].tolist(), first_page):
        begin = max(start, page * pool.span)
        stop = min(end, (page + 1) * pool.span)
        if block == -1 and stop <= length - length % 2:
            continue
        if not 0 <= block < pool.tensor.shape[0]:
            return None
        for offset in range(entries):
            last = stop - 1 - (stop - 1 - owned_start - offset) % cycle
            if last >= begin:
                target = block * pool.tensor.stride(0) + offset * width
                plan[target] = last - start
    if plan:
        indices = torch.tensor(tuple(plan.items()), dtype=torch.int64).to(pool.tensor.device, non_blocking=True)
        _kernels().store_prefill_csa[(len(plan),)](pool.tensor, indices, rows, scores, width, scores is not None)
    owned = _owned_before(end, cycle, owned_start, entries) - _owned_before(start, cycle, owned_start, entries)
    return owned, len(plan)


def position_slots(
    positions: torch.Tensor,
    table: torch.Tensor,
    *,
    span: int,
    entries: int,
    ratio: int,
    full: bool,
    pool_blocks: int,
) -> torch.Tensor:
    """Map batch-major raw token positions to packed-entry physical slots."""
    if positions.ndim < 1 or positions.shape[0] > table.shape[0]:
        raise ValueError("Graph positions must have a batch axis covered by the table")
    positions = positions.to(dtype=torch.int64).contiguous()
    result = torch.empty_like(positions)
    count = positions.numel()
    if count == 0:
        return result
    columns = count // positions.shape[0]
    if positions.device.type == "cuda":
        kernels = _kernels()
        kernels.position_slots[(kernels.triton.cdiv(count, 128),)](
            positions,
            table,
            result,
            count,
            columns,
            table.shape[0],
            table.shape[1],
            table.stride(0),
            table.stride(1),
            span,
            entries,
            ratio,
            full,
            pool_blocks,
            BLOCK=128,
            num_warps=4,
        )
        return result
    flat = positions.reshape(positions.shape[0], columns)
    logical = flat.div(span, rounding_mode="floor")
    requests = torch.arange(flat.shape[0], device=table.device)[:, None]
    block = table[requests, logical.clamp(0, table.shape[1] - 1)].long()
    offset = (
        flat.remainder(span).div(ratio, rounding_mode="floor")
        if full
        else flat.remainder(entries)
    )
    valid = (
        (flat >= 0)
        & (logical < table.shape[1])
        & (block > 0)
        & (block < pool_blocks)
        & (offset < entries)
    )
    result.copy_(torch.where(valid, block * entries + offset, -1).reshape_as(result))
    return result


def try_plan_verify_publication(
    raw, requests, prefixes, ends, valid, table, output, *,
    span, entries, ratio, full, trim_csa, pool_blocks,
):
    """Write unique physical slots, accepted count, and an invalid-slot flag."""
    count = raw.numel()
    if (
        not raw.is_cuda or not 0 < count <= 64 or raw.shape != (count,)
        or valid.shape != raw.shape or valid.dtype != torch.bool
        or requests.shape != raw.shape or prefixes.ndim != 1
        or ends.shape != prefixes.shape or output.shape != (count + 2,)
        or output.dtype != torch.int64 or table.ndim != 2
        or table.shape[0] < count or table.dtype not in (torch.int32, torch.int64)
        or ratio not in (1, 2) or span <= 0 or entries <= 0
        or any(t.dtype not in (torch.int32, torch.int64) for t in (raw, requests, prefixes, ends))
        or any(
            t.device != raw.device or not t.is_contiguous()
            for t in (raw, requests, prefixes, ends, valid, output)
        )
        or table.device != raw.device
    ):
        return False
    kernels = _kernels()
    kernels.plan_verify_publication[(1,)](
        raw, requests, prefixes, ends, valid, table, output, count,
        prefixes.numel(), table.shape[1], *table.stride(), span, entries, ratio, full, trim_csa,
        pool_blocks, kernels.triton.next_power_of_2(count), num_warps=4,
    )
    return True


class BatchedVerifyPublication:
    """Static pool descriptors for one B32 Verify journal and its native cache."""

    @classmethod
    def try_create(cls, pools, rows, device):
        if rows != 32 or device.type != "cuda":
            return None
        plan, transfer, storage, ranges = [], [], [], []
        words_per_row = max_words = 0
        for raw, payload, pool, table, span, entries, ratio, full, trim, planar in pools:
            if (
                any(t.device != device or t.data_ptr() % 4 for t in (raw, payload, pool, table))
                or raw.shape != (rows,) or raw.dtype != torch.int64 or not raw.is_contiguous()
                or payload.ndim != 2 or payload.shape[0] != rows or not payload.is_contiguous()
                or pool.ndim != 2 or pool.dtype != payload.dtype or not pool.is_contiguous()
                or pool.dtype not in (torch.uint8, torch.float32) or pool.shape[0] <= 1
                or table.ndim != 2 or min(table.shape) <= 0 or table.dtype != torch.int32
                or span <= 0 or entries <= 0 or ratio not in (1, 2)
            ):
                return None
            element_bytes = pool.element_size()
            width = payload.shape[1] * element_bytes
            stride = pool.stride(0) * element_bytes
            if (
                not 0 < width <= 4096 or width % 4 or stride % 4
                or pool.shape[1] < payload.shape[1] * entries
                or (planar and (pool.dtype != torch.uint8 or width != 68))
            ):
                return None
            width_words = width // 4
            plan.append((
                raw.data_ptr(), table.data_ptr(), *table.shape, *table.stride(),
                span, entries, ratio, full, trim, pool.shape[0],
            ))
            transfer.append((
                pool.data_ptr(), payload.data_ptr(), entries, stride // 4,
                width_words, planar, words_per_row * rows,
            ))
            words_per_row += width_words
            max_words = max(max_words, width_words)
            storage.extend((raw, payload, pool, table))
            ranges.append((pool.data_ptr(), pool.data_ptr() + pool.numel() * element_bytes))
        ranges.sort()
        # Pool-level launch ordering matters if a native allocation aliases
        # another pool. Keep the ordered fallback for those layouts.
        if not plan or any(end > start for (_, end), (start, _) in zip(ranges, ranges[1:])):
            return None
        return cls(plan, transfer, storage, ranges, rows, words_per_row, max_words, device)

    def __init__(self, plan, transfer, storage, ranges, rows, words_per_row, max_words, device):
        self.plan = torch.tensor(plan, dtype=torch.int64, device=device)
        self.transfer = torch.tensor(transfer, dtype=torch.int64, device=device)
        # Integer descriptors borrow addresses. Keep the referenced tensors
        # alive for at least as long as the descriptors.
        self._storage = storage
        self._native_ranges = ranges
        self._table_layouts = frozenset(
            (table.data_ptr(), table.shape, table.stride()) for table in storage[3::4]
        )
        self._joint_stream = None
        self._joint_workspace_lock = Lock()
        self._joint_workspace = None
        self.rows = rows
        self.backup_words = rows * words_per_row
        self.max_words = max_words

    def _record_joint_use(self, stream):
        if stream != self._joint_stream:
            for tensor in (self.plan, self.transfer, *self._storage):
                tensor.record_stream(stream)
            self._joint_stream = stream

    def _copy(self, plans, backup, *, write=False, restore=False, status=None):
        kernels = _kernels()
        kernels.copy_verify_publication_batch[(plans.shape[0], self.rows)](
            self.transfer, plans, backup, plans if status is None else status,
            self.rows, self.transfer.shape[1],
            BLOCK=kernels.triton.next_power_of_2(self.max_words),
            WRITE=write, RESTORE=restore, CHECKED=status is not None, num_warps=4,
        )

    def _plan(self, requests, prefixes, ends, valid):
        kernels = _kernels()
        pools = self.plan.shape[0]
        plans = self.plan.new_empty((pools, self.rows + 2))
        kernels.plan_verify_publication_batch[(pools,)](
            self.plan, requests, prefixes, ends, valid, plans,
            prefixes.numel(), self.rows, self.plan.shape[1], num_warps=4,
        )
        return plans

    def publish(self, requests, prefixes, ends, valid):
        if (
            requests.shape != (self.rows,) or requests.dtype != torch.int64
            or valid.shape != requests.shape or valid.dtype != torch.bool
            or prefixes.ndim != 1 or ends.shape != prefixes.shape
            or prefixes.dtype not in (torch.int32, torch.int64) or ends.dtype != prefixes.dtype
            or any(
                t.device != self.plan.device or not t.is_contiguous()
                for t in (requests, prefixes, ends, valid)
            )
        ):
            return None
        plans = self._plan(requests, prefixes, ends, valid)
        count, invalid = plans[:, self.rows:].sum(dim=0).tolist()
        if invalid:
            raise RuntimeError("accepted verify journal row has no native slot")
        if count:
            backup = torch.empty(self.backup_words, dtype=torch.int32, device=plans.device)
            self._copy(plans, backup)
            try:
                self._copy(plans, backup, write=True)
            except Exception:
                self._copy(plans, backup, write=True, restore=True)
                raise
        return count


def _verify_pair_inputs_supported(side, batch, rows, device):
    publisher, requests, prefixes, valid, checks = side[:5]
    return (
        isinstance(publisher, BatchedVerifyPublication)
        and publisher.rows == rows and publisher.plan.device == device
        and all(isinstance(t, torch.Tensor) for t in (requests, prefixes, valid))
        and requests.shape == (rows,) and requests.dtype == torch.int64
        and valid.shape == (rows,) and valid.dtype == torch.bool
        and prefixes.shape == (batch,) and prefixes.dtype in (torch.int32, torch.int64)
        and all(t.device == device and t.is_contiguous() for t in (requests, prefixes, valid))
        and isinstance(checks, tuple) and bool(checks)
        and all(
            isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor)
            and a.ndim == 2 and b.ndim == 2 and a.device == device and b.device == device
            and a.dtype == b.dtype and a.dtype in (torch.int32, torch.int64)
            and min(a.size(0), b.size(0)) >= batch and 0 < b.size(1) <= a.size(1)
            for _, a, b in checks
        )
        and publisher._table_layouts == {
            (captured.data_ptr(), captured.shape, captured.stride()) for _, captured, _ in checks
        }
    )


def _launch_joint_table_checks(single, batched, captured, live, geometry, offsets, flags, batch, block, stream):
    for snapshot, current in zip(captured, live):
        snapshot.record_stream(stream)
        current.record_stream(stream)
    if len(captured) <= 8:
        batched[(offsets[-1],)](
            captured, live, flags, batch, block, geometry, offsets, num_warps=4,
        )
        return
    for snapshot, current, shape, start, end in zip(captured, live, geometry, offsets, offsets[1:]):
        single[(end - start,)](
            snapshot, current, flags[start:end], batch, *shape, block, num_warps=4,
        )

def _copy_verify_pair(kernels, target, draft, plans, backup, status, *, write=False):
    kernels.copy_verify_publication_pair[(plans.shape[0], target.rows)](
        target.transfer, draft.transfer, plans, backup, status,
        target.plan.shape[0], target.backup_words, target.rows, target.transfer.shape[1],
        TARGET_BLOCK=kernels.triton.next_power_of_2(target.max_words),
        DRAFT_BLOCK=kernels.triton.next_power_of_2(draft.max_words),
        WRITE=write, num_warps=4,
    )


class _JointVerifyWorkspace:
    @classmethod
    def try_create(cls, target, draft, stream, batch, flag_capacity):
        try:
            host_status = torch.empty(1, dtype=torch.int32, device="cpu", pin_memory=True)
        except RuntimeError:
            return None
        return cls(target, draft, stream, batch, flag_capacity, host_status)

    def __init__(self, target, draft, stream, batch, flag_capacity, host_status):
        self.peer = weakref.ref(draft)
        self.stream = stream
        self.host_status = host_status
        device = target.plan.device
        target_pools = target.plan.shape[0]
        total_pools = target_pools + draft.plan.shape[0]
        self.layout = torch.empty(batch + 1, dtype=torch.int64, device=device)
        self.status = torch.empty_like(host_status, device=device)
        self.flags = torch.empty(flag_capacity, dtype=torch.int32, device=device)
        self.plans = target.plan.new_empty((total_pools, target.rows + 2))
        self.target_plans = self.plans[:target_pools]
        self.draft_plans = self.plans[target_pools:]
        self.ready = torch.cuda.Event()
        self.backup = None
        self.backup_words = target.backup_words + draft.backup_words

    def get_backup(self):
        if self.backup is None:
            self.backup = torch.empty(self.backup_words, dtype=torch.int32, device=self.plans.device)
        return self.backup


@contextmanager
def _borrow_joint_workspace(target, draft, stream, batch, flag_count):
    lock = target._joint_workspace_lock
    if lock.acquire(blocking=False):
        try:
            workspace = target._joint_workspace
            if workspace is None:
                workspace = _JointVerifyWorkspace.try_create(target, draft, stream, batch, 1024)
                target._joint_workspace = workspace
            if workspace is None or (workspace.peer() is draft and workspace.stream == stream):
                # Hold the lease until the host has read the pinned result.
                # Prior writes and exception rollback are ordered on this stream.
                yield workspace
                return
        finally:
            lock.release()
    yield _JointVerifyWorkspace.try_create(target, draft, stream, batch, flag_count)


def _verify_tensor_layout(tensor):
    return tensor.data_ptr(), tensor.shape, tensor.stride(), tensor.dtype, tensor.device


class _PreparedVerifyPair:
    def __init__(self, target, draft, prefixes, expected, stream, kernels, tables, table_args, block):
        self.target, self.draft, self.prefixes = target, draft, prefixes
        self.device = prefixes.device
        self.expected, self.stream = expected, stream
        self.kernels, self.tables = kernels, tables
        self.table_args, self.block = table_args, block
        self.layouts = tuple(
            (tensor, _verify_tensor_layout(tensor))
            for tensor in (prefixes, *table_args[0], *table_args[1])
        )
        self.target_pools, self.draft_pools = target[0].plan.shape[0], draft[0].plan.shape[0]

    def publish(self, counts) -> bool:
        """Launch the guarded pair only after the host approves its row counts."""
        batch, width, rows, table_block = 8, 4, 32, 256
        if (
            not isinstance(counts, tuple) or len(counts) != batch
            or any(type(count) is not int or not 0 <= count <= width for count in counts)
        ):
            return False
        packed_counts = sum(count << (3 * request) for request, count in enumerate(counts))
        prefixes = self.prefixes
        device = self.device
        stream = self.stream
        kernels = self.kernels
        target_pub, target_requests, target_prefixes, target_valid, target_checks = self.target
        draft_pub, draft_requests, draft_prefixes, draft_valid, draft_checks, lengths = self.draft
        target_pools, draft_pools = self.target_pools, self.draft_pools
        offsets = self.table_args[-1]
        flag_count = offsets[-1]
        with torch.cuda.device(device):
            if torch.cuda.is_current_stream_capturing() or torch.cuda.current_stream(device) != stream:
                return False
            if any(_verify_tensor_layout(tensor) != layout for tensor, layout in self.layouts):
                return False
            with _borrow_joint_workspace(target_pub, draft_pub, stream, batch, flag_count) as workspace:
                if workspace is None:
                    return False
                target_pub._record_joint_use(stream)
                draft_pub._record_joint_use(stream)
                for value in (
                    prefixes, lengths, target_requests, target_prefixes, target_valid,
                    draft_requests, draft_prefixes, draft_valid,
                ):
                    value.record_stream(stream)
                layout, status, flags, plans = workspace.layout, workspace.status, workspace.flags, workspace.plans
                target_plans, draft_plans = workspace.target_plans, workspace.draft_plans
                kernels.prepare_verify_pair[(1,)](
                    prefixes, target_prefixes, draft_prefixes, lengths, layout, packed_counts,
                    *self.expected, batch, width, num_warps=1,
                )
                kernels.plan_verify_publication_pair[(target_pools + draft_pools,)](
                    target_pub.plan, target_requests, target_prefixes, target_valid,
                    draft_pub.plan, draft_requests, draft_prefixes, draft_valid, layout, plans,
                    target_pools, batch, rows, target_pub.plan.shape[1], num_warps=4,
                )
                _launch_joint_table_checks(
                    *self.tables, *self.table_args, flags, batch, table_block, stream,
                )
                kernels.validate_verify_pair[(1,)](
                    layout, target_plans, draft_plans, flags, status, batch, rows, flag_count,
                    target_pools, draft_pools, self.block, num_warps=4,
                )
                workspace.host_status.copy_(status, non_blocking=True)
                workspace.ready.record(stream)
                if packed_counts:
                    target_words = target_pub.backup_words
                    backup = workspace.get_backup()
                    _copy_verify_pair(kernels, target_pub, draft_pub, plans, backup, status)
                    try:
                        _copy_verify_pair(kernels, target_pub, draft_pub, plans, backup, status, write=True)
                    except Exception:
                        rollback_error = None
                        # A combined launch may have reached both journals. Restore
                        # both saved ranges in reverse order on the same stream.
                        for pub, side_plan, saved in (
                            (draft_pub, draft_plans, backup[target_words:]),
                            (target_pub, target_plans, backup[:target_words]),
                        ):
                            try:
                                pub._copy(side_plan, saved, write=True, restore=True, status=status)
                            except Exception as error:
                                rollback_error = error
                        if rollback_error is not None:
                            raise RuntimeError("joint Verify KV publication rollback failed") from rollback_error
                        raise
                workspace.ready.synchronize()
                outcome = int(workspace.host_status.item())
        if outcome == 1:
            raise ValueError("stale acceptance: scheduler request/order/prefix mismatch")
        if outcome == 2:
            raise ValueError("private Draft KV acceptance does not match the replay layout")
        if outcome in (3, 4):
            side = "verify journal" if outcome == 3 else "private Draft KV journal"
            raise RuntimeError(f"accepted {side} row has no native slot")
        if outcome >= 5:
            first = outcome - 5
            for (key, _, _), end in zip((*target_checks, *draft_checks), offsets[1:]):
                if first < end:
                    raise RuntimeError(f"verify journal native block table changed for {key}")
        return True


def prepare_verify_pair(target, draft, prefixes, expected_prefixes) -> _PreparedVerifyPair | None:
    """Prepare fixed host metadata without submitting device work."""
    batch, rows, table_block = 8, 32, 256
    if not isinstance(prefixes, torch.Tensor):
        return None
    device = prefixes.device
    if (
        device.type != "cuda" or prefixes.shape != (batch,) or not prefixes.is_contiguous()
        or prefixes.dtype not in (torch.int32, torch.int64)
        or not isinstance(expected_prefixes, tuple) or len(expected_prefixes) != batch
        or any(type(prefix) is not int or not 0 <= prefix < (1 << 21) for prefix in expected_prefixes)
        or not isinstance(target, tuple) or len(target) != 5
        or not isinstance(draft, tuple) or len(draft) != 6
        or not _verify_pair_inputs_supported(target, batch, rows, device)
        or not _verify_pair_inputs_supported(draft, batch, rows, device)
    ):
        return None
    lengths = draft[5]
    if (
        not isinstance(lengths, torch.Tensor)
        or lengths.shape != (batch,) or lengths.dtype not in (torch.int32, torch.int64)
        or lengths.device != device or not lengths.is_contiguous()
    ):
        return None
    target_pub, draft_pub = target[0], draft[0]
    if (
        target_pub.plan.shape[1] != draft_pub.plan.shape[1]
        or target_pub.transfer.shape[1] != draft_pub.transfer.shape[1]
    ):
        return None
    ranges = sorted(target_pub._native_ranges + draft_pub._native_ranges)
    if any(end > start for (_, end), (start, _) in zip(ranges, ranges[1:])):
        return None
    try:
        table_kernel, batch_tables, cdiv = _verify_table_check()
        kernels = _kernels()
    except ImportError:
        return None
    checks = (*target[4], *draft[4])
    geometry = tuple(
        (live.size(1), *captured.stride(), *live.stride()) for _, captured, live in checks
    )
    offsets = [0]
    for columns, *_ in geometry:
        offsets.append(offsets[-1] + cdiv(batch * columns, table_block))
    block = kernels.triton.next_power_of_2(
        max(offsets[-1], target_pub.plan.shape[0], draft_pub.plan.shape[0])
    )
    if block > 1024:
        return None
    expected = []
    for start in range(0, batch, 3):
        word = sum(prefix << (21 * offset) for offset, prefix in enumerate(expected_prefixes[start:start + 3]))
        # Keep small words in the signed i64 ABI. The decoder masks out bit 63.
        expected.append(word if word >= (1 << 32) else word - (1 << 63))
    with torch.cuda.device(device):
        if torch.cuda.is_current_stream_capturing():
            return None
        stream = torch.cuda.current_stream(device)
    table_args = (
        tuple(captured for _, captured, _ in checks),
        tuple(live for _, _, live in checks),
        geometry,
        tuple(offsets),
    )
    return _PreparedVerifyPair(
        target, draft, prefixes, tuple(expected), stream, kernels,
        (table_kernel, batch_tables), table_args, block,
    )


def try_publish_verify_pair(target, draft, prefixes, counts, expected_prefixes) -> bool:
    """Validate both B8/Q4 journals once and publish them on the current stream."""
    prepared = prepare_verify_pair(target, draft, prefixes, expected_prefixes)
    return prepared is not None and prepared.publish(counts)

def read_packed_rows(
    pool: torch.Tensor,
    slots: torch.Tensor,
    *,
    width: int,
    entries: int,
    planar: bool,
) -> torch.Tensor:
    """Gather rows without changing native physical stride or planar layout."""
    slots = slots.to(dtype=torch.int64).contiguous()
    rows = torch.empty((*slots.shape, width), dtype=pool.dtype, device=pool.device)
    if not slots.numel():
        return rows
    if pool.device.type == "cuda":
        kernels = _kernels()
        kernels.packed_rows[(slots.numel(), kernels.triton.cdiv(width, 128))](
            pool, slots, rows, slots.numel(), width, entries, pool.shape[0],
            pool.stride(0), planar, False, BLOCK=128, num_warps=4,
        )
        return rows
    flat = slots.reshape(-1)
    block, offset = flat.div(entries, rounding_mode="floor"), flat.remainder(entries)
    column = torch.arange(width, device=pool.device)[None, :]
    if planar:
        physical = torch.where(
            column < 64, offset[:, None] * 64 + column,
            entries * 64 + offset[:, None] * 4 + column - 64,
        )
    else:
        physical = offset[:, None] * width + column
    values = pool[block.clamp(0, pool.shape[0] - 1)[:, None], physical]
    valid = (flat >= 0) & (block > 0) & (block < pool.shape[0])
    rows.copy_(torch.where(valid[:, None], values, 0).reshape_as(rows))
    return rows


def write_packed_rows(
    pool: torch.Tensor,
    slots: torch.Tensor,
    rows: torch.Tensor,
    *,
    width: int,
    entries: int,
    planar: bool,
) -> None:
    """Scatter rows; negative slots and block zero have no side effects."""
    if tuple(rows.shape) != (*slots.shape, width):
        raise ValueError("Graph cache row shape does not match slots and packed width")
    slots = slots.to(dtype=torch.int64).contiguous()
    rows = rows.to(dtype=pool.dtype).contiguous()
    if not slots.numel():
        return
    if pool.device.type == "cuda":
        kernels = _kernels()
        kernels.packed_rows[(slots.numel(), kernels.triton.cdiv(width, 128))](
            pool, slots, rows, slots.numel(), width, entries, pool.shape[0],
            pool.stride(0), planar, True, BLOCK=128, num_warps=4,
        )
        return
    flat = slots.reshape(-1)
    block, offset = flat.div(entries, rounding_mode="floor"), flat.remainder(entries)
    valid = (flat >= 0) & (block > 0) & (block < pool.shape[0])
    column = torch.arange(width, device=pool.device)[None, :]
    off = offset[valid, None]
    physical = (
        torch.where(column < 64, off * 64 + column, entries * 64 + off * 4 + column - 64)
        if planar else off * width + column
    )
    pool[block[valid, None], physical] = rows.reshape(-1, width)[valid]
