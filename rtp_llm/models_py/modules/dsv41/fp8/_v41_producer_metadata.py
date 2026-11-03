"""Forward-local integers for global raw source tiles."""

from bisect import bisect_right
from dataclasses import dataclass

import torch

try:
    import triton
    import triton.language as tl
except ModuleNotFoundError as error:
    if error.name not in ("triton", "triton.language"):
        raise
    triton = None


_MAX_METADATA_BYTES = 16 * 1024**2
_RAW_TILE_ROWS = 32768
_MAX_REQUESTS = 1024


def _slice_rows(tensor, start, end):
    return tensor if start == 0 and end == tensor.shape[0] else tensor[start:end]


@dataclass(frozen=True)
class SlotLayout:
    table: torch.Tensor
    entries_per_block: int
    tokens_per_block: int
    owner_tokens_per_block: int
    cp_size: int
    cp_rank: int


@dataclass(frozen=True)
class ProducerMetadata:
    segments: dict
    indices: tuple
    positions: torch.Tensor
    requests: torch.Tensor
    main_slots: torch.Tensor
    index_slots: torch.Tensor
    state_slots: torch.Tensor | None
    seq_ends: torch.Tensor
    raw_indices: torch.Tensor | None = None
    key_counts: torch.Tensor | None = None
    previous_slots: torch.Tensor | None = None
    starts: torch.Tensor | None = None
    lengths: torch.Tensor | None = None

    def tile(self, start, end):
        _, first, stop, phase = self.segments[start]
        return (
            (
                _slice_rows(self.raw_indices, first, stop)
                if self.raw_indices is not None
                else _slice_rows(self.indices[phase], 0, stop - first)
            ),
            _slice_rows(self.positions, first, stop),
            _slice_rows(self.requests, first, stop),
            _slice_rows(self.main_slots, first, stop),
            _slice_rows(self.index_slots, first, stop),
            (
                _slice_rows(self.state_slots, start, end)
                if self.state_slots is not None
                else None
            ),
        )


def _raw_plan(cp, ratio, tile_rows=_RAW_TILE_ROWS):
    lengths = getattr(cp, "input_lengths_global_host", None)
    starts = getattr(cp, "prefix_lengths_host", None)
    if (
        type(tile_rows) is not int
        or tile_rows != _RAW_TILE_ROWS
        or getattr(cp, "cp_size", 0) < 1
        or ratio not in (1, 2)
        or not isinstance(lengths, (tuple, list))
        or not isinstance(starts, (tuple, list))
        or not 2 <= len(lengths) <= 128
        or len(starts) != len(lengths)
        or any(type(n) is not int or n <= 0 for n in lengths)
        or any(
            type(p) is not int or p < 0 or p > 2**63 - 1 - n
            for p, n in zip(starts, lengths)
        )
    ):
        return None
    rows = sum(lengths)
    count = sum((n + p % ratio) // ratio for p, n in zip(starts, lengths))
    required = 8 * (5 * count + (rows if ratio == 2 else 0) + len(lengths))
    if required > _MAX_METADATA_BYTES:
        return None
    segments = {}
    first = 0
    for start in range(0, rows, tile_rows):
        end = min(start + tile_rows, rows)
        offset = completed = 0
        for prefix, size in zip(starts, lengths):
            left, right = max(start, offset), min(end, offset + size)
            if right > left:
                phase = (ratio - 1 - (prefix + left - offset)) % ratio
                completed += (right - left + ratio - 1 - phase) // ratio
            offset += size
        segments[start] = (end, first, first + completed, 0)
        first += completed
    return segments, rows, count


def raw_tile_plan(lengths, prefixes, ratio, tile_rows=_RAW_TILE_ROWS):
    """Yield global row and closed-entry ranges without whole-batch row tables."""
    if (
        ratio not in (1, 2)
        or tile_rows != _RAW_TILE_ROWS
        or len(lengths) != len(prefixes)
        or not 1 <= len(lengths) <= _MAX_REQUESTS
        or any(type(n) is not int or n <= 0 for n in lengths)
        or any(
            type(p) is not int or p < 0 or p > 2**63 - 1 - n
            for p, n in zip(prefixes, lengths)
        )
    ):
        raise ValueError("invalid producer tile geometry")
    row_ends = [0]
    entry_ends = [0]
    for n, p in zip(lengths, prefixes):
        row_ends.append(row_ends[-1] + n)
        entry_ends.append(entry_ends[-1] + (n + p % ratio) // ratio)

    def count_before(row):
        request = bisect_right(row_ends, row) - 1
        if request == len(lengths):
            return entry_ends[-1]
        return (
            entry_ends[request]
            + (row - row_ends[request] + prefixes[request] % ratio) // ratio
        )

    for start in range(0, row_ends[-1], tile_rows):
        end = min(start + tile_rows, row_ends[-1])
        yield start, end, count_before(start), count_before(end)


def _slot_layouts_supported(layouts, ratio, batch, device):
    if not isinstance(layouts, tuple) or len(layouts) != (3 if ratio == 2 else 2):
        return False
    for i, layout in enumerate(layouts):
        if not isinstance(layout, SlotLayout):
            return False
        table = layout.table
        if not (
            isinstance(table, torch.Tensor)
            and table.device == device
            and table.ndim == 2
            and table.shape[0] == batch
            and table.shape[1] > 0
            and table.dtype in (torch.int32, torch.int64)
            and table.stride(1) == 1
            and table.stride(0) >= table.shape[1]
            and all(
                type(v) is int and v > 0
                for v in (
                    layout.entries_per_block,
                    layout.tokens_per_block,
                    layout.owner_tokens_per_block,
                    layout.cp_size,
                )
            )
            and type(layout.cp_rank) is int
            and 0 <= layout.cp_rank < layout.cp_size
            and (
                i == 2
                or (
                    layout.owner_tokens_per_block % layout.tokens_per_block == 0
                    and layout.tokens_per_block % ratio == 0
                    and layout.entries_per_block >= layout.tokens_per_block // ratio
                )
            )
        ):
            return False
    return (
        len({(x.cp_size, x.cp_rank) for x in layouts}) == 1
        and len({x.table.dtype for x in layouts}) == 1
    )


if triton is not None:

    @triton.jit
    def _raw_slot(
        pos,
        req,
        table,
        valid,
        cols,
        stride,
        end,
        EB: tl.constexpr,
        TPB: tl.constexpr,
        OWNER: tl.constexpr,
        CP: tl.constexpr,
        RANK: tl.constexpr,
        RATIO: tl.constexpr,
        STATE: tl.constexpr,
        HAS_END: tl.constexpr,
    ):
        # Same owner/page and STATE ring arithmetic as _prefill_slots_kernel.
        valid &= pos >= 0
        pos = tl.maximum(pos, 0)
        if STATE:
            raw_column = pos // TPB
            column = raw_column % cols
            offset = pos % (EB * CP)
            valid &= offset // EB == RANK
            offset %= EB
            if HAS_END:
                valid &= pos + EB * CP >= tl.minimum((raw_column + 1) * TPB, end)
        else:
            owner = pos // OWNER
            column = owner // CP * (OWNER // TPB) + pos % OWNER // TPB
            offset = pos % TPB // RATIO
            valid &= (owner % CP == RANK) & (column < cols)
            valid &= (pos + 1) % RATIO == 0
        block = tl.load(table + req * stride + column, valid, other=0).to(tl.int64)
        return tl.where(valid & (block > 0), block * EB + offset, -1)

    @triton.jit(
        do_not_specialize=[
            "BATCH",
            "ROW_START",
            "ROWS",
            "ENTRY_START",
            "ENTRIES",
            "SS",
            "LS",
            "MC",
            "IC",
            "SC",
            "MS",
            "IS",
            "TS",
        ],
        do_not_specialize_on_alignment=[
            "starts",
            "lengths",
            "scratch",
            "main_table",
            "index_table",
            "state_table",
        ],
    )
    def _fused_raw_tile_metadata_kernel(
        starts,
        lengths,
        scratch,
        main_table,
        index_table,
        state_table,
        BATCH,
        ROW_START,
        ROWS,
        ENTRY_START,
        ENTRIES,
        SS,
        LS,
        MC,
        IC,
        SC,
        MS,
        IS,
        TS,
        M: tl.constexpr,
        I: tl.constexpr,
        S: tl.constexpr,
        RATIO: tl.constexpr,
        CAPACITY: tl.constexpr,
        REQUEST_BLOCK: tl.constexpr,
        REQUEST_STEPS: tl.constexpr,
    ):
        lane = tl.arange(0, 128)
        request_lane = tl.arange(0, REQUEST_BLOCK)
        prefix = tl.load(starts + request_lane * SS, request_lane < BATCH, other=0).to(
            tl.int64
        )
        size = tl.load(lengths + request_lane * LS, request_lane < BATCH, other=0).to(
            tl.int64
        )
        row_end = tl.cumsum(size)
        row_begin = row_end - size
        entry_size = (size + prefix % RATIO) // RATIO
        entry_end = tl.cumsum(entry_size)
        entry_begin = entry_end - entry_size
        pid = tl.program_id(0)
        entry_blocks = tl.cdiv(ENTRIES, 128)

        if RATIO == 2 and pid < entry_blocks:
            local_entry = pid * 128 + lane
            global_entry = ENTRY_START + local_entry
            lo = tl.full((128,), 0, tl.int32)
            hi = tl.full((128,), REQUEST_BLOCK, tl.int32)
            for _ in range(REQUEST_STEPS):
                mid = (lo + hi) // 2
                stop = tl.gather(entry_end, tl.minimum(mid, REQUEST_BLOCK - 1), 0)
                advance = (mid < REQUEST_BLOCK) & (global_entry >= stop)
                lo = tl.where(advance, mid + 1, lo)
                hi = tl.where(advance, hi, mid)
            request = tl.minimum(lo, REQUEST_BLOCK - 1)
            begin = tl.gather(row_begin, request, 0)
            first = 1 - tl.gather(prefix, request, 0) % 2
            row = (
                begin + first + 2 * (global_entry - tl.gather(entry_begin, request, 0))
            )
            local_row = row - ROW_START
            position = tl.gather(prefix, request, 0) + row - begin
            valid = local_entry < ENTRIES
            main = _raw_slot(
                position,
                request,
                main_table,
                valid,
                MC,
                MS,
                0,
                M[0],
                M[1],
                M[2],
                M[3],
                M[4],
                RATIO,
                False,
                False,
            )
            index = _raw_slot(
                position,
                request,
                index_table,
                valid,
                IC,
                IS,
                0,
                I[0],
                I[1],
                I[2],
                I[3],
                I[4],
                RATIO,
                False,
                False,
            )
            tl.store(scratch + 2 * CAPACITY + local_entry, local_row, valid)
            tl.store(scratch + 3 * CAPACITY + local_entry, position, valid)
            tl.store(scratch + 4 * CAPACITY + local_entry, request, valid)
            tl.store(scratch + 5 * CAPACITY + local_entry, main, valid)
            tl.store(scratch + 6 * CAPACITY + local_entry, index, valid)
        else:
            local_row = (
                (pid - entry_blocks) * 128 + lane if RATIO == 2 else pid * 128 + lane
            ).to(tl.int64)
            row = ROW_START + local_row
            lo = tl.full((128,), 0, tl.int32)
            hi = tl.full((128,), REQUEST_BLOCK, tl.int32)
            for _ in range(REQUEST_STEPS):
                mid = (lo + hi) // 2
                stop = tl.gather(row_end, tl.minimum(mid, REQUEST_BLOCK - 1), 0)
                advance = (mid < REQUEST_BLOCK) & (row >= stop)
                lo = tl.where(advance, mid + 1, lo)
                hi = tl.where(advance, hi, mid)
            request = tl.minimum(lo, REQUEST_BLOCK - 1)
            position = (
                tl.gather(prefix, request, 0) + row - tl.gather(row_begin, request, 0)
            )
            valid = local_row < ROWS
            tl.store(scratch + local_row, position, valid)
            tl.store(scratch + CAPACITY + local_row, request, valid)
            if RATIO == 2:
                end = tl.gather(prefix, request, 0) + tl.gather(size, request, 0)
                state = _raw_slot(
                    position,
                    request,
                    state_table,
                    valid,
                    SC,
                    TS,
                    end,
                    S[0],
                    S[1],
                    S[2],
                    S[3],
                    S[4],
                    RATIO,
                    True,
                    True,
                )
                tl.store(scratch + 7 * CAPACITY + local_row, state, valid)
            else:
                main = _raw_slot(
                    position,
                    request,
                    main_table,
                    valid,
                    MC,
                    MS,
                    0,
                    M[0],
                    M[1],
                    M[2],
                    M[3],
                    M[4],
                    RATIO,
                    False,
                    False,
                )
                index = _raw_slot(
                    position,
                    request,
                    index_table,
                    valid,
                    IC,
                    IS,
                    0,
                    I[0],
                    I[1],
                    I[2],
                    I[3],
                    I[4],
                    RATIO,
                    False,
                    False,
                )
                tl.store(scratch + 2 * CAPACITY + local_row, local_row, valid)
                tl.store(scratch + 3 * CAPACITY + local_row, position, valid)
                tl.store(scratch + 4 * CAPACITY + local_row, request, valid)
                tl.store(scratch + 5 * CAPACITY + local_row, main, valid)
                tl.store(scratch + 6 * CAPACITY + local_row, index, valid)

    @triton.jit(
        do_not_specialize=[
            "BATCH",
            "ROWS",
            "COUNT",
            "PS",
            "RS",
            "SS",
            "LS",
            "MC",
            "IC",
            "SC",
            "MS",
            "IS",
            "TS",
        ],
        do_not_specialize_on_alignment=[
            "positions",
            "requests",
            "starts",
            "lengths",
            "indices",
            "boundary_pos",
            "boundary_req",
            "main_table",
            "index_table",
            "state_table",
            "main_slots",
            "index_slots",
            "state_slots",
            "previous_slots",
            "small_long",
            "key_counts",
        ],
    )
    def _fused_raw_metadata_kernel(
        positions,
        requests,
        starts,
        lengths,
        indices,
        boundary_pos,
        boundary_req,
        main_table,
        index_table,
        state_table,
        main_slots,
        index_slots,
        state_slots,
        previous_slots,
        small_long,
        key_counts,
        BATCH,
        ROWS,
        COUNT,
        PS,
        RS,
        SS,
        LS,
        MC,
        IC,
        SC,
        MS,
        IS,
        TS,
        M: tl.constexpr,
        I: tl.constexpr,
        S: tl.constexpr,
        RATIO: tl.constexpr,
        TILE_ROWS: tl.constexpr,
    ):
        ids = tl.arange(0, 128)
        prefix = tl.load(starts + ids * SS, ids < BATCH, other=0).to(tl.int64)
        size = tl.load(lengths + ids * LS, ids < BATCH, other=0).to(tl.int64)
        pid = tl.program_id(0)
        if pid == 0:
            tl.store(small_long + ids, prefix + size, ids < BATCH)
            tl.store(small_long + BATCH + ids, prefix, ids < BATCH)
            tl.store(small_long + 2 * BATCH + ids, size, ids < BATCH)
            tl.store(
                key_counts + ids, ((prefix + size) // RATIO).to(tl.int32), ids < BATCH
            )
            if RATIO == 2:
                previous = _raw_slot(
                    tl.maximum(prefix - 1, 0),
                    ids,
                    state_table,
                    ids < BATCH,
                    SC,
                    TS,
                    0,
                    S[0],
                    S[1],
                    S[2],
                    S[3],
                    S[4],
                    RATIO,
                    True,
                    False,
                )
                tl.store(previous_slots + ids, previous, ids < BATCH)
        if pid < tl.cdiv(COUNT, 128):
            row = pid * 128 + ids
            if RATIO == 1:
                # No compact search or redundant position/request copies.
                global_index = row
            else:
                offset = tl.cumsum(size) - size
                compact = (size + prefix % RATIO) // RATIO
                compact_end = tl.cumsum(compact)
                compact_start = compact_end - compact
                lo = tl.full((128,), 0, tl.int32)
                hi = tl.full((128,), 128, tl.int32)
                for _ in range(8):
                    mid = (lo + hi) // 2
                    end = tl.gather(compact_end, tl.minimum(mid, 127), 0)
                    advance = (mid < 128) & (row >= end)
                    lo = tl.where(advance, mid + 1, lo)
                    hi = tl.where(advance, hi, mid)
                req = tl.minimum(lo, 127)
                global_index = (
                    tl.gather(offset, req, 0)
                    + 1
                    - tl.gather(prefix, req, 0) % 2
                    + 2 * (row - tl.gather(compact_start, req, 0))
                )
            pos = tl.load(positions + global_index * PS, row < COUNT, other=0).to(
                tl.int64
            )
            req = tl.load(requests + global_index * RS, row < COUNT, other=0).to(
                tl.int64
            )
            valid = (row < COUNT) & (req >= 0) & (req < BATCH)
            main = _raw_slot(
                pos,
                req,
                main_table,
                valid,
                MC,
                MS,
                0,
                M[0],
                M[1],
                M[2],
                M[3],
                M[4],
                RATIO,
                False,
                False,
            )
            index = _raw_slot(
                pos,
                req,
                index_table,
                valid,
                IC,
                IS,
                0,
                I[0],
                I[1],
                I[2],
                I[3],
                I[4],
                RATIO,
                False,
                False,
            )
            tl.store(indices + row, global_index % TILE_ROWS, row < COUNT)
            tl.store(main_slots + row, main, row < COUNT)
            tl.store(index_slots + row, index, row < COUNT)
            if RATIO == 2:
                tl.store(boundary_pos + row, pos, row < COUNT)
                tl.store(boundary_req + row, req, row < COUNT)
        elif RATIO == 2:
            row = (pid - tl.cdiv(COUNT, 128)) * 128 + ids
            pos = tl.load(positions + row * PS, row < ROWS, other=0).to(tl.int64)
            req = tl.load(requests + row * RS, row < ROWS, other=0).to(tl.int64)
            valid = (row < ROWS) & (req >= 0) & (req < BATCH)
            # Read caller inputs, never CTA0's newly published seq_ends.
            end = tl.load(starts + req * SS, valid, other=0).to(tl.int64) + tl.load(
                lengths + req * LS, valid, other=0
            ).to(tl.int64)
            state = _raw_slot(
                pos,
                req,
                state_table,
                valid,
                SC,
                TS,
                end,
                S[0],
                S[1],
                S[2],
                S[3],
                S[4],
                RATIO,
                True,
                True,
            )
            tl.store(state_slots + row, state, row < ROWS)

    @triton.jit(do_not_specialize=["BATCH", "COUNT", "PS", "RS", "SS", "LS"])
    def _raw_boundaries_kernel(
        positions,
        requests,
        starts,
        lengths,
        indices,
        boundary_pos,
        boundary_req,
        seq_ends,
        BATCH,
        COUNT,
        PS,
        RS,
        SS,
        LS,
        RATIO: tl.constexpr,
        TILE_ROWS: tl.constexpr,
    ):
        ids = tl.arange(0, 128)
        prefix = tl.load(starts + ids * SS, ids < BATCH, other=0).to(tl.int64)
        size = tl.load(lengths + ids * LS, ids < BATCH, other=0).to(tl.int64)
        offset = tl.cumsum(size) - size
        compact = (size + prefix % RATIO) // RATIO
        compact_end = tl.cumsum(compact)
        compact_start = compact_end - compact
        row = tl.program_id(0) * 128 + ids
        # Upper-bound search skips requests with zero closed pairs. Every output
        # row maps to its original full-stream index, including request crossings.
        lo = tl.full((128,), 0, tl.int32)
        hi = tl.full((128,), 128, tl.int32)
        for _ in range(8):
            mid = (lo + hi) // 2
            end = tl.gather(compact_end, tl.minimum(mid, 127), 0)
            advance = (mid < 128) & (row >= end)
            lo = tl.where(advance, mid + 1, lo)
            hi = tl.where(advance, hi, mid)
        req = tl.minimum(lo, 127)
        phase = RATIO - 1 - tl.gather(prefix, req, 0) % RATIO
        global_index = (
            tl.gather(offset, req, 0)
            + phase
            + RATIO * (row - tl.gather(compact_start, req, 0))
        )
        valid = row < COUNT
        pos = tl.load(positions + global_index * PS, valid, other=0)
        request = tl.load(requests + global_index * RS, valid, other=0)
        tl.store(indices + row, global_index % TILE_ROWS, valid)
        tl.store(boundary_pos + row, pos, valid)
        tl.store(boundary_req + row, request, valid)
        if tl.program_id(0) == 0:
            tl.store(seq_ends + ids, prefix + size, ids < BATCH)


def prepare_raw(
    cp,
    positions,
    requests,
    starts,
    lengths,
    ratio,
    slots,
    regions,
    *,
    tile_rows=_RAW_TILE_ROWS,
    slot_layouts=None,
):
    """Metadata for global raw tiles, including request crossings.

    Host mirrors must describe the supplied GPU lengths/prefixes.
    Returns the same tile() tuple; never repartitions GEMMs or compressor rows.
    One integer launch builds compact boundaries/relative indices/sequence ends;
    existing slot callbacks run once per region. No H2D row table or retained cache.
    tile_rows must match the producer's fixed 32768-row slices.
    Warm reachable tile sizes, ratios and slot layouts before graph capture.
    Positive-stride int64 vectors are supported; unsupported inputs return None
    before allocation or launch.

    Optional slot_layouts replaces callbacks with one fused launch. Positions
    and requests stay int64; starts/lengths share int32 or int64 dtype, and all
    tables share int32 or int64 dtype. Fused pointers do not specialize on
    alignment. Each descriptor retains its own physical entries/page geometry.
    """
    plan = _raw_plan(cp, ratio, tile_rows)
    if plan is None or triton is None or not isinstance(positions, torch.Tensor):
        return None
    segments, rows, count = plan
    batch = len(cp.input_lengths_global_host)
    if (
        not positions.is_cuda
        or torch.version.hip is not None
        or len(regions) != 3
        or not callable(slots)
        or any(
            not isinstance(t, torch.Tensor)
            or t.shape != (n,)
            or t.dtype
            not in (
                (torch.int32, torch.int64)
                if slot_layouts is not None
                else (torch.int64,)
            )
            or t.device != positions.device
            or t.stride(0) <= 0
            for t, n in (
                (positions, rows),
                (requests, rows),
                (starts, batch),
                (lengths, batch),
            )
        )
    ):
        return None
    if slot_layouts is not None:
        if (
            positions.dtype != torch.int64
            or requests.dtype != torch.int64
            or starts.dtype != lengths.dtype
        ):
            return None
        if not _slot_layouts_supported(slot_layouts, ratio, batch, positions.device):
            return None
        required = (
            8
            * ((5 * count + rows + 4 * batch) if ratio == 2 else (3 * rows + 3 * batch))
            + 4 * batch
        )
        if required > _MAX_METADATA_BYTES:
            return None
        return _prepare_fused_raw(
            plan, positions, requests, starts, lengths, ratio, slot_layouts, tile_rows
        )
    boundaries = torch.empty((3, count), device=positions.device, dtype=torch.int64)
    indices, boundary_pos, boundary_req = boundaries.unbind(0)
    seq_ends = torch.empty((batch,), device=positions.device, dtype=torch.int64)
    _raw_boundaries_kernel[(max(1, triton.cdiv(count, 128)),)](
        positions,
        requests,
        starts,
        lengths,
        indices,
        boundary_pos,
        boundary_req,
        seq_ends,
        batch,
        count,
        positions.stride(0),
        requests.stride(0),
        starts.stride(0),
        lengths.stride(0),
        ratio,
        tile_rows,
        num_warps=4,
    )
    # Empty compact output is valid (e.g. all length-one even-prefix requests).
    # The existing slot fast path rejects empty vectors, so avoid calling it.
    main_slots = slots(regions[0], boundary_pos, boundary_req) if count else indices
    index_slots = slots(regions[1], boundary_pos, boundary_req) if count else indices
    state_slots = (
        slots(regions[2], positions, requests, state_end=seq_ends)
        if ratio == 2
        else None
    )
    if (
        main_slots is None
        or index_slots is None
        or (ratio == 2 and state_slots is None)
    ):
        return None
    return ProducerMetadata(
        segments,
        (),
        boundary_pos,
        boundary_req,
        main_slots,
        index_slots,
        state_slots,
        seq_ends,
        indices,
    )


def _prepare_fused_raw(
    plan, positions, requests, starts, lengths, ratio, layouts, tile_rows
):
    segments, rows, count = plan
    batch, device = starts.numel(), positions.device
    # key_counts has independent small storage: caching it cannot retain row buffers.
    key_counts = torch.empty(batch, dtype=torch.int32, device=device)
    small_long = torch.empty((3, batch), dtype=torch.int64, device=device)
    data = torch.empty(
        (5 if ratio == 2 else 3, count), dtype=torch.int64, device=device
    )
    if ratio == 2:
        indices, boundary_pos, boundary_req, main, index = data.unbind(0)
        state = torch.empty(rows, dtype=torch.int64, device=device)
        previous = torch.empty(batch, dtype=torch.int64, device=device)
    else:
        indices, main, index = data.unbind(0)
        boundary_pos, boundary_req = positions, requests
        state, previous = None, None
    m, i = layouts[:2]
    s = layouts[2] if ratio == 2 else m

    def constants(layout):
        return (
            layout.entries_per_block,
            layout.tokens_per_block,
            layout.owner_tokens_per_block,
            layout.cp_size,
            layout.cp_rank,
        )

    grid = max(
        1, triton.cdiv(count, 128) + (triton.cdiv(rows, 128) if ratio == 2 else 0)
    )
    _fused_raw_metadata_kernel[(grid,)](
        positions,
        requests,
        starts,
        lengths,
        indices,
        boundary_pos,
        boundary_req,
        m.table,
        i.table,
        s.table,
        main,
        index,
        state if state is not None else main,
        previous if previous is not None else main,
        small_long,
        key_counts,
        batch,
        rows,
        count,
        positions.stride(0),
        requests.stride(0),
        starts.stride(0),
        lengths.stride(0),
        m.table.shape[1],
        i.table.shape[1],
        s.table.shape[1],
        m.table.stride(0),
        i.table.stride(0),
        s.table.stride(0),
        constants(m),
        constants(i),
        constants(s),
        ratio,
        tile_rows,
        num_warps=4,
    )
    return ProducerMetadata(
        segments,
        (),
        boundary_pos,
        boundary_req,
        main,
        index,
        state,
        small_long[0],
        indices,
        key_counts,
        previous,
        small_long[1],
        small_long[2],
    )


def prepare_fused_tile(starts, lengths, ratio, slot_layouts, tile, scratch):
    """Generate one bounded source tile's row and pool metadata."""
    if triton is None:
        raise RuntimeError("Triton is required for fused source tile metadata")
    row_start, row_end, entry_start, entry_end = tile
    rows, entries = row_end - row_start, entry_end - entry_start
    if (
        starts.ndim != 1
        or lengths.shape != starts.shape
        or starts.dtype != torch.int64
        or lengths.dtype != torch.int64
        or starts.device != lengths.device
        or not starts.is_cuda
        or not _slot_layouts_supported(slot_layouts, ratio, len(starts), starts.device)
        or scratch.dtype != torch.int64
        or scratch.device != starts.device
        or scratch.shape != (8, _RAW_TILE_ROWS)
        or not scratch.is_contiguous()
        or not 0 < rows <= _RAW_TILE_ROWS
        or not 0 <= entries <= rows
        or row_start < 0
        or entry_start < 0
    ):
        raise ValueError("invalid fused source tile")
    m, i = slot_layouts[:2]
    state = slot_layouts[2] if ratio == 2 else m
    request_block = max(128, triton.next_power_of_2(len(starts)))

    def constants(layout):
        return (
            layout.entries_per_block,
            layout.tokens_per_block,
            layout.owner_tokens_per_block,
            layout.cp_size,
            layout.cp_rank,
        )

    _fused_raw_tile_metadata_kernel[
        (triton.cdiv(entries, 128) + triton.cdiv(rows, 128),)
    ](
        starts,
        lengths,
        scratch,
        m.table,
        i.table,
        state.table,
        len(starts),
        row_start,
        rows,
        entry_start,
        entries,
        starts.stride(0),
        lengths.stride(0),
        m.table.shape[1],
        i.table.shape[1],
        state.table.shape[1],
        m.table.stride(0),
        i.table.stride(0),
        state.table.stride(0),
        constants(m),
        constants(i),
        constants(state),
        ratio,
        _RAW_TILE_ROWS,
        request_block,
        request_block.bit_length(),
        num_warps=4,
    )
    state_slots = scratch[7, :rows] if ratio == 2 else None
    return (
        scratch[0, :rows],
        scratch[1, :rows],
        scratch[2, :entries],
        scratch[3, :entries],
        scratch[4, :entries],
        scratch[5, :entries],
        scratch[6, :entries],
        state_slots,
    )
