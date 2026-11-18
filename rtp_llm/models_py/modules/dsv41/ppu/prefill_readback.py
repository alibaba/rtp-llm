"""Batch the readback of CP-sharded full pools without changing their layout."""

import torch
import triton.language as tl
from triton import cdiv, jit

from rtp_llm.models_py.modules.dsv41._profiler import record_function_range
from rtp_llm.models_py.modules.dsv41.cp import cp_gather_rank_major
from rtp_llm.models_py.modules.dsv41.cp_cache import round_robin_coordinates_jit


_SCRATCH_BYTES = 128 * 1024**2


@jit
def _restore_rows(
    gathered,
    stripes,
    output,
    scale_bytes,
    stripe_count,
    OWNER_ROWS: tl.constexpr,
    PAGE_ROWS: tl.constexpr,
    CP_SIZE: tl.constexpr,
    BLOCK_BYTES: tl.constexpr,
    PAGE_BYTES: tl.constexpr,
    ENTRIES: tl.constexpr,
    WIDTH: tl.constexpr,
    PLANAR: tl.constexpr,
    SPLIT_INDEXER: tl.constexpr,
    TILE: tl.constexpr,
    PAGE_PLANAR_OUTPUT: tl.constexpr = False,
):
    stripe = tl.program_id(0)
    fields = 5 if PAGE_PLANAR_OUTPUT else 4
    count = tl.load(stripes + stripe * fields + 2)
    start = tl.load(stripes + stripe * fields + 3)
    byte = tl.program_id(1) * TILE + tl.arange(0, TILE)
    row, column = byte // WIDTH, byte % WIDTH
    owner, page, entry = round_robin_coordinates_jit(
        row, OWNER_ROWS, PAGE_ROWS, CP_SIZE
    )
    if PLANAR:
        page_offset = tl.where(
            column < 64, entry * 64 + column, ENTRIES * 64 + entry * 4 + column - 64
        )
    else:
        page_offset = entry * WIDTH + column
    block = (owner * stripe_count + stripe).to(tl.int64)
    valid = row < count
    value = tl.load(
        gathered + block * BLOCK_BYTES + page * PAGE_BYTES + page_offset, valid, other=0
    )
    if PAGE_PLANAR_OUTPUT:
        padded_count = tl.load(stripes + stripe * fields + 4)
        out_row = start + row
        out_page, out_entry = out_row // 256, out_row % 256
        out_byte = out_page * (256 * 68) + tl.where(
            column < 64, out_entry * 64 + column, 256 * 64 + out_entry * 4 + column - 64,
        )
        # Only the last stripe of a private tail owns the page padding. Its
        # extra stores write zero without touching another request's page.
        tl.store(output + out_byte, value, row < padded_count)
    elif SPLIT_INDEXER:
        is_payload = column < 64
        out_row = start + row
        tl.store(output + out_row * 64 + column, value, valid & is_payload)
        tl.store(scale_bytes + out_row * 4 + column - 64, value, valid & ~is_payload)
    else:
        tl.store(output + start * WIDTH + byte, value, valid)


def _stripe_groups(counts, stripe_rows, per_group, shared_prefix_stripes=0, *, compact=False, indexer_pages=False):
    # One stripe is one physical block per rank. Host lengths produce the
    # same groups on every rank, including ranks with no live tail rows.
    shared_rows = shared_prefix_stripes * stripe_rows
    if compact:
        from rtp_llm.models_py.modules.dsv41.ppu.cp_prefill_shared_global import compact_offsets

        tails, _ = compact_offsets(counts, shared_rows)
    if indexer_pages:
        if compact:
            raise ValueError("global compact and Indexer page layouts are independent")
        from rtp_llm.models_py.modules.dsv41.ppu.cp_prefill_shared_indexer import PAGE, page_layout

        layout = page_layout(counts, shared_rows)
        tails = tuple(page * PAGE for page in layout.tail_page_bases)
    stripes = []
    start = 0
    for request, count in enumerate(counts):
        for column, first in enumerate(range(0, count, stripe_rows)):
            if request and column < shared_prefix_stripes:
                continue
            destination = (first if not request and first < shared_rows
                           else tails[request] + first - shared_rows) if compact or indexer_pages else start + first
            live = min(count - first, stripe_rows)
            record = (request, column, live, destination)
            if indexer_pages:
                padded = live
                if first >= shared_rows and first + live == count:
                    tail_pages = (count - shared_rows + PAGE - 1) // PAGE
                    padded = (layout.tail_page_bases[request] + tail_pages) * PAGE - destination
                record += (padded,)
            stripes.append(record)
        start += count
    for first in range(0, len(stripes), per_group):
        group = stripes[first : first + per_group]
        yield len(group), max(row[4] if indexer_pages else row[2] for row in group), group


def gather_pool(
    pool, placement, counts, *, width, planar, split_indexer=False, plans=None,
    receive=None, shared_prefix_stripes=0, compact_shared_prefix=False, page_planar_indexer=False,
):
    """Return logical views, or an explicitly requested shared global bank.

    A split Indexer read writes contiguous FP4 payload and packed scale planes
    directly, returning (int8 [N, 64], int32 [N]) for each request.
    Compact global output retains a proven prefix once; its BF16 consumer
    remaps request tails rather than materializing duplicate prefix bytes.
    """
    if split_indexer and (not planar or width != 68 or pool.tensor.dtype != torch.uint8):
        raise ValueError("split Indexer readback requires planar 68-byte rows")

    size, ratio = placement.size, placement.ratio
    owner_rows, page_rows = placement.physical_span // ratio, pool.span // ratio
    if (
        pool.span % ratio
        or page_rows <= 0
        or pool.entries < page_rows
        or owner_rows != page_rows * placement.pages
    ):
        raise ValueError("CP readback requires complete compressed kernel pages")

    stripe_rows = size * owner_rows
    shared_rows = shared_prefix_stripes * stripe_rows
    if shared_rows and (len(counts) < 2 or shared_rows > min(counts)
                        or (not page_planar_indexer and len(set(counts)) != 1)):
        raise ValueError("shared readback prefix requires complete stripes and compatible lengths")
    if compact_shared_prefix and (
        size != 8 or width != 288 or planar or split_indexer
        or pool.tensor.dtype != torch.uint8 or not shared_rows
    ):
        raise ValueError("compact global readback requires a proven CP8 288-byte shared prefix")
    if page_planar_indexer and (
        size != 8 or width != 68 or not planar or pool.tensor.dtype != torch.uint8
        or not shared_rows or shared_rows % 256 or compact_shared_prefix
    ):
        raise ValueError("paged Indexer readback requires a page-aligned CP8 shared prefix")
    table = placement.table
    required_columns = cdiv(max(counts, default=0), stripe_rows)
    if table.shape[0] < len(counts) or table.shape[1] < required_columns:
        raise ValueError("native CP physical table is too short for batched readback")

    start = sum(counts) - (len(counts) - 1) * shared_rows if compact_shared_prefix else sum(counts)
    if page_planar_indexer:
        from rtp_llm.models_py.modules.dsv41.ppu.cp_prefill_shared_indexer import PAGE, WIDTH, page_layout

        layout = page_layout(counts, shared_rows)
        # Keep a versioned private owner even when the model forward runs
        # under inference_mode; public aliases must invalidate on mutation.
        with torch.inference_mode(False):
            output = pool.tensor.new_empty((layout.pages, PAGE * WIDTH))
    else:
        output = pool.tensor.new_empty((start, 64 if split_indexer else width))
    scale_bytes = pool.tensor.new_empty((start, 4)) if split_indexer and not page_planar_indexer else None
    page_bytes = pool.tensor.shape[1]
    block_bytes = placement.pages * page_bytes
    blocks = pool.tensor.view(-1, block_bytes)
    per_group = max(1, _SCRATCH_BYTES // ((size + 1) * block_bytes))
    if plans is None:
        groups = _stripe_groups(counts, stripe_rows, per_group, shared_prefix_stripes,
                                compact=compact_shared_prefix, indexer_pages=page_planar_indexer)
    else:
        from rtp_llm.models_py.modules.dsv41.fp8._v41_swa_metadata import _upload

        key = ("pool_readback", tuple(counts), stripe_rows, per_group, shared_prefix_stripes,
               compact_shared_prefix, page_planar_indexer, table.device)
        groups = plans.get(key)
        if groups is None:
            groups = tuple(
                (size, rows, _upload(group, table.device, torch.int64))
                for size, rows, group in _stripe_groups(counts, stripe_rows, per_group, shared_prefix_stripes,
                                                       compact=compact_shared_prefix, indexer_pages=page_planar_indexer)
            )
            plans[key] = groups
    for group_size, rows, group in groups:
        plan = torch.tensor(group, device=table.device, dtype=torch.int64) if plans is None else group
        physical = table[plan[:, 0], plan[:, 1]]
        live = plan[:, 2] > placement.rank * owner_rows
        # Nonowners send a dummy block; their bytes are never consumed.
        physical = torch.where(live, physical, 0)
        local = blocks.index_select(0, physical)
        with record_function_range("dsv41.ppu.prefill.pool_blocks.batch"):
            gathered = receive.gather(local) if receive is not None else cp_gather_rank_major(local, size)
        _restore_rows[(group_size, cdiv(rows * width, 1024))](
            gathered,
            plan,
            output,
            scale_bytes,
            group_size,
            owner_rows,
            page_rows,
            size,
            block_bytes,
            page_bytes,
            pool.entries,
            width,
            planar,
            split_indexer and not page_planar_indexer,
            1024,
            **({"PAGE_PLANAR_OUTPUT": True} if page_planar_indexer else {}),
        )
        # Drop the local packet and this view before constructing the next group.
        del local, gathered
    if compact_shared_prefix:
        from rtp_llm.models_py.modules.dsv41.ppu.cp_prefill_shared_global import SharedGlobalBank

        return SharedGlobalBank.create(output, shared_rows, counts)
    if page_planar_indexer:
        from rtp_llm.models_py.modules.dsv41.ppu.cp_prefill_shared_indexer import IndexerReadbackPages

        return IndexerReadbackPages.create(output, counts, shared_rows)
    if shared_rows:
        # Keep each consumer's logical request layout and packed bytes.
        view = output.view(len(counts), counts[0], -1)
        view[1:, :shared_rows].copy_(view[:1, :shared_rows])
        if scale_bytes is not None:
            view = scale_bytes.view(len(counts), counts[0], 4)
            view[1:, :shared_rows].copy_(view[:1, :shared_rows])
    if scale_bytes is not None:
        return tuple(
            zip(
                output.view(torch.int8).split(counts),
                scale_bytes.view(torch.int32).reshape(-1).split(counts),
            )
        )
    return output.split(counts)
