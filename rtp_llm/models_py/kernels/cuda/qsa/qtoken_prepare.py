"""Prepare FlashInfer G8 QToken routes and publish Qwen4-M11's BF16 KV pages."""

import torch
import triton
import triton.language as tl

from .qtoken_qkv import prepare_qkv


@triton.jit
def _request_for_token(cu, token, requests, steps: tl.constexpr):
    begin = 0
    end = requests
    for _ in range(steps):
        middle = (begin + end) // 2
        boundary = tl.load(cu + middle + 1, middle < requests, other=2147483647)
        later = boundary <= token
        begin = tl.where(later, middle + 1, begin)
        end = tl.where(later, end, middle)
    return begin


@triton.jit
def _selected_blocks(
    selected,
    first,
    n,
    visible,
    stride,
    query,
    slot,
    INDEX_STRIDE: tl.constexpr,
):
    index = (first + query).to(tl.int64) * stride + slot * INDEX_STRIDE
    logical = tl.load(selected + index, (query < n) & (slot < 512), other=-1)
    block = logical // INDEX_STRIDE
    complete = (logical >= 0) & (block < visible // 4)
    partial = (slot == 512) & (visible % 4 != 0)
    return tl.where(partial, visible // 4, block), (query < n) & (complete | partial)


@triton.jit
def _store_locators(
    block,
    member,
    rank,
    live,
    table,
    page_ids,
    masks,
    columns,
    pages,
    BLOCKS_PER_PAGE: tl.constexpr,
    LOCATOR_HEADS: tl.constexpr,
):
    block = block.to(tl.int32)
    page = block // BLOCKS_PER_PAGE
    phys = tl.load(table + page, live & (page < columns), other=0)
    valid = (phys > 0) & (phys < pages)
    locator = phys * (BLOCKS_PER_PAGE * LOCATOR_HEADS) + block % BLOCKS_PER_PAGE
    tl.store(page_ids + rank, tl.where(valid, locator, -1), live)
    tl.store(masks + rank, tl.where(valid, member, 0), live)


@triton.jit
def _or_membership(hist, block, query, valid):
    shift = (block % 4 * 8 + query).to(tl.uint32)
    tl.inline_asm_elementwise(
        "{ .reg .pred use; setp.ne.u32 use, $3, 0; @use atom.shared.or.b32 $0, [$1], $2; }",
        constraints="=r,r,r,r",
        args=(hist + (block // 4) * 4, 1 << shift, valid.to(tl.int32)),
        dtype=tl.uint32,
        is_pure=False,
        pack=1,
    )


@triton.jit
def _write_histogram_tile(
    hist,
    word,
    active_words,
    base,
    table,
    page_ids,
    masks,
    columns,
    pages,
    BLOCKS_PER_PAGE: tl.constexpr,
    LOCATOR_HEADS: tl.constexpr,
    MASKED: tl.constexpr,
):
    if MASKED:
        packed = tl.inline_asm_elementwise(
            "{ .reg .pred use; setp.ne.u32 use, $2, 0; mov.u32 $0, 0;"
            " @use ld.volatile.shared.b32 $0, [$1]; }",
            constraints="=r,r,r",
            args=(hist + word * 4, (word < active_words).to(tl.int32)),
            dtype=tl.uint32,
            is_pure=False,
            pack=1,
        )
    else:
        packed = tl.inline_asm_elementwise(
            "ld.volatile.shared.b32 $0, [$1];",
            constraints="=r,r",
            args=(hist + word * 4,),
            dtype=tl.uint32,
            is_pure=False,
            pack=1,
        )
    flags = packed | (packed >> 4)
    flags |= flags >> 2
    flags = (flags | (flags >> 1)) & 0x01010101
    counts = ((flags * 0x01010101) >> 24).to(tl.int32)
    rank = base + tl.cumsum(counts) - counts
    for byte in tl.static_range(4):
        member = (packed >> (byte * 8)) & 255
        live = member != 0
        _store_locators(
            word * 4 + byte,
            member,
            rank,
            live,
            table,
            page_ids,
            masks,
            columns,
            pages,
            BLOCKS_PER_PAGE,
            LOCATOR_HEADS,
        )
        rank += live.to(tl.int32)
    return tl.sum(counts, 0)


@triton.jit
def _histogram_route(
    selected,
    first,
    n,
    first_visible,
    stride,
    table,
    page_ids,
    masks,
    columns,
    pages,
    BLOCKS_PER_PAGE: tl.constexpr,
    LOCATOR_HEADS: tl.constexpr,
    HISTOGRAM_BLOCKS: tl.constexpr,
    INDEX_STRIDE: tl.constexpr,
    TILED: tl.constexpr,
    SINGLETON_HISTOGRAM: tl.constexpr,
):
    # Four sparse blocks share a u32. Its four bytes store eight query bits each.
    words: tl.constexpr = HISTOGRAM_BLOCKS // 4
    hist = tl.inline_asm_elementwise(
        f"{{ .shared .align 16 .b32 members[{words}]; mov.u32 $0, members; }}",
        constraints="=r",
        args=(),
        dtype=tl.uint32,
        is_pure=True,
        pack=1,
    )
    word = tl.arange(0, words)
    active_words = tl.cdiv(tl.maximum(first_visible + n - 1, 0), 16)
    if TILED:
        tl.inline_asm_elementwise(
            "{ .reg .pred use; setp.ne.u32 use, $2, 0;"
            " @use st.shared.b32 [$1], 0; mov.u32 $0, 0; }",
            constraints="=r,r,r",
            args=(hist + word * 4, (word < active_words).to(tl.int32)),
            dtype=tl.uint32,
            is_pure=False,
            pack=1,
        )
    else:
        tl.inline_asm_elementwise(
            "st.shared.b32 [$1], 0; mov.u32 $0, 0;",
            constraints="=r,r",
            args=(hist + word * 4,),
            dtype=tl.uint32,
            is_pure=False,
            pack=1,
        )
    tl.debug_barrier()

    # Padded draft routes cover one query even when the live CU changes on replay.
    QUERY_CAPACITY: tl.constexpr = 1 if SINGLETON_HISTOGRAM else 8
    item = tl.arange(0, QUERY_CAPACITY * 512)
    query = item // 512
    logical, valid = _selected_blocks(
        selected,
        first,
        n,
        first_visible + query,
        stride,
        query,
        item % 512,
        INDEX_STRIDE,
    )
    _or_membership(hist, logical, query, valid & (logical < HISTOGRAM_BLOCKS))
    tail_query = tl.arange(0, QUERY_CAPACITY)
    tail_visible = first_visible + tail_query
    _or_membership(
        hist,
        tail_visible // 4,
        tail_query,
        (tail_query < n)
        & (tail_visible % 4 != 0)
        & (tail_visible // 4 < HISTOGRAM_BLOCKS),
    )
    tl.debug_barrier()

    if TILED:
        # Scan only the visible prefix, and keep the route's metadata branch
        # from holding a full long-context histogram in registers.
        tile: tl.constexpr = min(words, 1024)
        count = 0
        for start in range(tl.cdiv(active_words, tile)):
            tile_word = start * tile + tl.arange(0, tile)
            count += _write_histogram_tile(
                hist,
                tile_word,
                active_words,
                count,
                table,
                page_ids,
                masks,
                columns,
                pages,
                BLOCKS_PER_PAGE,
                LOCATOR_HEADS,
                True,
            )
    else:
        count = _write_histogram_tile(
            hist,
            word,
            words,
            0,
            table,
            page_ids,
            masks,
            columns,
            pages,
            BLOCKS_PER_PAGE,
            LOCATOR_HEADS,
            False,
        )
    return count


@triton.jit
def _single_query_route(
    selected,
    row,
    visible,
    stride,
    table,
    page_ids,
    masks,
    columns,
    pages,
    BLOCKS_PER_PAGE: tl.constexpr,
    LOCATOR_HEADS: tl.constexpr,
    INDEX_STRIDE: tl.constexpr,
):
    # Complete selected blocks of a singleton route can stay in indexer
    # order. Only its implicit partial block must be last for causal masking.
    slot = tl.arange(0, 512)
    pos = tl.load(selected + row.to(tl.int64) * stride + slot * INDEX_STRIDE)
    block = pos // INDEX_STRIDE
    valid = (pos >= 0) & (block < visible // 4)
    rank = tl.cumsum(valid.to(tl.int32)) - 1
    _store_locators(
        block,
        1,
        rank,
        valid,
        table,
        page_ids,
        masks,
        columns,
        pages,
        BLOCKS_PER_PAGE,
        LOCATOR_HEADS,
    )
    count = tl.sum(valid.to(tl.int32), 0)
    partial = visible % 4 != 0
    _store_locators(
        visible // 4,
        1,
        count,
        partial,
        table,
        page_ids,
        masks,
        columns,
        pages,
        BLOCKS_PER_PAGE,
        LOCATOR_HEADS,
    )
    return count + partial.to(tl.int32)


@triton.jit
def _shared_load(table, slot):
    return tl.inline_asm_elementwise(
        "ld.volatile.shared.b32 $0, [$1];",
        constraints="=r,r",
        args=(table + slot * 4,),
        dtype=tl.uint32,
        is_pure=False,
        pack=1,
    )


@triton.jit
def _hash_insert(table, slot, entry, active):
    return tl.inline_asm_elementwise(
        """{
            .reg .pred use, match;
            .reg .b32 prev, old_key, key, bit;
            setp.ne.u32 use, $3, 0;
            mov.b32 prev, 0;
            @use atom.shared.cas.b32 prev, [$1], 0, $2;
            and.b32 old_key, prev, 0xffffff00;
            and.b32 key, $2, 0xffffff00;
            and.b32 bit, $2, 255;
            setp.eq.u32 match, old_key, key;
            and.pred match, match, use;
            @match atom.shared.or.b32 old_key, [$1], bit;
            mov.b32 $0, prev;
        }""",
        constraints="=r,r,r,r",
        args=(table + slot * 4, entry, active.to(tl.int32)),
        dtype=tl.uint32,
        is_pure=False,
        pack=1,
    )


@triton.jit
def _hash_route(
    selected,
    first,
    n,
    first_visible,
    stride,
    table,
    page_ids,
    masks,
    columns,
    pages,
    BLOCKS_PER_PAGE: tl.constexpr,
    LOCATOR_HEADS: tl.constexpr,
    INDEX_STRIDE: tl.constexpr,
):
    # At most 4096 historical candidates keep this table at most half full.
    capacity: tl.constexpr = 8192
    bins = tl.inline_asm_elementwise(
        "{ .shared .align 16 .b32 bins[8193]; mov.u32 $0, bins; }",
        constraints="=r",
        args=(),
        dtype=tl.uint32,
        is_pure=True,
        pack=1,
    )
    slot = tl.arange(0, capacity)
    tl.inline_asm_elementwise(
        "st.shared.b32 [$1], 0; mov.u32 $0, 0;",
        constraints="=r,r",
        args=(bins + slot * 4,),
        dtype=tl.uint32,
        is_pure=False,
        pack=1,
    )
    tl.inline_asm_elementwise(
        "st.shared.b32 [$1], 0; mov.u32 $0, 0;",
        constraints="=r,r",
        args=(bins + capacity * 4,),
        dtype=tl.uint32,
        is_pure=False,
        pack=1,
    )
    tl.debug_barrier()
    item = tl.arange(0, 4096)
    query = item // 512
    visible = first_visible + query
    boundary = (first_visible - 1) // 4
    block, valid = _selected_blocks(
        selected,
        first,
        n,
        visible,
        stride,
        query,
        item % 512,
        INDEX_STRIDE,
    )
    # Only atoms intersecting the route's live query rows require order.
    tail = block - boundary
    _or_membership(bins + capacity * 4, tail, query, valid & (tail >= 0))
    lane = tl.arange(0, 8)
    visible_tail = first_visible + lane
    tail_slot = visible_tail // 4 - boundary
    _or_membership(
        bins + capacity * 4,
        tail_slot,
        lane,
        (lane < n) & (visible_tail % 4 != 0),
    )
    pending = valid & (tail < 0)
    key = (block + 1).to(tl.uint32)
    entry = (key << 8) | (1 << query.to(tl.uint32))
    hash_key = key * 0x9E3779B1
    bucket = (hash_key ^ (hash_key >> 16)) & (capacity - 1)
    while tl.sum(pending.to(tl.int32), 0) > 0:
        prior = _hash_insert(bins, bucket, entry, pending)
        pending &= (prior != 0) & ((prior >> 8) != key)
        bucket = (bucket + 1) & (capacity - 1)
    tl.debug_barrier()
    compact = _shared_load(bins, slot)
    live = compact != 0
    rank = tl.cumsum(live.to(tl.int32)) - 1
    _store_locators(
        (compact >> 8) - 1,
        compact & 255,
        rank,
        live,
        table,
        page_ids,
        masks,
        columns,
        pages,
        BLOCKS_PER_PAGE,
        LOCATOR_HEADS,
    )
    count = tl.sum(live.to(tl.int32), 0)
    future = tl.arange(0, 4)
    remainder = _shared_load(bins, capacity)
    member = (remainder >> (future * 8)) & 255
    live = member != 0
    rank = count + tl.cumsum(live.to(tl.int32)) - 1
    _store_locators(
        boundary + future,
        member,
        rank,
        live,
        table,
        page_ids,
        masks,
        columns,
        pages,
        BLOCKS_PER_PAGE,
        LOCATOR_HEADS,
    )
    return count + tl.sum(live.to(tl.int32), 0)


@triton.jit
def _zero_tail(
    cache,
    table,
    visible,
    padding,
    columns,
    pages,
    CACHE_STRIDE: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
):
    # Only the partial final atom shares full TMA loads with visible tokens.
    # Clear its masked K/V slots so 0 * NaN in the attention kernel stays safe.
    column = (visible - 1) // PAGE_SIZE
    page = tl.load(table + column, (column >= 0) & (column < columns), other=0)
    if (page > 0) & (page < pages):
        feature = tl.arange(0, 8 * 256)
        head, dim = feature // 256, feature % 256
        offset = (
            page.to(tl.int64) * CACHE_STRIDE
            + head * (PAGE_SIZE * 256)
            + ((visible - 1) % PAGE_SIZE) * 256
            + dim
        )
        for i in tl.static_range(1, 4):
            tl.store(cache + offset + i * 256, 0, i <= padding)


@triton.jit(
    do_not_specialize=[
        "ROUTES",
        "REQUESTS",
        "TABLE_COLUMNS",
        "TABLE_STRIDE",
        "CACHE_PAGES",
        "SELECTED_STRIDE",
        "PAGE_CAPACITY",
        "MEMBERSHIP_BYTES",
    ],
)
def _qsa_qtoken_prepare(
    Qkv,
    Selected,
    Positions,
    QGamma,
    KGamma,
    Cache,
    Table,
    Cu,
    Prefixes,
    Routes,
    Query,
    PageIndices,
    MembershipBytes,
    Lengths,
    ROUTES,
    REQUESTS,
    TABLE_COLUMNS,
    TABLE_STRIDE,
    CACHE_PAGES,
    SELECTED_STRIDE,
    PAGE_CAPACITY,
    MEMBERSHIP_BYTES,
    CACHE_STRIDE: tl.constexpr,
    QKV_STRIDE: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    LOCATOR_HEADS: tl.constexpr,
    SEARCH_STEPS: tl.constexpr,
    HEAD_TILES: tl.constexpr,
    BLOCK_HEADS: tl.constexpr,
    RMS_ROWS: tl.constexpr,
    EPS: tl.constexpr,
    PREFILL: tl.constexpr,
    DECODE_Q1: tl.constexpr,
    HISTOGRAM_BLOCKS: tl.constexpr,
    Q_ONLY: tl.constexpr,
    PADDED_ROWS: tl.constexpr,
    INDEX_STRIDE: tl.constexpr,
    ZERO_TAIL: tl.constexpr,
    TILED_HISTOGRAM: tl.constexpr,
    BASE_ROPE: tl.constexpr,
    SINGLETON_HISTOGRAM: tl.constexpr,
    INTERLEAVE_PREFILL: tl.constexpr,
):
    pid = tl.program_id(0)
    if INTERLEAVE_PREFILL:
        tl.static_assert(
            BASE_ROPE
            and PREFILL
            and not DECODE_Q1
            and not Q_ONLY
            and not PADDED_ROWS
            and SEARCH_STEPS == 0
        )
        # Interleave each eight-token route with its QKV tiles. The grid size
        # truncates the last group to the live token count.
        GROUP_CTAS: tl.constexpr = 1 + 8 * HEAD_TILES
        group = pid // GROUP_CTAS
        offset = pid % GROUP_CTAS
        pid = tl.where(offset == 0, group, ROUTES + pid - group - 1)
    if pid < ROUTES:
        if DECODE_Q1:
            first = pid
            first_visible = tl.load(Prefixes + pid)
            request = pid
            n = 1
        else:
            first = tl.load(Routes + pid)
            finish = tl.load(Routes + pid + 1)
            request = _request_for_token(Cu, first, REQUESTS, SEARCH_STEPS)
            if PADDED_ROWS:
                finish = tl.minimum(finish, tl.load(Cu + REQUESTS))
                request = tl.minimum(request, REQUESTS - 1)
            first_visible = (
                tl.load(Prefixes + request) + first - tl.load(Cu + request) + 1
            )
            n = finish - first
        table = Table + request.to(tl.int64) * TABLE_STRIDE
        page_ids = PageIndices + pid.to(tl.int64) * PAGE_CAPACITY
        masks = MembershipBytes + pid.to(tl.int64) * MEMBERSHIP_BYTES
        count = 0
        if DECODE_Q1:
            column = (first_visible - 1) // PAGE_SIZE
            phys = tl.load(
                table + column, (column >= 0) & (column < TABLE_COLUMNS), other=0
            )
            if (phys > 0) & (phys < CACHE_PAGES):
                count = _single_query_route(
                    Selected,
                    first,
                    first_visible,
                    SELECTED_STRIDE,
                    table,
                    page_ids,
                    masks,
                    TABLE_COLUMNS,
                    CACHE_PAGES,
                    PAGE_SIZE // 4,
                    LOCATOR_HEADS,
                    INDEX_STRIDE,
                )
        elif HISTOGRAM_BLOCKS:
            count = _histogram_route(
                Selected,
                first,
                n,
                first_visible,
                SELECTED_STRIDE,
                table,
                page_ids,
                masks,
                TABLE_COLUMNS,
                CACHE_PAGES,
                PAGE_SIZE // 4,
                LOCATOR_HEADS,
                HISTOGRAM_BLOCKS,
                INDEX_STRIDE,
                TILED_HISTOGRAM,
                SINGLETON_HISTOGRAM,
            )
        else:
            count = _hash_route(
                Selected,
                first,
                n,
                first_visible,
                SELECTED_STRIDE,
                table,
                page_ids,
                masks,
                TABLE_COLUMNS,
                CACHE_PAGES,
                PAGE_SIZE // 4,
                LOCATOR_HEADS,
                INDEX_STRIDE,
            )
        # The final logical block is the only partially visible fragment at the
        # end of the route. Attention masks its tail using this compact length.
        last_visible = first_visible + n - 1
        padding = (-last_visible) & 3
        tl.store(Lengths + pid, tl.where(count > 0, count * 4 - padding, 1))
        tl.store(page_ids, -1, count == 0)
        end = tl.maximum(1, count)
        byte = count + tl.arange(0, 4)
        tl.store(masks + byte, 0, byte < tl.cdiv(end, 4) * 4)
        if ZERO_TAIL:
            if DECODE_Q1:
                last_route = count > 0
            else:
                last_route = (n > 0) & (first + n == tl.load(Cu + request + 1))
            if last_route & (padding != 0):
                _zero_tail(
                    Cache,
                    table,
                    last_visible,
                    padding,
                    TABLE_COLUMNS,
                    CACHE_PAGES,
                    CACHE_STRIDE,
                    PAGE_SIZE,
                )
    elif Qkv is not None:
        token = (pid - ROUTES) // HEAD_TILES
        head_tile = (pid - ROUTES) % HEAD_TILES
        active = True
        if PADDED_ROWS:
            active = token < tl.load(Cu + REQUESTS)
        if Q_ONLY or not active:
            feature = head_tile * BLOCK_HEADS * 256 + tl.arange(0, BLOCK_HEADS * 256)
            live_q = feature < 64 * 256
            if Q_ONLY:
                value = tl.load(
                    Qkv + token.to(tl.int64) * QKV_STRIDE + feature,
                    live_q & active,
                    other=0,
                )
            else:
                value = tl.full((BLOCK_HEADS * 256,), 0, tl.bfloat16)
            tl.store(Query + token.to(tl.int64) * (64 * 256) + feature, value, live_q)
        else:
            if DECODE_Q1:
                request = token
                position = tl.load(Prefixes + token) - 1
            else:
                request = _request_for_token(Cu, token, REQUESTS, SEARCH_STEPS)
                position = tl.load(Prefixes + request) + token - tl.load(Cu + request)
            column = position // PAGE_SIZE
            phys = tl.load(
                Table + request.to(tl.int64) * TABLE_STRIDE + column,
                (column >= 0) & (column < TABLE_COLUMNS),
                other=0,
            )
            prepare_qkv(
                Qkv,
                Query,
                Positions,
                QGamma,
                KGamma,
                Cache,
                token,
                head_tile,
                position,
                phys,
                (phys > 0) & (phys < CACHE_PAGES),
                QKV_STRIDE=QKV_STRIDE,
                CACHE_STRIDE=CACHE_STRIDE,
                PAGE_SIZE=PAGE_SIZE,
                RMS_ROWS=RMS_ROWS,
                EPS=EPS,
                BLOCK_HEADS=BLOCK_HEADS,
                PREFILL=PREFILL,
                BASE_ROPE=BASE_ROPE,
            )


def prepare_qtoken(
    qkv,
    selected,
    positions,
    q_gamma,
    k_gamma,
    cache,
    table,
    cu,
    prefixes,
    route_offsets,
    query,
    page_indices,
    memberships,
    seq_lens,
    *,
    max_kv_len,
    eps,
    prefill,
    locator_heads=1,
    decode_q1=False,
    q_only=False,
    padded_rows=False,
    zero_tail=False,
    base_rope=False,
):
    """Fill packed Q and FlashInfer's bound locator, mask and length buffers.

    Each route is a nonempty contiguous subset of at most eight tokens in one
    request. Prefixes are counts of prior tokens. For ordinary Q1 decode the
    input is instead the already-visible length, and each route is a singleton.
    The indexer can supply 512 logical block IDs or the existing expanded 2051
    token IDs. The causal partial block comes from each query's visible length.

    With qkv=None, the caller has already written Q and its KV pages. q_only
    additionally compacts the query view after an existing writer. In padded
    graph prefill, route offsets may include singleton rows after the live CU.
    base_rope selects the uncached text RoPE with three-axis position storage.
    zero_tail requires the final partially written atom to be on a private
    page. It clears masked slots that FlashInfer may still read with TMA.
    """
    tokens = selected.shape[0]
    requests = table.shape[0]
    routes, page_capacity = page_indices.shape
    if not tokens or not requests or not routes:
        raise ValueError(
            "QToken preparation requires nonempty tokens, requests and routes"
        )
    if not 0 < max_kv_len <= min(table.shape[1] * 128, 2**26 - 4):
        raise ValueError(
            "QToken maximum KV length exceeds the page-table or route-key range"
        )
    if decode_q1 and not requests == routes == tokens:
        raise ValueError("Q1 decode requires one QToken route per request")
    if selected.shape[1] not in (512, 2051) or selected.stride(1) != 1:
        raise ValueError("QToken selection must contain 512 blocks or 2051 token IDs")
    if q_only and qkv is not None and qkv.data_ptr() == query.data_ptr():
        qkv = None
    histogram_blocks = 0
    if not decode_q1 and max_kv_len <= 131072:
        histogram_blocks = triton.next_power_of_2(max(4, triton.cdiv(max_kv_len, 4)))
    tiled_histogram = histogram_blocks > 16384 or (
        histogram_blocks > 8192 and tokens >= 2048
    )
    tune_base = (
        base_rope
        and qkv is not None
        and not q_only
        and requests == 1
        and torch.cuda.get_device_capability(qkv.device) == (10, 3)
    )
    singleton_histogram = (
        tune_base and padded_rows and routes == tokens == 4 and histogram_blocks == 8192
    )
    block_heads = 16 if prefill else 8
    num_warps = 8 if histogram_blocks > 16384 or singleton_histogram else 4
    if tune_base and decode_q1:
        # Finer head tiles spread one projected token across more SMs.
        block_heads = 4
    head_tiles = triton.cdiv(64 if q_only else 72, block_heads)
    launch_options = {}
    interleave_prefill = False
    if (
        qkv is not None
        and not q_only
        and 2048 <= tokens
        and max_kv_len <= tokens
        and 0 < histogram_blocks <= 16384
    ):
        # The short metadata branch can otherwise limit residency for all QKV
        # CTAs on cold prefill. The cap is counterproductive for short batches.
        tuned_prefill = tune_base and prefill and not padded_rows and tokens <= 16464
        launch_options["maxnreg"] = 56 if tuned_prefill else 64
        interleave_prefill = (
            tuned_prefill and routes >= 512 and routes == triton.cdiv(tokens, 8)
        )
    _qsa_qtoken_prepare[(routes + (tokens * head_tiles if qkv is not None else 0),)](
        qkv,
        selected,
        positions,
        q_gamma,
        k_gamma,
        cache,
        table,
        cu,
        prefixes,
        route_offsets,
        query,
        page_indices,
        memberships.view(torch.uint8),
        seq_lens,
        routes,
        requests,
        table.shape[1],
        table.stride(0),
        cache.shape[0],
        selected.stride(0),
        page_capacity,
        memberships.shape[1] * 4,
        cache.stride(0),
        0 if qkv is None else qkv.stride(0),
        128,
        locator_heads,
        0 if requests == 1 else requests.bit_length(),
        head_tiles,
        block_heads,
        1 if tokens < 4 else 4,
        eps,
        prefill,
        decode_q1,
        histogram_blocks,
        q_only,
        padded_rows,
        1 if selected.shape[1] == 512 else 4,
        zero_tail,
        tiled_histogram,
        base_rope,
        singleton_histogram,
        interleave_prefill,
        num_warps=num_warps,
        enable_fp_fusion=False,
        **launch_options,
    )
