"""Request-local eviction with selection-wide ownership and Host-read deduplication."""

import triton
import triton.language as tl

# Sticky error bits shared by selection, admission and resident writes.
ERROR_INVALID_ROW = tl.constexpr(1)
ERROR_INVALID_REQUEST = tl.constexpr(2)
ERROR_INVALID_REQUEST_KEY = tl.constexpr(4)
ERROR_PARTITION_CAPACITY = tl.constexpr(8)
ERROR_REQUEST_QUOTA = tl.constexpr(16)
ERROR_MISSING_ADMISSION = tl.constexpr(32)


@triton.jit
def mark_requests(
    Ids,
    RequestIds,
    Seen,
    Epoch,
    PendingCount,
    Error,
    NUM_ROWS: tl.constexpr,
    ROWS_PER_QUERY: tl.constexpr,
    NUM_REQUESTS: tl.constexpr,
    MAX_REQUESTS: tl.constexpr,
    ANONYMOUS: tl.constexpr,
    HBM_ROWS: tl.constexpr,
    TOTAL_ROWS: tl.constexpr,
    TILE_SIZE: tl.constexpr,
):
    i = tl.program_id(0) * TILE_SIZE + tl.arange(0, TILE_SIZE)
    token = tl.load(Ids + i, i < NUM_ROWS, other=-1)
    if ANONYMOUS:
        request = tl.full((TILE_SIZE,), 0, tl.int32)
    else:
        request = tl.load(RequestIds + i // ROWS_PER_QUERY, i < NUM_ROWS, other=-1)
    valid = (i < NUM_ROWS) & (token >= 0) & (token < TOTAL_ROWS)
    bad_row = (i < NUM_ROWS) & ((token < -1) | (token >= TOTAL_ROWS))
    bad_request = valid & ((request < 0) | (request >= NUM_REQUESTS))
    bits = tl.where(tl.sum(bad_row.to(tl.int32), 0) > 0, ERROR_INVALID_ROW, 0)
    bits |= tl.where(tl.sum(bad_request.to(tl.int32), 0) > 0, ERROR_INVALID_REQUEST, 0)
    tl.atomic_or(Error, bits, bits != 0, sem="relaxed")
    # HBM-only queries need no hot partition, including an idle dynamic pool.
    valid &= (token >= HBM_ROWS) & (request >= 0) & (request < NUM_REQUESTS)
    epoch = tl.load(Epoch)
    if ANONYMOUS or ROWS_PER_QUERY % TILE_SIZE == 0:
        # An aligned tile belongs to one query, so its Host rows share a mark.
        seen_request = tl.max(tl.where(valid, request, -1), 0)
        tl.atomic_xchg(Seen + seen_request, epoch, seen_request >= 0, sem="relaxed")
    else:
        tl.atomic_xchg(Seen + request, epoch, valid)
    tl.store(PendingCount + i, 0, i < MAX_REQUESTS)


@triton.jit
def match_requests(
    Keys,
    Generations,
    Seen,
    BoundKeys,
    BoundGenerations,
    Parts,
    First,
    Kept,
    ActiveRows,
    Epoch,
    Error,
    NUM_REQUESTS: tl.constexpr,
    MAX_REQUESTS: tl.constexpr,
    ROWS_PER_REQUEST: tl.constexpr,
    KEY_STRIDE: tl.constexpr,
    ROWS_PER_PAGE: tl.constexpr,
    ROWS_PER_BLOCK: tl.constexpr,
    TOTAL_ROWS: tl.constexpr,
    DYNAMIC: tl.constexpr,
    ANONYMOUS: tl.constexpr,
    TILE_SIZE: tl.constexpr,
):
    request = tl.program_id(0)
    epoch = tl.load(Epoch)
    seen = tl.load(Seen + request) == epoch
    if ANONYMOUS:
        key = tl.full((), 1, tl.int64)
        generation = tl.full((), 0, tl.int64)
    else:
        key = tl.load(Keys + request * KEY_STRIDE).to(tl.int64)
        key_valid = (key > 0) & (key < TOTAL_ROWS // ROWS_PER_PAGE)
        tl.atomic_or(Error, ERROR_INVALID_REQUEST_KEY, seen & ~key_valid, sem="relaxed")
        generation = tl.load(
            Generations + key * ROWS_PER_PAGE // ROWS_PER_BLOCK,
            seen & key_valid,
            other=-1,
        )
    valid = (key > 0) & seen & (tl.load(Error) == 0)
    limit = (
        tl.minimum(MAX_REQUESTS, tl.load(ActiveRows) // ROWS_PER_REQUEST)
        if DYNAMIC
        else MAX_REQUESTS
    )
    found = tl.full((), 0x7FFFFFFF, tl.int32)
    lane = tl.arange(0, TILE_SIZE)
    for start in range(0, MAX_REQUESTS, TILE_SIZE):
        part = start + lane
        old = tl.load(BoundKeys + part, part < MAX_REQUESTS, other=0)
        gen = tl.load(BoundGenerations + part, part < MAX_REQUESTS, other=-1)
        match = valid & (part < limit) & (old == key) & (gen == generation)
        found = tl.minimum(found, tl.min(tl.where(match, part, 0x7FFFFFFF), 0))
    found = tl.where(found < MAX_REQUESTS, found, -1)
    tl.store(Parts + request, found)
    tl.atomic_xchg(Kept + found, epoch, found >= 0)
    # Warmup can repeat a sentinel-backed identity. Bind it once, just as
    # several MTP queries share one real request's quota.
    first = request
    if not ANONYMOUS:
        for start in range(0, NUM_REQUESTS, TILE_SIZE):
            other = start + lane
            other_key = tl.load(
                Keys + other * KEY_STRIDE, other < NUM_REQUESTS, other=0
            )
            other_seen = tl.load(Seen + other, other < NUM_REQUESTS, other=0)
            match = (
                valid & (other < request) & (other_key == key) & (other_seen == epoch)
            )
            first = tl.minimum(first, tl.min(tl.where(match, other, request), 0))
    tl.store(First + request, tl.where(valid, first, -1))


@triton.jit
def assign_requests(
    Keys,
    Generations,
    BoundKeys,
    BoundGenerations,
    Parts,
    First,
    Kept,
    Reset,
    ActiveRows,
    Epoch,
    Error,
    NUM_REQUESTS: tl.constexpr,
    MAX_REQUESTS: tl.constexpr,
    ROWS_PER_REQUEST: tl.constexpr,
    KEY_STRIDE: tl.constexpr,
    ROWS_PER_PAGE: tl.constexpr,
    ROWS_PER_BLOCK: tl.constexpr,
    TOTAL_ROWS: tl.constexpr,
    DYNAMIC: tl.constexpr,
    ANONYMOUS: tl.constexpr,
    TILE_SIZE: tl.constexpr,
):
    if tl.load(Error) != 0:
        return
    i = tl.arange(0, TILE_SIZE)
    epoch = tl.load(Epoch)
    part = tl.load(Parts + i, i < NUM_REQUESTS, other=-1)
    first = tl.load(First + i, i < NUM_REQUESTS, other=-1)
    new = (i < NUM_REQUESTS) & (first == i) & (part < 0)
    if tl.sum(new.to(tl.int32), 0) == 0:
        return
    limit = (
        tl.minimum(MAX_REQUESTS, tl.load(ActiveRows) // ROWS_PER_REQUEST)
        if DYNAMIC
        else MAX_REQUESTS
    )
    kept = tl.load(Kept + i, i < MAX_REQUESTS, other=epoch)
    free = (i < limit) & (kept != epoch)
    if tl.sum(new.to(tl.int32), 0) > tl.sum(free.to(tl.int32), 0):
        tl.atomic_or(Error, ERROR_PARTITION_CAPACITY, sem="relaxed")
        return
    ordered = tl.sort(tl.where(free, i, 0x7FFFFFFF), descending=False)
    rank = tl.cumsum(new.to(tl.int32), 0) - 1
    assigned = tl.gather(ordered, tl.where(new, rank, 0), 0)
    if ANONYMOUS:
        key = tl.full((TILE_SIZE,), 1, tl.int64)
        generation = tl.full((TILE_SIZE,), 0, tl.int64)
    else:
        key = tl.load(Keys + i * KEY_STRIDE, new, other=0).to(tl.int64)
        generation = tl.load(
            Generations + key * ROWS_PER_PAGE // ROWS_PER_BLOCK, new, other=-1
        )
    tl.store(Parts + i, assigned, new)
    tl.store(BoundKeys + assigned, key, new)
    tl.store(BoundGenerations + assigned, generation, new)
    tl.store(Reset + assigned, epoch, new)


@triton.jit
def reset_partitions(
    Reset,
    Epoch,
    Tags,
    Map,
    Versions,
    SlotEpochs,
    Protected,
    Clock,
    Error,
    ROWS_PER_REQUEST: tl.constexpr,
    DYNAMIC: tl.constexpr,
    TILE_SIZE: tl.constexpr,
):
    part = tl.program_id(0)
    if (tl.load(Error) != 0) | (tl.load(Reset + part) != tl.load(Epoch)):
        return
    for start in range(0, ROWS_PER_REQUEST, TILE_SIZE):
        j = start + tl.arange(0, TILE_SIZE)
        slot = part * ROWS_PER_REQUEST + j
        old = tl.load(Tags + slot, j < ROWS_PER_REQUEST, other=-1)
        # Reset precedes all lookups/admissions, so no new mappings race here.
        mapped = tl.load(Map + old, (j < ROWS_PER_REQUEST) & (old >= 0), other=-1)
        tl.store(Map + old, -1, (j < ROWS_PER_REQUEST) & (old >= 0) & (mapped == slot))
        tl.store(Tags + slot, -1, j < ROWS_PER_REQUEST)
        tl.store(Versions + slot, -1, j < ROWS_PER_REQUEST)
        tl.store(Protected + slot, 0, j < ROWS_PER_REQUEST)
        if DYNAMIC:
            tl.store(SlotEpochs + slot, -1, j < ROWS_PER_REQUEST)
    tl.store(Clock + part, 0)


@triton.jit
def collect_misses(
    Ids,
    Slots,
    Owners,
    Map,
    Misses,
    MissCount,
    Generations,
    Versions,
    RequestIds,
    Parts,
    First,
    Pending,
    PendingCount,
    Error,
    NUM_ROWS: tl.constexpr,
    ROWS_PER_QUERY: tl.constexpr,
    ROWS_PER_BLOCK: tl.constexpr,
    ROWS_PER_REQUEST: tl.constexpr,
    NUM_REQUESTS: tl.constexpr,
    ANONYMOUS: tl.constexpr,
    TILE_SIZE: tl.constexpr,
):
    if tl.load(Error) != 0:
        return
    i = tl.program_id(0) * TILE_SIZE + tl.arange(0, TILE_SIZE)
    refresh = tl.load(Slots + i, i < NUM_ROWS, other=-1) == -2
    token = tl.load(Ids + i, refresh, other=-1)
    owner = tl.load(Owners + token, refresh, other=-1)
    owned = refresh & (owner == i)
    slot = tl.load(Map + token, owned, other=-1)
    if ANONYMOUS:
        request = tl.full((TILE_SIZE,), 0, tl.int32)
    else:
        request = tl.load(RequestIds + i // ROWS_PER_QUERY, owned, other=-1)
    first = tl.load(
        First + request, owned & (request >= 0) & (request < NUM_REQUESTS), other=-1
    )
    part = tl.load(
        Parts + first, owned & (first >= 0) & (first < NUM_REQUESTS), other=-1
    )
    # A stale copy left by an earlier request must not be refreshed into
    # that request's partition. Its unique miss is admitted into this quota.
    foreign = owned & (slot >= 0) & (slot // ROWS_PER_REQUEST != part)
    tl.store(Map + token, -1, foreign)
    slot = tl.where(foreign, -1, slot)
    generation = tl.load(Generations + token // ROWS_PER_BLOCK, owned, other=-1)
    count = tl.sum(owned.to(tl.int32), 0)
    start = tl.atomic_add(MissCount, count, sem="relaxed")
    rank = tl.cumsum(owned.to(tl.int32), 0) - 1
    # One entry per unique backing row across ALL requests/queries.
    tl.store(Misses + start + rank, i, owned)
    tl.store(Versions + slot, generation, owned & (slot >= 0))
    pending = owned & (slot < 0)
    tl.atomic_or(
        Error,
        ERROR_PARTITION_CAPACITY,
        tl.sum((pending & (part < 0)).to(tl.int32), 0) > 0,
        sem="relaxed",
    )
    pending &= part >= 0
    if ANONYMOUS or ROWS_PER_QUERY % TILE_SIZE == 0:
        # Reserve one interval for the tile and give each pending row an offset.
        partition = tl.max(tl.where(pending, part, -1), 0)
        count = tl.sum(pending.to(tl.int32), 0)
        start = tl.atomic_add(PendingCount + partition, count, count > 0, sem="relaxed")
        offset = start + tl.cumsum(pending.to(tl.int32), 0) - 1
    else:
        offset = tl.atomic_add(PendingCount + part, 1, pending, sem="relaxed")
    tl.atomic_or(
        Error,
        ERROR_REQUEST_QUOTA,
        tl.sum((pending & (offset >= ROWS_PER_REQUEST)).to(tl.int32), 0) > 0,
        sem="relaxed",
    )
    tl.store(
        Pending + part * ROWS_PER_REQUEST + offset,
        i,
        pending & (offset < ROWS_PER_REQUEST),
    )


@triton.jit
def admit(
    Ids,
    Pending,
    PendingCount,
    Map,
    Tags,
    Versions,
    SlotEpochs,
    HotEpoch,
    Protected,
    Epoch,
    Clock,
    Generations,
    Error,
    ROWS_PER_REQUEST: tl.constexpr,
    ROWS_PER_BLOCK: tl.constexpr,
    LRU: tl.constexpr,
    DYNAMIC: tl.constexpr,
    TILE_SIZE: tl.constexpr,
):
    part = tl.program_id(0)
    count = tl.load(PendingCount + part)
    if (count == 0) | (tl.load(Error) != 0):
        return
    if count > ROWS_PER_REQUEST:
        tl.atomic_or(Error, ERROR_REQUEST_QUOTA, sem="relaxed")
        return
    epoch = tl.load(Epoch)
    j = tl.arange(0, TILE_SIZE)
    if LRU:
        slot = part * ROWS_PER_REQUEST + j
        age = tl.load(Protected + slot, j < ROWS_PER_REQUEST, other=epoch)
        # All rows selected in this round are protected, even if several
        # requests/queries reuse the same canonical resident copy.
        available = (j < ROWS_PER_REQUEST) & (age != epoch)
        if DYNAMIC:
            allocation = tl.load(SlotEpochs + slot, j < ROWS_PER_REQUEST, other=-1)
            expected = tl.load(HotEpoch + slot, j < ROWS_PER_REQUEST, other=-2)
            age = tl.where(allocation == expected, age, 0)
        key = tl.where(available, age * TILE_SIZE + j, 0x7FFFFFFFFFFFFFFF)
        if tl.sum(available.to(tl.int32), 0) < count:
            tl.atomic_or(Error, ERROR_REQUEST_QUOTA, sem="relaxed")
            return
        ordered = tl.sort(key, descending=False)
        victim = part * ROWS_PER_REQUEST + (ordered % TILE_SIZE).to(tl.int32)
        selected = j < count
        i = tl.load(Pending + part * ROWS_PER_REQUEST + j, selected, other=0)
        token = tl.load(Ids + i, selected, other=-1)
        old = tl.load(Tags + victim, selected, other=-1)
        mapped = tl.load(Map + old, selected & (old >= 0), other=-1)
        valid = selected & (old >= 0) & (mapped == victim)
        if DYNAMIC:
            prior = tl.load(SlotEpochs + victim, selected, other=-1)
            allocation = tl.load(HotEpoch + victim, selected, other=-2)
            valid &= prior == allocation
            tl.store(SlotEpochs + victim, allocation, selected)
        tl.store(Map + old, -1, valid)
        tl.store(Tags + victim, token, selected)
        generation = tl.load(Generations + token // ROWS_PER_BLOCK, selected, other=-1)
        tl.store(Versions + victim, generation, selected)
        tl.store(Protected + victim, epoch, selected)
        tl.store(Map + token, victim, selected)
    else:
        cursor = tl.load(Clock + part) % ROWS_PER_REQUEST
        admitted = tl.full((), 0, tl.int32)
        scanned = tl.full((), 0, tl.int32)
        while (admitted < count) & (scanned < ROWS_PER_REQUEST):
            offset = (cursor + j) % ROWS_PER_REQUEST
            candidate = part * ROWS_PER_REQUEST + offset
            available = (j < tl.minimum(TILE_SIZE, ROWS_PER_REQUEST - scanned)) & (
                tl.load(Protected + candidate) != epoch
            )
            rank = tl.cumsum(available.to(tl.int32), 0) - 1
            selected = available & (rank < count - admitted)
            i = tl.load(
                Pending + part * ROWS_PER_REQUEST + admitted + rank, selected, other=0
            )
            token = tl.load(Ids + i, selected, other=-1)
            old = tl.load(Tags + candidate, selected, other=-1)
            mapped = tl.load(Map + old, selected & (old >= 0), other=-1)
            valid = selected & (old >= 0) & (mapped == candidate)
            if DYNAMIC:
                prior = tl.load(SlotEpochs + candidate, selected, other=-1)
                allocation = tl.load(HotEpoch + candidate, selected, other=-2)
                valid &= prior == allocation
                tl.store(SlotEpochs + candidate, allocation, selected)
            tl.store(Map + old, -1, valid)
            tl.store(Tags + candidate, token, selected)
            generation = tl.load(
                Generations + token // ROWS_PER_BLOCK, selected, other=-1
            )
            tl.store(Versions + candidate, generation, selected)
            tl.store(Protected + candidate, epoch, selected)
            tl.store(Map + token, candidate, selected)
            taken = tl.sum(selected.to(tl.int32), 0)
            last = tl.max(tl.where(selected, j + 1, 0), 0)
            advance = tl.where(
                admitted + taken == count,
                last,
                tl.minimum(TILE_SIZE, ROWS_PER_REQUEST - scanned),
            )
            admitted += taken
            scanned += advance
            cursor = (cursor + advance) % ROWS_PER_REQUEST
        if admitted != count:
            tl.atomic_or(Error, ERROR_REQUEST_QUOTA, sem="relaxed")
        tl.store(Clock + part, cursor)
