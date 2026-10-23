"""Per-request zigzag ownership with checkpoint boundaries retained."""
from bisect import bisect_right
from functools import lru_cache
from math import lcm


@lru_cache(maxsize=128)
def make_plan(cu, world, checkpoint_interval):
    if len(cu) < 2 or cu[0] != 0 or any(b <= a for a, b in zip(cu, cu[1:])):
        raise ValueError('Expected strictly increasing packed sequence boundaries')
    lengths = [b - a for a, b in zip(cu, cu[1:])]
    # The tested cuLA path has 64-token aligned segment boundaries. Reject
    # unsupported shapes explicitly rather than silently changing ownership.
    if any(n % (2 * world * 64) for n in lengths):
        raise ValueError('Zigzag cuLA experiment requires Q lengths divisible by 2*CP*64')
    if checkpoint_interval <= 0 or checkpoint_interval % 64:
        raise ValueError('Expected a positive 64-token aligned checkpoint interval')
    local_tokens = cu[-1] // world
    boundaries = set(cu)
    checkpoints, chunks = [], []
    request_offset = 0
    for seq, (begin, end) in enumerate(zip(cu, cu[1:])):
        pair = (end - begin) // (2 * world)
        points = list(range(begin + checkpoint_interval, end, checkpoint_interval)) + [end]
        checkpoints.extend(points)
        boundaries.update(points)
        for slot in range(2 * world):
            start = begin + slot * pair
            rank = slot if slot < world else 2 * world - 1 - slot
            offset = request_offset + (0 if slot < world else pair)
            chunks.append((start, start + pair, seq, rank, offset))
            boundaries.update((start, start + pair))
        request_offset += 2 * pair
    starts = [c[0] for c in chunks]
    segments, local_offsets = [], []
    by_rank = [[] for _ in range(world)]
    by_seq = [[] for _ in lengths]
    ordered = sorted(boundaries)
    for begin, end in zip(ordered, ordered[1:]):
        chunk = chunks[bisect_right(starts, begin) - 1]
        cs, ce, seq, rank, offset = chunk
        assert cs <= begin < end <= ce
        index = len(segments)
        segments.append((begin, end, seq, rank, -1))
        local_offsets.append(offset + begin - cs)
        by_rank[rank].append(index)
        by_seq[seq].append(index)
    for rank, ids in enumerate(by_rank):
        ids.sort(key=local_offsets.__getitem__)
        cursor = 0
        for slot, i in enumerate(ids):
            begin, end, seq, owner, _ = segments[i]
            assert owner == rank and local_offsets[i] == cursor
            segments[i] = (begin, end, seq, owner, slot)
            cursor += end - begin
        assert cursor == local_tokens
    padded_segments = max(map(len, by_rank))
    physical_indices = [r * padded_segments + s for _, _, _, r, s in segments]
    checkpoint_set = set(checkpoints)
    checkpoint_indices = [physical_indices[i] for i, (_, end, _, _, _) in enumerate(segments)
                          if end in checkpoint_set]
    merge_starts, counts, offsets, seq_ids = [], [], [], []
    initial_indices = [None] * len(segments)
    non_first = 0
    for seq, ids in enumerate(by_seq):
        initial_indices[ids[0]] = seq
        if len(ids) > 1:
            merge_starts.append(ids[0]); counts.append(len(ids))
            offsets.append(non_first); seq_ids.append(seq)
            for j, i in enumerate(ids[1:]):
                initial_indices[i] = len(lengths) + non_first + j
            non_first += len(ids) - 1
    local_merge_counts = [[max(i for i in by_seq[seq] if segments[i][3] == rank) - start + 1
                           for seq, start in zip(seq_ids, merge_starts)] for rank in range(world)]
    return {'layout': 'zigzag', 'local_tokens': local_tokens,
            'segments': segments, 'local_offsets': local_offsets,
            'chunks': chunks, 'by_rank': by_rank, 'padded_segments': padded_segments,
            'flattened_indices': physical_indices,
            'checkpoint_indices': checkpoint_indices, 'initial_indices': initial_indices,
            'merge_starts': merge_starts, 'merge_counts': counts,
            'local_merge_counts': local_merge_counts,
            'merge_offsets': offsets, 'merge_seq_ids': seq_ids, 'non_first': non_first}


@lru_cache(maxsize=128)
def make_checkpoint_plan(cu, world, page_size, summary_interval=32768):
    """Separate summary segments from dense, canonical cache-page checkpoints.

    cuLA emits packed FP32 checkpoints inside each local summary segment. If a
    zigzag boundary cuts a cache page, retain the original page-split segments:
    their partial end states are filtered out by the canonical checkpoint map.
    """
    if page_size <= 0 or page_size % 64:
        raise ValueError('Expected a positive 64-token aligned checkpoint interval')
    if summary_interval <= 0 or summary_interval % 64:
        raise ValueError('Expected a positive 64-token aligned summary interval')
    aligned = all(((b - a) // (2 * world)) % page_size == 0
                  for a, b in zip(cu, cu[1:]))
    span = lcm(page_size, summary_interval) if aligned else page_size
    plan = make_plan(cu, world, span)
    offsets = []
    for ids in plan['by_rank']:
        row = [0]
        for i in ids:
            begin, end, *_ = plan['segments'][i]
            row.append(row[-1] + (end - begin + page_size - 1) // page_size)
        offsets.append(tuple(row))
    padded = max(row[-1] for row in offsets)
    physical = {}
    for begin, end, _, rank, slot in plan['segments']:
        points = list(range(begin + page_size, end, page_size)) + [end]
        for j, point in enumerate(points):
            physical[point] = rank * padded + offsets[rank][slot] + j
    indices = []
    for begin, end in zip(cu, cu[1:]):
        for point in list(range(begin + page_size, end, page_size)) + [end]:
            indices.append(physical[point])
    return {'plan': plan, 'summary_interval': span,
            'checkpoint_offsets': tuple(offsets), 'padded_checkpoints': padded,
            'checkpoint_indices': tuple(indices)}
