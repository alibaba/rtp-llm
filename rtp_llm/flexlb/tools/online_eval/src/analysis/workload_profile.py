"""Offline prefix-sharing and fully associative LRU diagnostics for token plans.

These are workload properties, not a prediction of multi-engine tree eviction,
Memory handover, admission, pinned blocks or in-flight publication semantics.
"""

from collections import OrderedDict, Counter


def profile(requests, capacities=(128, 256, 512, 1024)):
    caches = {n: OrderedDict() for n in capacities}
    hit_tokens = dict.fromkeys(capacities, 0)
    seen, families, gaps, last = set(), Counter(), [], {}
    count = tokens = potential = 0
    previous_ts = None
    for row in requests:
        count += 1
        tokens += row.input_tokens
        families[row.family] += 1
        keys = row.prefix_keys
        for key in keys:
            if key not in seen:
                break
            potential += row.block_size
        for capacity, cache in caches.items():
            for key in keys:
                if key not in cache:
                    break
                hit_tokens[capacity] += row.block_size
            for key in keys:
                cache[key] = None
                cache.move_to_end(key)
                while len(cache) > capacity:
                    cache.popitem(last=False)
        for key in keys:
            if key in last:
                gaps.append(row.arrival_ms - last[key])
            last[key] = row.arrival_ms
        seen.update(keys)
        if previous_ts is not None and row.arrival_ms < previous_ts:
            raise ValueError("unordered trace")
        previous_ts = row.arrival_ms
    gaps.sort()
    return dict(
        requests=count,
        input_tokens=tokens,
        distinct_prefix_blocks=len(seen),
        family_requests=dict(families),
        infinite_cache_potential_hit=potential / tokens if tokens else None,
        lru_hit_by_capacity={
            str(n): v / tokens if tokens else None for n, v in hit_tokens.items()
        },
        repeat_gap_ms={
            str(q): gaps[min(len(gaps) - 1, int(q * (len(gaps) - 1)))] if gaps else None
            for q in (0.5, 0.9, 0.99)
        },
        realism="NOT_VALIDATED",
        semantics=__doc__,
    )
