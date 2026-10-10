"""Offline prefix-sharing and fully associative LRU diagnostics for token plans.

These are workload properties, not a prediction of multi-engine tree eviction,
Memory handover, admission, pinned blocks or in-flight publication semantics.
"""

import argparse
from collections import OrderedDict, Counter
import hashlib
import json
from pathlib import Path
import struct


def profile(path, capacities=(128, 256, 512, 1024)):
    caches = {n: OrderedDict() for n in capacities}
    hit_tokens = dict.fromkeys(capacities, 0)
    seen, families, gaps, last = set(), Counter(), [], {}
    count = tokens = potential = 0
    previous_ts = None
    with Path(path).open() as source:
        for line in source:
            row = json.loads(line)
            count += 1
            tokens += row["il"]
            families[row.get("family", "unknown")] += 1
            key, keys = b"", []
            block = row["cache_key_block_size"]
            if 'input_token_blocks' in row:
                physical = row['input_token_blocks'][:row['il']//block]
                blocks = (v if isinstance(v,list) else [v]*block for v in physical)
            else:
                blocks = (row['input_ids'][offset:offset+block] for offset in range(0,len(row['input_ids'])-block+1,block))
            for values in blocks:
                key = hashlib.sha256(
                    key + struct.pack("<" + "i" * block, *values)
                ).digest()
                keys.append(key)
            for key in keys:
                if key not in seen:
                    break
                potential += block
            for capacity, cache in caches.items():
                for key in keys:
                    if key not in cache:
                        break
                    hit_tokens[capacity] += block
                for key in keys:
                    cache[key] = None
                    cache.move_to_end(key)
                    while len(cache) > capacity:
                        cache.popitem(last=False)
            for key in keys:
                if key in last:
                    gaps.append(row["ts"] - last[key])
                last[key] = row["ts"]
            seen.update(keys)
            if previous_ts is not None and row["ts"] < previous_ts:
                raise ValueError("unordered trace")
            previous_ts = row["ts"]
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


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("trace", type=Path)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--capacities", type=int, nargs="+", default=[128, 256, 512, 1024])
    a = p.parse_args()
    if any(n < 1 for n in a.capacities):
        p.error("capacities must be positive")
    a.output.write_text(json.dumps(profile(a.trace, a.capacities), indent=2) + "\n")


if __name__ == "__main__":
    main()
