#!/usr/bin/env python3
"""Check the sealed anonymous capture from every Master before replay conversion."""

import argparse
from collections import Counter
import io
import json
from pathlib import Path

import zstandard


def read_shard(path):
    with path.open("rb") as source:
        with zstandard.ZstdDecompressor().stream_reader(source) as decoded:
            yield from map(json.loads, io.TextIOWrapper(decoded, encoding="utf-8"))


def load_shard(path):
    rows = read_shard(path)
    header = next(rows)
    if len(header) != 6 or header[:2] != ["header", 1] or header[3] <= 0:
        raise ValueError(f"{path}: unsupported capture header")
    nodes = {0: (None, 0)}
    requests, outcomes = {}, {}
    trailer = None
    for kind, *fields in rows:
        if trailer is not None:
            raise ValueError(f"{path}: data after seal")
        if kind == "node":
            node, parent, key = fields
            if node != len(nodes) or parent not in nodes or len(bytes.fromhex(key)) != 16:
                raise ValueError(f"{path}: invalid prefix node")
            nodes[node] = (key, nodes[parent][1] + 1)
        elif kind == "request":
            attempt, request, offset, block, tokens, _, _, vit_only, leaf = fields
            if (attempt in requests or leaf not in nodes or len(bytes.fromhex(request)) != 16
                    or not 0 <= offset < header[3] or tokens < 0
                    or leaf and (block <= 0 or nodes[leaf][1] * block > tokens)):
                raise ValueError(f"{path}: invalid request row")
            requests[attempt] = (request, offset, block, tokens, vit_only, nodes[leaf][0])
        elif kind == "result":
            attempt, outcome, _ = fields
            if attempt not in requests or attempt in outcomes or outcome not in {"routed", "rejected", "cancelled"}:
                raise ValueError(f"{path}: invalid schedule result")
            outcomes[attempt] = outcome
        elif kind == "end":
            trailer = fields
        else:
            raise ValueError(f"{path}: unknown capture row type")
    if trailer is None:
        raise ValueError(f"{path}: unsealed capture")
    arrivals, dropped_arrivals, dropped_results = trailer
    if arrivals != len(requests) + dropped_arrivals or min(dropped_arrivals, dropped_results) < 0:
        raise ValueError(f"{path}: inconsistent capture counts")
    return header[2:5], header[5], requests, outcomes, dropped_arrivals, dropped_results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--masters", required=True, type=int, help="Expected number of Master shard files")
    parser.add_argument("shards", nargs="+", type=Path)
    args = parser.parse_args()
    if args.masters < 1 or len({p.resolve() for p in args.shards}) != args.masters or len(args.shards) != args.masters:
        parser.error("Expected one distinct shard per Master")
    window = None
    totals, outcomes = Counter(), Counter()
    pd_ids, shard_ids = set(), set()
    for path in args.shards:
        shard_window, shard_id, requests, results, dropped, lost_results = load_shard(path)
        if shard_id in shard_ids:
            raise ValueError("Duplicate Master shard")
        shard_ids.add(shard_id)
        if window is not None and shard_window != window:
            raise ValueError("Master shards have different windows or anonymization keys")
        window = shard_window
        totals.update(arrivals=len(requests) + dropped, dropped_arrivals=dropped,
                      dropped_results=lost_results, missing_results=len(requests) - len(results))
        for request, _, block, tokens, vit_only, leaf in requests.values():
            if vit_only:
                totals["vit_only_attempts"] += 1
                continue
            pd_ids.add(request)
            totals["pd_attempts"] += 1
            if leaf is None:
                totals["empty_prefix_attempts"] += 1
                if not block or tokens >= block:
                    totals["empty_prefix_with_unknown_or_complete_block"] += 1
        outcomes.update(results.values())
    complete = not any(totals[k] for k in ("dropped_arrivals", "dropped_results", "missing_results"))
    print(json.dumps({
        "start_ms": window[0], "duration_ms": window[1], "shards": args.masters,
        "complete": complete, **totals, "distinct_pd_request_ids": len(pd_ids),
        "schedule_outcomes": outcomes,
    }, indent=2))
    return 0 if complete else 1


if __name__ == "__main__":
    raise SystemExit(main())
