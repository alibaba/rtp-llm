#!/usr/bin/env python3
"""Prefill batching of a CEP4PP2 P during a run, from its engine.log.

  an_p_batches.py <engine.log> [<engine.log> ...]

Every PP stage-1 rank logs one "publish pd plan kv cache, request <id>" line per request when it finishes the request's
layers. Lines of rank 4 (first stage-1 rank) within 15 ms of each other belong to one prefill pass. Prints the request
rate through stage 1, the distribution of requests per pass, and the pass-start interval p50 / mean.
"""
import collections
import re
import statistics
import sys

PAT = re.compile(r"^\[\d{4}-\d\d-\d\d (\d\d):(\d\d):([\d.]+)\].*\[RANK 4\].*publish pd plan kv cache, request")


def main():
    for path in sys.argv[1:]:
        ts = []
        for line in open(path, errors="ignore"):
            m = PAT.match(line)
            if m:
                ts.append(int(m.group(1)) * 3600 + int(m.group(2)) * 60 + float(m.group(3)))
        if len(ts) < 2:
            print("%s: too few publish lines" % path)
            continue
        ts.sort()
        groups, cur = [], [ts[0]]
        for t in ts[1:]:
            if t - cur[-1] < 0.015:
                cur.append(t)
            else:
                groups.append(cur)
                cur = [t]
        groups.append(cur)
        sizes = collections.Counter(len(g) for g in groups)
        iv = [b[0] - a[0] for a, b in zip(groups, groups[1:])]
        span = ts[-1] - ts[0]
        print("%s: %d requests over %.0f s = %.2f req/s; requests per pass %s; pass interval p50 %.0f ms, mean %.0f ms" % (
            path, len(ts), span, len(ts) / span, dict(sorted(sizes.items())), statistics.median(iv) * 1e3,
            statistics.mean(iv) * 1e3))


if __name__ == "__main__":
    main()
