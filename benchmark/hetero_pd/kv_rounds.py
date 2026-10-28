#!/usr/bin/env python3
"""Per-round NIC traffic for one PD request: TX bytes between consecutive rank-0 [PPSTEP] round starts,
plus the burst pattern inside the request (10-ms buckets)."""
import re
import sys
from datetime import datetime

nic, eng, which = (
    sys.argv[1],
    sys.argv[2],
    int(sys.argv[3]) if len(sys.argv) > 3 else -1,
)
rt, tx = [], []
for line in open(nic):
    p = line.split()
    rt.append(float(p[0]))
    tx.append(int(p[2]))
ts = lambda l: datetime.strptime(l[1:27], "%Y-%m-%d %H:%M:%S.%f").timestamp()
ft, r0, pubs = [], [], []
for l in open(eng, errors="ignore"):
    if "[RANK 0]" in l and "publish kv cache for pd sep, stream" in l:
        ft.append(ts(l))
    elif "[RANK 0]" in l and "[PPSTEP]" in l and "streams=1" in l:
        r0.append(ts(l))
    elif "publish pd plan kv cache, request" in l:
        r = int(re.search(r"\[RANK (\d)\]", l).group(1))
        pubs.append((ts(l), r, int(re.search(r"blocks (\d+)", l).group(1))))
ft = [f for f in ft if rt[0] < f < rt[-1]]
f = ft[which]


def at(t):
    lo, hi = 0, len(rt) - 1
    while lo < hi:
        m = (lo + hi) // 2
        if rt[m] < t:
            lo = m + 1
        else:
            hi = m
    return tx[lo]


steps = [s for s in r0 if f - 2.0 < s <= f]
steps = steps[-8:]
edges = steps + [f + 0.05]
print(f"request with first token at {f:.3f}; {len(steps)} rank-0 rounds")
for i in range(len(steps)):
    a, b = edges[i], edges[i + 1]
    nb = at(b) - at(a)
    pp = [(round(1e3 * (t - steps[0])), r, n) for t, r, n in pubs if a <= t < b]
    print(
        f" round {i}: t={1e3*(a-steps[0]):7.1f}..{1e3*(b-steps[0]):7.1f} ms  tx {nb/2**20:6.1f} MiB  "
        f"{nb/(b-a)/1e9:5.2f} GB/s  publishes(t_ms,rank,blocks)={pp}"
    )
print(" 10-ms buckets (MiB) from first round start:")
t = steps[0] - 0.05
row = []
while t < f + 0.05:
    row.append(f"{(at(t + 0.01) - at(t)) / 2**20:.1f}")
    t += 0.01
print(" ", " ".join(row))
