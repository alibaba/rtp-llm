#!/usr/bin/env python3
"""Backlog-mode burst timeline for one request: NIC rate (2-ms samples) around each stage's publish/store-done."""
import re
import sys
from datetime import datetime

run, which = sys.argv[1], int(sys.argv[2])
rt, tx = [], []
for line in open(run + "/nic.txt"):
    p = line.split()
    rt.append(float(p[0]))
    tx.append(int(p[2]))
ts = lambda l: datetime.strptime(l[1:27], "%Y-%m-%d %H:%M:%S.%f").timestamp()
ft = []
ev = {}
for l in open(run + "/logs/engine.log", errors="ignore"):
    m = re.search(r"\[RANK 0\].*publish kv cache for pd sep, stream (\d+)", l)
    if m:
        ft.append((m.group(1), ts(l)))
    m = re.search(
        r"\[RANK (\d)\].*publish pd plan kv cache, request (\d+), rank \d+, model_id \d+, blocks (\d+)",
        l,
    )
    if m:
        ev.setdefault(m.group(2), []).append(
            (ts(l), "pub r" + m.group(1), int(m.group(3)))
        )
    m = re.search(
        r"\[RANK (\d)\].*publish pd plan kv cache store done, request (\d+)", l
    )
    if m:
        ev.setdefault(m.group(2), []).append((ts(l), "done r" + m.group(1), 0))
rid, t_ft = ft[which]
evs = sorted(ev[rid])
t0 = evs[0][0]
print(
    "request",
    rid,
    "events (ms rel. first publish):",
    [(round(1e3 * (t - t0), 1), n, b) for t, n, b in evs],
    "first token",
    round(1e3 * (t_ft - t0), 1),
)


def idx(t):
    lo, hi = 0, len(rt) - 1
    while lo < hi:
        m = (lo + hi) // 2
        if rt[m] < t:
            lo = m + 1
        else:
            hi = m
    return lo


i0, i1 = idx(t0 - 0.005), idx(t_ft + 0.06)
busy = []
for i in range(i0 + 1, i1):
    r = (tx[i] - tx[i - 1]) / (rt[i] - rt[i - 1])
    if r > 50e6:
        busy.append(i)
    if r > 50e6 or i % 5 == 0:
        print(
            f"  {1e3*(rt[i]-t0):7.1f} ms  {r/1e9:6.2f} GB/s  cum {(tx[i]-tx[i0])/2**20:6.1f} MiB"
        )
# bursts: groups of busy samples separated by > 6 ms idle
bursts, cur = [], [busy[0]]
for i in busy[1:]:
    if rt[i] - rt[cur[-1]] > 0.006:
        bursts.append(cur)
        cur = [i]
    else:
        cur.append(i)
bursts.append(cur)
for b in bursts:
    s, e = b[0] - 1, b[-1]
    nb = tx[e] - tx[s]
    d = rt[e] - rt[s]
    print(
        f"burst {1e3*(rt[s]-t0):6.1f}..{1e3*(rt[e]-t0):6.1f} ms: {nb/2**20:6.1f} MiB in {1e3*d:5.1f} ms = {nb/d/1e9:5.2f} GB/s"
    )
