#!/usr/bin/env python3
"""Per-layer KV burst timing on the P-side eRDMA NIC counter (RDMA mode).

  an_bursts.py <tag> [length]      reads wire-<tag>.jsonl + len-<tag>-<length>-*.json

For every request, groups NIC tx increments into bursts (a burst ends after a
6ms gap) and reports, relative to tok1: the number of bursts, the duration and
size of the last (final-layer) burst, the effective rate of the large (~6MB)
csa-layer bursts, and the time the last KV byte left P.
"""
import bisect, glob, json, re, statistics, sys


def main():
    tag = sys.argv[1]
    length = sys.argv[2] if len(sys.argv) > 2 else "32k"
    wire = []
    for line in open("wire-%s.jsonl" % tag):
        try:
            wire.append(json.loads(line))
        except Exception:
            pass
    ts = [w["t"] for w in wire]
    files = sorted(glob.glob("len-%s-%s-*.json" % (tag, length)), key=lambda f: int(re.search(r"-(\d+)\.json$", f).group(1)))
    last_dur, last_mb, big_rate, nb, kv_last = [], [], [], [], []
    for f in files:
        r = json.load(open(f))["rows"][0]
        t0 = r["t_send_wall"]
        t1 = t0 + r["chunks"][0]
        seg = wire[bisect.bisect_left(ts, t0 - 0.02):bisect.bisect_right(ts, t1 + 0.02)]
        bursts, cur, prev, prev_t = [], None, seg[0]["n"][0], seg[0]["t"]
        for w in seg[1:]:
            d = w["n"][0] - prev
            if d > 64 * 1024:
                if cur and (w["t"] - cur[1]) * 1000 <= 6:
                    cur[1] = w["t"]
                    cur[2] += d
                else:
                    if cur:
                        bursts.append(cur)
                    cur = [prev_t, w["t"], d]
            prev, prev_t = w["n"][0], w["t"]
        if cur:
            bursts.append(cur)
        if not bursts:
            continue
        nb.append(len(bursts))
        last = bursts[-1]
        last_dur.append((last[1] - last[0]) * 1000)
        last_mb.append(last[2] / 1e6)
        kv_last.append((last[1] - t1) * 1000)
        for b in bursts:
            dur = (b[1] - b[0]) * 1000
            if b[2] > 4e6 and dur > 0:
                big_rate.append(b[2] * 8 / (dur / 1000) / 1e9)
    med = statistics.median
    print("%s %s: n=%d bursts/req p50 %d | last burst %.1fMB dur p50 %.1fms (min %.1f max %.1f) | kv_last rel tok1 p50 %.1fms"
          % (tag, length, len(nb), med(nb), med(last_mb), med(last_dur), min(last_dur), max(last_dur), med(kv_last)))
    print("   large (>4MB) bursts: n=%d, sampled rate p50 %.1f Gb/s (p10 %.1f, p90 %.1f) -- upper bound, ~2.2ms sampling"
          % (len(big_rate), med(big_rate), sorted(big_rate)[len(big_rate) // 10], sorted(big_rate)[len(big_rate) * 9 // 10]))


if __name__ == "__main__":
    main()
