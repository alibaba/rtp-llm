#!/usr/bin/env python3
"""KV bytes on the wire during a constant-rate run (pd_wire_sampler.py output next to the pd_uniform_load.py --out).

  an_wire_rate.py <wire-T.jsonl> [<wire-T.jsonl> ...]      (T.jsonl, the client file, must sit next to it)

Window = [first send + 5 s, last send]. TCP cache-store mode: sum of bytes_sent deltas of the P-side KV sockets
(local port 140xx). RDMA mode (file name contains "rdma"): delta of the P NIC tx byte counter. Prints the mean rate,
the peak over 100 ms windows, KV MB per request over the whole run, the share of 10 ms samples that carried KV
(> 64 KB), and the rate while sending (p50 / p90 over those samples).
"""
import json
import os
import statistics
import sys


def series(path, rdma):
    prev, pn, out = {}, None, []
    for line in open(path):
        try:
            x = json.loads(line)
        except ValueError:
            continue
        if rdma:
            n = x.get("n")
            if not n:
                continue
            d = 0 if pn is None else max(0, n[0] - pn)
            pn = n[0]
        else:
            cur, d = {}, 0
            for e in x.get("s", []):
                if 14000 <= int(e[0]) < 14100:
                    k = (e[0], e[1])
                    cur[k] = e[4]
                    if k in prev:
                        d += max(0, e[4] - prev[k])
            prev = cur
        out.append((x["t"], d))
    return out


def main():
    for wp in sys.argv[1:]:
        jp = os.path.join(os.path.dirname(wp), os.path.basename(wp)[len("wire-"):])
        R = [json.loads(l) for l in open(jp)]
        R = [r for r in R if r.get("chunks")]
        t0, t1 = min(r["t_send"] for r in R), max(r["t_send"] for r in R)
        tend = max(r["t_send"] + r["chunks"][1] for r in R) + 0.5
        S = series(wp, "rdma" in os.path.basename(wp))
        total = sum(d for t, d in S if t0 - 1 <= t <= tend)
        W = [(t, d) for t, d in S if t0 + 5 <= t <= t1]
        if len(W) < 10:
            print("%s: too few samples" % wp)
            continue
        span = W[-1][0] - W[0][0]
        mean = sum(d for _, d in W[1:]) * 8 / span / 1e9
        cum, acc = [], 0
        for t, d in W:
            acc += d
            cum.append((t, acc))
        peak, j = 0.0, 0
        for i in range(len(cum)):
            while cum[i][0] - cum[j][0] > 0.1:
                j += 1
            if cum[i][0] - cum[j][0] >= 0.09:
                peak = max(peak, (cum[i][1] - cum[j][1]) * 8 / (cum[i][0] - cum[j][0]) / 1e9)
        act = [d * 8 / (b[0] - a[0]) / 1e9 for a, b in zip(W, W[1:]) for d in [b[1]] if d > 64 * 1024 and b[0] > a[0]]
        act.sort()
        print("%-36s mean %5.1f Gb/s  peak(100ms) %5.1f Gb/s  %6.1f MB/request  sending in %3.0f%% of 10ms samples, "
              "rate while sending p50 %4.1f p90 %4.1f Gb/s" % (
                  os.path.basename(wp), mean, peak, total / len(R) / 1e6, len(act) / (len(W) - 1) * 100,
                  statistics.median(act) if act else 0, act[int(0.9 * (len(act) - 1))] if act else 0))


if __name__ == "__main__":
    main()
