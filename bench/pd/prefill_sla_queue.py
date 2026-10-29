#!/usr/bin/env python3
"""Prefill queueing under a TTFT SLA: how much of the saturation capacity is usable.

A P instance is modelled from two measured numbers: the start interval T = 1 / saturation req/s (a new request can
start every T seconds; PP2 overlap makes T shorter than the single-request latency on 5kpro) and the single-request
TTFT L. TTFT = wait for a start slot + L. k instances behind an ideal load balancer (each arrival goes to the
instance that can start it earliest). Arrivals are uniform (our benchmark) or Poisson (bursty production traffic).
For each SLA, the largest arrival rate whose p90 (or p99) TTFT stays within the SLA is reported as a fraction of
the k-instance saturation rate.

usage: prefill_sla_queue.py [--sla 1.5 2.0] [--pct 90] [--n 40000]
"""
import argparse
import heapq
import random

# (name, start interval T = 1/saturation, single-request TTFT L), seconds; measured (§96, §97)
CASES = [
    ("5kpro CEP4PP2 冷", 1 / 0.98, 1.26),
    ("5kpro CEP4PP2 50%", 1 / 1.742, 0.739),
    ("5kpro CEP4PP2 93.75%", 1 / 10.07, 0.236),
    ("PPU TP4 冷", 1 / 0.48, 2.02),
    ("PPU TP4 50%", 1 / 0.93, 1.109),
    ("PPU TP4 93.75%", 1 / 5.14, 0.238),
]


def p_ttft(T, L, k, lam, n, pct, poisson, seed=1):
    rng = random.Random(seed)
    free = [0.0] * k  # earliest start time per instance
    heapq.heapify(free)
    t = 0.0
    ttft = []
    for _ in range(n):
        t += rng.expovariate(lam) if poisson else 1 / lam
        s = max(t, heapq.heappop(free))
        heapq.heappush(free, s + T)
        ttft.append(s - t + L)
    ttft = sorted(ttft[n // 10 :])  # drop warm-up
    return ttft[min(len(ttft) - 1, int(pct / 100 * len(ttft)))]


def max_rate(T, L, k, sla, pct, poisson, n):
    if L > sla:
        return 0.0
    lo, hi = 0.0, k / T * 0.9999
    if p_ttft(T, L, k, hi, n, pct, poisson) <= sla:
        return hi
    for _ in range(30):
        mid = (lo + hi) / 2
        if p_ttft(T, L, k, mid, n, pct, poisson) <= sla:
            lo = mid
        else:
            hi = mid
    return lo


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sla", type=float, nargs="+", default=[1.5, 2.0])
    ap.add_argument("--pct", type=float, default=90)
    ap.add_argument("--k", type=int, nargs="+", default=[1, 2, 3, 4])
    ap.add_argument("--n", type=int, default=40000)
    a = ap.parse_args()
    for sla in a.sla:
        print(
            "## TTFT p%g ≤ %.1fs：可用吞吐 / 饱和吞吐（k 个 P 实例，理想负载均衡）\n"
            % (a.pct, sla)
        )
        print(
            "| P | 饱和 req/s | 单条 TTFT | 均匀到达 k=1 | "
            + " | ".join("泊松 k=%d" % k for k in a.k)
            + " |"
        )
        print("|---|---|---|---|" + "---|" * len(a.k))
        for name, T, L in CASES:
            u = max_rate(T, L, 1, sla, a.pct, False, a.n) * T
            ps = [max_rate(T, L, k, sla, a.pct, True, a.n) * T / k for k in a.k]
            print(
                "| %s | %.2f | %.2fs | %.0f%% | %s |"
                % (
                    name,
                    1 / T,
                    L,
                    100 * u,
                    " | ".join("%.0f%%" % (100 * p) for p in ps),
                )
            )
        print()


if __name__ == "__main__":
    main()
