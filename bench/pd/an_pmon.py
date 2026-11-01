#!/usr/bin/env python3
"""P-host CPU during a measured load window (pmon.sh jsonl + the pd_uniform_load.py --out of the same run).

  an_pmon.py <pmon.jsonl> <load.jsonl> [<pmon.jsonl> <load.jsonl> ...]

Window = [first send, last send + its TTFT]. Prints the cores used by the 8 rank processes in total and by each of
their thread groups (pmon.sh strips the trailing number from thread names). /proc/stat inside the container does not
reflect the host, so machine-wide busy cores are not reported.
"""
import collections
import json
import sys


def window(load):
    t0, t1 = None, None
    for line in open(load):
        r = json.loads(line)
        if not r.get("chunks"):
            continue
        a, b = r["t_send"], r["t_send"] + r["chunks"][0]
        t0 = a if t0 is None else min(t0, a)
        t1 = b if t1 is None else max(t1, b)
    return t0, t1


def main():
    args = sys.argv[1:]
    for pm, load in zip(args[::2], args[1::2]):
        t0, t1 = window(load)
        M = [json.loads(l) for l in open(pm)]
        C = collections.Counter()
        span = 0.0
        for a, x in zip(M, M[1:]):
            if not (t0 <= x["t"] <= t1):
                continue
            span += x["t"] - a["t"]
            for k, v in x["thr"].items():
                C[k] += v
        if not span:
            print("== %s: no samples in window" % pm)
            continue
        print("== %s  window %.0fs  rank processes %.1f cores" % (pm, span, sum(C.values()) / 100 / span))
        print("   " + ", ".join("%s %.1f" % (k, v / 100 / span) for k, v in C.most_common(8)))


if __name__ == "__main__":
    main()
