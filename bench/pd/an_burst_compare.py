#!/usr/bin/env python3
"""TCP vs RDMA comparison of gated whole-batch runs, pooled over repeated runs.

  an_burst_compare.py <out_dir> <tcp_tag>[,<tcp_tag>...] <rdma_tag>[,<rdma_tag>...]

Reads the len-<tag>-32k-<i>.json rows of pd_prefix_gated_client.py. Every request of a gated round shares one
prefill forward and one decode first step, so the round (not the request) is the independent sample: per round
take the median over its requests of TTFT, time to the 2nd token, gap12 (tok1 -> tok2) and TPOT
((t_last - t_2nd) / (tokens - 2)), all from the gate release. Round 0 (first batched forward after a restart)
and rounds with a failed request are dropped and reported. Rounds of the same transport are pooled over its runs;
the result is the mean over rounds with the standard error, and the RDMA gain on the 2nd-token time for the
pooled means and for every TCP/RDMA run pair.
"""
import collections
import glob
import json
import os
import statistics
import sys


def rounds(out_dir, tag):
    files = glob.glob(os.path.join(out_dir, "len-%s-32k-*.json" % tag))
    rows = [r for f in files for r in json.load(open(f))["rows"]]
    bad = {r["round"] for r in rows if r.get("err")}
    ok = sum(1 for r in rows if not r.get("err") and r.get("city_ok"))
    per_round = collections.defaultdict(list)
    for r in rows:
        if r.get("err") or not r.get("chunks") or r["round"] == 0 or r["round"] in bad:
            continue
        c = r["chunks"]
        base = r["t_release_wall"] - r["t_send_wall"]
        tpot = (c[-1] - c[1]) / max(r["completion_tokens"] - 2, 1) * 1e3
        per_round[r["round"]].append(((c[0] - base) * 1e3, (c[1] - base) * 1e3, (c[1] - c[0]) * 1e3, tpot))
    med = [[statistics.median(x[i] for x in v) for i in range(4)] for v in per_round.values()]
    return med, sorted(bad), ok, sum(1 for r in rows if not r.get("err")), sum(1 for r in rows if r.get("err"))


def mean_se(v):
    return statistics.mean(v), (statistics.stdev(v) / len(v) ** 0.5 if len(v) > 1 else 0.0)


def main():
    out_dir = sys.argv[1]
    groups = {"TCP": sys.argv[2].split(","), "RDMA": sys.argv[3].split(",")}
    pooled = {}
    per_run = {}
    for name, tags in groups.items():
        pooled[name], per_run[name] = [], []
        for tag in tags:
            med, bad, ok, n_ok, n_err = rounds(out_dir, tag)
            if not med:
                print("%-4s %s: no usable rounds" % (name, tag))
                continue
            pooled[name] += med
            per_run[name].append(statistics.mean(x[1] for x in med))
            print(
                "%-4s %-40s rounds %d, failed requests %d (rounds %s), correct %d/%d, TTFT %.0f, 2nd %.0f, gap12 %.0f, TPOT %.1f"
                % (name, tag, len(med), n_err, bad, ok, n_ok, *[statistics.mean(x[i] for x in med) for i in range(4)])
            )
    for name in ("TCP", "RDMA"):
        v = pooled[name]
        if not v:
            return
        cols = [mean_se([x[i] for x in v]) for i in range(3)]
        print(
            "%-4s pooled %d rounds: TTFT %.0f +- %.0f, 2nd %.0f +- %.0f, gap12 %.0f +- %.0f, TPOT %.1f"
            % (name, len(v), *[a for c in cols for a in c], statistics.mean(x[3] for x in v))
        )
    (t2, et), (r2, er) = mean_se([x[1] for x in pooled["TCP"]]), mean_se([x[1] for x in pooled["RDMA"]])
    print(
        "2nd token: RDMA %.0f vs TCP %.0f ms, RDMA lower by %.0f ms = %.1f%% (+- %.1f%%)"
        % (r2, t2, t2 - r2, (t2 - r2) / t2 * 100, (et**2 + er**2) ** 0.5 / t2 * 100)
    )
    print("per TCP/RDMA run pair: " + " ".join("%.1f%%" % ((a - b) / a * 100) for a in per_run["TCP"] for b in per_run["RDMA"]))


if __name__ == "__main__":
    main()
