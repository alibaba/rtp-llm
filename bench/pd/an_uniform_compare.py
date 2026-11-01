#!/usr/bin/env python3
"""TCP vs RDMA under constant-rate load, per prefix-hit level, pooled over repeated runs.

  an_uniform_compare.py [--skip S] <dir> <level> <tcp_jsonl>[,<tcp_jsonl>...] <rdma_jsonl>[,<rdma_jsonl>...]
  an_uniform_compare.py [--skip S] <dir> --table <prefix_tcp_runs> <prefix_rdma_runs> <level> ...

--skip S drops the requests scheduled in the first S seconds of every run (the ramp from idle to the steady
decode concurrency), so the statistics cover the steady part of one continuous constant-rate stream; the run line
also reports the TTFT p50 per 4 s of the skipped part, which shows how long the ramp took.

Inputs are pd_uniform_load.py --out files. Per request: TTFT = first chunk, tok2 = second chunk (both from the
request's own send time), gap12 = tok2 - TTFT, TPOT = (last chunk - tok2) / (tokens after the 2nd). Requests of a
run are pooled; each run also reports the per-quarter TTFT median (a stable run has a flat trend), the worst
deviation of the send gap from 1/qps, failures and the prefill reuse length range. The RDMA vs TCP difference is
given for the p50 / p90 of TTFT, tok2, gap12 and TPOT, and the run-to-run spread of the per-run p50s.
"""
import glob
import json
import os
import statistics
import sys


def pct(xs, q):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(round(q * (len(xs) - 1))))] if xs else float("nan")


SKIP = 0.0


def load(path):
    rows = [json.loads(l) for l in open(path) if l.strip()]
    t0 = min(r["t_target"] for r in rows)
    ramp = sorted((r for r in rows if r["t_target"] - t0 < SKIP and r["chunks"]), key=lambda r: r["t_target"])
    rows = [r for r in rows if r["t_target"] - t0 >= SKIP]
    ok = [r for r in rows if not r["err"] and len(r["chunks"]) >= 3]
    m = {"ttft": [], "tok2": [], "gap12": [], "tpot": []}
    for r in ok:
        c, t = r["chunks"], r["toks"]
        m["ttft"].append(c[0] * 1e3)
        m["tok2"].append(c[1] * 1e3)
        m["gap12"].append((c[1] - c[0]) * 1e3)
        if t[-1] > t[1]:
            m["tpot"].append((c[-1] - c[1]) * 1e3 / (t[-1] - t[1]))
    by = sorted(ok, key=lambda r: r["idx"])
    q = max(1, len(by) // 4)
    quarters = [statistics.median(r["chunks"][0] * 1e3 for r in by[i:i + q]) for i in range(0, len(by), q)][:4]
    sends = sorted(r["t_send"] for r in rows)
    gaps = [b - a for a, b in zip(sends, sends[1:])]
    target = statistics.median(gaps) if gaps else 0
    reuse = [r.get("prefill_reuse") for r in ok if r.get("prefill_reuse") is not None]
    bins = {}
    for r in ramp:
        bins.setdefault(int((r["t_target"] - t0) // 4), []).append(r["chunks"][0] * 1e3)
    info = {
        "ramp": [statistics.median(bins[b]) for b in sorted(bins)],
        "n": len(rows), "ok": len(ok), "failed": len(rows) - len(ok),
        "ttft_quarters": quarters,
        "send_dev_ms": max(abs(g - target) for g in gaps) * 1e3 if gaps else float("nan"),
        "reuse": (min(reuse), max(reuse)) if reuse else None,
    }
    return m, info


def pooled(paths):
    P = {"ttft": [], "tok2": [], "gap12": [], "tpot": []}
    runs = []
    for p in paths:
        m, info = load(p)
        for k in P:
            P[k] += m[k]
        runs.append((p, m, info))
    return P, runs


def compare(level, tcp_paths, rdma_paths, out=print):
    T, truns = pooled(tcp_paths)
    R, rruns = pooled(rdma_paths)
    out("== %s" % level)
    for name, runs in (("TCP", truns), ("RDMA", rruns)):
        for p, m, info in runs:
            out("  %-4s %-40s ok %d/%d, send gap dev %.1f ms, reuse %s, skipped-part TTFT/4s %s, TTFT p50 by quarter %s, p50 TTFT %.0f tok2 %.0f gap12 %.0f TPOT %.1f" % (
                name, os.path.basename(p), info["ok"], info["n"], info["send_dev_ms"], info["reuse"],
                "/".join("%.0f" % x for x in info["ramp"]) or "-",
                "/".join("%.0f" % x for x in info["ttft_quarters"]),
                statistics.median(m["ttft"]), statistics.median(m["tok2"]), statistics.median(m["gap12"]), statistics.median(m["tpot"])))
    res = {}
    for k in ("ttft", "tok2", "gap12", "tpot"):
        for q in (0.5, 0.9):
            t, r = pct(T[k], q), pct(R[k], q)
            res["%s_p%d" % (k, q * 100)] = (t, r)
        tr = [statistics.median(m[k]) for _, m, _ in truns]
        rr = [statistics.median(m[k]) for _, m, _ in rruns]
        res["%s_runs" % k] = (tr, rr)
    for k, unit in (("ttft", "ms"), ("tok2", "ms"), ("gap12", "ms"), ("tpot", "ms")):
        (t5, r5), (t9, r9) = res[k + "_p50"], res[k + "_p90"]
        tr, rr = res[k + "_runs"]
        out("  %-5s p50 TCP %7.1f RDMA %7.1f  TCP-RDMA %+7.1f (%+.1f%%) | p90 TCP %7.1f RDMA %7.1f (%+.1f%%) | per-run p50 TCP %s RDMA %s" % (
            k, t5, r5, t5 - r5, (t5 - r5) / r5 * 100, t9, r9, (t9 - r9) / r9 * 100,
            "/".join("%.0f" % x if k != "tpot" else "%.1f" % x for x in tr), "/".join("%.0f" % x if k != "tpot" else "%.1f" % x for x in rr)))
    return res


def main():
    global SKIP
    argv = sys.argv[1:]
    if argv[0] == "--skip":
        SKIP = float(argv[1])
        argv = argv[2:]
    d = argv[0]
    if argv[1] == "--table":
        tpat, rpat = argv[2], argv[3]
        for level in argv[4:]:
            tp = sorted(p for pat in tpat.split(",") for p in glob.glob(os.path.join(d, "%s-%s.jsonl" % (pat, level))))
            rp = sorted(p for pat in rpat.split(",") for p in glob.glob(os.path.join(d, "%s-%s.jsonl" % (pat, level))))
            if tp and rp:
                compare(level, tp, rp)
        return
    compare(argv[1], [os.path.join(d, p) for p in argv[2].split(",")], [os.path.join(d, p) for p in argv[3].split(",")])


if __name__ == "__main__":
    main()
