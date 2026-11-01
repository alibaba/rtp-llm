#!/usr/bin/env python3
"""Summarise end-to-end runs written by pd_uniform_load.py (one jsonl per run).

usage: an_e2e_tab.py [--md] [FORCE:]run.jsonl ...
  FORCE: prefix -> the D ran with force_sp_accept on a 4-token draft (MTP 3+1) and the TPOT is rescaled by 4/3.3
  (5kpro TP4 D accepts 4 tokens per step when forced, the reported TPOT is quoted at an accept length of 3.3).

Columns: requests sent, failed, achieved send QPS, TTFT p50/p90, time to 2nd token p50, TPOT p50/p90 (2nd token -> last),
TTFT median of every 15 s of send time (flat = no queueing), output tok/s (completed tokens / send window) and the mean
number of requests in decode (Little's law: sum of decode durations / send window).
"""
import collections
import json
import os
import statistics as st
import sys


def pct(v, a):
    return v[min(len(v) - 1, int(a * (len(v) - 1)))]


def load(f):
    """pd_uniform_load.py jsonl (full chunk timeline) or the compact archive form (ttft / t2 / t_last per request)."""
    R = [json.loads(l) for l in open(f)]
    for r in R:
        if "chunks" not in r and r.get("ttft") is not None and r.get("n_chunks", 0) > 2:
            r["chunks"] = [r["ttft"], r["t2"], r["t_last"]]
            r["toks"] = [1, r["tok2"], r["tok_last"]]
    return R


def run(f, scale=1.0):
    R = load(f)
    ok = [r for r in R if r.get("chunks") and len(r["chunks"]) > 2]
    if not ok:
        return None
    t0 = min(r["t_send"] for r in R)
    t1 = max(r["t_send"] for r in R)
    win = max(t1 - t0, 1e-6)
    b = collections.defaultdict(list)
    for r in ok:
        b[int((r["t_send"] - t0) // 15)].append(r["chunks"][0] * 1e3)
    trend = [round(st.median(v)) for _, v in sorted(b.items())]
    tt = sorted(r["chunks"][0] * 1e3 for r in ok)
    t2 = sorted(r["chunks"][1] * 1e3 for r in ok)
    tp = sorted(
        (r["chunks"][-1] - r["chunks"][1]) / (r["toks"][-1] - r["toks"][1]) * 1e3 * scale
        for r in ok
        if r["toks"][-1] > r["toks"][1]
    )
    toks = sum(r["toks"][-1] for r in ok)
    dec = sum(r["chunks"][-1] - r["chunks"][0] for r in ok)
    return dict(
        n=len(R),
        fail=len(R) - len(ok),
        qps=(len(R) - 1) / win,
        ttft50=pct(tt, 0.5),
        ttft90=pct(tt, 0.9),
        tok2=pct(t2, 0.5),
        tpot50=pct(tp, 0.5),
        tpot90=pct(tp, 0.9),
        trend=trend,
        out_tok_s=toks / win,
        d_cc=dec / win,
    )


def main():
    md = "--md" in sys.argv
    files = [a for a in sys.argv[1:] if a != "--md"]
    if md:
        print("| run | n | fail | QPS | TTFT p50 / p90 | tok2 p50 | TPOT p50 / p90 | TTFT per 15 s | out tok/s | D cc |")
        print("|---|---|---|---|---|---|---|---|---|---|")
    for f in files:
        sc = 1.0
        if f.startswith("FORCE:"):
            f, sc = f[6:], 4 / 3.3
        m = run(f, sc)
        if not m:
            print("%s: no completed request" % os.path.basename(f))
            continue
        name = os.path.basename(f).replace(".jsonl", "")
        if md:
            print(
                "| %s | %d | %d | %.2f | %.0f / %.0f | %.0f | %.1f / %.1f | %s | %.0f | %.0f |"
                % (name, m["n"], m["fail"], m["qps"], m["ttft50"], m["ttft90"], m["tok2"], m["tpot50"], m["tpot90"],
                   " → ".join(map(str, m["trend"])), m["out_tok_s"], m["d_cc"])
            )
        else:
            print(
                "%-30s n=%d fail=%d qps=%.2f TTFT p50/p90 %.0f/%.0f tok2 %.0f TPOT p50/p90 %.1f/%.1f trend(15s) %s "
                "out %.0f tok/s Dcc %.0f"
                % (name, m["n"], m["fail"], m["qps"], m["ttft50"], m["ttft90"], m["tok2"], m["tpot50"], m["tpot90"],
                   m["trend"], m["out_tok_s"], m["d_cc"])
            )


if __name__ == "__main__":
    main()
