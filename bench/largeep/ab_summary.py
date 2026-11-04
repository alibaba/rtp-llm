#!/usr/bin/env python3
"""Summarise one ab_run.sh result directory: step / TPOT / acceptance (p50, min, per-group for cc63).
usage: ab_summary.py /tmp/ab_<TAG> [...]"""
import collections
import json
import re
import sys


def last_json(p):
    try:
        lines = [l for l in open(p) if l.startswith("{")]
        return json.loads(lines[-1]) if lines else {}
    except OSError:
        return {}


def acc(p):
    g = collections.defaultdict(list)
    try:
        for l in open(p):
            if l.startswith("rid") and " chunks=0 " not in l and "err=None" in l:
                c = int(re.search(r"chunks=(\d+)", l).group(1))
                t = int(re.search(r"toks=(\d+)", l).group(1))
                g[int(re.search(r"grp(\d)", l).group(1))].append(t / c)
    except OSError:
        pass
    allv = sorted(x for v in g.values() for x in v)
    return allv, g


for d in sys.argv[1:]:
    tag = d.rstrip("/").split("ab_")[-1]
    f = last_json(f"{d}/forced.log")
    r = last_json(f"{d}/real.log")
    r1 = last_json(f"{d}/real1k.log")
    c = last_json(f"{d}/cc63.log")
    a, _ = acc(f"{d}/real.log")
    a1, _ = acc(f"{d}/real1k.log")
    _, g63 = acc(f"{d}/cc63.log")
    p50 = lambda v: v[len(v) // 2] if v else float("nan")
    print(
        f"{tag:14s} forced {f.get('step_ms_p50')} (rate {f.get('step_ms_from_rate')}) | real {r.get('step_ms_p50')} "
        f"tpot {r.get('tpot_ms')} acc {p50(a):.3f}/{(a[0] if a else float('nan')):.3f} | real1k {r1.get('step_ms_p50')} "
        f"acc {p50(a1):.3f} | cc63 {c.get('step_ms_p50')} grp-acc "
        + " ".join(f"{k}:{sum(v)/len(v):.2f}" for k, v in sorted(g63.items()))
    )
