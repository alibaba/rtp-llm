#!/usr/bin/env python3
"""KV-transfer bandwidth per PD request, from prefill-host NIC counter samples + prefill engine log.

Inputs (both from the prefill host, so one clock):
  nic   : nic_sampler.py output  "<realtime s> <monotonic s> <tx_bytes> <rx_bytes> <call_ms>"
  engine: prefill engine.log (realtime timestamps; [PPSTEP] lines need RTP_LLM_PP_OBJ_LOG=1)
  bench : pd_bench.py json (server_first_token_ms per request, in order)

Per request (anchored on rank 0's "publish kv cache for pd sep" = first token on prefill):
  bytes     : tx counter delta over the request's transfer window
  window    : first and last 2-ms sample with tx > idle threshold, inside [alloc - 0.5 s, first token + 2 s]
  bandwidth : bytes / (last_busy - first_busy)          (mean while transferring)
  peak_10ms : max rate over 10-ms windows
  after_ft  : bytes that left the NIC after the first token (these can delay the first decode step)
Time resolution is the sampling period (~2 ms); windows shorter than ~20 ms are reported but flagged.
"""
import argparse
import json
import re
from datetime import datetime

ap = argparse.ArgumentParser()
ap.add_argument("nic")
ap.add_argument("engine")
ap.add_argument(
    "--idle-bps",
    type=float,
    default=50e6,
    help="rate below which the NIC counts as idle",
)
ap.add_argument("--json-out", default=None)
a = ap.parse_args()

rt, tx = [], []
for line in open(a.nic):
    p = line.split()
    rt.append(float(p[0]))
    tx.append(int(p[2]))
ts = lambda l: datetime.strptime(l[1:27], "%Y-%m-%d %H:%M:%S.%f").timestamp()
alloc, firsttok, r0steps = [], [], []
for l in open(a.engine, errors="ignore"):
    if "legacy peer without PD handoff" in l:
        alloc.append(ts(l))
    elif "[RANK 0]" in l and "publish kv cache for pd sep, stream" in l:
        firsttok.append(ts(l))
    elif "[RANK 0]" in l and "[PPSTEP]" in l and "streams=1" in l:
        r0steps.append(ts(l))


def idx(t):
    lo, hi = 0, len(rt) - 1
    while lo < hi:
        m = (lo + hi) // 2
        if rt[m] < t:
            lo = m + 1
        else:
            hi = m
    return lo


out = []
for k, ft in enumerate(firsttok):
    al = max([x for x in alloc if x < ft] or [ft - 1.5])
    lo, hi = idx(al - 0.5), idx(ft + 2.0)
    busy = [
        i
        for i in range(lo + 1, hi + 1)
        if (tx[i] - tx[i - 1]) / max(rt[i] - rt[i - 1], 1e-6) > a.idle_bps
    ]
    if not busy:
        out.append(dict(req=k, bytes=0))
        continue
    b0, b1 = busy[0] - 1, busy[-1]
    nbytes = tx[b1] - tx[b0]
    dur = rt[b1] - rt[b0]
    i_ft = idx(ft)
    after = tx[b1] - tx[min(max(i_ft, b0), b1)]
    peak = 0.0
    j = b0
    while j < b1:
        m = j
        while m < b1 and rt[m] - rt[j] < 0.010:
            m += 1
        if m > j:
            peak = max(peak, (tx[m] - tx[j]) / (rt[m] - rt[j]))
        j = m
    steps = [s for s in r0steps if al <= s <= ft]
    rec = dict(
        req=k,
        bytes=nbytes,
        MiB=round(nbytes / 2**20, 1),
        busy_ms=round(1e3 * dur, 1),
        mean_GBps=round(nbytes / dur / 1e9, 2) if dur > 0 else None,
        peak10ms_GBps=round(peak / 1e9, 2),
        start_vs_firsttoken_ms=round(1e3 * (rt[b0] - ft), 1),
        end_vs_firsttoken_ms=round(1e3 * (rt[b1] - ft), 1),
        MiB_after_firsttoken=round(after / 2**20, 1),
        prefill_rounds=len(steps),
        flag_short_window=dur < 0.02,
    )
    out.append(rec)
    print(json.dumps(rec))
if a.json_out:
    json.dump(out, open(a.json_out, "w"), indent=1)
