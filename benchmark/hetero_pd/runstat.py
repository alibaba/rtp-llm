#!/usr/bin/env python3
"""One-line stats per run dir: TTFT/gap/TPOT from m*.json, NIC from kv_bw.json, per-burst drain rate (backlog runs)."""
import glob
import json
import os
import re
import statistics as st
import subprocess
import sys

BURST = os.path.join(os.path.dirname(os.path.abspath(__file__)), "burst.py")
for run in sys.argv[1:]:
    rows = []
    for f in sorted(glob.glob(run + "/m*.json")):
        d = json.load(open(f))
        rows += [r for r in d["rows"] if r["tag"] == "measure"]
    ok = [r for r in rows if not r["error"]]
    m = lambda k: st.mean([r[k] for r in ok if r.get(k)]) if ok else float("nan")
    bw = [b for b in json.load(open(run + "/kv_bw.json")) if b.get("bytes")]
    mb = lambda k: st.mean([b[k] for b in bw]) if bw else float("nan")
    ids = set(tuple(r.get("output_ids") or []) for r in ok)
    print(
        f"{run}: {len(ok)}/{len(rows)} ok  srvTTFT {m('server_first_token_ms'):.1f}  cliTTFT {m('client_ttft_ms'):.1f}  gap12 {m('gap12_ms'):.1f}  TPOT {m('client_tpot_ms'):.2f}"
        f" | NIC {mb('MiB'):.1f} MiB  busy {mb('busy_ms'):.1f} ms  after_ft {mb('MiB_after_firsttoken'):.1f} MiB  end_vs_ft {mb('end_vs_firsttoken_ms'):.1f} ms  peak10 {mb('peak10ms_GBps'):.2f} GB/s"
        f" | distinct outputs {len(ids)}"
    )
    rates = {0: [], 1: []}
    for k in range(len(bw)):
        out = subprocess.run(
            [sys.executable, BURST, run, str(k + 1)], capture_output=True, text=True
        ).stdout
        bs = re.findall(
            r"burst .*?: +([\d.]+) MiB in +([\d.]+) ms = +([\d.]+) GB/s", out
        )
        if len(bs) == 2:
            for i in (0, 1):
                rates[i].append(float(bs[i][2]))
    if rates[0] and all(float(x) for x in rates[0]) and max(mb("busy_ms"), 0) < 400:
        print(
            f"   burst drain GB/s (stage0 / stage1, payload+hdr): {st.mean(rates[0]):.2f} / {st.mean(rates[1]):.2f}  (n={len(rates[0])})"
        )
