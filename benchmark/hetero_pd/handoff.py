#!/usr/bin/env python3
"""Per-request prefill-side PD handoff timeline from PD_HANDOFF_TRACE lines (PREFILL_TRACE_LOG_ENABLE=1)."""
import glob
import json
import re
import statistics as st
import sys

run = sys.argv[1]
ev = {}
for l in open(run + "/logs/engine.log", errors="ignore"):
    m = re.search(r"PD_HANDOFF_TRACE request=(\d+) event=(\w+) t_us=(\d+)", l)
    if m:
        ev.setdefault(m.group(1), {})[m.group(2)] = int(m.group(3)) / 1e3
gaps = []
for f in sorted(glob.glob(run + "/m*.json")):
    r = [x for x in json.load(open(f))["rows"] if x["tag"] == "measure"][0]
    gaps.append(r["gap12_ms"])
rows = []
for rid, e in ev.items():
    if all(
        k in e
        for k in (
            "first_token_written",
            "load_done_received",
            "generate_sent",
            "first_remote_output",
        )
    ):
        t0 = e["first_token_written"]
        rows.append(
            (
                e["load_done_received"] - t0,
                e["generate_sent"] - e["load_done_received"],
                e["first_remote_output"] - e["generate_sent"],
                e["first_remote_output"] - t0,
            )
        )
rows = rows[-len(gaps) :] if gaps else rows
for r, g in zip(rows, gaps):
    print(
        "wait load-done %5.1f | send GENERATE %4.1f | GENERATE->decode token %5.1f | total %5.1f | client gap12 %5.1f"
        % (*r, g)
    )
if rows:
    print(
        "mean:  wait load-done %.1f | send GENERATE %.1f | GENERATE->decode token %.1f | total %.1f | client gap12 %.1f"
        % (
            st.mean(r[0] for r in rows),
            st.mean(r[1] for r in rows),
            st.mean(r[2] for r in rows),
            st.mean(r[3] for r in rows),
            st.mean(gaps),
        )
    )
