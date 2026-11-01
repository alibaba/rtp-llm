#!/usr/bin/env python3
"""Split the 1st -> 2nd token time of a constant-rate run into handoff segments, from the prefill's
PD_HANDOFF_TRACE lines (P started with PREFILL_TRACE_LOG_ENABLE=1).

  an_handoff_split.py <run.jsonl> <p_engine.log> [<run.jsonl> <p_engine.log> ...]

For every measured request of pd_uniform_load.py (run.jsonl, written on the P host so both share one clock) the
P trace event whose first_token_written is closest to the client's first token (within 20 ms) is taken. Per request:
  wait_load = load_done_received - first_token_written   KV still arriving at D when P emitted the first token
  send_gen  = generate_sent - load_done_received          P hands the request over to D
  d_first   = first_remote_output - generate_sent         D's first decode step for the request
Prints p50 / p90 of each, the share of requests whose KV finished more than 5 ms after the first token, and the
match rate. Several (jsonl, log) pairs are pooled.
"""
import collections
import json
import re
import sys

PAT = re.compile(r"PD_HANDOFF_TRACE request=(-?\d+) event=(\w+) t_us=(\d+)")
NEED = ("first_token_written", "load_done_received", "generate_sent", "first_remote_output")


def pct(v, q):
    v = sorted(v)
    return v[min(len(v) - 1, int(round(q * (len(v) - 1))))] if v else float("nan")


def trace(log):
    ev = collections.defaultdict(dict)
    for line in open(log, errors="ignore"):
        m = PAT.search(line)
        if m:
            ev[m.group(1)].setdefault(m.group(2), int(m.group(3)) / 1e6)
    rows = sorted((e for e in ev.values() if all(k in e for k in NEED)), key=lambda e: e["first_token_written"])
    return rows


def main():
    args = sys.argv[1:]
    seg = {"wait_load": [], "send_gen": [], "d_first": [], "tok1_to_tok2_client": []}
    n_req = n_match = 0
    for run, log in zip(args[0::2], args[1::2]):
        tr = trace(log)
        times = [e["first_token_written"] for e in tr]
        import bisect

        for line in open(run):
            r = json.loads(line)
            if r["err"] or len(r["chunks"]) < 2:
                continue
            n_req += 1
            ft = r["t_send"] + r["chunks"][0]
            i = bisect.bisect_left(times, ft)
            best = min((j for j in (i - 1, i) if 0 <= j < len(tr)), key=lambda j: abs(times[j] - ft), default=None)
            if best is None or abs(times[best] - ft) > 0.02:
                continue
            e = tr[best]
            n_match += 1
            seg["wait_load"].append((e["load_done_received"] - e["first_token_written"]) * 1e3)
            seg["send_gen"].append((e["generate_sent"] - e["load_done_received"]) * 1e3)
            seg["d_first"].append((e["first_remote_output"] - e["generate_sent"]) * 1e3)
            seg["tok1_to_tok2_client"].append((r["chunks"][1] - r["chunks"][0]) * 1e3)
    late = sum(1 for v in seg["wait_load"] if v > 5)
    print("requests %d, matched %d, KV finished > 5 ms after the first token: %d (%.0f%%)" % (
        n_req, n_match, late, late * 100 / max(n_match, 1)))
    for k, v in seg.items():
        print("  %-20s p50 %7.1f  p90 %7.1f  mean %7.1f ms" % (k, pct(v, 0.5), pct(v, 0.9), sum(v) / max(len(v), 1)))


if __name__ == "__main__":
    main()
