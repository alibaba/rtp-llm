#!/usr/bin/env python3
"""Count how many requests P prefilled together per round (batched-prefill check).

  prefill_batches.py <engine.log> <client-glob> [rank]

Reads P's INFO lines "publish pd plan kv cache, request <id>, rank <r>, ..."
(emitted once per request per stage-1 rank right after that rank's prefill
forward) for the time window of the client runs, and groups them by timestamp
(lines within 25ms belong to one forward). Prints the batch-size histogram and,
per batch, the start time relative to the first request of the run.
"""
import datetime
import glob
import json
import re
import subprocess
import sys


def main():
    log, pattern = sys.argv[1], sys.argv[2]
    rank = sys.argv[3] if len(sys.argv) > 3 else "4"
    rows = [json.load(open(f))["rows"][0] for f in glob.glob(pattern)]
    if not rows:
        print("no client rows")
        return
    t0 = min(r["t_send_wall"] for r in rows) - 1
    t1 = (
        max(r["t_send_wall"] + (r["chunks"][0] if r["chunks"] else 0) for r in rows) + 1
    )
    stamp = datetime.datetime.fromtimestamp(t0).strftime("%Y-%m-%d %H:%M:%S")[:-1]
    off = (
        subprocess.run(
            'grep -a -b -m1 "^\\[%s" %s | cut -d: -f1' % (stamp, log),
            shell=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        or "1"
    )
    out = subprocess.run(
        'tail -c +%s %s | head -c 600000000 | grep -a "publish pd plan kv cache, request" | grep -a "rank %s,"'
        % (off, log, rank),
        shell=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    ev = []
    for l in out:
        t = datetime.datetime.strptime(l[1:27], "%Y-%m-%d %H:%M:%S.%f").timestamp()
        if t0 <= t <= t1:
            ev.append(t)
    ev.sort()
    # The PD publisher logs one line per request after a forward, serially, so
    # one batched forward shows up as a burst of lines a few ms apart. Gaps
    # between forwards are >= one prefill (~60ms+), so split on > 25ms.
    batches, cur = [], []
    for t in ev:
        if cur and t - cur[-1] > 0.025:
            batches.append(cur)
            cur = []
        cur.append(t)
    if cur:
        batches.append(cur)
    hist = {}
    for b in batches:
        hist[len(b)] = hist.get(len(b), 0) + 1
    print(
        "requests published (rank %s): %d in %d forwards; batch-size histogram %s"
        % (rank, len(ev), len(batches), dict(sorted(hist.items())))
    )
    print("forward sizes in order:", [len(b) for b in batches])


if __name__ == "__main__":
    main()
