#!/usr/bin/env python3
"""Scheduler run-queue wait of the decode host's threads (from /proc/<pid>/task/<tid>/schedstat), by thread name.

  sched_wait.py <out.json> <seconds>

schedstat = (on-CPU ns, run-queue wait ns, timeslices). Over the window, per thread group: CPU cores used, run-queue
wait per second (how long runnable threads waited for a CPU), and the mean wait per timeslice (per wake-up). Only
processes whose command line contains rtp_llm.
"""
import collections
import json
import os
import re
import sys
import time


def snap():
    d = {}
    for p in os.listdir("/proc"):
        if not p.isdigit():
            continue
        try:
            if b"rtp_llm" not in open("/proc/%s/cmdline" % p, "rb").read():
                continue
            tids = os.listdir("/proc/%s/task" % p)
        except OSError:
            continue
        for t in tids:
            try:
                name = open("/proc/%s/task/%s/comm" % (p, t)).read().strip()
                run, wait, n = map(int, open("/proc/%s/task/%s/schedstat" % (p, t)).read().split())
            except (OSError, ValueError):
                continue
            d[t] = (re.sub(r"[-_]?\d+$", "", name), run, wait, n)
    return d


def main():
    out, secs = sys.argv[1], float(sys.argv[2])
    a, t0 = snap(), time.time()
    time.sleep(secs)
    b, dt = snap(), time.time() - t0
    g = collections.defaultdict(lambda: [0, 0, 0, 0])
    for t, (name, run, wait, n) in b.items():
        if t not in a:
            continue
        x = g[name]
        x[0] += run - a[t][1]
        x[1] += wait - a[t][2]
        x[2] += n - a[t][3]
        x[3] += 1
    rows = {k: {"threads": v[3], "cores": v[0] / dt / 1e9, "wait_ms_per_s": v[1] / dt / 1e6,
                "wait_us_per_slice": v[1] / max(v[2], 1) / 1e3} for k, v in g.items() if v[0] > 0}
    json.dump({"secs": dt, "groups": rows}, open(out, "w"), indent=1)
    for k, v in sorted(rows.items(), key=lambda kv: -kv[1]["wait_ms_per_s"])[:12]:
        print("%-20s threads %4d  cores %6.2f  run-queue wait %8.1f ms/s  per slice %7.1f us" % (
            k, v["threads"], v["cores"], v["wait_ms_per_s"], v["wait_us_per_slice"]))


if __name__ == "__main__":
    main()
