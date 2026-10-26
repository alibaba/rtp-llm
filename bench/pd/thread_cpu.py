#!/usr/bin/env python3
"""CPU-seconds per thread name for all rtp_llm processes over a window.

  thread_cpu.py snap  <out.json>
  thread_cpu.py diff  <before.json> <after.json> [top]

snap reads utime+stime (clock ticks) of every thread of every process whose
cmdline mentions rtp_llm; diff prints CPU-seconds per (thread comm) between the
two snapshots, summed over processes, plus the wall time between them.
"""
import json
import os
import re
import sys
import time


def snap():
    hz = os.sysconf("SC_CLK_TCK")
    out = {"t": time.time(), "threads": {}}
    for pid in os.listdir("/proc"):
        if not pid.isdigit():
            continue
        try:
            cmd = (
                open("/proc/%s/cmdline" % pid, "rb")
                .read()
                .replace(b"\0", b" ")
                .decode("utf-8", "ignore")
            )
        except Exception:
            continue
        if "rtp_llm" not in cmd:
            continue
        try:
            tids = os.listdir("/proc/%s/task" % pid)
        except Exception:
            continue
        for tid in tids:
            try:
                st = open("/proc/%s/task/%s/stat" % (pid, tid)).read()
            except Exception:
                continue
            m = re.match(r"\d+ \((.*)\) (.*)", st)
            comm, rest = m.group(1), m.group(2).split()
            ticks = int(rest[11]) + int(rest[12])
            out["threads"]["%s/%s" % (pid, tid)] = [comm, ticks / hz]
    return out


def main():
    if sys.argv[1] == "snap":
        json.dump(snap(), open(sys.argv[2], "w"))
        return
    a, b = json.load(open(sys.argv[2])), json.load(open(sys.argv[3]))
    top = int(sys.argv[4]) if len(sys.argv) > 4 else 15
    agg = {}
    for k, (comm, sec) in b["threads"].items():
        prev = a["threads"].get(k, [comm, 0.0])[1]
        name = re.sub(r"[-_]?\d+$", "", comm)
        agg[name] = agg.get(name, 0.0) + max(0.0, sec - prev)
    wall = b["t"] - a["t"]
    total = sum(agg.values())
    print("wall %.1fs  total CPU %.1fs (%.1f cores avg)" % (wall, total, total / wall))
    for name, sec in sorted(agg.items(), key=lambda kv: -kv[1])[:top]:
        print("  %-24s %7.1f s  (%5.2f cores)" % (name, sec, sec / wall))


if __name__ == "__main__":
    main()
