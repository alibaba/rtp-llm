#!/usr/bin/env python3
"""Per gated round: when each PP stage of the prefill finished its requests, from the prefill engine log.

  an_burst_stages.py <out_dir> <tag> <engine.log> [ranks_per_stage=4]

Every rank logs "publish pd plan kv cache, request X, rank R" when its stage has finished request X's layers; ranks
0..n-1 are stage 0, the rest stage 1. Prints, averaged over rounds (round 0 skipped) and from the gate release:
first / last publish of each stage and first / last first token.
"""
import collections
import glob
import json
import os
import re
import statistics
import sys
from datetime import datetime

PUBLISH = re.compile(r"^\[(\S+ \S+)\].*publish pd plan kv cache, request (\d+), rank (\d+)")


def main():
    out_dir, tag, log = sys.argv[1:4]
    per_stage = int(sys.argv[4]) if len(sys.argv) > 4 else 4
    events = []
    for line in open(log, errors="ignore"):
        m = PUBLISH.search(line)
        if m:
            t = datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S.%f").timestamp()
            events.append((t, int(m.group(3)) // per_stage))
    release = collections.defaultdict(list)
    for f in glob.glob(os.path.join(out_dir, "len-%s-32k-*.json" % tag)):
        for r in json.load(open(f))["rows"]:
            if r.get("err") or not r.get("chunks") or r["round"] == 0:
                continue
            release[r["t_release_wall"]].append(r["t_send_wall"] + r["chunks"][0])
    acc = collections.defaultdict(list)
    for t0, first_tokens in release.items():
        end = max(first_tokens) + 1
        for stage in (0, 1):
            ts = [t for t, s in events if s == stage and t0 < t < end]
            if ts:
                acc["stage%d first" % stage].append((min(ts) - t0) * 1e3)
                acc["stage%d last" % stage].append((max(ts) - t0) * 1e3)
        acc["first token"].append((min(first_tokens) - t0) * 1e3)
        acc["last first token"].append((max(first_tokens) - t0) * 1e3)
    print("%s, %d rounds, ms from release: " % (tag, len(acc["first token"])) + ", ".join(
        "%s %.0f" % (k, statistics.mean(v)) for k, v in acc.items()))


if __name__ == "__main__":
    main()
