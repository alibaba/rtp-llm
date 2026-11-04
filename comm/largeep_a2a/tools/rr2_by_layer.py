#!/usr/bin/env python3
"""Per-layer routed2 stage medians from the device timestamp rings (all ranks). Layer = position of the MoE call
inside its forward (verify forwards have 43 calls; 1-call forwards are the MTP draft). usage: rr2_by_layer.py prefix"""
import json
import statistics as S
import sys

pre = sys.argv[1]
acc = {}
for r in range(16):
    ts = json.load(open(f"{pre}.r{r}.json"))["ts"]
    c = sorted([t for t in ts if all(t[:12]) and t[0] <= t[11]], key=lambda t: t[0])
    fw = [[c[0]]]
    for a, b in zip(c, c[1:]):
        if (b[0] - a[11]) / 1e3 > 600:
            fw.append([b])
        else:
            fw[-1].append(b)
    for f in fw:
        for j, t in enumerate(f):
            key = j if len(f) >= 40 else "draft"
            acc.setdefault(key, []).append(
                (
                    (t[11] - t[0]) / 1e3,
                    (t[5] - t[3]) / 1e3,
                    (t[6] - t[5]) / 1e3,
                    (t[11] - t[6]) / 1e3,
                )
            )
print("layer   n   span  d2(wait+fwd)  experts  combine   (p50 us over ranks x calls)")
keys = [k for k in acc if k != "draft"] + ["draft"]
for k in keys:
    v = acc[k]
    print(
        f"{str(k):5s} {len(v):4d} "
        + "  ".join(f"{S.median(x[i] for x in v):7.1f}" for i in range(4))
    )
