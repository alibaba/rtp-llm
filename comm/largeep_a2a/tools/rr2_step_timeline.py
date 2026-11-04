#!/usr/bin/env python3
"""Unprofiled per-step timeline from the routed2 device timestamp ring of one rank (globaltimer, ns).
Each MoE call has d1 start (col 0) and c2 end (col 11). Calls sorted by d1 start; a step = 43 verify calls + the
draft calls; step boundaries are found as the gaps > --boundary-us. Reports per step: MoE time (sum of call spans),
gaps between consecutive MoE calls inside a forward (attention + hc + norm/gate + glue), and the inter-forward gaps.
usage: rr2_step_timeline.py ts_prefix rank [--boundary-us 600]"""
import argparse
import json
import statistics as S

ap = argparse.ArgumentParser()
ap.add_argument("pre")
ap.add_argument("rank", type=int)
ap.add_argument("--boundary-us", type=float, default=600.0)
a = ap.parse_args()
ts = json.load(open(f"{a.pre}.r{a.rank}.json"))["ts"]
c = sorted([t for t in ts if all(t[:12]) and t[0] <= t[11]], key=lambda t: t[0])
span = [(t[11] - t[0]) / 1e3 for t in c]
gap = [(c[i + 1][0] - c[i][11]) / 1e3 for i in range(len(c) - 1)]
# group into forwards separated by big gaps
fw, cur = [], [0]
for i, g in enumerate(gap):
    if g > a.boundary_us:
        fw.append((cur, g))
        cur = [i + 1]
    else:
        cur.append(i + 1)
fw.append((cur, None))
sizes = [len(f[0]) for f in fw]
print("forward sizes (MoE calls):", sizes[:40])
# steps: verify forward (>=40 calls) followed by draft forwards (1 call each) until next verify
steps, i = [], 0
while i < len(fw):
    if len(fw[i][0]) >= 40:
        j = i + 1
        while j < len(fw) and len(fw[j][0]) < 40:
            j += 1
        if j < len(fw):
            steps.append((i, j))
        i = j
    else:
        i += 1
rows = []
for i, j in steps:
    v = fw[i][0]
    t_start = c[v[0]][0]
    t_end_next = c[fw[j][0][0]][0]
    step = (t_end_next - t_start) / 1e3
    moe_v = sum(span[k] for k in v)
    intra_v = sum(gap[k] for k in v[:-1])
    drafts = [fw[k][0] for k in range(i + 1, j)]
    moe_d = sum(span[k] for d in drafts for k in d)
    inter = [fw[k][1] for k in range(i, j)]
    rows.append(
        dict(
            step=step,
            verify_moe=moe_v,
            verify_nonmoe=intra_v,
            verify_calls=len(v),
            draft_moe=moe_d,
            n_draft=len(drafts),
            inter_gaps=sum(inter),
            inter_list=[round(x) for x in inter],
        )
    )
for r in rows[:6]:
    print({k: (round(v, 1) if isinstance(v, float) else v) for k, v in r.items()})
med = lambda k: S.median(r[k] for r in rows)
print(
    f"steps={len(rows)}  step {med('step')/1e3:.2f} ms = verify MoE {med('verify_moe')/1e3:.2f} + verify non-MoE (between MoE calls) "
    f"{med('verify_nonmoe')/1e3:.2f} + draft MoE {med('draft_moe')/1e3:.2f} + inter-forward gaps {med('inter_gaps')/1e3:.2f} ms"
)
