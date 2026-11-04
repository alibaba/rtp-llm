#!/usr/bin/env python3
"""Stage breakdown of routed2 dispatch/combine from the library's device timestamp rings (all 16 ranks).

Input: <prefix>.r<rank>.json written by the transport's TS dump (DSV4_SM120_RAIL_TS_DUMP). Ring slot i of
every rank holds the same call t (calls are globally sequenced), columns (globaltimer ns, comparable within
one node only):
  0 d1 start   1 d1 doorbell (RDMA send posted)   2 d1 end (same-node P2P rows written)
  3 d2 start   4 d2 partner's rail rows arrived    5 d2 end (all inbound legs in, -1 invalidation done)
  6 c1 start   7 c1 end (partial rows pushed to owners / rail partners)
  8 c2 same-node partials arrived   9 c2 pre-reduce RDMA posted   10 c2 remote pre-reduced row arrived
  11 c2 end (TP all-gather of y done)
usage: rr2_ts_breakdown.py <prefix> [n_ranks=16] [lw=8]
"""
import json
import statistics
import sys

pre = sys.argv[1]
W = int(sys.argv[2]) if len(sys.argv) > 2 else 16
LW = int(sys.argv[3]) if len(sys.argv) > 3 else 8
R = [json.load(open(f"{pre}.r{r}.json"))["ts"] for r in range(W)]
n = min(len(x) for x in R)


def sane(t):
    t = t[:12] if len(t) >= 12 else t
    return (
        all(t)
        and t[0]
        <= t[1]
        <= t[2]
        <= t[3]
        <= t[4]
        <= t[5]
        <= t[6]
        <= t[7]
        <= t[8]
        <= t[11]
        and t[10] <= t[11]
    )


# per-rank stage stats: every complete slot (each rank dumped at its own moment; slots differ in call t)
per_rank_valid = {r: [i for i in range(n) if sane(R[r][i])] for r in range(W)}


# cross-rank stats: slots holding the same call on all 8 ranks of a node (d1 starts within 1 ms; a ring lap is ~200 ms)
def node_slots(node):
    rs = range(node * LW, (node + 1) * LW)
    return [
        i
        for i in range(n)
        if all(sane(R[r][i]) for r in rs)
        and max(R[r][i][0] for r in rs) - min(R[r][i][0] for r in rs) < 1_000_000
    ]


q = lambda v, p: sorted(v)[min(len(v) - 1, int(len(v) * p))] if v else float("nan")
med = statistics.median


def stage(name, f):
    vals = [f(R[r][i]) / 1e3 for r in range(W) for i in per_rank_valid[r]]
    print(f"  {name:46s} p50 {med(vals):7.1f}  p90 {q(vals, .9):7.1f}  us")
    return med(vals)


print(f"ranks={W} calls/rank={min(len(v) for v in per_rank_valid.values())} (ring)")
print("[per rank, per call]")
d_tot = stage("dispatch total  (d1 start -> d2 end)", lambda t: t[5] - t[0])
stage("  d1: build masks + RDMA staging + doorbell", lambda t: t[1] - t[0])
stage("  d1: same-node P2P row writes", lambda t: t[2] - t[1])
stage("  d1 end -> d2 start (launch gap)", lambda t: t[3] - t[2])
stage("  d2: wait partner's rail rows (RDMA)", lambda t: t[4] - t[3])
stage("  d2: forward rail rows + wait all legs + -1", lambda t: t[5] - t[4])
stage("  doorbell -> rail rows in (RDMA one-way, cross rank)", lambda t: t[4] - t[1])
mid = stage("experts  (d2 end -> c1 start)", lambda t: t[6] - t[5])
c_tot = stage("combine total  (c1 start -> c2 end)", lambda t: t[11] - t[6])
stage("  c1: push partial rows (P2P)", lambda t: t[7] - t[6])
stage("  c1 end -> same-node partials in (wait peers)", lambda t: t[8] - t[7])
stage("  c2: pre-reduce partner rows + RDMA post", lambda t: t[9] - t[8])
stage("  c2: wait remote pre-reduced row (RDMA)", lambda t: t[10] - t[9])
stage("  c2: owner reduce + y all-gather + copy out", lambda t: t[11] - t[10])
if all(len(R[r][0]) >= 14 for r in range(W)):
    # protocol 5: 12 = owner reduce + y P2P writes to the 4 TP mates done (last block), 13 = all mates' y slices
    # arrived (block 0). Both measured on this rank's clock.
    stage("    owner reduce + y writes to TP mates (10->12)", lambda t: t[12] - t[10])
    stage("    wait TP mates' y slices (12->13, may be <0)", lambda t: t[13] - t[12])
    stage("    copy mates' rows out (13->11)", lambda t: t[11] - t[13])
    stage("    remote row in -> mates' y in (10->13)", lambda t: t[13] - t[10])

# Cross-rank alignment is not attempted: %globaltimer is per GPU (not synchronized across GPUs) and each
# rank dumps its ring at its own moment, so slot i is not the same call on every rank. Use the kineto traces
# (host clock) for cross-rank skew (a2a_skew.py / expert_skew.py).
print(
    f"[summary] dispatch {d_tot:.1f} + experts {mid:.1f} + combine {c_tot:.1f} us per call"
)
