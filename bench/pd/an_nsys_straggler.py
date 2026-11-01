#!/usr/bin/env python3
"""Where the extra prefill time of TCP mode goes on the GPU timeline (nsys sqlite of the 5kpro CEP4PP2 P).

  an_nsys_straggler.py <capture.sqlite> [<capture.sqlite> ...]

1. Kernel time by name for stage 0 (GPUs 0-3): compute kernels vs NCCL kernels.
2. Each CP all-gather (ncclDevKernel_AllGather_*) runs on the 4 GPUs of a stage and finishes on all of them at about the
   same time. Instances are matched across the 4 GPUs by end time (within 0.5 ms). For each matched collective:
   wait = last start - own start (time a GPU spends inside the NCCL kernel waiting for the slowest rank), and the idle
   gap on the GPU of the last-arriving rank just before its NCCL kernel (its stream was empty: the host had not yet
   launched the collective).
3. Host side: CUDA runtime call durations (cudaLaunchKernel, cudaMemcpyAsync, cudaStreamSynchronize, event calls) by
   calling thread, and the time each thread spent blocked in pthread rwlock / mutex calls (OS runtime trace).
"""
import bisect
import collections
import sqlite3
import statistics
import sys


def pct(v, q):
    v = sorted(v)
    return v[min(len(v) - 1, int(q * (len(v) - 1)))] if v else float("nan")


def main():
    for path in sys.argv[1:]:
        db = sqlite3.connect(path)
        sid = dict(db.execute("SELECT id, value FROM StringIds"))
        K = collections.defaultdict(list)
        tot = collections.Counter()
        for d, s, e, n in db.execute("SELECT deviceId, start, end, shortName FROM CUPTI_ACTIVITY_KIND_KERNEL"):
            K[d].append((s, e, sid[n]))
            if d < 4:
                tot["nccl" if sid[n].startswith("nccl") else "compute"] += e - s
        print("== %s" % path)
        print("  stage0 kernel time per GPU: compute %.0f ms, NCCL %.0f ms" % (tot["compute"] / 4e6, tot["nccl"] / 4e6))
        for d in K:
            K[d].sort()
        ag = {d: [(s, e) for s, e, n in K[d] if n.startswith("ncclDevKernel_AllGather")] for d in range(4)}
        ends = {d: [e for _, e in ag[d]] for d in range(4)}
        waits, last_gap, nmatch = [], [], 0
        for s0, e0 in ag[0]:
            grp = {0: (s0, e0)}
            for d in (1, 2, 3):
                i = bisect.bisect_left(ends[d], e0)
                best = None
                for j in (i - 1, i):
                    if 0 <= j < len(ends[d]) and abs(ends[d][j] - e0) < 500_000:
                        if best is None or abs(ends[d][j] - e0) < abs(ends[d][best] - e0):
                            best = j
                if best is None:
                    break
                grp[d] = ag[d][best]
            if len(grp) < 4:
                continue
            nmatch += 1
            last = max(grp, key=lambda d: grp[d][0])
            ls = grp[last][0]
            for d, (s, e) in grp.items():
                if d != last:
                    waits.append((ls - s) / 1e3)
            starts = [k[0] for k in K[last]]
            i = bisect.bisect_left(starts, ls)
            if i > 0:
                last_gap.append(max(0, ls - K[last][i - 1][1]) / 1e3)
        if nmatch:
            print("  CP all-gathers matched on 4 GPUs: %d | wait for the last rank p50 %.0f p90 %.0f mean %.0f us | "
                  "idle gap before the last rank's kernel p50 %.0f p90 %.0f mean %.0f us" % (
                      nmatch, pct(waits, .5), pct(waits, .9), statistics.mean(waits), pct(last_gap, .5), pct(last_gap, .9),
                      statistics.mean(last_gap)))
        q = ("SELECT r.globalTid, r.end - r.start, s.value FROM CUPTI_ACTIVITY_KIND_RUNTIME r JOIN StringIds s ON r.nameId = s.id "
             "WHERE s.value LIKE 'cudaLaunchKernel%' OR s.value LIKE 'cudaMemcpyAsync%' OR s.value LIKE 'cudaStreamSynchronize%' "
             "OR s.value LIKE 'cudaEventQuery%' OR s.value LIKE 'cudaEventRecord%'")
        tname = {}
        try:
            for tid, nid in db.execute("SELECT globalTid, nameId FROM ThreadNames"):
                tname[tid] = sid.get(nid, "?")
        except sqlite3.Error:
            pass
        by = collections.defaultdict(list)
        for tid, dur, api in db.execute(q):
            name = tname.get(tid, "?")
            name = name.rstrip("0123456789-_") or name
            by[(api.split("_")[0], name)].append(dur / 1e3)
        rows = sorted(by.items(), key=lambda kv: -len(kv[1]))
        for (api, name), v in rows[:10]:
            print("  %-24s %-18s calls %7d  host p50 %6.1f p99 %7.1f mean %6.1f us" % (api, name, len(v), pct(v, .5), pct(v, .99),
                                                                                      statistics.mean(v)))
        locks = collections.defaultdict(lambda: [0, 0])
        try:
            q = ("SELECT o.globalTid, s.value, o.end - o.start FROM OSRT_API o JOIN StringIds s ON o.nameId = s.id "
                 "WHERE s.value LIKE 'pthread_rwlock%' OR s.value LIKE 'pthread_mutex_lock%'")
            for tid, api, dur in db.execute(q):
                name = tname.get(tid, "?").rstrip("0123456789-_")
                locks[(name, api)][0] += 1
                locks[(name, api)][1] += dur
        except sqlite3.Error:
            pass
        for (name, api), (n, t) in sorted(locks.items(), key=lambda kv: -kv[1][1])[:6]:
            print("  lock wait %-18s %-24s calls %7d  total %7.0f ms" % (name, api, n, t / 1e6))


if __name__ == "__main__":
    main()
