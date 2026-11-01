#!/usr/bin/env python3
"""GPU idle bubbles inside prefill forwards, from an nsys sqlite export of the P (5kpro CEP4PP2, nsys launch + a
few-second nsys start/stop capture during constant-rate load).

  an_nsys_bubbles.py <capture.sqlite> [<capture.sqlite> ...]

Per GPU (deviceId), kernels are merged into busy intervals; gaps between consecutive busy intervals are idle time.
Gaps > 50 ms are treated as "no work queued" (between forwards / waiting on the pipeline peer) and left out; the
remaining gaps of 20 us .. 50 ms are bubbles inside the work stream: the GPU waited for the host to launch the next
kernel. Reports per stage (devices 0-3 = PP stage 0, 4-7 = stage 1): kernel busy time, bubble time, bubble share,
gap histogram, and the host-side cudaLaunchKernel duration p50 / p99 (runtime API rows).
"""
import sqlite3
import statistics
import sys

BIG_GAP_NS = 50_000_000
SMALL_GAP_NS = 20_000


def kernels(db):
    q = "SELECT deviceId, start, end FROM CUPTI_ACTIVITY_KIND_KERNEL ORDER BY deviceId, start"
    by = {}
    for d, s, e in db.execute(q):
        by.setdefault(d, []).append((s, e))
    return by


def merge(iv):
    out = []
    for s, e in iv:
        if out and s <= out[-1][1]:
            if e > out[-1][1]:
                out[-1][1] = e
        else:
            out.append([s, e])
    return out


def launch_stats(db):
    try:
        ids = [r[0] for r in db.execute("SELECT id FROM StringIds WHERE value LIKE 'cudaLaunchKernel%' OR value LIKE 'cuLaunchKernel%'")]
    except sqlite3.Error:
        return None
    if not ids:
        return None
    q = "SELECT end - start FROM CUPTI_ACTIVITY_KIND_RUNTIME WHERE nameId IN (%s)" % ",".join(map(str, ids))
    d = sorted(r[0] for r in db.execute(q))
    if not d:
        return None
    return len(d), d[len(d) // 2] / 1e3, d[int(0.99 * (len(d) - 1))] / 1e3, statistics.mean(d) / 1e3


def main():
    for path in sys.argv[1:]:
        db = sqlite3.connect(path)
        by = kernels(db)
        print("== %s" % path)
        for stage, devs in (("stage0", (0, 1, 2, 3)), ("stage1", (4, 5, 6, 7))):
            busy = bub = 0
            nb = 0
            hist = [0, 0, 0, 0]
            for d in devs:
                m = merge(by.get(d, []))
                for (s0, e0), (s1, e1) in zip(m, m[1:]):
                    g = s1 - e0
                    if SMALL_GAP_NS <= g <= BIG_GAP_NS:
                        bub += g
                        nb += 1
                        hist[0 if g < 100_000 else 1 if g < 1_000_000 else 2 if g < 10_000_000 else 3] += 1
                busy += sum(e - s for s, e in m)
            n = len(devs)
            if not busy:
                continue
            print("  %s per GPU: kernel busy %.0f ms, bubbles %.0f ms (%d gaps; 20-100us %d, 0.1-1ms %d, 1-10ms %d, 10-50ms %d), "
                  "bubble / (busy + bubble) %.1f%%" % (stage, busy / n / 1e6, bub / n / 1e6, nb // n, hist[0] // n, hist[1] // n,
                                                       hist[2] // n, hist[3] // n, bub / (busy + bub) * 100))
        ls = launch_stats(db)
        if ls:
            print("  cudaLaunchKernel calls %d, host duration p50 %.1f us, p99 %.1f us, mean %.1f us" % ls)


if __name__ == "__main__":
    main()
