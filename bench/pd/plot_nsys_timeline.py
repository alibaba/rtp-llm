#!/usr/bin/env python3
"""Timeline figure for the TTFT mechanism: stage-0 GPUs of the 5kpro P, RDMA vs TCP, from two nsys sqlite exports.

  plot_nsys_timeline.py <rdma.sqlite> <tcp.sqlite> <out.png> [window_ms]

For each capture, a window of window_ms (default 120) starting at the first CP all-gather after 1 s into the capture is
drawn: one row per GPU 0-3 with compute kernels (blue), NCCL kernels (orange), and D2H copies of the cache store
(red, TCP only); above, for each GPU, the pthread rwlock waits of the host thread that launches its kernels (matched
through the CUPTI correlation id), from the OS-runtime trace.
"""
import sqlite3
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def rows(path, win_ms):
    db = sqlite3.connect(path)
    sid = dict(db.execute("SELECT id, value FROM StringIds"))
    t0 = db.execute("SELECT min(start) FROM CUPTI_ACTIVITY_KIND_KERNEL").fetchone()[0]
    ag = db.execute(
        "SELECT k.start FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds s ON k.shortName = s.id "
        "WHERE k.deviceId = 0 AND s.value LIKE 'ncclDevKernel_AllGather%' AND k.start > ? ORDER BY k.start LIMIT 1",
        (t0 + 1_000_000_000,)).fetchone()[0]
    a, b = ag - 5_000_000, ag - 5_000_000 + int(win_ms * 1e6)
    K = {d: [] for d in range(4)}
    for d, s, e, n in db.execute(
            "SELECT deviceId, start, end, shortName FROM CUPTI_ACTIVITY_KIND_KERNEL WHERE deviceId < 4 AND end > ? AND start < ?",
            (a, b)):
        K[d].append(((s - a) / 1e6, (e - s) / 1e6, sid[n].startswith("nccl")))
    M = {d: [] for d in range(4)}
    for d, s, e in db.execute(
            "SELECT deviceId, start, end FROM CUPTI_ACTIVITY_KIND_MEMCPY WHERE copyKind = 2 AND deviceId < 4 AND end > ? AND start < ?",
            (a, b)):
        M[d].append(((s - a) / 1e6, max((e - s) / 1e6, 0.02)))
    dev_tid = {}
    for d in range(4):
        r = db.execute(
            "SELECT r.globalTid, count(*) c FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN CUPTI_ACTIVITY_KIND_RUNTIME r "
            "ON k.correlationId = r.correlationId WHERE k.deviceId = ? AND k.start > ? AND k.start < ? "
            "GROUP BY r.globalTid ORDER BY c DESC LIMIT 1", (d, a, b)).fetchone()
        if r:
            dev_tid[d] = r[0]
    L = {dev_tid[d]: [] for d in dev_tid}
    for t, s, e in db.execute(
            "SELECT o.globalTid, o.start, o.end FROM OSRT_API o JOIN StringIds s ON o.nameId = s.id "
            "WHERE s.value LIKE 'pthread_rwlock_%lock' AND o.end > ? AND o.start < ?", (a, b)):
        if t in L:
            L[t].append(((s - a) / 1e6, max((e - s) / 1e6, 0.03)))
    return K, M, L, dev_tid


def main():
    rp, tp, out = sys.argv[1:4]
    win = float(sys.argv[4]) if len(sys.argv) > 4 else 120
    fig, axes = plt.subplots(2, 1, figsize=(14, 6.5), sharex=True)
    for ax, path, title in ((axes[0], rp, "RDMA"), (axes[1], tp, "TCP")):
        K, M, L, DT = rows(path, win)
        y = 0
        ticks, labels = [], []
        for d in range(4):
            ax.broken_barh([(s, w) for s, w, n in K[d] if not n], (y, 0.8), facecolors="#4c72b0")
            ax.broken_barh([(s, w) for s, w, n in K[d] if n], (y, 0.8), facecolors="#dd8452")
            if M[d]:
                ax.broken_barh(M[d], (y + 0.8, 0.2), facecolors="#c44e52")
            ticks.append(y + 0.4)
            labels.append("GPU%d" % d)
            y += 1.2
        for d, t in sorted(DT.items()):
            ax.broken_barh(L[t], (y, 0.5), facecolors="#8172b2")
            ticks.append(y + 0.25)
            labels.append("GPU%d host: lock wait" % d)
            y += 0.7
        ax.set_yticks(ticks)
        ax.set_yticklabels(labels, fontsize=8)
        ax.set_title("%s: stage-0 GPUs (blue compute, orange NCCL incl. waiting for peers, red D2H copy for TCP send), purple = launching thread blocked on a lock" % title,
                     fontsize=10)
        ax.set_xlim(0, win)
    axes[1].set_xlabel("ms")
    fig.tight_layout()
    fig.savefig(out, dpi=130)


if __name__ == "__main__":
    main()
