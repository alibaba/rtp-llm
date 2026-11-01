#!/usr/bin/env python3
"""Per-forward host wall time vs GPU time of the prefill (P started with DSV4_FWD_STATS=1 DSV4_FWD_GPU=1).

  an_fwd_gap.py <fwd.txt> [<fwd.txt> ...]

fwd.txt holds the [FWDT] / [FWDTG] lines a run printed to prefill.stdout. For every rank, forward n has
  [FWDT]  rank=R n=K T=<tokens> ... total=<host ms> ... loop=<host ms of the layer loop> ...
  [FWDTG] rank=R T=<tokens> cp=4 gpu_ms=<ms between CUDA events recorded at forward entry and exit>
The two streams are paired in order per rank. gpu_ms spans from the entry event to the exit event on the compute
stream, so it includes any time the stream sat idle waiting for the host to launch the next kernel; host total is the
CPU wall time of the same forward. Reports per stage (ranks 0-3 = PP stage 0, 4-7 = stage 1) the p50 / mean of both
and of the ratio. Comparing two runs that execute the same work (same prompts, same rate), a longer gpu_ms with the
same kernels means launch / scheduling bubbles on the GPU timeline.
"""
import collections
import re
import statistics
import sys

FWDT = re.compile(r"\[FWDT\] rank=(\d+) n=\d+ T=(\d+) cp=\d+ total=([\d.]+)")
FWDTG = re.compile(r"\[FWDTG\] rank=(\d+) T=(\d+) cp=\d+ gpu_ms=([\d.]+)")


def main():
    for path in sys.argv[1:]:
        host = collections.defaultdict(list)
        gpu = collections.defaultdict(list)
        for line in open(path, errors="ignore"):
            m = FWDT.search(line)
            if m:
                host[int(m.group(1))].append((int(m.group(2)), float(m.group(3))))
                continue
            m = FWDTG.search(line)
            if m:
                gpu[int(m.group(1))].append((int(m.group(2)), float(m.group(3))))
        print("== %s" % path)
        for stage, ranks in (("stage0", range(0, 4)), ("stage1", range(4, 8))):
            hs, gs = [], []
            for r in ranks:
                n = min(len(host[r]), len(gpu[r]))
                hs += [h for _, h in host[r][:n]]
                gs += [g for _, g in gpu[r][:n]]
            if not gs:
                continue
            print("  %s forwards %4d  host p50 %7.1f mean %7.1f ms | GPU span p50 %7.1f mean %7.1f ms" % (
                stage, len(gs), statistics.median(hs), statistics.mean(hs), statistics.median(gs), statistics.mean(gs)))


if __name__ == "__main__":
    main()
