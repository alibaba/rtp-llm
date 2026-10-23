#!/usr/bin/env python3
"""Sample an eRDMA port's hardware TX/RX byte counters at a fixed period on this host.

Counter source: the kernel's RDMA netlink statistics (`rdma -j statistic show link <dev>/<port>`),
fields hw_tx_bytes_cnt / hw_rx_bytes_cnt, which the erdma driver reads from the NIC.
One `rdma` call per sample (~0.6 ms); each sample is stamped at the midpoint of its call.

Output: one line per sample  "<CLOCK_REALTIME s> <CLOCK_MONOTONIC s> <tx_bytes> <rx_bytes>"
CLOCK_REALTIME is the same clock the rtp-llm engine log prints, so samples line up with
[PPSTEP]/publish lines from the same host.
usage: nic_sampler.py <out> [--dev erdma_0/1] [--period-ms 2] [--seconds 120] [--stop-file PATH]
"""
import argparse
import os
import re
import subprocess
import sys
import time

ap = argparse.ArgumentParser()
ap.add_argument("out")
ap.add_argument("--dev", default="erdma_0/1")
ap.add_argument("--period-ms", type=float, default=2.0)
ap.add_argument("--seconds", type=float, default=120.0)
ap.add_argument("--stop-file", default=None)
a = ap.parse_args()

tx_re = re.compile(r'"hw_tx_bytes_cnt":(\d+)')
rx_re = re.compile(r'"hw_rx_bytes_cnt":(\d+)')
cmd = ["rdma", "-j", "statistic", "show", "link", a.dev]
period = a.period_ms / 1000.0
end = time.monotonic() + a.seconds
n = 0
with open(a.out, "w") as f:
    nxt = time.monotonic()
    while time.monotonic() < end:
        if a.stop_file and n % 100 == 0 and os.path.exists(a.stop_file):
            break
        t_rt0 = time.time()
        t_mo0 = time.monotonic()
        line = subprocess.run(cmd, capture_output=True, text=True, timeout=5).stdout
        t_mo1 = time.monotonic()
        t_rt1 = time.time()
        mt, mr = tx_re.search(line), rx_re.search(line)
        if mt and mr:
            # timestamp = midpoint of the call window; column 5 = call duration (ms)
            f.write(
                f"{(t_rt0 + t_rt1) / 2:.6f} {(t_mo0 + t_mo1) / 2:.6f} {mt.group(1)} {mr.group(1)} {1e3 * (t_mo1 - t_mo0):.3f}\n"
            )
            n += 1
        nxt += period
        delay = nxt - time.monotonic()
        if delay > 0:
            time.sleep(delay)
        else:
            nxt = time.monotonic()
print(f"{n} samples -> {a.out}", file=sys.stderr)
