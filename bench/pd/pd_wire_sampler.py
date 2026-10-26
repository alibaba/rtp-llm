#!/usr/bin/env python3
"""Low-overhead PD wire sampler for one prefill host.

Every --interval seconds records, with the host wall clock:
  * per-socket bytes_acked / bytes_received / bytes_sent toward --peer
    (ss -tinH), which covers the P->D gRPC control stream (remote port 21001)
    and, in TCP cache-store mode, the KV data sockets. Use bytes_sent for send
    times: bytes_acked lags by the peer's delayed ACK;
  * eRDMA NIC tx/rx byte counters (rdma statistic show link <dev>/1), which
    capture RDMA-mode KV traffic (D reads from P -> P NIC tx).

  pd_wire_sampler.py --peer <decode-host-ip> --rdma-dev erdma_0 --secs 60 \
      --interval 0.002 --out wire.jsonl
"""
import argparse
import json
import re
import subprocess
import time

ACK = re.compile(r"bytes_acked:(\d+)")
RCV = re.compile(r"bytes_received:(\d+)")
SNT = re.compile(r"bytes_sent:(\d+)")
TX = re.compile(r"hw_tx_bytes_cnt (\d+)")
RX = re.compile(r"hw_rx_bytes_cnt (\d+)")


def socks(peer):
    out = subprocess.run(
        ["ss", "-tinH", "dst", peer], capture_output=True, text=True
    ).stdout.splitlines()
    rows = []
    for i in range(0, len(out) - 1, 2):
        head = out[i].split()
        if len(head) < 5:
            continue
        info = out[i + 1]
        a = ACK.search(info)
        r = RCV.search(info)
        s = SNT.search(info)
        rows.append(
            [
                head[3].rsplit(":", 1)[-1],
                head[4].rsplit(":", 1)[-1],
                int(a.group(1)) if a else 0,
                int(r.group(1)) if r else 0,
                int(s.group(1)) if s else 0,
            ]
        )
    return rows


def nic(dev):
    if not dev:
        return None
    out = subprocess.run(
        ["rdma", "statistic", "show", "link", dev + "/1"],
        capture_output=True,
        text=True,
    ).stdout
    t = TX.search(out)
    r = RX.search(out)
    return [int(t.group(1)) if t else 0, int(r.group(1)) if r else 0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--peer", required=True)
    ap.add_argument("--rdma-dev", default="")
    ap.add_argument("--secs", type=float, default=60)
    ap.add_argument("--interval", type=float, default=0.002)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    end = time.time() + a.secs
    n = 0
    with open(a.out, "w") as f:
        while time.time() < end:
            t = time.time()
            rec = {"t": t, "s": socks(a.peer), "n": nic(a.rdma_dev), "t2": time.time()}
            f.write(json.dumps(rec, separators=(",", ":")) + "\n")
            n += 1
            dt = a.interval - (time.time() - t)
            if dt > 0:
                time.sleep(dt)
    print(f"samples={n}")


if __name__ == "__main__":
    main()
