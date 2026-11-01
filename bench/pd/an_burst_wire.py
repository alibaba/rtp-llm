#!/usr/bin/env python3
"""Per gated round: KV bytes on the wire, send window and rate, and when the last KV bytes left relative to the
round's last first token.

  an_burst_wire.py <out_dir> <tag> [grpc_port_base=21000] [ranks=8] [port_step=9]

Inputs: wire-<tag>.jsonl from pd_wire_sampler.py and len-<tag>-32k-*.json from pd_prefix_gated_client.py, both
written on the prefill host. TCP mode: KV bytes are bytes_sent of every socket toward the decode host except the
decode ranks' HTTP / gRPC ports (base + step*r and base + step*r + 1). RDMA mode: P NIC tx bytes (counts protocol
overhead too). The send window runs from 1% to 99% of the round's bytes.
"""
import collections
import glob
import json
import os
import sys


def main():
    out_dir, tag = sys.argv[1], sys.argv[2]
    base = int(sys.argv[3]) if len(sys.argv) > 3 else 21000
    ranks = int(sys.argv[4]) if len(sys.argv) > 4 else 8
    step = int(sys.argv[5]) if len(sys.argv) > 5 else 9
    control = {str(base + step * r + k) for r in range(ranks) for k in (0, 1)}
    release = collections.defaultdict(list)
    for f in glob.glob(os.path.join(out_dir, "len-%s-32k-*.json" % tag)):
        for r in json.load(open(f))["rows"]:
            if r.get("err") or not r.get("chunks"):
                continue
            release[r["t_release_wall"]].append(r["t_send_wall"] + r["chunks"][0])
    wire = []
    for line in open(os.path.join(out_dir, "wire-%s.jsonl" % tag)):
        x = json.loads(line)
        tcp = sum(s[4] for s in x["s"] if s[1] not in control)
        wire.append((x["t"], tcp, x["n"][0] if x.get("n") else 0))
    for t0 in sorted(release):
        last_ft = max(release[t0])
        pts = [w for w in wire if t0 - 0.1 < w[0] < last_ft + 2.5]
        for idx, name in ((1, "tcp"), (2, "nic")):
            if not pts:
                continue
            b0, total = pts[0][idx], pts[-1][idx] - pts[0][idx]
            if total < 1e7:
                continue
            t1 = next(w[0] for w in pts if w[idx] - b0 > 0.01 * total)
            t99 = next(w[0] for w in pts if w[idx] - b0 >= 0.99 * total)
            sent_at_ft = max(w[idx] for w in pts if w[0] <= last_ft) - b0
            print(
                "%s %s: %.2f GB, +%.0f..+%.0f ms after release, %.1f Gb/s, %.0f%% sent at the last first token (+%.0f ms), 99%% at %+.0f ms"
                % (tag, name, total / 1e9, (t1 - t0) * 1e3, (t99 - t0) * 1e3, total * 8 / 1e9 / max(t99 - t1, 1e-3),
                   sent_at_ft * 100 / total, (last_ft - t0) * 1e3, (t99 - last_ft) * 1e3)
            )


if __name__ == "__main__":
    main()
