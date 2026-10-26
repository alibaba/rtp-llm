#!/usr/bin/env python3
"""Per-request PD exposure breakdown from pd_wire_sampler + pd_ttft_gap_client.

  an_exposed.py <wire.jsonl> <client-json> [<client-json> ...]

For each request (same host clock as the sampler):
  * KV wire bytes in [send, tok2]: eRDMA NIC tx delta (RDMA mode) and the
    cache-store TCP sockets' bytes_acked delta (TCP mode, local port 14xxx);
  * kv_last: time (rel tok1) the KV byte counter last advanced before tok2;
  * grpc events on the P->D control stream (remote port 21001): every sample
    where bytes_acked / bytes_received moved, rel tok1.
"""
import bisect
import json
import statistics
import sys


def load_wire(p):
    rows = []
    for line in open(p):
        try:
            rows.append(json.loads(line))
        except Exception:
            pass
    return rows


def main():
    wire = load_wire(sys.argv[1])
    ts = [w["t"] for w in wire]
    out = []
    for cf in sys.argv[2:]:
        r = json.load(open(cf))["rows"][0]
        c = r["chunks"]
        if len(c) < 6:
            continue
        t0 = r["t_send_wall"]
        t1, t2 = t0 + c[0], t0 + c[1]
        step = statistics.median([b - a for a, b in zip(c[2:], c[3:])])
        i0 = max(0, bisect.bisect_left(ts, t0 - 0.05))
        i1 = min(len(wire), bisect.bisect_right(ts, t2 + 0.15))
        seg = wire[i0:i1]
        if not seg:
            continue

        def kv_tcp(w):
            return sum(s[2] for s in w["s"] if s[0].startswith("140"))

        def nic_tx(w):
            return w["n"][0] if w.get("n") else 0

        def grpc(w):
            g = [s for s in w["s"] if s[1] == "21001"]
            return (g[0][2], g[0][3]) if g else None

        base_nic, base_tcp = nic_tx(seg[0]), kv_tcp(seg[0])
        ev = []
        prev_g = None
        nic_last = tcp_last = None
        prev_nic, prev_tcp = base_nic, base_tcp
        for w in seg:
            rel = (w["t"] - t1) * 1000
            n, k = nic_tx(w), kv_tcp(w)
            if w["t"] <= t2:
                if n - prev_nic > 64 * 1024:
                    nic_last = rel
                if k != prev_tcp:
                    tcp_last = rel
            prev_nic, prev_tcp = n, k
            g = grpc(w)
            if g and prev_g and g != prev_g:
                ev.append((round(rel, 1), g[0] - prev_g[0], g[1] - prev_g[1]))
            if g:
                prev_g = g
        at_t2 = [w for w in seg if w["t"] <= t2][-1]
        nic_mb = (nic_tx(at_t2) - base_nic) / 1e6
        tcp_mb = (kv_tcp(at_t2) - base_tcp) / 1e6
        rec = {
            "tag": json.load(open(cf))["tag"],
            "isl": r["prompt_tokens"],
            "ttft": c[0] * 1000,
            "gap12": (c[1] - c[0]) * 1000,
            "step": step * 1000,
            "exposed": (c[1] - c[0] - step) * 1000,
            "nic_mb": nic_mb,
            "tcp_mb": tcp_mb,
            "nic_last": nic_last,
            "tcp_last": tcp_last,
            "grpc": [
                e for e in ev if -c[0] * 1000 - 60 < e[0] < (c[1] - c[0]) * 1000 + 5
            ],
        }
        out.append(rec)
    for o in out:
        print(
            "%-12s ISL=%5d TTFT=%6.0f gap12=%5.1f exp=%5.1f  nicMB=%6.1f tcpMB=%6.1f  kv_last(rel tok1) nic=%s tcp=%s"
            % (
                o["tag"],
                o["isl"],
                o["ttft"],
                o["gap12"],
                o["exposed"],
                o["nic_mb"],
                o["tcp_mb"],
                "%.1f" % o["nic_last"] if o["nic_last"] is not None else "-",
                "%.1f" % o["tcp_last"] if o["tcp_last"] is not None else "-",
            )
        )
        print("     grpc(rel tok1, +sent, +recv):", o["grpc"])
    json.dump(out, open(sys.argv[1] + ".breakdown.json", "w"))


if __name__ == "__main__":
    main()
