#!/usr/bin/env python3
"""Per-round summary for pd_prefix_gated_client.py runs.

  an_prefix_gated.py <tag> <rdma|tcp> [per_request_MB]

Latencies are measured from the round's release time (gated runs) or from each
request's own send time (natural arrival). Prints per round: TTFT first / p50 /
last, time to 2nd token, gap12, steady TPOT (median of token gaps from the 3rd
token on), and the KV timeline from the wire sampler (P NIC tx for RDMA,
cache-store socket bytes_sent for TCP): MB sent per request, and the last KV
byte relative to the round's first and last tok1. Then an ALL line.
"""
import bisect
import glob
import json
import re
import statistics
import sys


def main():
    tag, mode = sys.argv[1], sys.argv[2]
    wire = []
    for line in open("wire-%s.jsonl" % tag):
        try:
            wire.append(json.loads(line))
        except Exception:
            pass
    ts = [w["t"] for w in wire]

    def kv(w):
        if mode == "rdma":
            return w["n"][0] if w.get("n") else 0
        return sum(s[4] for s in w["s"] if s[0].startswith("140"))

    fs = sorted(
        glob.glob("len-%s-32k-*.json" % tag),
        key=lambda f: int(re.search(r"-(\d+)\.json$", f).group(1)),
    )
    rows = [json.load(open(f))["rows"][0] for f in fs]
    ok = [r for r in rows if not r.get("err") and len(r["chunks"]) >= 8]
    rounds = {}
    for r in ok:
        rounds.setdefault(r["round"], []).append(r)
    med = statistics.median
    A = {
        "ttft": [],
        "tok2": [],
        "gap": [],
        "step": [],
        "first": [],
        "last": [],
        "mb": [],
        "kv_last": [],
    }
    print(
        "round n | TTFT first / p50 / last | tok2 p50 / last | gap12 p50 / max | TPOT | KV MB/req | last KV byte vs first / last tok1"
    )
    for rd, rs in sorted(rounds.items()):

        def base(r):
            return r["t_release_wall"] or r["t_send_wall"]

        ttft = sorted((r["t_send_wall"] + r["chunks"][0] - base(r)) * 1000 for r in rs)
        tok2 = sorted((r["t_send_wall"] + r["chunks"][1] - base(r)) * 1000 for r in rs)
        gap = [(r["chunks"][1] - r["chunks"][0]) * 1000 for r in rs]
        step = [
            med([(b - a) * 1000 for a, b in zip(r["chunks"][2:], r["chunks"][3:])])
            for r in rs
        ]
        t1s = sorted(r["t_send_wall"] + r["chunks"][0] for r in rs)
        t_lo = min(base(r) for r in rs)
        seg = wire[
            bisect.bisect_left(ts, t_lo - 0.01) : bisect.bisect_right(ts, t1s[-1] + 0.5)
        ]
        b0, prev, last = kv(seg[0]), kv(seg[0]), None
        for w in seg:
            if kv(w) - prev > (64 * 1024 if mode == "rdma" else 0):
                last = w["t"]
            prev = kv(w)
        mb = (kv(seg[-1]) - b0) / 1e6 / len(rs)
        A["ttft"] += ttft
        A["tok2"] += tok2
        A["gap"] += gap
        A["step"] += step
        A["first"].append(ttft[0])
        A["last"].append(ttft[-1])
        A["mb"].append(mb)
        if last:
            A["kv_last"].append((last - t1s[-1]) * 1000)
        print(
            "r%-3d %2d | %5.0f / %5.0f / %5.0f | %5.0f / %5.0f | %5.1f / %5.1f | %4.1f | %5.1f | %+7.1f / %+6.1f ms"
            % (
                rd,
                len(rs),
                ttft[0],
                med(ttft),
                ttft[-1],
                med(tok2),
                tok2[-1],
                med(gap),
                max(gap),
                med(step),
                mb,
                (last - t1s[0]) * 1000 if last else float("nan"),
                (last - t1s[-1]) * 1000 if last else float("nan"),
            )
        )
    city = [r.get("city_ok") for r in ok if r.get("city_ok") is not None]
    print(
        "ALL n=%d failed=%d rounds=%d | TTFT p50 %.0f mean %.0f (first-in-round mean %.0f, last-in-round mean %.0f) | "
        "tok2 p50 %.0f mean %.0f | gap12 p50 %.1f mean %.1f | TPOT p50 %.1f | KV MB/req %.1f | last KV byte vs last tok1 p50 %+.1f ms | city %d/%d | P reuse %s"
        % (
            len(ok),
            len(rows) - len(ok),
            len(rounds),
            med(A["ttft"]),
            statistics.mean(A["ttft"]),
            statistics.mean(A["first"]),
            statistics.mean(A["last"]),
            med(A["tok2"]),
            statistics.mean(A["tok2"]),
            med(A["gap"]),
            statistics.mean(A["gap"]),
            med(A["step"]),
            med(A["mb"]),
            med(A["kv_last"]) if A["kv_last"] else float("nan"),
            sum(city),
            len(city),
            sorted(set(r["aux"].get("prefill_total_reuse_len") for r in ok)),
        )
    )


if __name__ == "__main__":
    main()
