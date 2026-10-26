#!/usr/bin/env python3
"""Round-level breakdown for high-concurrency shared-prefix runs.

  an_prefix_cN.py <tag> <rdma|tcp> [per_request_MB]

For each round of N simultaneous requests (pd_prefix_c4_client.py output):
  * TTFT / time-to-second-token distribution (p50, mean, max, first, last)
  * gap12 and steady inter-token gap
  * KV progress: at the k-th first token of the round, how many requests' worth
    of KV had left P (P NIC tx for RDMA, cache-store socket bytes_sent for TCP),
    and when the round's last KV byte left relative to the round's last tok1
  * correctness: fraction of answers that name the expected suffix city
"""
import bisect
import glob
import json
import re
import statistics
import sys


def main():
    tag, mode = sys.argv[1], sys.argv[2]
    per_req = (
        float(sys.argv[3]) * 1e6
        if len(sys.argv) > 3
        else (147.1e6 if mode == "rdma" else 137.9e6)
    )
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
    ok_rows = [r for r in rows if not r.get("err") and len(r["chunks"]) >= 8]
    rounds = {}
    for r in ok_rows:
        rounds.setdefault(r["round"], []).append(r)
    med = statistics.median
    all_ttft, all_tok2, all_gap, all_step = [], [], [], []
    print(
        "round  n  TTFT p50 / mean / first / last | tok2 p50 / mean / max | gap12 p50 / mean / max | step | KV sent at k-th tok1 (req-equiv, k=1,N/4,N/2,3N/4,N) | last KV byte vs last tok1 | wireMB/req"
    )
    for rd, rs in sorted(rounds.items()):
        t0 = min(r["t_send_wall"] for r in rs)
        t1s = sorted(r["t_send_wall"] + r["chunks"][0] for r in rs)
        t2_last = max(r["t_send_wall"] + r["chunks"][1] for r in rs)
        seg = wire[
            bisect.bisect_left(ts, t0 - 0.01) : bisect.bisect_right(ts, t2_last + 0.2)
        ]
        base = kv(seg[0])
        at = []
        for t1 in t1s:
            w = wire[bisect.bisect_right(ts, t1) - 1]
            at.append((kv(w) - base) / per_req)
        last, prev = None, base
        for w in seg:
            if kv(w) - prev > (64 * 1024 if mode == "rdma" else 0):
                last = w["t"]
            prev = kv(w)
        total = (kv(seg[-1]) - base) / 1e6
        n = len(rs)
        ks = sorted(set([0, n // 4 - 1, n // 2 - 1, 3 * n // 4 - 1, n - 1]))
        ttft = [r["chunks"][0] * 1000 for r in rs]
        tok2 = [r["chunks"][1] * 1000 for r in rs]
        gap = [(r["chunks"][1] - r["chunks"][0]) * 1000 for r in rs]
        step = [
            med([(b - a) * 1000 for a, b in zip(r["chunks"][2:], r["chunks"][3:])])
            for r in rs
        ]
        all_ttft += ttft
        all_tok2 += tok2
        all_gap += gap
        all_step += step
        print(
            "r%-4d %2d  %5.0f / %5.0f / %5.0f / %5.0f | %5.0f / %5.0f / %5.0f | %5.1f / %5.1f / %5.1f | %4.1f | %s | %+6.1f ms | %.1f"
            % (
                rd,
                n,
                med(ttft),
                statistics.mean(ttft),
                min(ttft),
                max(ttft),
                med(tok2),
                statistics.mean(tok2),
                max(tok2),
                med(gap),
                statistics.mean(gap),
                max(gap),
                med(step),
                " ".join("%.2f" % at[k] for k in ks),
                ((last - t1s[-1]) * 1000) if last else float("nan"),
                total / n,
            )
        )
    city = [r.get("city_ok") for r in ok_rows if r.get("city_ok") is not None]
    reuse = sorted(set(r["aux"].get("prefill_total_reuse_len") for r in ok_rows))
    print(
        "ALL  n=%d (failed %d)  TTFT p50 %.0f mean %.0f max %.0f | tok2 p50 %.0f mean %.0f max %.0f | gap12 p50 %.1f mean %.1f max %.1f | step p50 %.1f | city ok %d/%d | P reuse %s"
        % (
            len(ok_rows),
            len(rows) - len(ok_rows),
            med(all_ttft),
            statistics.mean(all_ttft),
            max(all_ttft),
            med(all_tok2),
            statistics.mean(all_tok2),
            max(all_tok2),
            med(all_gap),
            statistics.mean(all_gap),
            max(all_gap),
            med(all_step),
            sum(city),
            len(city),
            reuse,
        )
    )


if __name__ == "__main__":
    main()
