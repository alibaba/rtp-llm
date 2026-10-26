#!/usr/bin/env python3
"""Per-length PD latency breakdown from run_len_sweep.sh output.

  an_len_sweep.py <tag> <rdma|tcp> [lengths...]

Reads wire-<tag>.jsonl (pd_wire_sampler) and len-<tag>-<L>-<i>.json
(pd_ttft_gap_client), all on the prefill host clock. Per request, relative to
tok1 (first token at the client):

  kv_last    last KV byte leaves P (RDMA: eRDMA hw_tx; TCP: cache-store sockets'
             bytes_sent, local port 14xxx)
  load_done  D's LOAD response reaches P on the gRPC stream (remote :21001,
             first bytes_received bump after the request was sent, payload < 110B)
  gen_sent   P sends GENERATE (first token) to D (bytes_sent bump >= 60B near tok1)
  tok2_at_p  D's second-token response reaches P (bytes_received bump >= 110B)

Derived:
  kv_exposed = max(0, load_done): P writes tok1 to the client first
               (pollLocalOutput), then blocks on D's LOAD reply
               (remoteLoadCacheEnd), then sends GENERATE. So KV is exposed only
               by how much the LOAD reply lands after tok1; resolution is one
               sampler period (~2.2ms)
  kv_slack   = -load_done: how long before tok1 D had all KV
  d_side     = tok2_at_p - gen_sent  (D: first decode step + handoff)
  step       = steady inter-token gap (median of chunk gaps from the 3rd on)
  handoff    = d_side - step
"""
import bisect
import glob
import json
import re
import statistics
import sys

KV_PORTS_PREFIX = "140"


def load_wire(p):
    out = []
    for line in open(p):
        try:
            out.append(json.loads(line))
        except Exception:
            pass
    return out


def p50(xs):
    xs = [x for x in xs if x is not None]
    return statistics.median(xs) if xs else float("nan")


def mean(xs):
    xs = [x for x in xs if x is not None]
    return statistics.mean(xs) if xs else float("nan")


def main():
    tag, mode = sys.argv[1], sys.argv[2]
    lengths = sys.argv[3:] or ["1k", "4k", "8k", "16k", "32k"]
    wire = load_wire("wire-%s.jsonl" % tag)
    ts = [w["t"] for w in wire]

    def kv_counter(w):
        if mode == "rdma":
            return w["n"][0] if w.get("n") else 0
        return sum(s[4] for s in w["s"] if s[0].startswith(KV_PORTS_PREFIX))

    def grpc(w):
        g = [s for s in w["s"] if s[1] == "21001"]
        return (g[0][4], g[0][3]) if g else None  # (bytes_sent, bytes_received)

    table = []
    per_req = []
    for L in lengths:
        rows = []
        files = sorted(
            glob.glob("len-%s-%s-*.json" % (tag, L)),
            key=lambda f: int(re.search(r"-(\d+)\.json$", f).group(1)),
        )
        for f in files:
            r = json.load(open(f))["rows"][0]
            c = r["chunks"]
            if r.get("err") or len(c) < 8:
                continue
            t0 = r["t_send_wall"]
            t1 = t0 + c[0]
            t2 = t0 + c[1]
            gaps = [(b - a) * 1000 for a, b in zip(c, c[1:])]
            step = statistics.median(gaps[2:])
            seg = wire[
                max(0, bisect.bisect_left(ts, t0 - 0.02)) : bisect.bisect_right(
                    ts, t2 + 0.1
                )
            ]
            if len(seg) < 3:
                continue
            kv0 = kv_counter(seg[0])
            kv_last = None
            kv_total = 0
            prev = kv0
            for w in seg:
                v = kv_counter(w)
                if w["t"] <= t2 and v - prev > (64 * 1024 if mode == "rdma" else 0):
                    kv_last = (w["t"] - t1) * 1000
                if w["t"] <= t2:
                    kv_total = v - kv0
                prev = v
            ev = []
            pg = None
            for w in seg:
                g = grpc(w)
                if g and pg and g != pg:
                    ev.append(((w["t"] - t1) * 1000, g[0] - pg[0], g[1] - pg[1]))
                if g:
                    pg = g
            req_ms = c[0] * 1000
            gen_sent = next((e[0] for e in ev if e[1] >= 60 and -6 < e[0] < 6), None)
            # D's LOAD reply: the first small inbound message after the ALLOCATE
            # exchange (send .. send+~50ms, ACKs included) and no later than GENERATE.
            small_in = [
                e
                for e in ev
                if 0 < e[2] < 110
                and e[0] > -req_ms + 60
                and (gen_sent is None or e[0] <= gen_sent + 0.5)
            ]
            load_done = small_in[0][0] if small_in else None
            tok2_at_p = next((e[0] for e in ev if e[2] >= 110 and e[0] > 3), None)
            kv_exposed = max(0.0, load_done) if load_done is not None else None
            d_side = (
                (tok2_at_p - gen_sent)
                if (tok2_at_p is not None and gen_sent is not None)
                else None
            )
            rec = {
                "len": L,
                "file": f,
                "isl": r["prompt_tokens"],
                "ttft": c[0] * 1000,
                "gap12": gaps[0],
                "step": step,
                "tpot_rest": statistics.mean(gaps[2:]),
                "kv_mb": kv_total / 1e6,
                "kv_last": kv_last,
                "load_done": load_done,
                "gen_sent": gen_sent,
                "tok2_at_p": tok2_at_p,
                "kv_exposed": kv_exposed,
                "d_side": d_side,
                "handoff": (d_side - step) if d_side is not None else None,
                "kv_slack": -load_done if load_done is not None else None,
            }
            rows.append(rec)
            per_req.append(rec)
        if not rows:
            continue
        g = lambda k: [x[k] for x in rows]
        table.append(
            (
                L,
                len(rows),
                p50(g("isl")),
                p50(g("ttft")),
                p50([a + b for a, b in zip(g("ttft"), g("gap12"))]),
                p50(g("gap12")),
                mean(g("gap12")),
                p50(g("step")),
                p50(g("kv_mb")),
                p50(g("kv_last")),
                p50(g("load_done")),
                p50(g("kv_exposed")),
                (
                    max(x for x in g("kv_exposed") if x is not None)
                    if any(x is not None for x in g("kv_exposed"))
                    else float("nan")
                ),
                p50(g("d_side")),
                p50(g("handoff")),
                p50(g("kv_slack")),
            )
        )
    hdr = (
        "len",
        "n",
        "ISL",
        "TTFT",
        "tok2",
        "gap12",
        "gap12mean",
        "step",
        "kvMB",
        "kvLast",
        "loadDone",
        "kvExp",
        "kvExpMax",
        "Dside",
        "handoff",
        "kvSlack",
    )
    print(("%-4s %2s %6s %7s %7s %6s %9s %5s %6s %7s %8s %5s %8s %6s %7s %7s") % hdr)
    for t in table:
        print(
            "%-4s %2d %6d %7.0f %7.0f %6.1f %9.1f %5.1f %6.1f %7.1f %8.1f %5.1f %8.1f %6.1f %7.1f %7.1f"
            % t
        )
    json.dump(
        {"tag": tag, "mode": mode, "requests": per_req},
        open("len-%s.summary.json" % tag, "w"),
    )


if __name__ == "__main__":
    main()
