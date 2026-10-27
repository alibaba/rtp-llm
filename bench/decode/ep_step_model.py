#!/usr/bin/env python3
"""Estimate TP4 vs wide-EP (EP16) decode step time from measured components.

Inputs are the outputs of the other tools in this directory:
  --breakdown-lo / --breakdown-hi  dsv4_step_breakdown.py --json of the TP4 server at two
                                   concurrencies (default cc8 / cc16) with the same MTP gamma
  --moe                            moe_tp_vs_ep_bench.py output (JSON lines)
  --coll                           nccl_decode_collectives.py output for the TP group (JSON lines)

Per 4 GPUs, a TP4 instance with c requests is compared with one 4-GPU DP group of EP16
serving c requests (16 GPUs = 4 groups):
  B  DP4 x TP4 attention + EP16 MoE: attention all-reduce becomes reduce-scatter (each rank
     dispatches T/4 tokens), all-gather after combine; shared expert runs in full on T/4.
  A  DP16 attention + EP16 MoE: each rank runs c/4 requests with full attention weights
     (``--tplin-full-us``), no intra-group all-reduce, indexer work /4.
Token-dependent TP4 components are interpolated linearly in verify tokens between the two
breakdowns; routed MoE comes from the MoE bench (``--routing uniform``) or is scaled from
the measured trace (``--routing trace``: distinct experts estimated from the measured MoE
time with the fitted TP-slice cost). Cross-node dispatch+combine per layer is
``2 * fixed + bytes / node_bw`` with fixed per-direction costs from ``--fixed-us``
(one-way RDMA latency + proxy/flag overhead; measure with rdma_ep_probe.sh).
"""
import argparse
import json
import random

L, H = 43, 4096


def jl(path):
    return [json.loads(l) for l in open(path) if l.startswith("{")]


def interp(tbl, x):
    ks = sorted(tbl)
    if x <= ks[0]:
        return tbl[ks[0]]
    for a, b in zip(ks, ks[1:]):
        if a <= x <= b:
            return tbl[a] + (tbl[b] - tbl[a]) * (x - a) / (b - a)
    a, b = ks[-2], ks[-1]
    return tbl[b] + (tbl[b] - tbl[a]) * (x - b) / (b - a)


def fit_line(pts):
    """Least squares; refit once without points off by > 2.5x the median residual."""

    def ls(p):
        n = len(p)
        mx = sum(x for x, _ in p) / n
        my = sum(y for _, y in p) / n
        sxx = sum((x - mx) ** 2 for x, _ in p)
        k = sum((x - mx) * (y - my) for x, y in p) / sxx
        return my - k * mx, k

    c, k = ls(pts)
    res = [abs(y - (c + k * x)) for x, y in pts]
    med = sorted(res)[len(res) // 2]
    keep = [p for p, r in zip(pts, res) if r <= 2.5 * med]
    return ls(keep) if 2 <= len(keep) < len(pts) else (c, k)


class Model:
    COMP = ["rep_lin", "tp_lin", "mla", "indexer", "attn_glue", "hc", "gate", "shared",
            "moe", "moe_glue", "ar", "a2a", "draft", "glue", "idle"]  # fmt: skip
    TOKDEP = ["rep_lin", "tp_lin", "mla", "indexer", "attn_glue", "hc", "gate", "shared",
              "moe_glue", "ar"]  # fmt: skip

    def __init__(self, a):
        self.a = a
        self.lo, self.hi = self._comp(a.breakdown_lo), self._comp(a.breakdown_hi)
        moe = jl(a.moe)
        tp = [
            (d["active_experts"], d["us_per_layer"]) for d in moe if d["mode"] == "tp"
        ]
        ep = [
            (d["active_experts"], d["us_per_layer"]) for d in moe if d["mode"] == "ep"
        ]
        self.tp_c, self.tp_k = fit_line(tp)
        self.ep_c, self.ep_k = fit_line(ep)
        self.coll = {d["tokens"]: d for d in jl(a.coll)}
        self.experts, self.topk = a.experts, a.topk
        self._mc = {}
        self.rng = random.Random(1)

    @staticmethod
    def _comp(path):
        d = json.load(open(path))
        g = lambda k: d.get(k, 0.0)
        nd = max(g("n_draft"), 1)
        return dict(
            rep_lin=g("lin_rep_wq_a") + g("lin_rep_wkv") + g("lin_rep_compressor")
            + g("lin_rep_idx_wq_b") + g("lin_rep_idx_wproj"),
            tp_lin=g("lin_tp_wq_b") + g("lin_tp_wo_a") + g("lin_tp_wo_b"),
            mla=g("attn_core_mla"), indexer=g("attn_core_indexer"),
            attn_glue=g("attn_glue_kv_compress") + g("attn_glue_small"),
            hc=g("hc"), gate=g("gate_gemm") + g("router"), shared=g("shared_expert"),
            moe=g("moe_gemm"), moe_glue=g("moe_glue") + g("ffn_glue_small"),
            ar=g("comm_ar_attn") + g("comm_ar_moe"),
            draft=g("draft_busy"), glue=g("glue_busy"), idle=g("idle"),
            n_draft=g("n_draft"),
            d_tplin=(g("draft:lin_tp_wq_b") + g("draft:lin_tp_wo_a") + g("draft:lin_tp_wo_b")) / nd,
            d_ar=(g("draft:comm_ar_attn") + g("draft:comm_ar_moe")) / nd,
        )  # fmt: skip

    def coll_us(self, tok, key):
        return interp({t: d[key] for t, d in self.coll.items() if key in d}, tok)

    def d_uniform(self, T):
        return self.experts * (1 - (1 - self.topk / self.experts) ** T)

    def ep_max_active(self, Dg, ranks, trials=3000):
        Dg = int(round(min(Dg, self.experts)))
        key = (Dg, ranks)
        if key not in self._mc:
            per = self.experts // ranks
            s = 0
            for _ in range(trials):
                cnt = [0] * ranks
                for e in self.rng.sample(range(self.experts), Dg):
                    cnt[e // per] += 1
                s += max(cnt)
            self._mc[key] = s / trials
        return self._mc[key]

    def T(self, c):
        return c * (self.a.gamma + 1)

    def d_tp(self, c, routing):
        if routing == "uniform":
            return self.d_uniform(self.T(c))
        D0 = (self.lo["moe"] / L - self.tp_c) / self.tp_k
        T0 = self.T(self.a.cc_lo)
        return min(self.experts, D0 * self.d_uniform(self.T(c)) / self.d_uniform(T0))

    def tp4(self, c, routing):
        a, lo, hi = self.a, self.lo, self.hi
        t0, t1 = self.T(a.cc_lo), self.T(a.cc_hi)
        f = (self.T(c) - t0) / (t1 - t0)
        fc = (c - a.cc_lo) / (a.cc_hi - a.cc_lo)
        o = {k: lo[k] + (hi[k] - lo[k]) * f for k in self.TOKDEP}
        o["moe"] = (self.tp_c + self.tp_k * self.d_tp(c, routing)) * L
        for k in ("draft", "glue", "idle"):
            o[k] = lo[k] + (hi[k] - lo[k]) * fc
        o["a2a"] = 0.0
        return o

    def a2a_layer(self, tok_per_rank, fixed_us):
        dest = 16 * (1 - (1 - 1 / 16) ** self.topk)
        remote = tok_per_rank * dest / 2 * self.a.imbalance
        node_bw = self.a.node_bw_gbps * 1e9 / 8 / 1e6
        return 2 * fixed_us + 8 * remote * H * (1.03 + 2) / node_bw

    def ep16(self, c, which, fixed_us, routing):
        a = self.a
        T = self.T(c)
        s = self.tp4(c, routing)
        Dg = self.experts * (1 - (1 - self.d_tp(c, routing) / self.experts) ** 4)
        s["moe"] = (self.ep_c + self.ep_k * self.ep_max_active(Dg, 16)) * L
        s["a2a"] = self.a2a_layer(T / 4, fixed_us) * L
        rs_ag = self.coll_us(T, "reduce_scatter_us") + self.coll_us(T, "all_gather_us")
        s["shared"] = a.shared_full_us * L
        if which == "B":
            s["ar"] *= rs_ag / (2 * self.coll_us(T, "allreduce_us"))
        else:
            s["tp_lin"] = a.tplin_full_us * L
            s["indexer"] /= 4
            s["ar"] = 0.0
            s["glue"] += (a.gamma + 1) * self.coll_us(c, "all_gather_us")
            s["draft"] += a.gamma * (
                a.tplin_full_us - self.lo["d_tplin"] - self.lo["d_ar"]
            )
        return s

    def total(self, s):
        return sum(s[k] for k in self.COMP)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--breakdown-lo", required=True)
    ap.add_argument("--breakdown-hi", required=True)
    ap.add_argument("--cc-lo", type=int, default=8)
    ap.add_argument("--cc-hi", type=int, default=16)
    ap.add_argument("--moe", required=True)
    ap.add_argument("--coll", required=True)
    ap.add_argument("--gamma", type=int, default=2, help="MTP draft tokens per step")
    ap.add_argument("--accept", type=float, default=2.3)
    ap.add_argument("--cc", default="8,16")
    ap.add_argument("--fixed-us", default="40,65,100")
    ap.add_argument("--node-bw-gbps", type=float, default=400.0, help="RDMA per node")
    ap.add_argument("--imbalance", type=float, default=1.15)
    ap.add_argument("--experts", type=int, default=256)
    ap.add_argument("--topk", type=int, default=6)
    ap.add_argument("--shared-full-us", type=float, default=40.0)
    ap.add_argument("--tplin-full-us", type=float, default=96.0)
    ap.add_argument("--detail", action="store_true", help="per-component table")
    a = ap.parse_args()
    m = Model(a)
    fixed = [float(v) for v in a.fixed_us.split(",")]
    print(f"MoE fit: TP slice {m.tp_c:.1f} + {m.tp_k:.3f}*experts us/layer, "
          f"EP rank {m.ep_c:.1f} + {m.ep_k:.2f}*experts")  # fmt: skip
    for routing in ("trace", "uniform"):
        print(f"\n== routing={routing}, MTP{a.gamma}, accept {a.accept}")
        for c in [int(v) for v in a.cc.split(",")]:
            b = m.tp4(c, routing)
            bt = m.total(b)
            line = (
                f"c={c:3d}: TP4 step {bt / 1e3:5.1f}ms TPOT {bt / a.accept / 1e3:5.1f}"
            )
            for which in ("B", "A"):
                parts = []
                for f in fixed:
                    st = m.total(m.ep16(c, which, f, routing))
                    parts.append(
                        f"{f:.0f}us {st / a.accept / 1e3:4.1f}({100 * (st / bt - 1):+3.0f}%)"
                    )
                z = m.ep16(c, which, 0.0, routing)
                be = (bt - (m.total(z) - z["a2a"])) / L
                line += (
                    f" | {which}: " + " ".join(parts) + f" breakeven {be:4.0f}us/layer"
                )
            print(line)
            if a.detail:
                cols = [("TP4", b)] + [
                    (f"{w}@{fixed[0]:.0f}", m.ep16(c, w, fixed[0], routing))
                    for w in "BA"
                ]
                for k in Model.COMP:
                    print(
                        f"    {k:10s} "
                        + " | ".join(f"{s[k] / 1e3:6.2f}" for _, s in cols)
                    )
                print(
                    f"    {'TOTAL':10s} "
                    + " | ".join(f"{m.total(s) / 1e3:6.2f}" for _, s in cols)
                )


if __name__ == "__main__":
    main()
