#!/usr/bin/env python3
"""EPLB placement plan for EP16 decode from a measured expert-load histogram.

Input: per-layer expert row counts (JSON {"layers": {layer_id: [count_e for e in 0..255]}}), produced by the
engine with DSV4_MOE_EXPERT_HIST=<file> (moe_layer counts topk ids per layer during decode).
Output:
  * the per-rank load imbalance of today's contiguous placement (rank r owns experts 16r..16r+15) vs a greedy
    balanced permutation (no replication: each rank still owns exactly 16 experts), per layer;
  * phy2log JSON (layer -> [256 logical ids in physical order]) that rtp_llm's loader accepts via
    --phy2log_path (LoadConfig.get_selected_experts), when --out is given.
--objective rows: min-max of per-rank row count. --objective active: min-max of the expected number of active
experts per rank, a_e = 1 - (1 - p_e)^N with N = routed rows per step (decode GEMM is weight-read bound, so its
cost tracks active experts: GEMM_us ~ 102.7 + 9.51*active - 0.215*rows, fitted on EP16 decode traces).
usage: eplb_plan.py hist.json [--out phy2log.json] [--ranks 16] [--objective rows|active] [--rows-per-step 1152]
"""
import argparse
import json


def greedy(loads, ranks, per_rank):
    order = sorted(range(len(loads)), key=lambda e: -loads[e])
    bins = [[] for _ in range(ranks)]
    tot = [0.0] * ranks
    for e in order:
        cands = [r for r in range(ranks) if len(bins[r]) < per_rank]
        r = min(cands, key=lambda r: tot[r])
        bins[r].append(e)
        tot[r] += loads[e]
    return bins, tot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("hist")
    ap.add_argument("--out", default="")
    ap.add_argument("--ranks", type=int, default=16)
    ap.add_argument("--layers", type=int, default=43)
    ap.add_argument("--objective", choices=("rows", "active"), default="rows")
    ap.add_argument("--rows-per-step", type=int, default=1152)
    a = ap.parse_args()
    h = json.load(open(a.hist))["layers"]
    phy2log = []
    s_before = s_after = s_mean = 0.0
    for l in range(a.layers):
        loads = h.get(str(l))
        if loads is None:
            phy2log.append(list(range(256)))
            continue
        E = len(loads)
        per = E // a.ranks
        contig = [sum(loads[r * per : (r + 1) * per]) for r in range(a.ranks)]
        if a.objective == "active":
            t = float(sum(loads))
            w = [1.0 - (1.0 - x / t) ** a.rows_per_step for x in loads]
            bins, _ = greedy(w, a.ranks, per)
            tot = [sum(loads[e] for e in b) for b in bins]
        else:
            bins, tot = greedy(loads, a.ranks, per)
        mean = sum(loads) / a.ranks
        s_before += max(contig)
        s_after += max(tot)
        s_mean += mean
        phy2log.append([e for b in bins for e in sorted(b)])
        if l in (0, 2, 10, 20, 30, 42):
            print(
                f"layer {l:2d}: contiguous max/mean {max(contig)/mean:.2f}  balanced max/mean {max(tot)/mean:.2f}"
            )
    print(
        f"all layers: max-rank rows / mean-rank rows  contiguous {s_before/s_mean:.3f}  balanced {s_after/s_mean:.3f}"
    )
    if a.out:
        json.dump(phy2log, open(a.out, "w"))
        print("wrote", a.out)


if __name__ == "__main__":
    main()
