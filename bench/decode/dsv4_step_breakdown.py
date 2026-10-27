#!/usr/bin/env python3
"""Per-module breakdown of DeepSeek-V4 MTP decode steps from an nsys sqlite export.

Targets the SM120 TP decode path (attention TP, routed experts through the
CUTLASS fused MoE, NCCL all-reduce). Device 0 only.

Step = one verify forward (43 layers, 86 ``mhc_pre_big_fuse_kernel``) + the draft
forwards (1 MTP layer each) + the kernels between them (lm_head, all-gather,
sampling). Inside a layer the attention region is [mhc_pre #1, mhc_pre #2) and
the FFN region is [mhc_pre #2, next layer). Kernels are assigned to modules by
name and grid shape; the grid-based linear split (``gridY`` of the FP8 GEMM)
matches DeepSeek-V4-Flash at TP4 and must be re-checked for other shapes
(``--dump-layer`` prints one layer's kernel sequence for that).

usage: dsv4_step_breakdown.py <file.sqlite> [--json out.json] [--dump-layer L]
"""
import argparse
import collections
import json
import re
import sqlite3
import statistics
import sys

PRE, HEAD = "mhc_pre_big_fuse_kernel", "mhc_head_fuse_kernel"
VERIFY_PRE = 86

ORDER = [
    "lin_rep_wq_a", "lin_rep_wkv", "lin_rep_compressor", "lin_rep_idx_wq_b",
    "lin_rep_idx_wproj", "lin_tp_wq_b", "lin_tp_wo_a", "lin_tp_wo_b",
    "attn_core_mla", "attn_core_indexer", "attn_glue_kv_compress", "attn_glue_small",
    "comm_ar_attn", "hc", "gate_gemm", "router", "shared_expert", "moe_gemm",
    "moe_glue", "comm_ar_moe", "ffn_glue_small", "lin_other",
]  # fmt: skip


def load(db):
    con = sqlite3.connect(db)
    tables = {r[0] for r in con.execute("select name from sqlite_master")}
    if "CUPTI_ACTIVITY_KIND_KERNEL" not in tables:
        sys.exit(f"{db}: no CUDA kernel records (capture saw no CUDA activity)")
    names = dict(con.execute("select id,value from StringIds"))
    q = (
        "select start,end,shortName,gridX,gridY,gridZ from CUPTI_ACTIVITY_KIND_KERNEL "
        "where deviceId=0 order by start"
    )
    return [
        (s, e, re.sub(r"<.*", "", names[n]), gx, gy, gz)
        for s, e, n, gx, gy, gz in con.execute(q)
    ]


def classify_attn(seg):
    out = []
    seen_sparse = False
    n64 = 0
    for r in seg:
        n, gx, gy, gz = r[2], r[3], r[4], r[5]
        if "sparse_mla" in n:
            seen_sparse = True
            c = "attn_core_mla"
        elif n == "device_kernel":
            if gy == 8:
                c = "lin_rep_wq_a"
            elif gy == 4:
                c = "lin_rep_wkv"
            elif gy == 64:
                n64 += 1
                c = "lin_tp_wq_b" if n64 == 1 else "lin_rep_idx_wq_b"
            elif gy == 32 and seen_sparse:
                c = "lin_tp_wo_b"
            else:
                c = "lin_other"
        elif n == "sm120_fp8_fp4_gemm_1d1d_impl":
            c = "lin_tp_wo_a"
        elif n in ("Kernel2", "splitKreduce_kernel"):
            big = gz >= 6 or (n == "splitKreduce_kernel" and gx >= 8)
            c = "lin_rep_compressor" if big else "lin_rep_idx_wproj"
        elif re.search(r"paged_score|main_kernel|indexer_q", n):
            c = "attn_core_indexer"
        elif re.search(r"AllReduce", n):
            c = "comm_ar_attn"
        elif re.search(r"mhc|hc_prenorm", n):
            c = "hc"
        elif re.search(r"compress|save_partial|quantize_and_insert", n):
            c = "attn_glue_kv_compress"
        else:
            c = "attn_glue_small"
        out.append((c, r))
    return out


def classify_ffn(seg):
    out = []
    for r in seg:
        n, gy = r[2], r[4]
        if n == "device_kernel" and gy == 110:
            c = "moe_gemm"
        elif n == "device_kernel" and gy == 32:
            c = "shared_expert"
        elif n in ("Kernel2", "splitKreduce_kernel"):
            c = "gate_gemm"
        elif "gate_sqrtsoftplus" in n:
            c = "router"
        elif re.search(
            r"expandInputRows|finalizeMoe|doActivation|computeStrides|"
            r"fusedBuildExpertMaps|quantize_with_block_size|_add_cast",
            n,
        ):
            c = "moe_glue"
        elif re.search(r"AllReduce", n):
            c = "comm_ar_moe"
        elif re.search(r"mhc|hc_prenorm", n):
            c = "hc"
        else:
            c = "ffn_glue_small"
        out.append((c, r))
    return out


def layer_regions(seg):
    p = [i for i, r in enumerate(seg) if r[2] == PRE]
    regions = []
    for L in range(len(p) // 2):
        a0, a1 = p[2 * L], p[2 * L + 1]
        f1 = p[2 * L + 2] if 2 * L + 2 < len(p) else len(seg) - 1
        regions.append((a0, a1, f1))
    return regions


def forward_breakdown(seg):
    agg = collections.defaultdict(float)
    for a0, a1, f1 in layer_regions(seg):
        for c, r in classify_attn(seg[a0:a1]) + classify_ffn(seg[a1:f1]):
            agg[c] += (r[1] - r[0]) / 1e3
    span = (seg[-1][1] - seg[0][0]) / 1e3
    busy = sum((r[1] - r[0]) / 1e3 for r in seg)
    return agg, span, busy


def glue_category(r):
    n = r[2]
    if "AllGather" in n:
        return "allgather"
    if "AllReduce" in n or "Broadcast" in n:
        return "allreduce/bcast"
    if n == "Kernel2" and r[4] > 100:
        return "lm_head"
    if n in ("Kernel2", "splitKreduce_kernel"):
        return "bf16_gemm"
    if re.search(
        r"SoftMax|TopP|AirTopP|Temperature|distribution|reduce_kernel|rejection|sampl",
        n,
        re.I,
    ):
        return "sampling"
    if n == "device_kernel":
        return "fp8_gemm"
    return "small/meta"


def split_forwards(rows):
    """[(kind, forward_rows, rows_before_forward)], kind V(erify) or D(raft)."""
    heads = [i for i, r in enumerate(rows) if r[2] == HEAD]
    seq, prev = [], 0
    for i in heads:
        seg = rows[prev : i + 1]
        npre = sum(1 for r in seg if r[2] == PRE)
        if npre == VERIFY_PRE or 1 <= npre <= 4:
            start = next(k for k in range(prev, i + 1) if rows[k][2] == PRE)
            kind = "V" if npre == VERIFY_PRE else "D"
            seq.append((kind, rows[start : i + 1], rows[prev:start]))
        prev = i + 1
    return seq


def steps_of(seq):
    steps, k = [], 0
    while k < len(seq):
        if seq[k][0] != "V":
            k += 1
            continue
        j = k + 1
        while j < len(seq) and seq[j][0] == "D":
            j += 1
        if j < len(seq) and seq[j][0] == "V":
            steps.append((seq[k], seq[k + 1 : j], seq[j]))
        k = j
    return steps


def analyze(rows):
    steps = steps_of(split_forwards(rows))
    if not steps:
        raise SystemExit("no complete verify->verify step found (MTP trace?)")
    res = collections.defaultdict(list)
    cats = collections.defaultdict(list)
    for v, ds, vnext in steps:
        agg, vspan, vbusy = forward_breakdown(v[1])
        res["step"].append((vnext[1][0][0] - v[1][0][0]) / 1e3)
        res["verify_span"].append(vspan)
        res["verify_busy"].append(vbusy)
        for c, d in agg.items():
            cats[c].append(d)
        res["n_draft"].append(len(ds))
        res["draft_span"].append(sum((d[1][-1][1] - d[1][0][0]) / 1e3 for d in ds))
        res["draft_busy"].append(
            sum(sum((r[1] - r[0]) / 1e3 for r in d[1]) for d in ds)
        )
        dagg = collections.defaultdict(float)
        for d in ds:
            for c, x in forward_breakdown(d[1])[0].items():
                dagg[c] += x
        for c, x in dagg.items():
            cats["draft:" + c].append(x)
        glue_rows = [r for d in ds for r in d[2]] + list(vnext[2])
        res["glue_busy"].append(sum((r[1] - r[0]) / 1e3 for r in glue_rows))
        gcat = collections.defaultdict(float)
        for r in glue_rows:
            gcat[glue_category(r)] += (r[1] - r[0]) / 1e3
        for c, x in gcat.items():
            cats["glue:" + c].append(x)
    med = statistics.median
    summary = {k: med(v) for k, v in cats.items()}
    summary.update({k: med(v) for k, v in res.items()})
    summary["idle"] = (
        summary["step"]
        - summary["verify_busy"]
        - summary["draft_busy"]
        - summary["glue_busy"]
    )
    summary["n_steps"] = len(steps)
    return summary


def report(summary, n_layers=43):
    p = print
    p(
        f"steps={summary['n_steps']} drafts/step={summary['n_draft']:.0f}  "
        f"step {summary['step']:.0f}us | verify span {summary['verify_span']:.0f} "
        f"busy {summary['verify_busy']:.0f} | draft busy {summary['draft_busy']:.0f} | "
        f"glue busy {summary['glue_busy']:.0f} | idle {summary['idle']:.0f}"
    )
    p("\n[verify forward, busy us per step]")
    tot = 0.0
    for c in ORDER:
        if c in summary:
            tot += summary[c]
            p(f"  {c:24s} {summary[c]:8.0f}   ({summary[c] / n_layers:6.1f}/layer)")
    p(f"  {'TOTAL':24s} {tot:8.0f}")
    for prefix, title in (("draft:", "draft forwards"), ("glue:", "between forwards")):
        p(f"\n[{title}, busy us per step]")
        for c in sorted(k for k in summary if k.startswith(prefix)):
            p(f"  {c:32s} {summary[c]:8.0f}")


def dump_layer(rows, L):
    steps = steps_of(split_forwards(rows))
    seg = steps[len(steps) // 2][0][1]
    a0, a1, f1 = layer_regions(seg)[L]
    prev_end = None
    for c, r in classify_attn(seg[a0:a1]) + classify_ffn(seg[a1:f1]):
        gap = (r[0] - prev_end) / 1e3 if prev_end else 0.0
        prev_end = r[1]
        print(
            f"{(r[1] - r[0]) / 1e3:7.1f}us gap{gap:5.1f} g({r[3]},{r[4]},{r[5]}) "
            f"{c:22s} {r[2][:60]}"
        )


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("sqlite")
    ap.add_argument("--json", default="", help="write the summary here")
    ap.add_argument("--dump-layer", type=int, default=-1)
    args = ap.parse_args()
    rows = load(args.sqlite)
    if not rows:
        sys.exit("no CUDA kernels in this export")
    if args.dump_layer >= 0:
        dump_layer(rows, args.dump_layer)
        return
    summary = analyze(rows)
    report(summary)
    if args.json:
        json.dump(summary, open(args.json, "w"), indent=1)


if __name__ == "__main__":
    main()
