#!/usr/bin/env python3
"""DSV4 MTP decode-step breakdown from a torch profiler (kineto) trace, rank 0.

Same layer split and categories for EP16 (rail MoE) and TP4 (fused CUTLASS MoE + NCCL all-reduce):
  verify forward = 43 layers (86 mhc_pre), draft forwards = 1 MTP layer each, glue = between forwards.
  Inside a layer: attention region [mhc_pre#1, mhc_pre#2), FFN region [mhc_pre#2, next mhc_pre#1).
Each full engine step (engine.normal.execute window) is analysed separately; medians are reported.

usage: dsv4_trace_breakdown.py trace.json [--json out.json] [--label NAME]
"""
import argparse
import collections
import json
import re
import statistics

PRE, HEAD = "mhc_pre_big_fuse_kernel", "mhc_head_fuse"


def short(n):
    return re.sub(r"<.*", "", n.split("(")[0]).replace("void ", "").split("::")[-1]


def cat_attn(n, l):
    if "sparse_mla" in l:
        return "attn.core_mla"
    if (
        "paged_score" in l
        or "indexer_q" in l
        or "main_kernel" in n
        or "canonical_topk" in l
    ):
        return "attn.core_indexer"
    if (
        "nccl" in l
        or "ipc_rsag_push" in l
        or "pcie_ipc" in l
        or "flat_one_pack" in l
        or "topology_one_pack" in l
        or "flat_cyclic" in l
        or "topology_cyclic" in l
    ):
        return "comm.attn_allreduce"
    if "flat_push_kernel" in l or "recursive_doubling" in l:
        return "comm.attn_allgather"
    if "mhc" in l or "hc_prenorm" in l:
        return "hc"
    if (
        "compress" in l
        or "save_partial" in l
        or "quantize_and_insert" in l
        or "pack_slots_to_paged" in l
    ):
        return "attn.kv_compress_insert"
    if (
        "device_kernel" in n
        or "cutlass_3x_gemm_fp8" in l
        or "sm120_fp8_fp4_gemm_1d1d" in l
        or "wmma" in l
        or "s16816gemm" in l
        or "splitkreduce" in l
        or "kernel2" == short(n).lower()
    ):
        return "attn.linear"
    return "attn.glue_small"


def cat_ffn(n, l):
    if l.startswith("k_") or "rr2" in l:
        return "comm.moe_a2a_rail"
    if "nccl" in l or "ipc_rsag_push" in l or "pcie_ipc" in l:
        return (
            "comm.moe_allreduce"
            if ("allreduce" in l or "ipc_rsag" in l or "pcie_ipc" in l)
            else "comm.moe_nccl_other"
        )
    if (
        "flat_one_pack" in l
        or "flat_cyclic" in l
        or "topology_one_pack" in l
        or "topology_cyclic" in l
    ):
        return "comm.attn_allreduce"
    if "flat_push_kernel" in l or "recursive_doubling" in l:
        return "comm.attn_allgather"
    if "sm120_fp8_fp4_gemm_1d1d_impl<0u, 4096u" in n:
        return "moe.routed_gemm"
    if "sm120_fp8_fp4_gemm_1d1d_impl<0u, 1024u" in n:
        return "moe.shared_expert"
    if "sqrtsoftplus" in l:
        return "moe.router"
    if (
        "silu_mul_masked" in l
        or "ep_scatter" in l
        or "ep_gather" in l
        or "recompute_topk" in l
        or "quant_pack_dispatch" in l
        or "fp32_peer_sum" in l
        or "_add_cast" in l
        or "expandinputrows" in l
        or "finalizemoe" in l
        or "doactivation" in l
        or "computestrides" in l
        or "fusedbuildexpertmaps" in l
        or "quantize_with_block_size" in l
    ):
        return "moe.glue"
    if "mhc" in l or "hc_prenorm" in l:
        return "hc"
    return None


def classify_ffn_layer(seg):
    """TP4 MoE GEMMs are CUTLASS device_kernel (grouped) - split by grid like dsv4_step_breakdown (gridY 110 =
    routed, 32 = shared); kineto has the grid in args."""
    out = []
    for k in seg:
        n = k["name"]
        l = n.lower()
        c = cat_ffn(n, l)
        if c is None:
            g = k.get("args", {}).get("grid", [0, 0, 0])
            sn = short(n)
            if "GemmUniversal" in n and "GroupProblemShape" in n:
                c = "moe.routed_gemm"
            elif "cutlass_3x_gemm_fp8_blockwise" in n:
                c = "moe.shared_expert"
            elif sn == "device_kernel" and g[1] == 110:
                c = "moe.routed_gemm"
            elif sn == "device_kernel" and g[1] == 32:
                c = "moe.shared_expert"
            elif sn == "device_kernel":
                c = "moe.routed_gemm"
            elif (
                sn in ("Kernel2", "splitKreduce_kernel")
                or "wmma" in l
                or "s16816gemm" in l
            ):
                c = "moe.gate_gemm"
            else:
                c = "ffn.glue_small"
        out.append((c, k))
    return out


def glue_cat(k):
    n = k["name"]
    sn = short(n)
    l = n.lower()
    if "AllGather" in n or "flat_push_kernel" in n or "recursive_doubling" in n:
        return "glue.allgather"
    if "AllReduce" in n or "Broadcast" in n:
        return "glue.allreduce_bcast"
    g = k.get("args", {}).get("grid", [0, 0, 0])
    if sn == "Kernel2" and g[1] > 100:
        return "glue.lm_head"
    if re.search(
        r"SoftMax|TopP|AirTopP|Temperature|distribution|reduce_kernel|rejection|sampl|topk",
        n,
        re.I,
    ):
        return "glue.sampling"
    if sn in ("Kernel2", "splitKreduce_kernel", "device_kernel") or "gemm" in l:
        return "glue.gemm_other"
    return "glue.small_meta"


def forward_cats(seg):
    agg = collections.defaultdict(float)
    pres = [i for i, k in enumerate(seg) if PRE in k["name"]]
    for L in range(len(pres) // 2):
        a0, a1 = pres[2 * L], pres[2 * L + 1]
        f1 = pres[2 * L + 2] if 2 * L + 2 < len(pres) else len(seg)
        for k in seg[a0:a1]:
            agg[cat_attn(k["name"], k["name"].lower())] += k["dur"]
        for c, k in classify_ffn_layer(seg[a1:f1]):
            agg[c] += k["dur"]
    return agg


def union(iv):
    iv = sorted(iv)
    t, cs, ce = 0.0, None, None
    for s, e in iv:
        if cs is None:
            cs, ce = s, e
        elif s > ce:
            t += ce - cs
            cs, ce = s, e
        else:
            ce = max(ce, e)
    return t + (ce - cs if cs is not None else 0.0)


def analyze_step(ks):
    """ks: kernels of one engine step (sorted). Split into forwards at mhc_head_fuse."""
    heads = [i for i, k in enumerate(ks) if HEAD in k["name"]]
    fwd, prev = [], 0
    for h in heads:
        seg = ks[prev : h + 1]
        npre = sum(1 for k in seg if PRE in k["name"])
        if npre:
            st = next(j for j in range(prev, h + 1) if PRE in ks[j]["name"])
            fwd.append(("V" if npre >= 80 else "D", ks[st : h + 1], ks[prev:st]))
        prev = h + 1
    tail = ks[prev:]
    out = collections.defaultdict(float)
    nv = nd = 0
    for kind, seg, before in fwd:
        pre = "" if kind == "V" else "draft."
        nv += kind == "V"
        nd += kind == "D"
        for c, v in forward_cats(seg).items():
            out[pre + c] += v
        for k in before:
            out[glue_cat(k)] += k["dur"]
    for k in tail:
        out[glue_cat(k)] += k["dur"]
    return out, nv, nd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("trace")
    ap.add_argument("--json", default="")
    ap.add_argument("--label", default="")
    a = ap.parse_args()
    ev = json.load(open(a.trace))["traceEvents"]
    X = [e for e in ev if e.get("ph") == "X" and "dur" in e]
    cpu = [e for e in X if e.get("cat") == "cpu_op"]
    starts = sorted(
        e["ts"] for e in cpu if e["name"].startswith("engine.normal.execute(")
    )
    kern = sorted(
        (e for e in X if e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset")),
        key=lambda e: e["ts"],
    )
    # GPU-time periods: verify forward start -> next verify forward start (same as dsv4_step_breakdown);
    # the CPU engine.normal.execute window would cut the asynchronous GPU stream mid-forward.
    t_lo, t_hi = starts[0], starts[-1]
    ks_all = [k for k in kern if t_lo <= k["ts"] < t_hi + 200000]
    heads = [i for i, k in enumerate(ks_all) if HEAD in k["name"]]
    fwd, prev = [], 0
    for h in heads:
        seg = ks_all[prev : h + 1]
        npre = sum(1 for k in seg if PRE in k["name"])
        if npre:
            st = next(j for j in range(prev, h + 1) if PRE in ks_all[j]["name"])
            fwd.append(("V" if npre >= 80 else "D", st, h + 1))
        prev = h + 1
    vidx = [i for i, f in enumerate(fwd) if f[0] == "V"]
    rows = []
    for a_i, b_i in zip(vidx, vidx[1:]):
        p0, p1 = fwd[a_i][1], fwd[b_i][1]
        ks = ks_all[p0:p1]
        cats = collections.defaultdict(float)
        inside = set()
        nd = 0
        for kind, st, en in fwd[a_i:b_i]:
            pre = "" if kind == "V" else "draft."
            nd += kind == "D"
            for c, v in forward_cats(ks_all[st:en]).items():
                cats[pre + c] += v
            inside.update(range(st, en))
            cats["_busy_" + ("verify" if kind == "V" else "draft")] += sum(
                k["dur"] for k in ks_all[st:en]
            )
        for j in range(p0, p1):
            if j not in inside:
                cats[glue_cat(ks_all[j])] += ks_all[j]["dur"]
                cats["_busy_glue"] += ks_all[j]["dur"]
        step = ks_all[p1]["ts"] - ks_all[p0]["ts"]
        busy = union([(k["ts"], k["ts"] + k["dur"]) for k in ks])
        cats["_step"] = step
        cats["_busy"] = busy
        cats["_idle"] = step - busy
        cats["_ndraft"] = nd
        rows.append(cats)
    if not rows:
        raise SystemExit("no complete step with one verify forward")
    keys = sorted({k for r in rows for k in r})
    med = {k: statistics.median([r.get(k, 0.0) for r in rows]) for k in keys}
    med["_nsteps"] = len(rows)
    print(
        f"{a.label} steps={len(rows)} step(median) {med['_step']/1e3:.2f} ms busy {med['_busy']/1e3:.2f} "
        f"idle {med['_idle']/1e3:.2f}  drafts/step {med['_ndraft']:.0f}  "
        f"step range {min(r['_step'] for r in rows)/1e3:.2f}-{max(r['_step'] for r in rows)/1e3:.2f}  "
        f"busy verify {med['_busy_verify']/1e3:.2f} draft {med.get('_busy_draft',0)/1e3:.2f} glue {med.get('_busy_glue',0)/1e3:.2f}"
    )
    for k in sorted((k for k in keys if not k.startswith("_")), key=lambda k: -med[k]):
        print(f"  {k:32s} {med[k]/1e3:7.2f} ms")
    if a.json:
        json.dump(med, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
