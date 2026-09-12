#!/usr/bin/env python3
"""QPS saturation ladder sweep analyzer (Task #42).

Two modes
---------
analyze : walk one or more sweep dirs, load every cell's guardrail products
          (guardrail-summary.json + guardrail-raw.json + guardrail-affinity.jsonl
          + flexlb_master.log) and emit a compact per-cell matrix.json carrying
          every Task #42 metric.  Run this ON each remote machine over that
          machine's shard of cells.

report  : read one or more matrix.json (e.g. one per machine), aggregate by
          (qps, cap) over seeds, and emit ladder-curves.json + questions.json
          (the four Task #42 answers with numbers).  Run this locally after
          pulling the per-machine matrix.json files back.

SLO caliber: intrinsic (routing-independent) deadline = FORMULA(1,0,seqLen) x
M2, M2 = 2.6299 (golden cap250 pooled P95, frozen -- NOT recalibrated here so
the numbers stay comparable to the golden 5.60% self-referential baseline).

All latency/violation statistics are computed on the SUSTAIN phase (steady
saturation) unless a field is explicitly named *_allphase.
"""
import argparse
import glob
import json
import math
import os
from collections import Counter, defaultdict

# ── frozen SLO caliber (verbatim from golden-sweep slo_analysis_v2.py) ────────
M2 = 2.6299
INPUT_LEN = {0: 4224, 1: 4224, 2: 3072, 3: 3072, 4: 1024, 5: 1024}


def prefill_formula(batchSize, hitCacheTokens, computeTokens):
    h = hitCacheTokens / 1024.0
    c = computeTokens / 1024.0
    b = batchSize
    inner = (287.3980926717
        + 2.30134977837751 * b
        + 0.158123254797307 * h
        + 0.575522710053703 * c
        + 0.0517623430739831 * (c * c)
        + 0.0395308136993267 * (h * c)
        + 0.0104363634681015 * (h * h)
        + 0.575522710053703 * max(c - 16, 0)
        + 2.82077211814514 * max(c - 32, 0)
        - 0.0254671429192862 * max(c - 64, 0)
        + 2.15779213792494 * max(c - 96, 0)
        + 0.247806025472364 * max(h - 32, 0)
        - 0.444522654549492 * max(h - 64, 0)
        - 0.427317020061895 * max(h - 128, 0)
        + 0.347029077528455 * max(h - 256, 0)
        - 0.298742307762735 * max(h - 384, 0)
        + 2.30134977837751 * max(b - 8, 0)
        - 3.54884859699154 * max(b - 16, 0)
        - 11.3438560779984 * max(b - 24, 0)
        + 0.879751992138183 * max(c - 2, 0)
        + 0.636364578079591 * max(c - 4, 0)
        - 0.0513345988517118 * max(c - 8, 0)
        - 0.332584389129357 * max(h - 2, 0)
        + 0.305819761192588 * max(h - 4, 0)
        - 0.287610979974721 * max(h - 8, 0)
        + 0.191310200712013 * max(h - 12, 0)
        + 0.0130251644478961 * max(b - 8, 0) * h
        + 0.00981382840761646 * max(b - 16, 0) * h
        - 0.0299132587297009 * max(b - 24, 0) * h
        + 0.0447455122487382 * max(b - 8, 0) * c
        + 0.0104635312001851 * max(b - 16, 0) * c
        + 0.0542737877321807 * max(b - 24, 0) * c)
    return max(196.0, -68.612174288157 + 0.993068319341 * max(0.0, inner))


def intrinsic(seqLen):
    return prefill_formula(1, 0, seqLen)


INTRINSIC = {ilen: intrinsic(ilen) for ilen in sorted(set(INPUT_LEN.values()))}


def fam_abc(family):
    if family is None:
        return None
    if family < 2:
        return "A"
    if family < 4:
        return "B"
    if family < 6:
        return "C"
    return "bg"


# ── small stats helpers ──────────────────────────────────────────────────────
def pct(sv, q):
    if not sv:
        return None
    k = (len(sv) - 1) * q
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return sv[int(k)]
    return sv[f] * (c - k) + sv[c] * (k - f)


def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def r4(x):
    return None if x is None else round(x, 4)


def r6(x):
    return None if x is None else round(x, 6)


def dist(vals):
    """compact distribution summary of a numeric list."""
    vals = sorted(v for v in vals if v is not None)
    if not vals:
        return {"n": 0}
    return {
        "n": len(vals), "mean": r4(mean(vals)), "p50": r4(pct(vals, .5)),
        "p95": r4(pct(vals, .95)), "max": r4(vals[-1]),
    }


# ── master.log signals (admission rejects / capacity blocks) ──────────────────
BLOCK_TOKENS = ("REJECT", "CAPACITY_BLOCK", "OVER_CAP", "NO_CANDIDATE", "park")


def scan_master_log(fp):
    """count block/reject signal tokens in the master log (honest raw counts)."""
    sig = {t: 0 for t in BLOCK_TOKENS}
    route_trace_rows = 0
    if not fp or not os.path.exists(fp):
        return {"present": False, "signals": sig, "route_trace_rows": 0}
    with open(fp, "r", errors="replace") as f:
        for line in f:
            if line.startswith("OFFLINE_ROUTE_TRACE "):
                route_trace_rows += 1
            for t in BLOCK_TOKENS:
                if t in line:
                    sig[t] += 1
    return {"present": True, "signals": sig, "route_trace_rows": route_trace_rows}


# ── per-cell loader ──────────────────────────────────────────────────────────
def _find_one(root, pattern):
    hits = sorted(glob.glob(os.path.join(root, pattern), recursive=True))
    return hits[0] if hits else None


def load_cell(cell_dir):
    """Build the full Task #42 record for one cell dir. Returns dict (or error)."""
    summ_fp = _find_one(cell_dir, "**/guardrail-summary.json")
    if not summ_fp:
        return {"cell": os.path.basename(cell_dir), "status": "NO_SUMMARY",
                "cell_dir": cell_dir}
    idir = os.path.dirname(summ_fp)
    raw_fp = os.path.join(idir, "guardrail-raw.json")
    aff_fp = os.path.join(idir, "guardrail-affinity.jsonl")
    log_fp = _find_one(cell_dir, "**/flexlb_master.log")

    S = json.load(open(summ_fp))
    rec = {
        "cell": os.path.basename(cell_dir), "status": "OK", "cell_dir": cell_dir,
        "summary_path": summ_fp,
        # ── construction / validity ──
        "config": S.get("config"),
        "arm_cap_ms": S.get("arm_cap_ms"),
        "trace_sha256": S.get("trace_sha256"),
        "planned": S.get("planned"), "issued": S.get("issued"),
        "planned_eq_issued": S.get("planned") == S.get("issued"),
        "max_pacing_lag_s": r4(S.get("max_pacing_lag_s")),
        "execution_valid": S.get("execution_valid"),
        "construction_valid": S.get("construction_valid"),
        "harmful_affinity_observed": S.get("harmful_affinity_observed"),
        "control_clean": S.get("control_clean"),
    }
    cfg = S.get("config") or {}
    rec["qps_nominal"] = (cfg.get("peak_a_qps", 0) + cfg.get("b_qps", 0)
                          + cfg.get("c_qps", 0))
    rec["capacity_blocks"] = None  # filled by caller from run plan if desired
    cadence_tol = max(0.5, 2 / max(cfg.get("peak_a_qps", 1), 1))
    rec["cadence_ok"] = (S.get("max_pacing_lag_s") or 0) <= cadence_tol + 1e-9

    # ── master.log reject / block signals ──
    rec["master_log"] = scan_master_log(log_fp)

    # ── saturation metrics from summary.metrics (sustain phase) ──
    metrics = S.get("metrics") or []
    sus = next((m for m in metrics if m.get("id") == "sustain"), None)
    if sus is None and metrics:
        sus = max(metrics, key=lambda m: m.get("duration_s", 0))
    per_engine = {}
    if sus:
        dur = sus.get("duration_s") or 0
        for ename, pd in (sus.get("per_prefill") or {}).items():
            per_engine[ename] = {
                "busy_fraction": r4(pd.get("busy_fraction")),
                "completed": pd.get("completed"),
                "qps": r4(pd.get("completed") / dur) if dur else None,
                "hit_rate": r4(pd.get("hit_rate")),
                "input_tps": r4(pd.get("input_tps")),
                "compute_tps": r4(pd.get("compute_tps")),
            }
    rec["sustain"] = {
        "duration_s": r4(sus.get("duration_s")) if sus else None,
        "issued": sus.get("issued") if sus else None,
        "input_tps": r4(sus.get("input_tps")) if sus else None,
        "compute_tps": r4(sus.get("compute_tps")) if sus else None,
        "ttft_p95_s": r4(sus.get("ttft_p95_s")) if sus else None,
        "decode_waiting_peak": sus.get("decode_waiting_peak") if sus else None,
        "per_engine": per_engine,
        "busy_mean": r4(mean([v["busy_fraction"] for v in per_engine.values()])),
        "busy_active_mean": r4(mean([v["busy_fraction"] for v in per_engine.values()
                                     if (v["busy_fraction"] or 0) > 0.01])),
        "busy_max": r4(max([v["busy_fraction"] for v in per_engine.values()])) if per_engine else None,
        "active_engines": sum(1 for v in per_engine.values()
                              if (v["busy_fraction"] or 0) > 0.01),
    }
    # whole-run ttft (total caliber, all phases pooled) from summary.ttft_s
    rec["ttft_total_s"] = S.get("ttft_s")

    # ── KV pool / eviction (sustain) ──
    kv = (S.get("kv_pool_analysis") or {}).get("sustain") or {}
    ev = {}
    for ename, ed in (kv.get("per_engine") or {}).items():
        ev[ename] = {
            "eviction_age_avg_s": ed.get("eviction_age_avg_s"),
            "eviction_age_p95_s": ed.get("eviction_age_p95_s"),
            "lifetime_s": ed.get("lifetime_s"),
            "free_pct": ed.get("free_pct"),
            "eviction_rate_bps": ed.get("eviction_rate_bps"),
            "cache_capacity": ed.get("cache_capacity"),
        }
    lifetimes = [v["lifetime_s"] for v in ev.values() if v["lifetime_s"] is not None]
    rec["eviction"] = {
        "per_engine": ev,
        "lifetime_min_s": r4(min(lifetimes)) if lifetimes else None,
        "lifetime_mean_s": r4(mean(lifetimes)) if lifetimes else None,
        # aligned band = 200-300s eviction lifetime (golden signature)
        "out_of_band": bool(lifetimes) and (min(lifetimes) < 200 or max(lifetimes) > 300),
        "fast_swap": bool(lifetimes) and max(lifetimes) < 200,
    }

    # ── affinity aggregates from summary ──
    aff = S.get("affinity") or {}
    rec["affinity_summary"] = {
        "reasons": aff.get("reasons"),
        "selected_prefill_counts": aff.get("selected_prefill_counts"),
        "hotspot_survivors_ge2_ratio": aff.get("hotspot_survivors_ge2_ratio"),
        "net_gain_nonneg_ratio": aff.get("net_gain_nonneg_ratio"),
        "hotspot_phase": aff.get("hotspot_phase"),
    }

    # ── per-request recompute from raw.json (per-family TTFT + SLO) ──
    rec.update(_per_request(raw_fp, aff_fp))
    return rec


def _per_request(raw_fp, aff_fp):
    out = {
        "ttft_by_family": {}, "slo": {}, "crosstab": {}, "reason_branch": {},
        "delta_dist": {}, "pending_batch": {}, "raw_requests": None,
    }
    if not os.path.exists(raw_fp):
        out["error"] = "missing raw.json"
        return out
    raw = json.load(open(raw_fp))
    reqs = raw.get("requests", [])
    out["raw_requests"] = len(reqs)

    # addr -> physical engine name, from window samples' prefill grpc_addr.
    # NOTE: the affinity trace's `selected` mixes the mock control address
    # (mock_base-1) with engine names and reflects a LOGICAL routing label,
    # not physical execution (e.g. prefill-2 shows selected>0 but completed=0).
    # The family x engine crosstab and engine-spread count therefore use the
    # authoritative raw.json `prefill_addr` instead.
    addr2ename = {}
    for w in raw.get("windows", []):
        for s in w.get("samples", []):
            for ename, v in (s.get("prefill") or {}).items():
                ga = v.get("grpc_addr")
                if ga:
                    addr2ename[ga] = ename
        if addr2ename:
            break

    # sustain-phase requests with a TTFT
    sus_rows, all_rows = [], []
    for r in reqs:
        iss = r.get("issued_s")
        fo = (r.get("stream") or {}).get("first_output_s")
        ilen = r.get("input_len")
        fam = r.get("family")
        if iss is None or fo is None or ilen is None:
            continue
        pa = r.get("prefill_addr")
        row = {
            "family": fam, "abc": fam_abc(fam), "input_len": ilen,
            "ttft_ms": (fo - iss) * 1000.0, "phase": r.get("phase"),
            "intr": INTRINSIC.get(ilen, intrinsic(ilen)),
            "ename": addr2ename.get(pa, pa),
        }
        all_rows.append(row)
        if r.get("phase") == "sustain":
            sus_rows.append(row)

    def fam_stats(rows):
        by = {}
        for abc in ("A", "B", "C", "bg"):
            sub = [x for x in rows if x["abc"] == abc]
            if not sub:
                continue
            tt = sorted(x["ttft_ms"] for x in sub)
            by[abc] = {
                "n": len(sub),
                "ttft_ms_p50": r4(pct(tt, .5)), "ttft_ms_p95": r4(pct(tt, .95)),
                "ttft_ms_p99": r4(pct(tt, .99)), "ttft_ms_mean": r4(mean(tt)),
                "ttft_ms_max": r4(tt[-1]),
                # fast band = cache-hit short TTFT (<350ms); golden C band ~243ms
                "fast_band_frac_lt350": r6(sum(1 for x in tt if x < 350) / len(tt)),
                "fast_band_frac_lt300": r6(sum(1 for x in tt if x < 300) / len(tt)),
            }
        return by

    out["ttft_by_family"] = fam_stats(sus_rows)
    out["ttft_by_family_allphase"] = fam_stats(all_rows)

    # SLO violation (intrinsic x M2), sustain + all-phase, total + by family
    def viol(rows):
        nj = len(rows)
        if nj == 0:
            return {"n": 0}
        n_v = sum(1 for x in rows if x["ttft_ms"] > x["intr"] * M2)
        by = {}
        for abc in ("A", "B", "C", "bg"):
            sub = [x for x in rows if x["abc"] == abc]
            if sub:
                by[abc] = r6(sum(1 for x in sub
                                 if x["ttft_ms"] > x["intr"] * M2) / len(sub))
        return {"n": nj, "violations": n_v, "rate": r6(n_v / nj), "by_family": by}

    out["slo"] = {"sustain": viol(sus_rows), "allphase": viol(all_rows)}

    # pending (waiting) / batch (running) distributions in sustain from windows
    for w in raw.get("windows", []):
        if w.get("id") != "sustain":
            continue
        wait_sys, run_sys = [], []
        wait_pe, run_pe = defaultdict(list), defaultdict(list)
        for s in w.get("samples", []):
            pre = s.get("prefill", {})
            ws = [v.get("waiting", 0) for v in pre.values()]
            rs = [v.get("running", 0) for v in pre.values()]
            wait_sys.append(sum(ws))
            run_sys.append(sum(rs))
            for ename, v in pre.items():
                wait_pe[ename].append(v.get("waiting", 0))
                run_pe[ename].append(v.get("running", 0))
        out["pending_batch"] = {
            "n_samples": len(w.get("samples", [])),
            "pending_system": dist(wait_sys),
            "batch_system": dist(run_sys),
            "pending_per_engine": {e: dist(v) for e, v in wait_pe.items()},
            "batch_per_engine": {e: dist(v) for e, v in run_pe.items()},
        }
        break

    # ── family x engine crosstab (PHYSICAL, from raw.json prefill_addr) ──
    ct_sus = defaultdict(Counter)   # abc -> physical engine name -> n
    ct_all = defaultdict(Counter)
    for x in all_rows:
        if x["abc"] and x["ename"]:
            ct_all[x["abc"]][x["ename"]] += 1
    for x in sus_rows:
        if x["abc"] and x["ename"]:
            ct_sus[x["abc"]][x["ename"]] += 1
    out["crosstab"] = {
        "sustain": {a: dict(c) for a, c in ct_sus.items()},
        "all": {a: dict(c) for a, c in ct_all.items()},
    }
    # distinct PHYSICAL engines serving sustain traffic (routing spread, Q3)
    eng = set(x["ename"] for x in sus_rows if x["ename"])
    out["engines_used_sustain"] = len(eng)
    out["engines_used_sustain_list"] = sorted(eng)

    # ── reason branch (CL/NCL/OVER_CAP) + delta dist (affinity.jsonl) ──
    if os.path.exists(aff_fp):
        reasons = Counter()
        deltas = []
        with open(aff_fp) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                if d.get("phase") == "sustain":
                    if d.get("reason"):
                        reasons[d["reason"]] += 1
                    if d.get("delta_ms") is not None:
                        deltas.append(d["delta_ms"])
        tot = sum(reasons.values())
        out["reason_branch"] = {
            "counts": dict(reasons),
            "ratios": {k: r6(v / tot) for k, v in reasons.items()} if tot else {},
        }
        out["delta_dist"] = dist(deltas)
        out["delta_gt100_ratio"] = (r6(sum(1 for d in deltas if d > 100) / len(deltas))
                                    if deltas else None)
    return out


# ── analyze mode ─────────────────────────────────────────────────────────────
def cmd_analyze(args):
    cells = []
    seen = set()
    for d in args.sweep_dirs:
        for cd in sorted(glob.glob(os.path.join(d, "cell_*"))):
            if os.path.isdir(cd) and cd not in seen:
                seen.add(cd)
                cells.append(cd)
        # also allow a dir that IS a cell (has nested summary)
    matrix = []
    for cd in cells:
        try:
            matrix.append(load_cell(cd))
        except Exception as exc:
            matrix.append({"cell": os.path.basename(cd), "status": "ANALYZE_ERROR",
                           "error": "%s: %s" % (type(exc).__name__, exc)})
    # cross-reference lane manifests for rc/wall/status if present
    man = {}
    for d in args.sweep_dirs:
        for mf in glob.glob(os.path.join(d, "*manifest*.jsonl")):
            for line in open(mf):
                line = line.strip()
                if line:
                    m = json.loads(line)
                    man[m.get("cell")] = m
    for rec in matrix:
        m = man.get(rec["cell"])
        if m:
            rec["run_rc"] = m.get("rc")
            rec["run_wall_s"] = m.get("wall_s")
            rec["run_status"] = m.get("status")
            rec["capacity_blocks"] = m.get("capacity_blocks")
            rec["window"] = m.get("window")
            rec["qps_rung"] = m.get("qps")
            rec["cap"] = m.get("cap")
            rec["cap_label"] = m.get("cap_label")
            rec["seed"] = m.get("seed")
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    json.dump({"schema": "task42-matrix-v1", "M2": M2, "n_cells": len(matrix),
               "intrinsic_by_input_len": {str(k): r4(v) for k, v in INTRINSIC.items()},
               "cells": matrix},
              open(args.out, "w"), indent=1)
    nok = sum(1 for r in matrix if r.get("status") == "OK")
    print("[analyze] %d cells (%d OK) -> %s" % (len(matrix), nok, args.out))
    return 0


# ── report mode ──────────────────────────────────────────────────────────────
def _load_matrix(paths):
    cells = []
    for p in paths:
        d = json.load(open(p))
        cells += d.get("cells", d if isinstance(d, list) else [])
    return [c for c in cells if c.get("status") == "OK"]


def _merge_crosstabs(cts):
    """sum abc -> physical engine -> n across per-cell sustain crosstabs."""
    merged = defaultdict(Counter)
    for ct in cts:
        for abc, eng in (ct or {}).items():
            for e, n in eng.items():
                merged[abc][e] += n
    return {a: dict(c) for a, c in merged.items()}


def _merge_reasons(rbs):
    merged = Counter()
    for rb in rbs:
        for k, v in ((rb or {}).get("counts") or {}).items():
            merged[k] += v
    tot = sum(merged.values())
    return {"counts": dict(merged),
            "ratios": {k: r6(v / tot) for k, v in merged.items()} if tot else {}}


def cmd_report(args):
    cells = _load_matrix(args.matrix)
    # group by (qps_rung, cap_label)
    by = defaultdict(list)
    for c in cells:
        q = c.get("qps_rung")
        cl = c.get("cap_label")
        if q is None:
            # infer from config
            cfg = c.get("config") or {}
            q = cfg.get("peak_a_qps", 0) + cfg.get("b_qps", 0) + cfg.get("c_qps", 0)
        if cl is None:
            cap = c.get("arm_cap_ms")
            cl = "inf" if (cap or 0) >= 1e8 else str(cap)
        by[(q, cl)].append(c)

    def agg(cs):
        def g(path):
            vals = []
            for c in cs:
                cur = c
                for k in path:
                    cur = (cur or {}).get(k) if isinstance(cur, dict) else None
                if cur is not None:
                    vals.append(cur)
            return vals
        slo_sus = [c["slo"]["sustain"].get("rate") for c in cs
                   if c.get("slo", {}).get("sustain", {}).get("rate") is not None]
        slo_all = [c["slo"]["allphase"].get("rate") for c in cs
                   if c.get("slo", {}).get("allphase", {}).get("rate") is not None]
        cf = [c["ttft_by_family"].get("C", {}).get("fast_band_frac_lt350")
              for c in cs if c.get("ttft_by_family", {}).get("C")]
        return {
            "n_cells": len(cs),
            "seeds": sorted(c.get("seed") for c in cs),
            "busy_mean": r4(mean(g(["sustain", "busy_mean"]))),
            "busy_active_mean": r4(mean(g(["sustain", "busy_active_mean"]))),
            "busy_max": r4(mean(g(["sustain", "busy_max"]))),
            "active_engines": r4(mean(g(["sustain", "active_engines"]))),
            "engines_used_sustain": r4(mean([c.get("engines_used_sustain") for c in cs
                                             if c.get("engines_used_sustain") is not None])),
            "input_tps": r4(mean(g(["sustain", "input_tps"]))),
            "sustain_duration_s": r4(mean(g(["sustain", "duration_s"]))),
            "slo_viol_sustain": r6(mean(slo_sus)) if slo_sus else None,
            "slo_viol_sustain_seeds": [r6(x) for x in sorted(slo_sus)],
            "slo_viol_allphase": r6(mean(slo_all)) if slo_all else None,
            "slo_viol_by_family": {
                abc: r6(mean([c["slo"]["sustain"]["by_family"].get(abc) for c in cs
                              if c.get("slo", {}).get("sustain", {}).get("by_family", {}).get(abc) is not None]))
                for abc in ("A", "B", "C")
            },
            "c_fastband_frac": r6(mean(cf)) if cf else None,
            "ttft_p95_s": r4(mean(g(["sustain", "ttft_p95_s"]))),
            "ttft_by_family_p95": {
                abc: r4(mean([c["ttft_by_family"][abc]["ttft_ms_p95"] for c in cs
                              if c.get("ttft_by_family", {}).get(abc)]))
                for abc in ("A", "B", "C")
            },
            "ttft_by_family_p50": {
                abc: r4(mean([c["ttft_by_family"][abc]["ttft_ms_p50"] for c in cs
                              if c.get("ttft_by_family", {}).get(abc)]))
                for abc in ("A", "B", "C")
            },
            "lifetime_mean_s": r4(mean(g(["eviction", "lifetime_mean_s"]))),
            "lifetime_min_s": r4(mean(g(["eviction", "lifetime_min_s"]))),
            "out_of_band_frac": r4(mean([1.0 if c.get("eviction", {}).get("out_of_band") else 0.0
                                         for c in cs])),
            "reject_signals": {t: sum(c.get("master_log", {}).get("signals", {}).get(t, 0) for c in cs)
                               for t in BLOCK_TOKENS},
            "planned_eq_issued_all": all(c.get("planned_eq_issued") for c in cs),
            "cadence_ok_all": all(c.get("cadence_ok") for c in cs),
            "delta_gt100_ratio": r6(mean([c.get("delta_gt100_ratio") for c in cs
                                          if c.get("delta_gt100_ratio") is not None])),
            "crosstab_sustain": _merge_crosstabs(
                [c.get("crosstab", {}).get("sustain") for c in cs]),
            "reason_branch": _merge_reasons([c.get("reason_branch") for c in cs]),
        }

    qps_list = sorted({q for (q, cl) in by})
    caps_list = sorted({cl for (q, cl) in by},
                       key=lambda s: (s != "inf", float(s) if s != "inf" else 0))
    curves = {}
    for q in qps_list:
        curves[str(q)] = {}
        for cl in caps_list:
            cs = by.get((q, cl))
            if cs:
                curves[str(q)][cl] = agg(cs)

    # ── the four Task #42 questions ──
    questions = _answer_questions(curves, qps_list, caps_list, by, agg)

    json.dump({"schema": "task42-ladder-curves-v1", "M2": M2,
               "qps_list": qps_list, "caps": caps_list, "curves": curves},
              open(args.curves, "w"), indent=1)
    json.dump(questions, open(args.questions, "w"), indent=1)
    print("[report] %d OK cells, %d QPS rungs, %d caps" % (len(cells), len(qps_list), len(caps_list)))
    print("[report] curves -> %s ; questions -> %s" % (args.curves, args.questions))
    print(json.dumps(questions, indent=1, ensure_ascii=False))
    return 0


def _answer_questions(curves, qps_list, caps_list, by, agg):
    q = {}
    # Q1: C fast band disappearance across QPS (avg over caps)
    c_band = {}
    for qq in qps_list:
        fr = [curves[str(qq)][cl]["c_fastband_frac"] for cl in caps_list
              if cl in curves[str(qq)] and curves[str(qq)][cl]["c_fastband_frac"] is not None]
        c_band[qq] = r6(mean(fr)) if fr else None
    gone = [qq for qq in qps_list if c_band[qq] is not None and c_band[qq] < 0.05]
    q["Q1_c_fastband"] = {
        "frac_by_qps": c_band,
        "disappears_at_qps": min(gone) if gone else None,
        "note": "fast_band_frac_lt350 averaged over caps; <0.05 = band gone",
    }
    # Q2: cap monotonicity at busy 0.6-0.7 rungs
    q2 = {}
    for qq in qps_list:
        busy = None
        fr = [curves[str(qq)][cl]["busy_mean"] for cl in caps_list if cl in curves[str(qq)]]
        busy = mean([b for b in fr if b is not None]) if fr else None
        per_cap = {cl: curves[str(qq)][cl]["slo_viol_sustain"] for cl in caps_list
                   if cl in curves[str(qq)]}
        q2[qq] = {"busy_mean": r4(busy), "viol_by_cap": per_cap,
                  "in_busy_band": bool(busy is not None and 0.6 <= busy <= 0.7)}
    q["Q2_cap_monotonic"] = q2
    # Q3: routing spread 2->4 engines
    q3 = {}
    q3b = {}
    for qq in qps_list:
        eu = [curves[str(qq)][cl]["engines_used_sustain"] for cl in caps_list
              if cl in curves[str(qq)] and curves[str(qq)][cl]["engines_used_sustain"] is not None]
        ae = [curves[str(qq)][cl]["active_engines"] for cl in caps_list
              if cl in curves[str(qq)] and curves[str(qq)][cl]["active_engines"] is not None]
        q3[qq] = r4(mean(eu)) if eu else None
        q3b[qq] = r4(mean(ae)) if ae else None
    q["Q3_routing_spread"] = {"engines_used_by_qps": q3,
                              "active_engines_by_qps": q3b,
                              "spread_90_to_120": (q3.get(90), q3.get(120))}
    # Q4: online-comparable saturation violation + cap spread -> new gate baseline
    band = [qq for qq in qps_list
            if any(curves[str(qq)][cl].get("busy_mean") is not None
                   and 0.6 <= curves[str(qq)][cl]["busy_mean"] <= 0.7
                   for cl in caps_list if cl in curves[str(qq)])]
    q4 = {"online_comparable_qps": band, "by_cap": {}}
    for cl in caps_list:
        vals = [curves[str(qq)][cl]["slo_viol_sustain"] for qq in band
                if cl in curves[str(qq)] and curves[str(qq)][cl]["slo_viol_sustain"] is not None]
        q4["by_cap"][cl] = r6(mean(vals)) if vals else None
    finite = [v for v in q4["by_cap"].values() if v is not None]
    q4["cap_spread_max_minus_min"] = r6(max(finite) - min(finite)) if len(finite) >= 2 else None
    q["Q4_gate_baseline"] = q4
    return q


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="mode", required=True)
    a = sub.add_parser("analyze", help="cells -> matrix.json (run per machine)")
    a.add_argument("sweep_dirs", nargs="+", help="dirs containing cell_* subdirs")
    a.add_argument("--out", required=True, help="matrix.json output path")
    a.set_defaults(func=cmd_analyze)
    r = sub.add_parser("report", help="matrix.json(s) -> curves + questions")
    r.add_argument("matrix", nargs="+", help="one or more matrix.json files")
    r.add_argument("--curves", required=True)
    r.add_argument("--questions", required=True)
    r.set_defaults(func=cmd_report)
    args = ap.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
