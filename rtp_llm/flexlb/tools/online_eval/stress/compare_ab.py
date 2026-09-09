#!/usr/bin/env python3
"""compare_ab.py — A/B differential regression gate for online_eval runs.

Compares two run aggregates (run A = baseline, run B = candidate) metric by
metric and classifies every metric into one of three decision tiers:

  Tier 1  significant + critical    — regression gate trips (exit 1):
                                      invariant violations / SLA monotone
                                      regressions that a human MUST review.
  Tier 2  significant + secondary   — large relative deltas that are
                                      small-baseline artifacts / single-sample
                                      outliers / shape counters; reported,
                                      ignored by default (no gate).
  Tier 3  critical + unchanged      — critical metrics proven NOT regressed
                                      (delta inside significance thresholds
                                      or inside the small-baseline guard).
  Tier 4  informational             — reported, never judged.

Significance is a TWO-condition test (both must hold):
  1. relative delta exceeds the per-category relative threshold (floored by
     --noise-floor), AND
  2. absolute delta exceeds the per-category small-baseline guard.
This prevents sub-millisecond baselines from producing scary "-51%" style
relative deltas (route_submit_mean: abs ~1ms), and tiny gini values from
producing "+14%" out of 0.007 absolute noise.

Tier assignment:
  critical  + significant              -> Tier 1
  secondary + relative-significant     -> Tier 2 (either condition alone
                                           already lands here — the whole
                                           point of Tier 2 is to absorb
                                           relative-only false alarms)
  critical  + not significant          -> Tier 3
  informational                        -> Tier 4
Exit codes: 0 = gate passed (Tier 1 empty); 1 = gate tripped; 2 = precheck
failure (not the same experiment — trace/params mismatch, unreadable inputs).

Metric paths consume the aggregate.json produced by aggregate_canvas_run.py
(the 51 field paths were validated against current aggregates; the e2e/ttft
percentile families from the same summary section are additionally included
— they are all-zero in mock-engine runs and safe).

Usage:
  python3 compare_ab.py --run-a <runDirOrAggregate> --run-b <runDirOrAggregate> \
      [--steady-lo 30.0 --steady-hi 110.0] [--out ab_summary.json] [--html] \
      [--noise-floor 0.02]

Steady-state window: defaults derived from meta.duration_s
(lo = 0.25 * duration — after ramp-up, hi = 0.92 * duration — before
drain), overridable via --steady-lo/--steady-hi.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

# ---------------------------------------------------------------------------
# Significance thresholds per metric category.
#
# Provenance (human triage experience to be mechanized here):
#   * schedule p50 carries ~±10% run-to-run noise across runs, but <2%
#     (mostly <0.5%) for the same code — hence the 2% noise floor default
#     and a conservative 5% latency threshold.
#   * millisecond-scale small baselines inflate relative deltas (abs ~1ms
#     already shows -51%) — hence the absolute-ms guard.
#   * max-type percentiles are single-sample outliers — demoted to
#     secondary criticality, never gate.
#   * single-point gini values amplify small decimals — hence the 0.005
#     absolute guard (12/40-engine clusters wobble at the 0.00x scale).
# ---------------------------------------------------------------------------

LATENCY_REL_PCT = 5.0  # percentile/mean latency relative threshold
LATENCY_ABS_MS = 5.0  # small-baseline guard: abs diff must EXCEED 5ms
RATE_REL_PCT = 2.0  # QPS/TPS class (same-code noise <2%)
RATIO_REL_PP = 3.0  # percentage-point threshold for pct-type metrics
RATIO_ABS_PP = 1.0  # guard: abs diff must exceed 1pp
COUNT_REL_PCT = 10.0  # cumulative counters
COUNT_ABS_FRAC = 0.01  # guard: abs diff must exceed 1% of |B|
GINI_REL_PCT = 5.0
GINI_ABS = 0.005  # small-decimal amplification guard
SHAPE_REL_PCT = 10.0  # batch-size distribution shape
SHAPE_ABS_UNITS = 2.0  # guard: batch-unit-level diffs are scheduling noise
BLOCKS_REL_PCT = 10.0  # KV block pool counters / levels
BLOCKS_ABS_FRAC = 0.01  # guard: 1% of the role's total_blocks

DEFAULT_NOISE_FLOOR = 0.02  # same-code core-metric noise (fraction)

# Category -> (relative threshold, absolute guard kind). The noise floor
# raises the relative threshold: eff_rel = max(cat_rel, floor * 100).
CATEGORY_RULES = {
    "latency": ("rel_pct", LATENCY_REL_PCT),
    "rate": ("rel_pct", RATE_REL_PCT),
    "ratio": ("rel_pp", RATIO_REL_PP),
    "count": ("rel_pct", COUNT_REL_PCT),
    "gini": ("rel_pct", GINI_REL_PCT),
    "shape": ("rel_pct", SHAPE_REL_PCT),
    "blocks": ("rel_pct", BLOCKS_REL_PCT),
    "bool": ("equality", None),  # unequal => significant, no guards
}

# ---------------------------------------------------------------------------
# Criticality map (metric name -> (criticality, category, direction)).
#
# CRITICAL  = invariant / SLA-monotone mapping: test_valid, error_rate,
#             schedule & e2e & ttft percentiles, TPS trio, cache-hit ratios,
#             KV zero-invariants (evictions/admission fails/lack-mem),
#             decode KV reuse, send fidelity QPS (A/B must be same load).
# SECONDARY = shape descriptors / noisy-by-construction: stage mean/max,
#             small-baseline stages (route_submit & dispatch_ack families,
#             typically <5ms), single-point gini, batch_size_final counters,
#             KV final levels (drain-phase sensitive) & steady levels.
# INFO      = reported, never judged: total_requests, utilization echo,
#             capacity echo (total_blocks), completion counts.
# Direction: "worse"  — higher value degrades (latency, error, gini, ...);
#            "better" — higher value improves (TPS, hit ratio, reuse, ...);
#            "neutral" — no monotone semantics (levels, shape, echoes).
# ---------------------------------------------------------------------------

CRIT = "critical"
SEC = "secondary"
INFO = "info"

CRITICALITY = {
    # -- validity (invariants) --
    "test_valid": (CRIT, "bool", "better"),
    "validity_checks_passed": (CRIT, "count", "better"),
    # -- load & errors --
    "total_requests": (INFO, "count", "neutral"),
    "error_count": (CRIT, "count", "worse"),
    "error_rate": (CRIT, "ratio", "worse"),
    "actual_send_qps": (CRIT, "rate", "neutral"),  # load fidelity: must match
    "server_arrival_qps": (CRIT, "rate", "neutral"),
    # -- schedule latency percentiles (SLA monotone) --
    "schedule_latency_ms_p50": (CRIT, "latency", "worse"),
    "schedule_latency_ms_p90": (CRIT, "latency", "worse"),
    "schedule_latency_ms_p95": (CRIT, "latency", "worse"),
    "schedule_latency_ms_p99": (CRIT, "latency", "worse"),
    "schedule_latency_ms_mean": (CRIT, "latency", "worse"),
    "schedule_latency_ms_max": (SEC, "latency", "worse"),  # single-sample outlier
    # -- end-to-end latency (server full_e2e + client e2e + ttft) --
    "full_e2e_latency_ms_p50": (CRIT, "latency", "worse"),
    "full_e2e_latency_ms_p95": (CRIT, "latency", "worse"),
    "full_e2e_latency_ms_p99": (CRIT, "latency", "worse"),
    "full_e2e_latency_ms_count": (INFO, "count", "neutral"),
    "e2e_latency_ms_p50": (CRIT, "latency", "worse"),
    "e2e_latency_ms_p90": (CRIT, "latency", "worse"),
    "e2e_latency_ms_p95": (CRIT, "latency", "worse"),
    "e2e_latency_ms_p99": (CRIT, "latency", "worse"),
    "e2e_latency_ms_mean": (CRIT, "latency", "worse"),
    "e2e_latency_ms_max": (SEC, "latency", "worse"),  # single-sample outlier
    "e2e_latency_ms_count": (INFO, "count", "neutral"),
    "ttft_latency_ms_p50": (CRIT, "latency", "worse"),
    "ttft_latency_ms_p90": (CRIT, "latency", "worse"),
    "ttft_latency_ms_p95": (CRIT, "latency", "worse"),
    "ttft_latency_ms_p99": (CRIT, "latency", "worse"),
    "ttft_latency_ms_mean": (CRIT, "latency", "worse"),
    "ttft_latency_ms_max": (SEC, "latency", "worse"),  # single-sample outlier
    "ttft_latency_ms_count": (INFO, "count", "neutral"),
    # -- stage latency: batch_wait is the dominant schedule component --
    "batch_wait_ms_p50": (CRIT, "latency", "worse"),
    "batch_wait_ms_p90": (CRIT, "latency", "worse"),
    "batch_wait_ms_p95": (CRIT, "latency", "worse"),
    "batch_wait_ms_p99": (CRIT, "latency", "worse"),
    "batch_wait_ms_mean": (SEC, "latency", "worse"),  # tail-weighted vs p50s
    "batch_wait_ms_max": (SEC, "latency", "worse"),  # single-sample outlier
    # route_submit / dispatch_ack: small-baseline stages (typically <5ms),
    # relative deltas explode on ms-scale bases — whole family secondary.
    "route_submit_ms_p50": (SEC, "latency", "worse"),
    "route_submit_ms_p90": (SEC, "latency", "worse"),
    "route_submit_ms_p95": (SEC, "latency", "worse"),
    "route_submit_ms_p99": (SEC, "latency", "worse"),
    "route_submit_ms_mean": (SEC, "latency", "worse"),
    "route_submit_ms_max": (SEC, "latency", "worse"),
    "dispatch_ack_ms_p50": (SEC, "latency", "worse"),
    "dispatch_ack_ms_p90": (SEC, "latency", "worse"),
    "dispatch_ack_ms_p95": (SEC, "latency", "worse"),
    "dispatch_ack_ms_p99": (SEC, "latency", "worse"),
    "dispatch_ack_ms_mean": (SEC, "latency", "worse"),
    "dispatch_ack_ms_max": (SEC, "latency", "worse"),
    # -- cache & token throughput --
    "cache_hit_master_routing_hit_pct": (CRIT, "ratio", "better"),
    "cache_hit_engine_key_hit_pct": (CRIT, "ratio", "better"),
    "cache_hit_engine_token_hit_pct": (CRIT, "ratio", "better"),
    "cache_saved_tokens": (CRIT, "count", "better"),
    "input_token_tps": (CRIT, "rate", "better"),
    "output_token_tps": (CRIT, "rate", "better"),
    # -- mock self-reported TPS, steady-window means --
    "mock_tps_steady_context_tps": (CRIT, "rate", "better"),
    "mock_tps_steady_context_tps_with_cache": (CRIT, "rate", "better"),
    "mock_tps_steady_generate_tps": (CRIT, "rate", "better"),
}

# KV block-pool and balance/batch metrics are generated per-role / per-kind
# below; their criticality follows the same rationale:
KV_CRITICALITY = {
    "total_blocks": (INFO, "blocks", "neutral"),  # capacity echo
    "held_blocks_final": (SEC, "blocks", "neutral"),  # drain-phase sensitive
    "referenced_blocks_final": (SEC, "blocks", "neutral"),
    "available_blocks_final": (SEC, "blocks", "neutral"),
    "cache_evictions_cum": (CRIT, "count", "worse"),  # zero-invariant
    "kv_admission_fails_cum": (CRIT, "count", "worse"),  # zero-invariant
    "lack_mem_rejects_cum": (CRIT, "count", "worse"),  # zero-invariant
    "decode_reuse_blocks_cum": (CRIT, "count", "better"),  # PD KV reuse volume
    "held_blocks_steady": (SEC, "blocks", "neutral"),  # level shape
    "referenced_blocks_steady": (SEC, "blocks", "neutral"),
}
for _role in ("prefill", "decode"):
    for _metric, (_crit, _cat, _dirn) in KV_CRITICALITY.items():
        CRITICALITY[f"kv_{_role}_{_metric}"] = (_crit, _cat, _dirn)

BALANCE_CRITICALITY = {
    # single-point gini (whole-run cumulative, no time-window stability)
    "gini_prefill_requests": (SEC, "gini", "worse"),
    "gini_decode_requests": (SEC, "gini", "worse"),
    "util_prefill_gini": (SEC, "gini", "worse"),
    "util_decode_gini": (SEC, "gini", "worse"),
    # utilization echo — informational only
    "util_prefill_min_pct": (INFO, "ratio", "neutral"),
    "util_prefill_avg_pct": (INFO, "ratio", "neutral"),
    "util_decode_avg_pct": (INFO, "ratio", "neutral"),
    # batch-size distribution shape counters (fixed_window_timeout etc.)
    "batch_predicted_execution_cap_avg": (SEC, "shape", "neutral"),
    "batch_predicted_execution_cap_p50": (SEC, "shape", "neutral"),
    "batch_batch_full_avg": (SEC, "shape", "neutral"),
    "batch_batch_full_p50": (SEC, "shape", "neutral"),
    "batch_fixed_window_timeout_avg": (SEC, "shape", "neutral"),
    "batch_fixed_window_timeout_p50": (SEC, "shape", "neutral"),
}
CRITICALITY.update(BALANCE_CRITICALITY)

# Precheck parameter keys: A/B is only meaningful when these match.
# (key, aggregate-meta key, run_meta params key, only-in-run-meta)
PRECHECK_PARAM_KEYS = [
    ("n_prefill", None, "n_prefill", True),
    ("n_decode", None, "n_decode", True),
    ("send_mode", "send_mode", "send_mode", False),
    ("replay_speed", "replay_speed", "replay_speed", False),
    ("send_mode_qps", "send_mode_qps", "send_mode_qps", False),
    ("ramp_up_seconds", "ramp_up_seconds", "ramp_up_seconds", False),
    ("duration_s", "duration_s", "duration_s", False),
]

TIER1_TITLE = "显著且关键（必须人工审）"
TIER2_TITLE = "显著但次关键（小基数/单点噪声，默认忽略）"
TIER3_TITLE = "关键但未见显著变化（证明未劣化）"
TIER4_TITLE = "信息类"


class PrecheckError(Exception):
    """A/B preconditions violated — not the same experiment."""


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def load_json(path):
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def resolve_run(path):
    """Accept a run dir (contains aggregate.json [+ run_meta.json]) or a
    direct path to an aggregate.json. Returns a dict with aggregate / meta /
    run_meta / label."""
    if os.path.isdir(path):
        agg_path = os.path.join(path, "aggregate.json")
        if not os.path.isfile(agg_path):
            raise PrecheckError(f"{path}: run dir has no aggregate.json")
        meta_path = os.path.join(path, "run_meta.json")
    elif os.path.isfile(path):
        agg_path = path
        meta_path = os.path.join(
            os.path.dirname(os.path.abspath(path)), "run_meta.json"
        )
    else:
        raise PrecheckError(f"{path}: no such file or directory")
    aggregate = load_json(agg_path)
    meta = aggregate.get("meta") or {}
    run_meta = load_json(meta_path) if os.path.isfile(meta_path) else None
    label = meta.get("run_dir") or os.path.basename(
        os.path.dirname(os.path.abspath(agg_path))
    )
    return {
        "path": os.path.abspath(path),
        "aggregate_path": os.path.abspath(agg_path),
        "aggregate": aggregate,
        "meta": meta,
        "run_meta": run_meta,
        "label": str(label),
    }


def get_param(run, agg_key, rm_key):
    """Parameter lookup: aggregate.meta first (run's own product), then
    run_meta.params. Returns None when unavailable in both."""
    if agg_key and run["meta"].get(agg_key) is not None:
        return run["meta"][agg_key]
    params = (run["run_meta"] or {}).get("params") or {}
    if rm_key and params.get(rm_key) is not None:
        return params[rm_key]
    return None


def get_flexlb_config(run):
    raw = run["meta"].get("flexlb_config")
    if raw is None:
        params = (run["run_meta"] or {}).get("params") or {}
        raw = params.get("flexlb_config")
    if raw is None:
        return None
    if isinstance(raw, str):
        try:
            return json.loads(raw)
        except (ValueError, TypeError):
            return {"_raw": raw}
    return raw


# ---------------------------------------------------------------------------
# Precheck — fail-closed A/B preconditions
# ---------------------------------------------------------------------------


def precheck(run_a, run_b):
    """Returns (details_dict, warnings_list). Raises PrecheckError on hard
    mismatch (different trace / different experiment parameters)."""
    errors, warnings, details = [], [], {}

    sha_a = run_a["meta"].get("trace_file_sha256")
    sha_b = run_b["meta"].get("trace_file_sha256")
    if sha_a is None or sha_b is None:
        errors.append(
            "trace_file_sha256 missing (A=%s, B=%s) — cannot prove same trace"
            % (sha_a, sha_b)
        )
        details["trace_file_sha256_match"] = None
    elif sha_a != sha_b:
        errors.append(
            "trace_file_sha256 mismatch: A=%s... B=%s..." % (sha_a[:16], sha_b[:16])
        )
        details["trace_file_sha256_match"] = False
    else:
        details["trace_file_sha256_match"] = True

    details["params"] = {}
    for name, agg_key, rm_key, _rm_only in PRECHECK_PARAM_KEYS:
        va, vb = get_param(run_a, agg_key, rm_key), get_param(run_b, agg_key, rm_key)
        if va is None and vb is None:
            warnings.append(
                f"param {name}: unavailable on both sides — assertion skipped"
            )
            details["params"][name] = {"a": None, "b": None, "match": None}
        elif va is None or vb is None:
            warnings.append(
                f"param {name}: available on one side only (A={va}, B={vb})"
            )
            details["params"][name] = {"a": va, "b": vb, "match": None}
        elif va != vb:
            errors.append(f"param {name} mismatch: A={va} B={vb}")
            details["params"][name] = {"a": va, "b": vb, "match": False}
        else:
            details["params"][name] = {"a": va, "b": vb, "match": True}

    cfg_a, cfg_b = get_flexlb_config(run_a), get_flexlb_config(run_b)
    if cfg_a is None or cfg_b is None:
        warnings.append("flexlb_config unavailable on at least one side — not asserted")
        details["flexlb_config_match"] = None
    elif cfg_a != cfg_b:
        # Legitimate config-contrast experiments exist — warn, don't fail.
        warnings.append(
            "flexlb_config differs between A and B — verify this is the intended variable"
        )
        details["flexlb_config_match"] = False
    else:
        details["flexlb_config_match"] = True

    if run_a["run_meta"] is None:
        warnings.append(
            "run A: run_meta.json not found — n_prefill/n_decode not asserted"
        )
    if run_b["run_meta"] is None:
        warnings.append(
            "run B: run_meta.json not found — n_prefill/n_decode not asserted"
        )

    details["errors"] = errors
    if errors:
        raise PrecheckError("A/B precheck failed:\n  - " + "\n  - ".join(errors))
    return details, warnings


# ---------------------------------------------------------------------------
# Steady window
# ---------------------------------------------------------------------------


def derive_steady_window(meta, lo_opt, hi_opt):
    duration = meta.get("duration_s")
    if lo_opt is not None and hi_opt is not None:
        return float(lo_opt), float(hi_opt), "explicit"
    if not isinstance(duration, (int, float)) or duration <= 0:
        raise PrecheckError(
            "meta.duration_s unavailable — pass --steady-lo/--steady-hi explicitly"
        )
    # 0.25 * duration clears ramp-up (default ramp is 30s of a 120s run);
    # 0.92 * duration stays ahead of the drain phase.
    return duration * 0.25, duration * 0.92, f"derived from duration_s={duration}"


def steady_mean(rows, lo, hi, keys):
    sel = [r for r in rows if lo <= r["t"] <= hi]
    if not sel:
        return None, 0
    out = {k: sum(r[k] for r in sel) / len(sel) for k in keys if k in sel[0]}
    return out, len(sel)


# ---------------------------------------------------------------------------
# Metric collection (field paths follow the validated 51-path inventory)
# ---------------------------------------------------------------------------


def collect_metrics(run_a, run_b, lo, hi):
    """Returns list of metric dicts: name, group, a, b (+ ctx for KV blocks)."""
    agg_a, agg_b = run_a["aggregate"], run_b["aggregate"]
    sa, sb = agg_a["summary"], agg_b["summary"]
    out = []
    _collect_validity_load(sa, sb, out)
    _collect_latency_families(sa, sb, out)
    _collect_cache_tps(sa, sb, agg_a, agg_b, lo, hi, out)
    _collect_kv(agg_a, agg_b, lo, hi, out)
    _collect_balance_batch(agg_a, agg_b, out)
    return out


def _add(rows, name, group, a, b, ctx=None):
    rows.append({"name": name, "group": group, "a": a, "b": b, "ctx": ctx})


def _collect_validity_load(sa, sb, out):
    """G1 validity + G2 load & errors."""
    _add(out, "test_valid", "validity", sa.get("test_valid"), sb.get("test_valid"))
    vc_a = sa.get("validity_checks") or {}
    vc_b = sb.get("validity_checks") or {}
    _add(
        out,
        "validity_checks_passed",
        "validity",
        sum(1 for v in vc_a.values() if v),
        sum(1 for v in vc_b.values() if v),
    )
    _add(
        out,
        "total_requests",
        "load",
        sa.get("total_requests"),
        sb.get("total_requests"),
    )
    _add(out, "error_count", "load", sa.get("error_count"), sb.get("error_count"))
    _add(out, "error_rate", "load", sa.get("error_rate"), sb.get("error_rate"))
    _add(
        out,
        "actual_send_qps",
        "load",
        sa.get("actual_send_qps"),
        sb.get("actual_send_qps"),
    )
    _add(
        out,
        "server_arrival_qps",
        "load",
        sa.get("server_arrival_qps"),
        sb.get("server_arrival_qps"),
    )


def _collect_latency_families(sa, sb, out):
    """G3 schedule percentiles + G4 e2e families + G5 stage latency."""
    sla, slb = sa.get("schedule_latency_ms") or {}, sb.get("schedule_latency_ms") or {}
    for k in ("p50", "p90", "p95", "p99", "mean", "max"):
        _add(out, f"schedule_latency_ms_{k}", "schedule", sla.get(k), slb.get(k))

    for fam, label in (
        ("full_e2e_latency_ms", "full_e2e"),
        ("e2e_latency_ms", "e2e"),
        ("ttft_latency_ms", "ttft"),
    ):
        fa, fb = sa.get(fam) or {}, sb.get(fam) or {}
        keys = (
            ("p50", "p95", "p99", "count")
            if label == "full_e2e"
            else ("p50", "p90", "p95", "p99", "mean", "max", "count")
        )
        for k in keys:
            _add(out, f"{fam}_{k}", label, fa.get(k), fb.get(k))

    sta = sa.get("server_stage_latency_ms") or {}
    stb = sb.get("server_stage_latency_ms") or {}
    for stage in ("batch_wait_ms", "route_submit_ms", "dispatch_ack_ms"):
        va, vb = sta.get(stage) or {}, stb.get(stage) or {}
        for k in ("p50", "p90", "p95", "p99", "mean", "max"):
            _add(out, f"{stage}_{k}", "stage", va.get(k), vb.get(k))


def _collect_cache_tps(sa, sb, agg_a, agg_b, lo, hi, out):
    """G6 cache & token throughput + G7 mock self-reported TPS (steady)."""
    cha = sa.get("cache_hit_summary") or {}
    chb = sb.get("cache_hit_summary") or {}
    for k in ("master_routing_hit_pct", "engine_key_hit_pct", "engine_token_hit_pct"):
        _add(out, f"cache_hit_{k}", "cache", cha.get(k), chb.get(k))
    _add(
        out,
        "cache_saved_tokens",
        "cache",
        sa.get("cache_saved_tokens"),
        sb.get("cache_saved_tokens"),
    )
    _add(
        out,
        "input_token_tps",
        "cache",
        sa.get("input_token_tps"),
        sb.get("input_token_tps"),
    )
    _add(
        out,
        "output_token_tps",
        "cache",
        sa.get("output_token_tps"),
        sb.get("output_token_tps"),
    )

    keys = ("context_tps", "context_tps_with_cache", "generate_tps")
    mta, _ = steady_mean(agg_a.get("mock_tps_ts") or [], lo, hi, keys)
    mtb, _ = steady_mean(agg_b.get("mock_tps_ts") or [], lo, hi, keys)
    for k in keys:
        va = round(mta[k], 1) if mta and k in mta else None
        vb = round(mtb[k], 1) if mtb and k in mtb else None
        _add(out, f"mock_tps_steady_{k}", "tps", va, vb)


def _collect_kv(agg_a, agg_b, lo, hi, out):
    """G8 KV block pool per role: final / cumulative / steady levels."""
    for role in ("prefill", "decode"):
        ka_series = (agg_a.get("kv_blocks_ts_by_role") or {}).get(role) or []
        kb_series = (agg_b.get("kv_blocks_ts_by_role") or {}).get(role) or []
        ka = ka_series[-1] if ka_series else {}
        kb = kb_series[-1] if kb_series else {}
        total_ctx = ka.get("total_blocks")
        _add(
            out,
            f"kv_{role}_total_blocks",
            "kv",
            ka.get("total_blocks"),
            kb.get("total_blocks"),
        )
        for k in ("held_blocks", "referenced_blocks", "available_blocks"):
            _add(out, f"kv_{role}_{k}_final", "kv", ka.get(k), kb.get(k), ctx=total_ctx)
        for k in (
            "cache_evictions",
            "kv_admission_fails",
            "lack_mem_rejects",
            "decode_reuse_blocks",
        ):
            _add(out, f"kv_{role}_{k}_cum", "kv", ka.get(k), kb.get(k), ctx=total_ctx)
        keys = ("held_blocks", "referenced_blocks", "available_blocks", "total_blocks")
        sta_, _ = steady_mean(ka_series, lo, hi, keys)
        stb_, _ = steady_mean(kb_series, lo, hi, keys)
        for k in ("held_blocks", "referenced_blocks"):
            va = round(sta_[k], 1) if sta_ and k in sta_ else None
            vb = round(stb_[k], 1) if stb_ and k in stb_ else None
            _add(out, f"kv_{role}_{k}_steady", "kv", va, vb, ctx=total_ctx)


def _collect_balance_batch(agg_a, agg_b, out):
    """G9 balance (gini / utilization echo) + batch distribution shape."""
    ed_a, ed_b = agg_a.get("engine_dist") or {}, agg_b.get("engine_dist") or {}
    _add(
        out,
        "gini_prefill_requests",
        "balance",
        (ed_a.get("prefill") or {}).get("gini_cum"),
        (ed_b.get("prefill") or {}).get("gini_cum"),
    )
    _add(
        out,
        "gini_decode_requests",
        "balance",
        (ed_a.get("decode") or {}).get("gini_cum"),
        (ed_b.get("decode") or {}).get("gini_cum"),
    )
    ut_a = ed_a.get("utilization") or {}
    ut_b = ed_b.get("utilization") or {}
    _add(
        out,
        "util_prefill_gini",
        "balance",
        (ut_a.get("prefill") or {}).get("gini_cum"),
        (ut_b.get("prefill") or {}).get("gini_cum"),
    )
    _add(
        out,
        "util_decode_gini",
        "balance",
        (ut_a.get("decode") or {}).get("gini_cum"),
        (ut_b.get("decode") or {}).get("gini_cum"),
    )

    def _last(x):
        return x[-1] if isinstance(x, list) and x else x

    def _avg(x):
        return round(sum(x) / len(x), 2) if isinstance(x, list) and x else x

    _add(
        out,
        "util_prefill_min_pct",
        "balance",
        _last((ut_a.get("prefill") or {}).get("per_engine_pct")),
        _last((ut_b.get("prefill") or {}).get("per_engine_pct")),
    )
    _add(
        out,
        "util_prefill_avg_pct",
        "balance",
        _avg((ut_a.get("prefill") or {}).get("per_engine_pct")),
        _avg((ut_b.get("prefill") or {}).get("per_engine_pct")),
    )
    _add(
        out,
        "util_decode_avg_pct",
        "balance",
        _avg((ut_a.get("decode") or {}).get("per_engine_pct")),
        _avg((ut_b.get("decode") or {}).get("per_engine_pct")),
    )

    bs_a, bs_b = (
        agg_a.get("batch_size_final") or {},
        agg_b.get("batch_size_final") or {},
    )
    for kind in ("predicted_execution_cap", "batch_full", "fixed_window_timeout"):
        for k in ("avg", "p50"):
            _add(
                out,
                f"batch_{kind}_{k}",
                "batch",
                (bs_a.get(kind) or {}).get(k),
                (bs_b.get(kind) or {}).get(k),
            )


# ---------------------------------------------------------------------------
# Diff & significance
# ---------------------------------------------------------------------------


def compute_diff(metric, noise_floor_pct):
    """Fills abs_diff / rel_diff_pct / rel_significant / abs_significant /
    significant / tier / note onto the metric dict (in place)."""
    crit, cat, _direction = CRITICALITY.get(metric["name"], (INFO, "ratio", "neutral"))
    metric["criticality"], metric["category"] = crit, cat
    a, b = metric["a"], metric["b"]
    metric["abs_diff"] = None
    metric["rel_diff_pct"] = None

    if isinstance(a, bool) or isinstance(b, bool) or cat == "bool":
        metric["rel_significant"] = metric["abs_significant"] = metric[
            "significant"
        ] = (a != b)
        metric["abs_diff"] = None if a == b else "bool-mismatch"
        _assign_tier(metric)
        return

    if not _is_numeric_pair(a, b):
        metric["rel_significant"] = metric["abs_significant"] = metric[
            "significant"
        ] = False
        _assign_tier(metric)
        return

    abs_diff = b - a  # positive => candidate (B) higher than baseline (A)
    metric["abs_diff"] = abs_diff
    if a == b:
        metric["rel_diff_pct"] = 0.0
    elif a == 0:
        metric["rel_diff_pct"] = None  # undefined: baseline A == 0, B != 0
    else:
        metric["rel_diff_pct"] = abs_diff / a * 100.0

    kind, rel_thr = CATEGORY_RULES[cat]
    if kind == "rel_pp":
        # ratio class judged in percentage points directly
        rel_ok = abs(abs_diff) > max(rel_thr, noise_floor_pct)
        abs_ok = abs(abs_diff) > RATIO_ABS_PP
    else:
        rel_ok = _rel_guard_ok(
            cat, metric["rel_diff_pct"], a, b, max(rel_thr, noise_floor_pct)
        )
        abs_ok = _abs_guard_ok(cat, abs_diff, a, b, noise_floor_pct, metric.get("ctx"))

    metric["rel_significant"] = rel_ok
    metric["abs_significant"] = abs_ok
    metric["significant"] = bool(rel_ok and abs_ok)
    _assign_tier(metric)


def _is_numeric_pair(a, b):
    return isinstance(a, (int, float)) and isinstance(b, (int, float))


def _rel_guard_ok(cat, rel, a, b, eff_thr):
    if rel is None:
        # baseline == 0 while candidate != 0: unbounded relative change — the
        # strongest regression shape there is (0 -> nonzero), never
        # "insignificant".
        return a != b
    return abs(rel) > eff_thr


def _abs_guard_ok(cat, abs_diff, a, b, noise_floor_pct, ctx):
    """Small-baseline guard: only an absolute delta BEYOND the category
    floor counts as significant (strictly greater-than)."""
    if cat == "latency":
        return abs(abs_diff) > LATENCY_ABS_MS
    if cat == "rate":
        return abs(abs_diff) > noise_floor_pct / 100.0 * abs(a)
    if cat == "count":
        return abs(abs_diff) > COUNT_ABS_FRAC * abs(a)
    if cat == "gini":
        return abs(abs_diff) > GINI_ABS
    if cat == "shape":
        return abs(abs_diff) > SHAPE_ABS_UNITS
    if cat == "blocks":
        total = ctx if isinstance(ctx, (int, float)) else abs(a)
        return abs(abs_diff) > BLOCKS_ABS_FRAC * abs(total)
    return True


def _assign_tier(metric):
    if metric["criticality"] == CRIT:
        metric["tier"] = 1 if metric["significant"] else 3
    elif metric["criticality"] == SEC:
        # relative-significant alone lands in Tier 2 — that tier exists to
        # absorb exactly those relative-only small-baseline alarms.
        metric["tier"] = (
            2 if (metric["rel_significant"] or metric["significant"]) else 4
        )
    else:
        metric["tier"] = 4
    metric["note"] = _note(metric)


def _note(metric):
    if metric["tier"] != 2:
        if (
            metric["tier"] == 3
            and metric["rel_significant"]
            and not metric["abs_significant"]
        ):
            return "[小基数: abs未超保护]"
        return ""
    name = metric["name"]
    if name.endswith("_max"):
        return "[单次离群]"
    if metric["rel_significant"] and not metric["abs_significant"]:
        cat = metric["category"]
        if cat == "gini":
            return "[小数放大]"
        if cat == "shape":
            return "[批大小单位级]"
        if cat == "blocks":
            return "[块级噪声]"
        return "[小基数: abs<保护线]"
    return "[形态]"


def _direction_label(metric):
    """Direction label from the CANDIDATE's viewpoint (B vs baseline A):
    "← 劣化" means the candidate regressed relative to the baseline."""
    a, b = metric["a"], metric["b"]
    direction = CRITICALITY.get(metric["name"], (INFO, "ratio", "neutral"))[2]
    if isinstance(a, bool) and isinstance(b, bool):
        if a == b:
            return ""
        if direction == "neutral":
            return "A=True" if a else "B=True"
        b_degrades = (not b) if direction == "better" else b
        return "← 劣化" if b_degrades else "← 改善"
    if not isinstance(a, (int, float)) or not isinstance(b, (int, float)) or a == b:
        return ""
    if direction == "neutral":
        return "A高" if a > b else "B高"
    b_degrades = (b > a) if direction == "worse" else (b < a)
    return "← 劣化" if b_degrades else "← 改善"


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def _fmt_latency(v):
    if abs(v) < 1:
        return f"{v:.3f}ms"
    if abs(v) < 100:
        return f"{v:.1f}ms"
    return f"{v:,.0f}ms"


def _fmt_count(v):
    if abs(v) >= 1000:
        return f"{v:,.0f}"
    if isinstance(v, float):
        return f"{v:.1f}"
    return f"{v:,}"


def fmt_val(metric, side):
    v = metric[side]
    if isinstance(v, bool):
        return str(v)
    if v is None:
        return "N/A"
    if not isinstance(v, (int, float)):
        return str(v)
    cat = metric["category"]
    if cat == "latency":
        return _fmt_latency(v)
    if cat == "rate":
        return f"{v:,.3f}" if "qps" in metric["name"] else f"{v:,.0f}"
    if cat == "ratio":
        return f"{v:.2f}%"
    if cat == "gini":
        return f"{v:.4f}"
    if cat == "shape":
        return f"{v:.2f}"
    if cat == "count":
        return _fmt_count(v)
    if cat == "blocks":
        return f"{v:,.1f}" if isinstance(v, float) else f"{v:,}"
    return str(v)


def fmt_abs(metric):
    d = metric["abs_diff"]
    if d is None or isinstance(d, str):
        return ""
    cat = metric["category"]
    if cat == "latency":
        return f"{d:+.1f}ms"
    if cat == "ratio":
        return f"{d:+.2f}pp"
    if cat == "gini":
        return f"{d:+.4f}"
    if cat == "shape":
        return f"{d:+.2f}"
    if cat in ("count", "blocks"):
        return f"{d:+,.0f}"
    if cat == "rate":
        return f"{d:+,.3f}"
    return f"{d:+}"


def fmt_rel(metric):
    r = metric["rel_diff_pct"]
    if r is None:
        return "∞" if metric["a"] != metric["b"] else "0"
    return f"{r:+.2f}%"


def render_stdout(payload):
    lines = []
    run_a, run_b = payload["run_a"], payload["run_b"]
    lines.append(f"A/B 差分回归门  A={run_a['label']}  B={run_b['label']}")
    win = payload["steady_window"]
    lines.append(
        f"稳态窗口 [{win['lo_s']:.1f}s, {win['hi_s']:.1f}s]（{win['source']}；"
        f"TPS 采样 {win['rows_used']} 行）"
    )
    for w in payload["precheck_warnings"]:
        lines.append(f"WARNING: {w}")
    lines.append("")

    tiers = payload["tiers"]
    titles = {1: TIER1_TITLE, 2: TIER2_TITLE, 3: TIER3_TITLE, 4: TIER4_TITLE}
    for tier in (1, 2, 3, 4):
        rows = tiers[tier]
        lines.append(f"━━━ {titles[tier]} ━━━")
        if not rows:
            lines.append("  (无)")
        for m in rows:
            fa, fb = fmt_val(m, "a"), fmt_val(m, "b")
            rel = fmt_rel(m)
            abs_s = fmt_abs(m)
            abs_part = f" (abs {abs_s})" if abs_s else ""
            note = f"  {m['note']}" if m["note"] else ""
            dirn = _direction_label(m)
            # Tier 3 hides the arrow by design (spec format): the section
            # header already asserts "not regressed"; an arrow would read as
            # a contradiction. Full direction stays in the JSON output.
            dir_part = f"  {dirn}" if dirn and tier != 3 else ""
            if tier == 3 and not note:
                note = "  ✓ 噪声内"
            lines.append(
                f"  {m['name']:<42} A={fa:<14} B={fb:<14} {rel:>10}{abs_part}{note}{dir_part}"
            )
        lines.append("")

    summary = payload["classification_summary"]
    gate = payload["gate"]
    if gate["passed"]:
        gate_text = "PASS (exit 0)"
    else:
        gate_text = "TRIPPED (exit 1) — 显著且关键区非空，必须人工审"
    lines.append(
        f"判定: 显著且关键 {summary['significant_critical']} 项 | "
        f"显著但次关键 {summary['significant_secondary']} 项 | "
        f"关键未劣化 {summary['critical_unchanged']} 项 | "
        f"信息类 {summary['info']} 项"
    )
    lines.append(f"回归门: {gate_text}")
    return "\n".join(lines)


_HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="utf-8">
<title>A/B regression gate — {label_a} vs {label_b}</title>
<style>
body {{ font-family: 'SF Mono', Menlo, Consolas, monospace; margin: 24px auto;
       max-width: 1080px; color: #1a1a1a; background: #fafafa; }}
h1 {{ font-size: 18px; }}
.meta {{ color: #666; font-size: 13px; margin-bottom: 16px; }}
table {{ border-collapse: collapse; width: 100%; margin-bottom: 28px; font-size: 13px; }}
th, td {{ border: 1px solid #ddd; padding: 5px 10px; text-align: right; }}
th {{ background: #f0f0f0; }}
td.name {{ text-align: left; font-family: inherit; }}
h2.t1 {{ color: #c0392b; border-bottom: 2px solid #c0392b; }}
h2.t2 {{ color: #b8860b; border-bottom: 2px solid #b8860b; }}
h2.t3 {{ color: #1e7e34; border-bottom: 2px solid #1e7e34; }}
h2.t4 {{ color: #666; border-bottom: 2px solid #999; }}
.gate-pass {{ color: #1e7e34; font-weight: bold; }}
.gate-trip {{ color: #c0392b; font-weight: bold; }}
.worse {{ color: #c0392b; }}
.better {{ color: #1e7e34; }}
</style>
</head>
<body>
<h1>A/B 差分回归门</h1>
<div class="meta">A = {label_a} &nbsp;|&nbsp; B = {label_b} &nbsp;|&nbsp;
稳态窗口 [{lo:.1f}s, {hi:.1f}s]（{src}） &nbsp;|&nbsp;
噪声地板 {floor:.1%}</div>
{warnings}
{sections}
<p>回归门: <span class="{gate_cls}">{gate_text}</span></p>
</body>
</html>
"""


def render_html(payload):
    tier_titles = {1: TIER1_TITLE, 2: TIER2_TITLE, 3: TIER3_TITLE, 4: TIER4_TITLE}
    sections = []
    for tier in (1, 2, 3, 4):
        rows = payload["tiers"][tier]
        parts = [
            f'<h2 class="t{tier}">Tier {tier} · {tier_titles[tier]}（{len(rows)}）</h2>'
        ]
        parts.append(
            '<table><tr><th class="name">metric</th><th>A</th><th>B</th>'
            "<th>rel diff</th><th>abs diff</th><th>note</th><th>方向</th></tr>"
        )
        for m in rows:
            d = _direction_label(m)
            cls = "worse" if "劣化" in d else ("better" if "改善" in d else "")
            note = m.get("note", "").replace("[", "").replace("]", "")
            parts.append(
                '<tr><td class="name">%s</td><td>%s</td><td>%s</td><td>%s</td><td>%s</td>'
                '<td>%s</td><td class="%s">%s</td></tr>'
                % (
                    m["name"],
                    fmt_val(m, "a"),
                    fmt_val(m, "b"),
                    fmt_rel(m),
                    fmt_abs(m) or "—",
                    note,
                    cls,
                    d,
                )
            )
        parts.append("</table>")
        sections.append("\n".join(parts))
    warn_html = ""
    if payload["precheck_warnings"]:
        items = "".join(f"<li>{w}</li>" for w in payload["precheck_warnings"])
        warn_html = f'<ul style="color:#b8860b">{items}</ul>'
    passed = payload["gate"]["passed"]
    return _HTML_TEMPLATE.format(
        label_a=payload["run_a"]["label"],
        label_b=payload["run_b"]["label"],
        lo=payload["steady_window"]["lo_s"],
        hi=payload["steady_window"]["hi_s"],
        src=payload["steady_window"]["source"],
        floor=payload["noise_floor"],
        warnings=warn_html,
        sections="\n".join(sections),
        gate_cls="gate-pass" if passed else "gate-trip",
        gate_text=(
            "PASS — 显著且关键区为空"
            if passed
            else "TRIPPED — 显著且关键区非空，必须人工审"
        ),
    )


def build_payload(
    run_a, run_b, precheck_details, warnings, window, metrics, noise_floor
):
    tiers = {1: [], 2: [], 3: [], 4: []}
    for m in metrics:
        tiers[m["tier"]].append(m)
    passed = not tiers[1]
    return {
        "tool": "compare_ab",
        "run_a": {"label": run_a["label"], "path": run_a["path"]},
        "run_b": {"label": run_b["label"], "path": run_b["path"]},
        "precheck": precheck_details,
        "precheck_warnings": warnings,
        "steady_window": window,
        "noise_floor": noise_floor,
        "classification_summary": {
            "significant_critical": len(tiers[1]),
            "significant_secondary": len(tiers[2]),
            "critical_unchanged": len(tiers[3]),
            "info": len(tiers[4]),
        },
        "gate": {"passed": passed, "exit_code": 0 if passed else 1},
        "tiers": tiers,
    }


def metrics_to_json(payload):
    """Full-fidelity JSON view (all metrics, raw numbers) for downstream."""
    doc = {k: v for k, v in payload.items() if k != "tiers"}
    doc["metrics"] = [
        {
            "name": m["name"],
            "group": m["group"],
            "a": m["a"],
            "b": m["b"],
            "abs_diff": m["abs_diff"],
            "rel_diff_pct": m["rel_diff_pct"],
            "category": m["category"],
            "criticality": m["criticality"],
            "rel_significant": m["rel_significant"],
            "abs_significant": m["abs_significant"],
            "significant": m["significant"],
            "tier": m["tier"],
            "note": m.get("note", ""),
            "direction": _direction_label(m),
        }
        for m in sorted(
            (m for rows in payload["tiers"].values() for m in rows),
            key=lambda m: (m["tier"], m["name"]),
        )
    ]
    return doc


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args(argv=None):
    ap = argparse.ArgumentParser(
        description="A/B differential regression gate: every metric compared, "
        "three-tier classification (significant+critical / "
        "significant+secondary / critical+unchanged)."
    )
    ap.add_argument(
        "--run-a", required=True, help="baseline run dir or aggregate.json path"
    )
    ap.add_argument(
        "--run-b", required=True, help="candidate run dir or aggregate.json path"
    )
    ap.add_argument(
        "--steady-lo",
        type=float,
        default=None,
        help="steady window lower bound in seconds " "(default: 0.25 * duration_s)",
    )
    ap.add_argument(
        "--steady-hi",
        type=float,
        default=None,
        help="steady window upper bound in seconds " "(default: 0.92 * duration_s)",
    )
    ap.add_argument(
        "--out",
        default="ab_summary.json",
        help="JSON summary path ('-' for stdout-only, default " "ab_summary.json)",
    )
    ap.add_argument(
        "--html",
        action="store_true",
        help="also emit a self-contained ab_compare.html table",
    )
    ap.add_argument(
        "--noise-floor",
        type=float,
        default=DEFAULT_NOISE_FLOOR,
        help="same-code noise floor as a fraction (default 0.02); "
        "raises every category's relative threshold",
    )
    return ap.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    try:
        run_a = resolve_run(args.run_a)
        run_b = resolve_run(args.run_b)
        precheck_details, warnings = precheck(run_a, run_b)
        lo, hi, src = derive_steady_window(
            run_a["meta"], args.steady_lo, args.steady_hi
        )
    except PrecheckError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    metrics = collect_metrics(run_a, run_b, lo, hi)
    floor_pct = args.noise_floor * 100.0
    for m in metrics:
        compute_diff(m, floor_pct)

    steady_rows = len(
        [r for r in run_a["aggregate"].get("mock_tps_ts") or [] if lo <= r["t"] <= hi]
    )
    window = {"lo_s": lo, "hi_s": hi, "source": src, "rows_used": steady_rows}
    payload = build_payload(
        run_a, run_b, precheck_details, warnings, window, metrics, args.noise_floor
    )

    print(render_stdout(payload))

    if args.out != "-":
        out_dir = os.path.dirname(os.path.abspath(args.out))
        os.makedirs(out_dir, exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(metrics_to_json(payload), fh, indent=2, ensure_ascii=False)
        print(f"\nJSON summary -> {args.out}")
    if args.html:
        html_path = (
            "ab_compare.html"
            if args.out == "-"
            else os.path.join(
                os.path.dirname(os.path.abspath(args.out)), "ab_compare.html"
            )
        )
        os.makedirs(os.path.dirname(os.path.abspath(html_path)), exist_ok=True)
        with open(html_path, "w", encoding="utf-8") as fh:
            fh.write(render_html(payload))
        print(f"HTML report  -> {html_path}")

    return payload["gate"]["exit_code"]


if __name__ == "__main__":
    sys.exit(main())
