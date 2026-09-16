#!/usr/bin/env python3
"""compare_twin.py — L3 twin replay distribution comparison (mock vs real).

L3 = twin replay: the mock line and the real line each replay the SAME trace
on the SAME topology independently; this script compares DISTRIBUTIONS (not
trajectories) and answers "does the mock engine reproduce the real engine's
observable behavior well enough for scheduler-level conclusions?"

Five layers:

  1. Loading   — heterogeneous inputs normalized into one SideData:
                * mock side : aggregate.json (+ same-dir client_events.jsonl
                              when present; per-request families fall back to
                              the summary percentile quantile-approx expansion
                              when the raw rows were not kept).
                * real side : client_events.jsonl (JavaLoadClient
                              perRequestNode field names) + optional
                              prom_export.jsonl (1s metrics, long format
                              {t, metric, value} and wide format
                              {t, "<metric>": v} both auto-sniffed).
  2. Noise floor — with >=3 mock runs (--mock-runs) every comparable metric's
                pairwise distance median becomes the measured floor; with
                fewer runs the empirical defaults (compare_ab.py category
                thresholds) apply, flagged "经验地板非实测".
  3. Compare   — twelve metrics, distance-first (pure stdlib):
                Wasserstein-1 for TTFT/e2e/schedule-latency distributions
                (TTFT unified to the engine caliber on 20260903: 发出 →
                prefill 批完成 via the rid join; the client first-frame
                ttft_ms semantics is retired),
                KS statistic for batch-size and KV-level distributions,
                share-diff (pp) for dispatch reasons / cache hit / success /
                error rates, relative diff for TPS, gini reserved behind a
                >=2P2D gate (N/A on 1P1D).
  4. Verdict   — per metric ALIGNED (<2x floor) / DEVIATED (2-5x) /
                DIVERGED (>=5x); SKIP (data source missing, with the reason)
                / N/A (structurally inapplicable).
  5. Attribution — rule table over the verdict vector localizes likely
                causes for exceeded metrics (time-model bias, decode tail,
                batch-cut behavior, prefix/KV semantics, accounting caliber).

Exit codes: 0 = no DIVERGED metric; 1 = at least one DIVERGED; 2 = input
error (unreadable inputs / missing mandatory side).

Usage:
  python3 compare_twin.py \
      --mock-aggregate <mock run>/aggregate.json \
      [--mock-runs run1,run2,run3] \
      --real-client-events <real>/client_events.jsonl \
      [--real-prom <real>/prom_export.jsonl] \
      [--out twin] [--steady-lo 30 --steady-hi 110] [--no-html]

Outputs (prefix defaults to "twin"):
  <prefix>_summary.json — full raw numbers
  <prefix>_report.html  — self-contained colored table
  stdout               — aligned verdict table
"""
from __future__ import annotations

import argparse
import gzip
import json
import math
import os
import sys

# ---------------------------------------------------------------------------
# Constants: verdict multipliers, empirical floors, guards.
#
# Provenance: multipliers and empirical floors reuse compare_ab.py's triage
# calibration — LATENCY (5% rel / 5ms abs), RATE (2%), RATIO (3pp / 1pp),
# SHAPE (10%) — expressed here in each metric's own distance unit. Measured
# floors from --mock-runs pairwise distances supersede them; measured floors
# of exactly 0.0 fall back to the small guard minimum so a single duplicate
# run pair cannot make every deviation infinite.
# ---------------------------------------------------------------------------

ALIGNED_MAX_MULT = 2.0  # distance < 2x floor          -> ALIGNED
DIVERGED_MIN_MULT = 5.0  # distance >= 5x floor        -> DIVERGED

# Empirical (fallback) floors per distance kind; "measured" kind comes from
# --mock-runs pairwise medians. Kind -> (empirical floor fn, guard minimum).
FLOOR_GUARD_MIN = {
    "w1_ms": 1.0,  # Wasserstein in milliseconds
    "ks": 0.01,  # KS statistic (dimensionless)
    "pp": 0.2,  # percentage points
    "rel_pct": 0.5,  # relative percent
    "gini": 0.002,  # gini coefficient absolute
}

DISPATCH_REASONS = ("predicted_execution_cap", "batch_full", "fixed_window_timeout")

# Real-side prometheus metric name candidates per internal series key. The
# matcher tries (a) exact name, (b) suffix match after stripping Micrometer
# timer suffixes (_count/_sum/_max/_total), in order; wide-format column
# names and long-format metric names go through the same table.
PROM_SERIES_CANDIDATES = {
    "context_tps": ("context_wall_tps", "rtp_llm_context_tps", "context_tps"),
    "context_tps_with_cache": (
        "context_wall_tps_with_cache",
        "rtp_llm_context_tps_with_cache",
        "context_tps_with_cache",
    ),
    "generate_tps": ("generate_wall_tps", "rtp_llm_generate_tps", "generate_tps"),
    "cache_hit_token_pct": (
        "engine_token_hit_pct",
        "cache_hit_token_pct",
        "token_hit_ratio",
        "recent_cache_hit_ratio",
    ),
    "cache_hit_key_pct": (
        "engine_key_hit_pct",
        "cache_hit_key_pct",
        "recent_cache_key_hit_ratio",
    ),
    "kv_used_pct": (
        "kv_cache_used_pct",
        "flexlb_app_cache_used_pct",
        "kv_used_pct",
        "cache_used_pct",
    ),
}

# counter pairs -> derived ratio series (pct), used when no direct ratio
# metric exists: pct = 100 * delta(numerator) / delta(denominator).
PROM_RATIO_PAIRS = {
    "cache_hit_token_pct": (
        ("engine_hit_tokens", "cache_hit_tokens", "hit_tokens"),
        ("engine_input_tokens", "cache_input_tokens", "input_tokens"),
    ),
    "cache_hit_key_pct": (
        ("cache_key_hits", "recent_cache_key_hit_count", "key_hits"),
        ("cache_keys_requested", "recent_cache_total_count", "keys_requested"),
    ),
    "kv_used_pct": (
        ("kv_used_tokens", "used_tokens"),
        ("kv_capacity_tokens", "capacity_tokens", "total_tokens"),
    ),
}

# Long-format same-name same-t multi-row aggregation: counters sum (per-
# instance counters add up to the cluster value), gauges average (matches the
# mock side's per-engine avg for master batch_size). Kind "gauge" defaults to
# avg, everything else to sum.
GAUGE_METRIC_HINTS = ("batch_size",)

# master-side schedule latency (supplementary caliber, never gates): the
# auto_tpm.schedule.latency_ms timer family.
PROM_SCHEDULE_NAMES = (
    "auto_tpm_schedule_latency_ms",
    "auto_tpm.ttft",
    "auto_tpm_ttft",
)

# per-request row field aliases -> internal name (JavaLoadClient
# perRequestNode is canonical; tolerate the odd historical variant).
PER_REQUEST_ALIASES = {
    "rid": ("rid", "request_id_str", "source_rid"),
    "ts": ("ts",),
    "send_start": ("send_start_epoch_ms", "send_start_ms", "send_start"),
    "sched_done": ("sched_done_epoch_ms", "sched_done_ms", "sched_done"),
    "ttft_ms": ("ttft_ms", "ttftMs"),
    # engine 口径 ttft（20260903 统一）：加载层 rid join 引擎 prefill
    # 终态行注入（见 _inject_ttft_engine_ms）；合成 round-trip 行自带。
    "ttft_engine_ms": ("ttft_engine_ms",),
    "total_ms": ("total_ms", "totalMs"),
    "input_len": ("input_len", "il"),
    "output_len": ("output_len", "ol"),
    "status": ("status",),
    "priority": ("priority",),
    "schedule_ms": ("schedule_ms", "scheduleMs"),
    "error": ("error",),
    "prefill": ("prefill",),
    "decode": ("decode",),
}

VERDICT_ALIGNED = "ALIGNED"
VERDICT_DEVIATED = "DEVIATED"
VERDICT_DIVERGED = "DIVERGED"
VERDICT_SKIP = "SKIP"
VERDICT_NA = "N/A"


class InputError(Exception):
    """Bad CLI inputs — unreadable file, missing mandatory side, etc."""


# ---------------------------------------------------------------------------
# Numeric utilities (pure stdlib)
# ---------------------------------------------------------------------------


def _num(v):
    return v if isinstance(v, (int, float)) and not isinstance(v, bool) else None


def percentile_nr(values, p, nd=1):
    """Nearest-rank percentile, same rule as aggregate_canvas_run.latency_summary."""
    if not values:
        return 0.0
    s = sorted(values)
    k = max(0, min(len(s) - 1, math.ceil(p * len(s)) - 1))
    return round(float(s[k]), nd)


def wasserstein_1d(a, b):
    """Wasserstein-1 distance between two 1-D empirical distributions.

    Sorted merge walk: the ECDF difference integrates over the merged
    support. Returns None when either side has no samples.
    """
    if not a or not b:
        return None
    a = sorted(float(x) for x in a)
    b = sorted(float(x) for x in b)
    n, m = len(a), len(b)
    area = 0.0
    ai = bi = 0
    prev = None
    for x in sorted(a + b):
        if prev is not None:
            area += abs(ai / n - bi / m) * (x - prev)
        while ai < n and a[ai] == x:
            ai += 1
        while bi < m and b[bi] == x:
            bi += 1
        prev = x
    return area


def ks_statistic(a, b):
    """Two-sample Kolmogorov-Smirnov statistic (max ECDF gap)."""
    if not a or not b:
        return None
    a = sorted(float(x) for x in a)
    b = sorted(float(x) for x in b)
    n, m = len(a), len(b)
    best = 0.0
    ai = bi = 0
    for x in sorted(set(a) | set(b)):
        while ai < n and a[ai] == x:
            ai += 1
        while bi < m and b[bi] == x:
            bi += 1
        best = max(best, abs(ai / n - bi / m))
    return best


def gini_coef(values):
    """Gini coefficient of a count/amount vector (0 = perfectly even)."""
    vals = sorted(float(v) for v in values if _num(v) is not None)
    n = len(vals)
    if n == 0 or sum(vals) <= 0:
        return None
    cum = 0.0
    total = 0.0
    for i, v in enumerate(vals, 1):
        cum += v
        total += i * v
    return (2.0 * total) / (n * cum) - (n + 1.0) / n


def quantile_expand(summary, n_samples=1000):
    """Expand a latency_summary dict (p50/p90/p95/p99/max, count) into an
    approximate empirical sample list via piecewise-linear quantile inverse.

    Below p50 the curve extrapolates linearly from p50/2 to p50 (no lower
    quantiles are available in the summary); the approximation is flagged by
    the caller as mode="quantile-approx". Returns [] when count == 0 or the
    family is degenerate (max <= 0).
    """
    if not isinstance(summary, dict):
        return []
    if not summary.get("count") or _num(summary.get("count")) == 0:
        return []
    pts = []
    p50 = _num(summary.get("p50"))
    if p50 is not None and p50 > 0:
        pts.append((0.0, p50 * 0.5))
        pts.append((0.5, p50))
    for q, k in ((0.90, "p90"), (0.95, "p95"), (0.99, "p99"), (1.0, "max")):
        v = _num(summary.get(k))
        if v is not None and v > 0:
            pts.append((q, v))
    if len(pts) < 2:
        return []
    # enforce monotone non-decreasing values
    for i in range(1, len(pts)):
        if pts[i][1] < pts[i - 1][1]:
            pts[i] = (pts[i][0], pts[i - 1][1])
    out = []
    for k in range(n_samples):
        u = (k + 0.5) / n_samples
        for i in range(1, len(pts)):
            if u <= pts[i][0] or i == len(pts) - 1:
                q0, v0 = pts[i - 1]
                q1, v1 = pts[i]
                frac = 0.0 if q1 == q0 else (u - q0) / (q1 - q0)
                frac = max(0.0, min(1.0, frac))
                out.append(v0 + frac * (v1 - v0))
                break
    return out


def steady_window_from(duration_s, t_max):
    """[lo, hi] steady window: explicit duration first, else series extent."""
    if isinstance(duration_s, (int, float)) and duration_s > 0:
        dur = float(duration_s)
    elif t_max is not None and t_max > 0:
        dur = float(t_max)
    else:
        return 0.0, 0.0
    return dur * 0.25, dur * 0.92


def series_values_in_window(series, lo, hi):
    """Values of a [(t, v)] series restricted to the steady window."""
    if not series:
        return []
    return [v for t, v in series if lo <= t <= hi]


def series_steady_mean(series, lo, hi):
    vals = series_values_in_window(series, lo, hi)
    return (sum(vals) / len(vals)) if vals else None


def series_steady_pct_mean(series, lo, hi):
    """Steady mean of a ratio series, expressed in percent; rows may carry
    fractions ([0,1]) or percents — the mean of mixed input is normalized by
    the observed max."""
    vals = series_values_in_window(series, lo, hi)
    if not vals:
        return None
    mean = sum(vals) / len(vals)
    if max(vals) <= 1.0:
        mean *= 100.0
    return mean


# ---------------------------------------------------------------------------
# SideData — the single normalized structure both sides reduce to
# ---------------------------------------------------------------------------


class SideData:
    """Normalized side: per-request rows + run summary + 1s series.

    * per_request: dicts with internal field names from PER_REQUEST_ALIASES
      (rid/ts/send_start/sched_done/ttft_ms/total_ms/input_len/output_len/
      status/priority/schedule_ms/error/prefill/decode) plus the
      engine-caliber ttft_engine_ms injected at load time (20260903).
    * summary: total/error_count/error_rate/success_rate + latency families
      ("ttft"/"e2e"/"schedule" in latency_summary shape; ttft is the engine
      caliber — 发出 → prefill 批完成 — since 20260903)
      + cache_hit_token_pct
      + gini_prefill/gini_decode (None when unavailable).
    * series: internal key -> [(t_seconds_relative, v)] sorted by t.
    * approx_modes: internal field -> "quantile-approx" when the mock side's
      per-request samples were expanded from summary percentiles.
    """

    def __init__(self, label, kind):
        self.label = label
        self.kind = kind  # "mock" | "real"
        self.source_path = None  # resolved input path (floor dedupe key)
        self.per_request = []
        self.per_request_source = "absent"
        self.summary = {}
        self.series = {}
        self.series_sources = {}
        self.approx_modes = {}
        self.duration_s = None
        self.n_prefill = None
        self.n_decode = None
        self.warnings = []

    # -- derived helpers ---------------------------------------------------

    def steady_window(self, lo_opt=None, hi_opt=None):
        if lo_opt is not None and hi_opt is not None:
            return float(lo_opt), float(hi_opt)
        t_max = None
        for series in self.series.values():
            if series:
                t_max = max(t_max or 0.0, series[-1][0])
        return steady_window_from(self.duration_s, t_max)

    def ok_rows(self):
        """Success rows, same predicate as aggregate_canvas_run.is_ok."""
        out = []
        for d in self.per_request:
            err = d.get("error") or ""
            if d.get("status") == "ok" or (
                not err and d.get("status") not in ("schedule_error",)
            ):
                out.append(d)
        return out

    def latency_samples(self, family):
        """Per-request samples for a latency family, aggregate-caliber:

        ttft: ENGINE caliber (20260903 unified) — the injected
        ttft_engine_ms rows (rid join against engine_events.jsonl,
        prefill_done_ms − send_start; the client first-frame ttft_ms is
        retired and no longer sampled); e2e: ok rows with value > 0;
        schedule: ok rows, 0 allowed (same rule as sched_client_samples).
        Empty when no raw rows exist — quantile-approx samples live in
        self.approx and are NOT returned here.
        """
        if not self.per_request:
            return []
        key = {
            "ttft": "ttft_engine_ms",
            "e2e": "total_ms",
            "schedule": "schedule_ms",
        }[family]
        vals = []
        for d in self.ok_rows():
            v = _num(d.get(key))
            if v is None:
                continue
            if family in ("ttft", "e2e") and v <= 0:
                continue
            vals.append(v)
        return vals

    def approx_samples(self, family):
        """quantile-approx expanded samples for the family (mock fallback)."""
        return self.approx_modes.get(family) or []

    def gini_from_rows(self):
        """Per-engine request gini from per-request prefill/decode addrs."""
        out = {}
        for role in ("prefill", "decode"):
            counts = {}
            for d in self.per_request:
                addr = d.get(role)
                if isinstance(addr, str) and addr:
                    counts[addr] = counts.get(addr, 0) + 1
            g = gini_coef(list(counts.values())) if len(counts) >= 2 else None
            out[role] = g
            out[f"{role}_engines"] = len(counts)
        return out


def _normalize_request_row(raw):
    """Map one client_events.jsonl row onto the internal field names."""
    row = {}
    for internal, aliases in PER_REQUEST_ALIASES.items():
        for alias in aliases:
            if alias in raw and raw[alias] is not None:
                row[internal] = raw[alias]
                break
    return row


def _inject_ttft_engine_ms(raw_rows, run_dir):
    """Inject the engine-caliber ttft onto RAW client rows (20260903).

    ttft_engine_ms = prefill_done_ms − send_start_epoch_ms via the rid join
    against same-dir engine_events.jsonl(.gz) — the same caliber the
    aggregate layer now emits as ttft_*/ttft_latency_ms (client first-frame
    ttft_ms is retired). cancelled terminal rows are skipped; join misses
    (no prefill_done row / unparsable request_id) and negative diffs (clock
    anomaly) leave the row WITHOUT the key — never fabricated. Rows that
    already carry ttft_engine_ms (the synthesize round-trip writes it back)
    keep theirs when no engine stream sits next to them.
    """
    ev_path = None
    for _name in ("engine_events.jsonl", "engine_events.jsonl.gz"):
        _p = os.path.join(run_dir, _name)
        if os.path.isfile(_p):
            ev_path = _p
            break
    pf_done = {}
    if ev_path:
        _opener = gzip.open if ev_path.endswith(".gz") else open
        with _opener(ev_path, "rt", encoding="utf-8", errors="replace") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    ev = json.loads(line)
                except ValueError:
                    continue
                if ev.get("cancelled") or ev.get("event") != "prefill_done":
                    continue
                try:
                    pf_done[int(ev["rid"])] = int(ev["prefill_done_ms"])
                except (KeyError, TypeError, ValueError):
                    continue
    if not pf_done:
        return
    for raw in raw_rows:
        if raw.get("ttft_engine_ms") is not None:
            continue
        try:
            _rid = int(raw.get("request_id"))
        except (TypeError, ValueError):
            continue
        _send = _num(raw.get("send_start_epoch_ms"))
        _done = pf_done.get(_rid)
        if _done is None or _send is None or _done < _send:
            continue
        raw["ttft_engine_ms"] = _done - _send


def _load_jsonl_rows(path, limit_bytes=None):
    rows = []
    read = 0
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            read += len(line)
            try:
                rows.append(json.loads(line))
            except ValueError:
                continue  # tolerate a torn trailing line
            if limit_bytes and read > limit_bytes:
                break
    return rows


def _load_json_file(path):
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


# ---------------------------------------------------------------------------
# Loading layer — mock side
# ---------------------------------------------------------------------------


def load_mock_side(path):
    """Resolve a mock run dir (or direct aggregate.json) into SideData."""
    if os.path.isdir(path):
        agg_path = os.path.join(path, "aggregate.json")
        if not os.path.isfile(agg_path):
            raise InputError(f"{path}: run dir has no aggregate.json")
        label = os.path.basename(os.path.abspath(path)) or path
    elif os.path.isfile(path):
        agg_path = path
        label = os.path.basename(os.path.dirname(os.path.abspath(path))) or path
    else:
        raise InputError(f"{path}: no such file or directory")
    aggregate = _load_json_file(agg_path)
    side = SideData(label, "mock")
    side.source_path = os.path.abspath(agg_path)
    side.warnings.append(f"aggregate: {agg_path}")

    meta = aggregate.get("meta") or {}
    summary = aggregate.get("summary") or {}
    side.duration_s = _num(meta.get("duration_s"))

    # run_meta.json params carry n_prefill/n_decode (1P1D gini gate).
    run_meta_path = os.path.join(
        os.path.dirname(os.path.abspath(agg_path)), "run_meta.json"
    )
    if os.path.isfile(run_meta_path):
        try:
            params = (_load_json_file(run_meta_path).get("params")) or {}
            side.n_prefill = _num(params.get("n_prefill"))
            side.n_decode = _num(params.get("n_decode"))
        except (ValueError, OSError):
            pass

    # ---- per-request rows: same-dir client_events.jsonl when kept ----
    ce_path = os.path.join(
        os.path.dirname(os.path.abspath(agg_path)), "client_events.jsonl"
    )
    if os.path.isfile(ce_path):
        raw_rows = _load_jsonl_rows(ce_path)
        # ttft 统一 engine 口径（20260903）：rid join 引擎 prefill 终态行
        # 注入行级 ttft_engine_ms；原始 ttft_ms（client 首帧）保留在行上
        # 但不再进 ttft 族样本（见 latency_samples）。
        _inject_ttft_engine_ms(raw_rows, os.path.dirname(os.path.abspath(ce_path)))
        side.per_request = [_normalize_request_row(r) for r in raw_rows]
        side.per_request_source = ce_path
    else:
        # quantile-approx fallback for the latency families the summary
        # still carries; ttft_latency_ms is the ENGINE caliber since 20260903
        # (client first-frame samples retired), so the fallback aligns with
        # the row-level engine-join path.
        for family, agg_key in (
            ("ttft", "ttft_latency_ms"),
            ("e2e", "e2e_latency_ms"),
            ("schedule", "schedule_latency_ms"),
        ):
            samples = quantile_expand(summary.get(agg_key) or {})
            if samples:
                side.approx_modes[family] = samples
        if side.approx_modes:
            side.per_request_source = (
                "quantile-approx from aggregate summary percentiles "
                "(client_events.jsonl not kept next to the aggregate)"
            )

    # ---- summary ----
    side.summary["total_requests"] = _num(summary.get("total_requests"))
    side.summary["error_count"] = _num(summary.get("error_count"))
    side.summary["error_rate"] = _num(summary.get("error_rate"))
    success = _num(summary.get("success_count"))
    side.summary["success_count"] = success
    total = side.summary["total_requests"]
    if success is not None and total:
        side.summary["success_rate"] = 100.0 * success / total
    elif side.summary["error_rate"] is not None:
        side.summary["success_rate"] = 100.0 - side.summary["error_rate"]
    for family, agg_key in (
        ("ttft", "ttft_latency_ms"),
        ("e2e", "e2e_latency_ms"),
        ("schedule", "schedule_latency_ms"),
    ):
        side.summary[family] = summary.get(agg_key)
    chs = summary.get("cache_hit_summary") or {}
    side.summary["cache_hit_token_pct"] = _num(chs.get("engine_token_hit_pct"))
    side.summary["cache_hit_key_pct"] = _num(chs.get("engine_key_hit_pct"))

    # ---- gini (engine_dist) ----
    ed = aggregate.get("engine_dist") or {}
    side.summary["gini_prefill"] = _num((ed.get("prefill") or {}).get("gini_cum"))
    side.summary["gini_decode"] = _num((ed.get("decode") or {}).get("gini_cum"))
    if side.n_prefill is None:
        ec = _num((ed.get("prefill") or {}).get("engine_count"))
        if ec:
            side.n_prefill = int(ec)
    if side.n_decode is None:
        ec = _num((ed.get("decode") or {}).get("engine_count"))
        if ec:
            side.n_decode = int(ec)
    # per-engine request counts feed the synthesize helper (fake addrs).
    for role in ("prefill", "decode"):
        rpe = (ed.get(role) or {}).get("requests_per_engine")
        if isinstance(rpe, list):
            side.summary[f"{role}_requests_per_engine"] = [
                _num(v) for v in rpe if _num(v) is not None
            ]

    # ---- 1s series (keys normalized, t made relative) ----
    per_second = aggregate.get("per_second") or []
    for key, agg_key in (
        ("arrival_qps", "arrivals"),
        ("success_qps", "success"),
        ("error_qps", "errors"),
    ):
        side.series[key] = _series_from_rows(per_second, agg_key)
        side.series_sources[key] = f"aggregate.per_second.{agg_key}"
    tps_rows = aggregate.get("mock_tps_ts") or []
    for key in ("context_tps", "context_tps_with_cache", "generate_tps"):
        side.series[key] = _series_from_rows(tps_rows, key)
        side.series_sources[key] = f"aggregate.mock_tps_ts.{key}"
    side.series["cache_hit_token_pct"] = _series_from_rows(
        aggregate.get("cache_hit_ts") or [], "engine_token"
    )
    side.series_sources["cache_hit_token_pct"] = "aggregate.cache_hit_ts.engine_token"
    side.series["cache_hit_key_pct"] = _series_from_rows(
        aggregate.get("cache_hit_ts") or [], "engine_key"
    )
    side.series_sources["cache_hit_key_pct"] = "aggregate.cache_hit_ts.engine_key"
    side.series["kv_used_pct"] = _series_from_rows(
        aggregate.get("kv_ts") or [], "used_pct"
    )
    side.series_sources["kv_used_pct"] = "aggregate.kv_ts.used_pct"
    dr_rows = aggregate.get("dispatch_reason_ts") or []
    for reason in DISPATCH_REASONS:
        side.series[f"dispatch_reason_{reason}"] = _series_from_rows(dr_rows, reason)
        side.series_sources[f"dispatch_reason_{reason}"] = (
            f"aggregate.dispatch_reason_ts.{reason}"
        )
    bs_rows = aggregate.get("dispatch_batch_size_ts") or []
    for reason in DISPATCH_REASONS:
        side.series[f"dispatch_batch_size_{reason}"] = _series_from_rows(
            bs_rows, reason
        )
        side.series_sources[f"dispatch_batch_size_{reason}"] = (
            f"aggregate.dispatch_batch_size_ts.{reason}"
        )
    return side


def _series_from_rows(rows, key):
    """[(t, v)] from aggregate rows; t made relative to the first row,
    non-numeric / missing values skipped."""
    out = []
    t0 = None
    for r in rows:
        t = _num(r.get("t"))
        v = _num(r.get(key))
        if t is None:
            continue
        if t0 is None:
            t0 = t
        if v is None:
            continue
        out.append((float(t - t0), float(v)))
    out.sort(key=lambda p: p[0])
    return out


# ---------------------------------------------------------------------------
# Loading layer — real side
# ---------------------------------------------------------------------------


def load_real_side(client_events_path, prom_path=None):
    """Normalize the real side: client_events.jsonl (mandatory) + optional
    prom_export.jsonl into one SideData."""
    if not os.path.isfile(client_events_path):
        raise InputError(f"{client_events_path}: no such file")
    side = SideData(os.path.basename(os.path.dirname(client_events_path)), "real")
    side.warnings.append(f"client_events: {client_events_path}")
    _raw_rows = _load_jsonl_rows(client_events_path)
    # ttft 统一 engine 口径（20260903）：同 mock 侧，rid join 注入行级
    # ttft_engine_ms（同目录 engine_events.jsonl 缺失时行保持无该键，
    # ttft 族样本回退 quantile-approx/absent，不编造）。
    _inject_ttft_engine_ms(
        _raw_rows, os.path.dirname(os.path.abspath(client_events_path))
    )
    side.per_request = [_normalize_request_row(r) for r in _raw_rows]
    side.per_request_source = client_events_path
    if not side.per_request:
        raise InputError(f"{client_events_path}: no parsable rows")

    _aggregate_real_summary(side)
    if prom_path:
        if not os.path.isfile(prom_path):
            raise InputError(f"{prom_path}: no such file")
        side.warnings.append(f"prom: {prom_path}")
        _load_prom_series(side, prom_path)
    return side


def _aggregate_real_summary(side):
    """Run-level summary + per-second qps series from per-request rows."""
    rows = side.per_request
    ok = side.ok_rows()
    total = len(rows)
    side.summary["total_requests"] = total
    side.summary["error_count"] = total - len(ok)
    side.summary["error_rate"] = 100.0 * (total - len(ok)) / total if total else None
    side.summary["success_rate"] = 100.0 * len(ok) / total if total else None
    for family in ("ttft", "e2e", "schedule"):
        vals = side.latency_samples(family)
        side.summary[family] = {
            "count": len(vals),
            "p50": percentile_nr(vals, 0.50),
            "p90": percentile_nr(vals, 0.90),
            "p95": percentile_nr(vals, 0.95),
            "p99": percentile_nr(vals, 0.99),
            "max": round(max(vals), 1) if vals else 0.0,
            "mean": round(sum(vals) / len(vals), 1) if vals else 0.0,
        }
    gini_rows = side.gini_from_rows()
    side.summary["gini_prefill"] = gini_rows["prefill"]
    side.summary["gini_decode"] = gini_rows["decode"]

    # per-second birth-second buckets (same axis rule as the mock side's
    # per_second: arrival/success/error all bucketed by send second).
    starts = [
        _num(d.get("send_start"))
        for d in rows
        if _num(d.get("send_start")) is not None and _num(d.get("send_start")) > 0
    ]
    if starts:
        t0 = min(starts)
        arrival, success, error = {}, {}, {}
        for d in rows:
            s = _num(d.get("send_start"))
            if s is None or s <= 0:
                continue
            sec = int((s - t0) // 1000)
            arrival[sec] = arrival.get(sec, 0) + 1
            ok_row = d.get("status") == "ok" or (
                not (d.get("error") or "")
                and d.get("status") not in ("schedule_error",)
            )
            if ok_row:
                success[sec] = success.get(sec, 0) + 1
            else:
                error[sec] = error.get(sec, 0) + 1
        side.series["arrival_qps"] = [
            (float(k), float(v)) for k, v in sorted(arrival.items())
        ]
        side.series["success_qps"] = [
            (float(k), float(v)) for k, v in sorted(success.items())
        ]
        side.series["error_qps"] = [
            (float(k), float(v)) for k, v in sorted(error.items())
        ]
        for key in ("arrival_qps", "success_qps", "error_qps"):
            side.series_sources[key] = "client_events birth-second buckets"
        side.duration_s = float(max(arrival) + 1) if arrival else None


def _strip_micrometer_suffix(name):
    """Drop timer/counter decorations for suffix matching: foo_seconds_count
    -> foo_seconds -> foo; foo_total -> foo. Single layer — compound
    decorations are peeled by the caller's progressive loop."""
    for suffix in ("_count", "_sum", "_max", "_seconds"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    if name.endswith("_total"):
        return name[: -len("_total")]
    return name


def _name_matches(metric_name, candidate):
    """Exact, stripped-suffix, or suffix containment match (either side)."""
    if metric_name == candidate:
        return True
    # progressively strip Micrometer decorations — every intermediate form
    # is a valid match target (timer foo_seconds_count -> foo_seconds -> foo).
    name = metric_name
    for _ in range(3):
        nxt = _strip_micrometer_suffix(name)
        if nxt == name:
            break
        name = nxt
        if name == candidate:
            return True
    if metric_name.endswith("." + candidate):
        return True
    return metric_name.endswith(candidate) or candidate.endswith(metric_name)


def _looks_like_counter(values):
    """Monotone-ish detection with tolerance for resets: a series whose
    positive deltas dominate and whose values are non-negative is a counter
    (integrality is NOT required — rates accumulated at non-1s cadence and
    rounded rates break the integer check while staying counters)."""
    if len(values) < 3:
        return False
    ups = sum(1 for a, b in zip(values, values[1:]) if b >= a)
    nonneg = all(v >= 0 for v in values)
    return nonneg and ups >= len(values) - 1 and values[-1] >= values[0]


def _counter_to_rate(points):
    """[(t, v)] counter samples -> [(t, rate)] via positive deltas; resets
    (negative delta) drop the interval instead of fabricating a spike."""
    out = []
    for (t0, v0), (t1, v1) in zip(points, points[1:]):
        dt = t1 - t0
        if dt <= 0 or v1 < v0:
            continue
        out.append((t1, (v1 - v0) / dt))
    return out


def _parse_prom_export(path):
    """Read prom_export.jsonl into {metric_name: [(t, v)]}.

    Per-row sniffing (NOT whole-file): a row carrying "metric"+"value"
    keys is a LONG-format row, any other row with a "t" key is a WIDE-format
    row — mixed files are supported. Long rows aggregate same-name same-t
    (sum for counters, avg for gauge-hinted names); wide rows are one value
    per column. t is made relative to the first sample of the whole file.
    """
    rows = _load_jsonl_rows(path)
    if not rows:
        raise InputError(f"{path}: no parsable rows")
    collected = {}  # name -> {t: [values]}
    for r in rows:
        t = _num(r.get("t"))
        if t is None:
            continue
        if isinstance(r.get("metric"), str) and _num(r.get("value")) is not None:
            collected.setdefault(r["metric"], {}).setdefault(t, []).append(
                _num(r.get("value"))
            )
            continue
        for k, v in r.items():
            if k == "t":
                continue
            v = _num(v)
            if v is None:
                continue
            collected.setdefault(k, {}).setdefault(t, []).append(v)
    if not collected:
        raise InputError(f"{path}: no numeric series extracted")
    all_ts = sorted({t for by_t in collected.values() for t in by_t})
    t0 = all_ts[0]
    out = {}
    for name, by_t in collected.items():
        gauge = any(h in name for h in GAUGE_METRIC_HINTS)
        pts = []
        for t, vals in sorted(by_t.items()):
            v = sum(vals) / len(vals) if gauge else sum(vals)
            pts.append((float(t - t0), float(v)))
        pts.sort(key=lambda p: p[0])
        out[name] = pts
    return out


def _find_prom_series(parsed, candidates):
    """First candidate name that matches any parsed metric name."""
    for candidate in candidates:
        for name in parsed:
            if _name_matches(name, candidate):
                return name
    return None


def _load_prom_series(side, prom_path):
    """Map parsed prom series onto the internal series keys."""
    parsed = _parse_prom_export(prom_path)

    def assign(internal, name, transform=None):
        pts = parsed[name]
        if transform:
            pts = transform(pts)
        if pts:
            side.series[internal] = pts
            side.series_sources[internal] = f"prom:{name}"

    # direct TPS / ratio / kv series
    for internal, candidates in PROM_SERIES_CANDIDATES.items():
        name = _find_prom_series(parsed, candidates)
        if name:
            assign(internal, name)

    # counter pairs -> derived pct series
    for internal, (num_candidates, den_candidates) in PROM_RATIO_PAIRS.items():
        if side.series.get(internal):
            continue
        n_name = _find_prom_series(parsed, num_candidates)
        d_name = _find_prom_series(parsed, den_candidates)
        if not n_name or not d_name or n_name == d_name:
            continue
        if not (
            _looks_like_counter([v for _, v in parsed[n_name]])
            and _looks_like_counter([v for _, v in parsed[d_name]])
        ):
            continue

        def ratio_pair(num_pts, den_pts):
            dn = _counter_to_rate(num_pts)
            dd = _counter_to_rate(den_pts)
            by_t = {t: v for t, v in dd}
            out = []
            for t, nv in dn:
                dv = by_t.get(t)
                if dv and dv > 0:
                    pct = 100.0 * nv / dv
                    if pct <= 100.0:
                        out.append((t, pct))
            return out

        assign(
            internal,
            n_name,
            lambda pts, n=n_name, d=d_name: ratio_pair(parsed[n], parsed[d]),
        )

    # dispatch reason counters (labeled or flattened) -> per-second rate
    reason_series = _extract_reason_series(parsed, "dispatch_reason")
    reason_names = reason_series.pop("__name__", {})
    for reason, pts in reason_series.items():
        rate = (
            _counter_to_rate(pts) if _looks_like_counter([v for _, v in pts]) else pts
        )
        if rate:
            internal = f"dispatch_reason_{reason}"
            side.series[internal] = rate
            side.series_sources[internal] = (
                f"prom:{reason_names.get(reason, '?')} reason={reason}"
            )

    # master batch size gauges per reason
    bs_series = _extract_reason_series(parsed, "batch_size")
    bs_series.pop("__name__", None)
    for reason, pts in bs_series.items():
        if pts and reason in DISPATCH_REASONS:
            internal = f"dispatch_batch_size_{reason}"
            side.series[internal] = pts
            side.series_sources[internal] = f"prom batch_size reason={reason}"

    # supplementary master schedule latency (never gates, recorded only)
    sched_name = _find_prom_series(parsed, PROM_SCHEDULE_NAMES)
    if sched_name:
        pts = parsed[sched_name]
        if _looks_like_counter([v for _, v in pts]):
            # timer _count family: the values themselves are per-second means
            side.series["master_schedule_latency_ms"] = pts
        else:
            side.series["master_schedule_latency_ms"] = pts
        side.series_sources["master_schedule_latency_ms"] = f"prom:{sched_name}"
    # auto_tpm schedule timer as (sum, count) pair -> per-second mean
    sum_name = _find_prom_series(parsed, [n + "_sum" for n in PROM_SCHEDULE_NAMES])
    cnt_name = _find_prom_series(parsed, [n + "_count" for n in PROM_SCHEDULE_NAMES])
    if sum_name and cnt_name:
        sums = dict(parsed[sum_name])
        cnts = dict(parsed[cnt_name])
        out = []
        prev_s = prev_c = None
        for t in sorted(set(sums) & set(cnts)):
            s, c = sums[t], cnts[t]
            if prev_s is not None and c > prev_c and s >= prev_s:
                out.append((float(t), (s - prev_s) / (c - prev_c)))
            prev_s, prev_c = s, c
        if out:
            side.series["master_schedule_latency_ms"] = out
            side.series_sources["master_schedule_latency_ms"] = (
                f"prom:{sum_name}/{cnt_name} deltas"
            )
    if not sched_name and not sum_name:
        side.warnings.append(
            "real prom: no auto_tpm schedule-latency family — supplementary "
            "master caliber unavailable (client schedule_ms remains primary)"
        )


def _extract_reason_series(parsed, keyword):
    """Collect {reason: [(t, v)]} from metric names carrying the keyword,
    either as a label (`name{reason="X"}`) or a flattened suffix
    (`name_X` / `name_X_total`)."""
    out = {}
    names = {}
    for name in parsed:
        base = name.split("{", 1)[0]
        if keyword not in base:
            continue
        reason = None
        if "{" in name and 'reason="' in name:
            reason = name.split('reason="', 1)[1].split('"', 1)[0]
        else:
            for r in DISPATCH_REASONS:
                if base.endswith("_" + r) or base.endswith("_" + r + "_total"):
                    reason = r
                    break
        if reason and reason in DISPATCH_REASONS:
            existing = out.setdefault(reason, [])
            existing.extend(parsed[name])
            names[reason] = name
    for reason in list(out):
        out[reason].sort(key=lambda p: p[0])
    out["__name__"] = names  # type: ignore[assignment]
    return out


# ---------------------------------------------------------------------------
# Synthesize — serialize a (mock) SideData back into real-side input files.
# Doubles as the self-consistency round-trip (mock data fed as real input
# must compare ALIGNED) and as a bootstrap helper for future real twin runs.
# ---------------------------------------------------------------------------


def synthesize_real_inputs(side, out_dir):
    """Write <out_dir>/client_events.jsonl + prom_export.jsonl from SideData.

    * latency-family rows come from real per-request rows when available,
      else from quantile-approx samples (status "ok", send_start spread
      over the steady window, fake engine addrs allocated proportionally
      from engine_dist.requests_per_engine).
    * series are emitted as a MIXED export: dispatch-reason counters in LONG
      format (labeled names, the counter-diff path), everything else in
      WIDE format columns (gauge/ratio path) — both parser branches get
      exercised by the round-trip.
    """
    os.makedirs(out_dir, exist_ok=True)
    ce_path = os.path.join(out_dir, "client_events.jsonl")
    prom_path = os.path.join(out_dir, "prom_export.jsonl")

    lo, hi = side.steady_window()
    real_rows = [
        d
        for d in side.per_request
        if _num(d.get("send_start")) is not None and _num(d.get("send_start")) > 0
    ]
    rows_out = []
    if real_rows:
        for d in real_rows:
            rows_out.append(_denormalize_request_row(d))
    else:
        # approx path: one row per ok request, latency values sampled
        # uniformly from the quantile-expanded approximations; the ok-row
        # count honors the summary's success share (latency summaries are
        # computed over ok rows only) — the rest of total becomes error rows
        # below so success/error rates round-trip.
        approx = {
            f: side.approx_modes.get(f) or [] for f in ("ttft", "e2e", "schedule")
        }
        n = max((len(v) for v in approx.values()), default=0)
        if n:
            t0 = 1_700_000_000_000.0
            total = _num(side.summary.get("total_requests"))
            success = _num(side.summary.get("success_count"))
            n_ok = n
            if success is not None and total is not None and 0 < success <= total:
                n_ok = max(1, int(round(success)))
            pre_names = _fake_engine_addrs(side, "prefill", n_ok)
            dec_names = _fake_engine_addrs(side, "decode", n_ok)

            def pick(fam, i):
                vals = approx[fam]
                if not vals:
                    return 0.0
                if len(vals) >= n_ok:
                    return vals[i * len(vals) // n_ok]
                return vals[i % len(vals)]

            for i in range(n_ok):
                spread = lo + (hi - lo) * (i + 0.5) / n_ok if hi > lo else lo
                send = t0 + spread * 1000.0
                row = {
                    "rid": f"synthetic-{i}",
                    "trace_id": "twin-synthetic",
                    "request_id": i,
                    "ts": spread * 1000.0,
                    "input_len": 1000,
                    "output_len": 200,
                    "status": "scheduled",
                    "send_start_epoch_ms": send,
                    "sched_done_epoch_ms": send + pick("schedule", i),
                    "schedule_ms": pick("schedule", i),
                    # ttft 统一 engine 口径（20260903）：合成行无 client
                    # 首帧源，ttft_ms 占 0；engine 口径值写进
                    # ttft_engine_ms（round-trip 再次加载时行级直读）。
                    "ttft_ms": 0.0,
                    "ttft_engine_ms": pick("ttft", i),
                    "total_ms": pick("e2e", i),
                    "enqueued_by_master": True,
                    "prefill": pre_names[i],
                    "decode": dec_names[i],
                    "route_path": "master",
                    "wall_clock_ts": send / 1000.0,
                    "send_due_epoch_ms": send,
                    "pacing_lag_ms": 0.0,
                    "priority": 50,
                }
                rows_out.append(row)

    # error-row top-up: the approx path emits ok rows only — pad up to the
    # summary total with schedule_error rows (no latency samples, so they
    # stay outside the is_ok latency calibers; engine addrs are still
    # allocated so the per-engine gini caliber round-trips).
    total = _num(side.summary.get("total_requests"))
    if total is not None and len(rows_out) < int(total):
        n_err = int(total) - len(rows_out)
        t0 = 1_700_000_000_000.0
        pre_names = _fake_engine_addrs(side, "prefill", n_err)
        dec_names = _fake_engine_addrs(side, "decode", n_err)
        for i in range(n_err):
            spread = lo + (hi - lo) * (i + 0.5) / n_err if hi > lo else lo
            send = t0 + spread * 1000.0
            rows_out.append(
                {
                    "rid": f"synthetic-err-{i}",
                    "trace_id": "twin-synthetic",
                    "request_id": 1_000_000 + i,
                    "ts": spread * 1000.0,
                    "input_len": 1000,
                    "output_len": 0,
                    "status": "schedule_error",
                    "error": "synthesized_error",
                    "send_start_epoch_ms": send,
                    "sched_done_epoch_ms": send,
                    "schedule_ms": 0.0,
                    "ttft_ms": 0.0,
                    "total_ms": 0.0,
                    "enqueued_by_master": True,
                    "prefill": pre_names[i],
                    "decode": dec_names[i],
                    "route_path": "master",
                    "wall_clock_ts": send / 1000.0,
                    "send_due_epoch_ms": send,
                    "pacing_lag_ms": 0.0,
                    "priority": 50,
                }
            )
    with open(ce_path, "w", encoding="utf-8") as fh:
        for r in rows_out:
            fh.write(json.dumps(r) + "\n")

    # ---- prom export ----
    wide_rows = {}
    long_rows = []
    series_out = {}

    def wide_col(name, pts, scale=None):
        for t, v in pts:
            if scale is not None:
                v = v * scale
            wide_rows.setdefault(round(t, 1), {})[name] = round(v, 4)

    # TPS (production wall-caliber names), kv, cache-hit ratio series
    tps_map = {
        "context_tps": "context_wall_tps",
        "context_tps_with_cache": "context_wall_tps_with_cache",
        "generate_tps": "generate_wall_tps",
    }
    for internal, col in tps_map.items():
        pts = side.series.get(internal) or []
        wide_col(col, pts)
        series_out[internal] = col
    kv_pts = side.series.get("kv_used_pct") or []
    wide_col("kv_cache_used_pct", kv_pts)
    series_out["kv_used_pct"] = "kv_cache_used_pct"
    # cache hit: a CONSTANT column reproducing the mock run-level caliber —
    # the real side has no run-level scalar, its steady mean of this series
    # must equal the mock summary scalar exactly in the round-trip.
    hit_scalar = side.summary.get("cache_hit_token_pct")
    hit_pts = side.series.get("cache_hit_token_pct") or []
    if hit_scalar is not None:
        anchor_ts = [t for t, _ in (kv_pts or tps_series_anchor(side))]
        hit_pts = [(t, hit_scalar) for t in anchor_ts]
    wide_col("engine_token_hit_pct", hit_pts)
    series_out["cache_hit_token_pct"] = "engine_token_hit_pct"

    # batch-size gauges: flattened reason names
    for reason in DISPATCH_REASONS:
        pts = side.series.get(f"dispatch_batch_size_{reason}") or []
        col = f"flexlb_app_engine_balancing_master_batch_size_{reason}"
        wide_col(col, pts)
        series_out[f"dispatch_batch_size_{reason}"] = col

    # supplementary master schedule latency series (per-second mean)
    msched = side.series.get("master_schedule_latency_ms")
    if not msched:
        # synthesize from per_second sched summaries when the mock side
        # carries none: use per-request schedule p50 per second is not
        # available; steady mean scalar keeps the round-trip lossless.
        pass
    wide_col(
        "auto_tpm_schedule_latency_ms",
        side.series.get("master_schedule_latency_ms") or [],
    )

    # dispatch-reason counters: LONG format, cumulative
    for reason in DISPATCH_REASONS:
        pts = side.series.get(f"dispatch_reason_{reason}") or []
        cum = 0.0
        prev_t = None
        for t, v in pts:
            if prev_t is None:
                prev_t = t
                continue
            cum += v * (t - prev_t)
            prev_t = t
            long_rows.append(
                {
                    "t": round(t, 1),
                    "metric": (
                        "flexlb_app_engine_balancing_master_dispatch_reason_total"
                        f'{{reason="{reason}"}}'
                    ),
                    "value": round(cum, 2),
                }
            )
        series_out[f"dispatch_reason_{reason}"] = (
            f"dispatch_reason_total{{reason={reason}}}"
        )

    with open(prom_path, "w", encoding="utf-8") as fh:
        for t in sorted(wide_rows):
            row = {"t": t}
            row.update(wide_rows[t])
            fh.write(json.dumps(row) + "\n")
        for r in long_rows:
            fh.write(json.dumps(r) + "\n")
    return {"client_events": ce_path, "prom": prom_path, "series": series_out}


def tps_series_anchor(side):
    """Fallback t-axis for constant columns when no kv series exists."""
    pts = (
        side.series.get("context_tps_with_cache")
        or side.series.get("context_tps")
        or []
    )
    return pts or [(0.0, 0.0)]


def _denormalize_request_row(d):
    """Internal row -> JavaLoadClient perRequestNode field names."""
    return {
        "rid": d.get("rid"),
        "trace_id": "twin",
        "request_id": 0,
        "ts": d.get("ts"),
        "input_len": d.get("input_len"),
        "output_len": d.get("output_len"),
        "status": d.get("status") or "scheduled",
        "schedule_ms": d.get("schedule_ms"),
        "sched_done_epoch_ms": d.get("sched_done"),
        "ttft_ms": d.get("ttft_ms"),
        "ttft_engine_ms": d.get("ttft_engine_ms"),
        "total_ms": d.get("total_ms"),
        "enqueued_by_master": True,
        "prefill": d.get("prefill"),
        "decode": d.get("decode"),
        "error": d.get("error"),
        "route_path": "master",
        "wall_clock_ts": None,
        "send_due_epoch_ms": d.get("send_start"),
        "send_start_epoch_ms": d.get("send_start"),
        "pacing_lag_ms": 0.0,
        "priority": d.get("priority", 50),
    }


def _fake_engine_addrs(side, role, n):
    """n engine addrs allocated proportionally to engine_dist counts so the
    synthesized rows reproduce the mock side's per-engine gini."""
    counts = side.summary.get(f"{role}_requests_per_engine") or []
    if not counts or n <= 0:
        return ["engine-0"] * n
    total = sum(counts)
    names = []
    allocated = 0
    for i, c in enumerate(counts[:-1]):
        k = int(round(c / total * n))
        names.extend([f"engine-{role}-{i}"] * k)
        allocated += k
    names.extend([f"engine-{role}-{len(counts) - 1}"] * (n - allocated))
    return names[:n] if len(names) >= n else (names + [names[-1]] * (n - len(names)))


# ---------------------------------------------------------------------------
# Noise floor layer
# ---------------------------------------------------------------------------


def compute_floors(mock_side, extra_mock_sides, lo, hi):
    """Per-metric floor: measured (pairwise-median over mock runs) when >=2
    distinct mock sides exist (the primary + extras; the task's >=3 runs
    recommendation maps to >=3 pairwise distances for a robust median),
    else empirical per distance kind.

    Returns {metric_name: {"floor": float, "source": str}} plus the
    "_measured" meta entry (n runs + whether any measured floor was used).
    """
    sides = [mock_side] + list(extra_mock_sides or [])
    # dedupe by resolved aggregate path (an identical run must not
    # fabricate a "measured" floor out of one run's self-distance)
    seen = set()
    uniq = []
    for s in sides:
        key = s.source_path or s.series_sources.get("arrival_qps") or s.label
        if key not in seen:
            seen.add(key)
            uniq.append(s)
    floors = {}
    any_measured = False
    for spec in METRIC_SPECS:
        dists = []
        if len(uniq) >= 2:
            for i in range(len(uniq)):
                for j in range(i + 1, len(uniq)):
                    res = spec["evaluate"](uniq[i], uniq[j], lo, hi)
                    d = res.get("distance")
                    if d is not None:
                        dists.append(d)
        if dists:
            dists.sort()
            mid = len(dists) // 2
            med = dists[mid] if len(dists) % 2 else (dists[mid - 1] + dists[mid]) / 2.0
            if med > 0:
                any_measured = True
                floors[spec["name"]] = {
                    "floor": med,
                    "source": "measured (median of %d pairwise distances over %d mock runs)"
                    % (len(dists), len(uniq)),
                }
                continue
            floors[spec["name"]] = {
                "floor": FLOOR_GUARD_MIN[spec["kind"]],
                "source": "measured-zero → guard minimum (duplicate runs)",
            }
            any_measured = True
            continue
        floors[spec["name"]] = {
            "floor": spec["empirical_floor"](mock_side, mock_side, lo, hi),
            "source": "经验地板非实测 (empirical)",
        }
    floors["_measured"] = {
        "floor": None,
        "source": "%d mock runs, measured=%s" % (len(uniq), any_measured),
    }  # type: ignore[assignment]
    return floors


# ---------------------------------------------------------------------------
# Compare layer — the twelve metrics
# ---------------------------------------------------------------------------


def _side_latency_samples(side, family):
    """Per-request samples, or quantile-approx samples as mock fallback."""
    vals = side.latency_samples(family)
    if vals:
        return vals, "per-request"
    approx = side.approx_modes.get(family) or []
    return (approx, "quantile-approx") if approx else ([], "absent")


def _eval_latency_family(a, b, family):
    va, mode_a = _side_latency_samples(a, family)
    vb, mode_b = _side_latency_samples(b, family)
    if not va and not vb:
        return {
            "skip_reason": "两侧均无 %s 样本" % family,
            "mock_value": None,
            "real_value": None,
            "distance": None,
            "details": {"mock_mode": mode_a, "real_mode": mode_b},
        }
    if not va:
        return {
            "skip_reason": "mock 侧无 %s 样本" % family,
            "mock_value": None,
            "real_value": percentile_nr(vb, 0.50),
            "distance": None,
            "details": {"mock_mode": mode_a, "real_mode": mode_b},
        }
    if not vb:
        return {
            "skip_reason": "real 侧无 %s 样本" % family,
            "mock_value": percentile_nr(va, 0.50),
            "real_value": None,
            "distance": None,
            "details": {"mock_mode": mode_a, "real_mode": mode_b},
        }
    w = wasserstein_1d(va, vb)
    return {
        "mock_value": percentile_nr(va, 0.50),
        "real_value": percentile_nr(vb, 0.50),
        "distance": w,
        "abs_diff": percentile_nr(vb, 0.50) - percentile_nr(va, 0.50),
        "details": {
            "mock_mode": mode_a,
            "real_mode": mode_b,
            "mock_p90": percentile_nr(va, 0.90),
            "real_p90": percentile_nr(vb, 0.90),
            "mock_n": len(va),
            "real_n": len(vb),
            "p50_diff": percentile_nr(vb, 0.50) - percentile_nr(va, 0.50),
            "p90_diff": percentile_nr(vb, 0.90) - percentile_nr(va, 0.90),
        },
    }


def _eval_batch_size(a, b, lo, hi):
    per_reason = {}
    distances = []
    for reason in DISPATCH_REASONS:
        va = [v for t, v in (a.series.get(f"dispatch_batch_size_{reason}") or [])]
        vb = [v for t, v in (b.series.get(f"dispatch_batch_size_{reason}") or [])]
        per_reason[reason] = {
            "mock_mean": round(sum(va) / len(va), 2) if va else None,
            "real_mean": round(sum(vb) / len(vb), 2) if vb else None,
            "mock_n": len(va),
            "real_n": len(vb),
        }
        if va and vb:
            per_reason[reason]["ks"] = ks_statistic(va, vb)
            distances.append((per_reason[reason]["ks"], reason))
    if not distances:
        missing = (
            "real 侧 prom 缺 batch_size 指标"
            if not any(per_reason[r]["real_n"] for r in DISPATCH_REASONS)
            else "mock 侧缺 dispatch_batch_size_ts"
        )
        return {
            "skip_reason": missing,
            "mock_value": None,
            "real_value": None,
            "distance": None,
            "details": per_reason,
        }
    distances.sort(reverse=True)
    worst_d, worst_reason = distances[0]
    return {
        "mock_value": next(
            (
                per_reason[r]["mock_mean"]
                for r in DISPATCH_REASONS
                if per_reason[r]["mock_mean"] is not None
            ),
            None,
        ),
        "real_value": next(
            (
                per_reason[r]["real_mean"]
                for r in DISPATCH_REASONS
                if per_reason[r]["real_mean"] is not None
            ),
            None,
        ),
        "distance": worst_d,
        "abs_diff": None,
        "details": {"per_reason": per_reason, "worst_reason": worst_reason},
    }


def _reason_shares(side):
    totals = {}
    for reason in DISPATCH_REASONS:
        totals[reason] = sum(
            v for _, v in (side.series.get(f"dispatch_reason_{reason}") or [])
        )
    grand = sum(totals.values())
    if grand <= 0:
        return None, totals
    return {r: 100.0 * v / grand for r, v in totals.items()}, totals


def _eval_dispatch_reason_share(a, b, lo, hi):
    share_a, totals_a = _reason_shares(a)
    share_b, totals_b = _reason_shares(b)
    if share_a is None and share_b is None:
        return {
            "skip_reason": "两侧均无 dispatch_reason 数据",
            "mock_value": None,
            "real_value": None,
            "distance": None,
        }
    if share_a is None:
        return {
            "skip_reason": "mock 侧缺 dispatch_reason_ts",
            "mock_value": None,
            "real_value": share_b,
            "distance": None,
        }
    if share_b is None:
        return {
            "skip_reason": "real 侧 prom 缺 dispatch_reason 指标",
            "mock_value": share_a,
            "real_value": None,
            "distance": None,
        }
    diffs = {r: share_b[r] - share_a[r] for r in DISPATCH_REASONS}
    worst = max(diffs, key=lambda r: abs(diffs[r]))
    return {
        "mock_value": share_a[worst],
        "real_value": share_b[worst],
        "distance": abs(diffs[worst]),
        "abs_diff": diffs[worst],
        "details": {
            "mock_shares": {r: round(v, 2) for r, v in share_a.items()},
            "real_shares": {r: round(v, 2) for r, v in share_b.items()},
            "share_diff_pp": {r: round(v, 2) for r, v in diffs.items()},
            "worst_reason": worst,
            "mock_totals": totals_a,
            "real_totals": totals_b,
        },
    }


def _eval_tps(a, b, key, lo, hi):
    va = series_steady_mean(a.series.get(key) or [], lo, hi)
    vb = series_steady_mean(b.series.get(key) or [], lo, hi)
    if va is None and vb is None:
        return {
            "skip_reason": "两侧均无 %s 时序" % key,
            "mock_value": None,
            "real_value": None,
            "distance": None,
        }
    if va is None:
        return {
            "skip_reason": "mock 侧缺 mock_tps_ts.%s" % key,
            "mock_value": None,
            "real_value": round(vb, 1),
            "distance": None,
        }
    if vb is None:
        return {
            "skip_reason": "real 侧 prom 缺 %s（wall 口径）" % key,
            "mock_value": round(va, 1),
            "real_value": None,
            "distance": None,
        }
    if va == 0:
        # zero baseline: identical is aligned, any nonzero is unbounded
        rel = 0.0 if vb == 0 else float("inf")
    else:
        rel = (vb - va) / va * 100.0
    return {
        "mock_value": round(va, 1),
        "real_value": round(vb, 1),
        "distance": abs(rel),
        "abs_diff": round(vb - va, 1),
        "rel_diff_pct": (round(rel, 2) if rel != float("inf") else "inf"),
    }


def _eval_ratio_pp(a, b, key, lo, hi):
    def side_value(side):
        v = side.summary.get(key)
        if v is not None:
            return v
        return series_steady_pct_mean(side.series.get(key) or [], lo, hi)

    va, vb = side_value(a), side_value(b)
    if va is None and vb is None:
        return {
            "skip_reason": "两侧均无 %s" % key,
            "mock_value": None,
            "real_value": None,
            "distance": None,
        }
    if va is None:
        return {
            "skip_reason": "mock 侧缺 %s" % key,
            "mock_value": None,
            "real_value": round(vb, 2),
            "distance": None,
        }
    if vb is None:
        return {
            "skip_reason": "real 侧缺 %s（prom 无该指标）" % key,
            "mock_value": round(va, 2),
            "real_value": None,
            "distance": None,
        }
    return {
        "mock_value": round(va, 2),
        "real_value": round(vb, 2),
        "distance": abs(vb - va),
        "abs_diff": round(vb - va, 2),
    }


def _eval_kv_level(a, b, lo, hi):
    va = series_values_in_window(a.series.get("kv_used_pct") or [], lo, hi)
    vb = series_values_in_window(b.series.get("kv_used_pct") or [], lo, hi)
    if not va and not vb:
        return {
            "skip_reason": "两侧均无 KV 水位时序",
            "mock_value": None,
            "real_value": None,
            "distance": None,
        }
    if not va:
        return {
            "skip_reason": "mock 侧缺 kv_ts.used_pct",
            "mock_value": None,
            "real_value": round(sum(vb) / len(vb), 1),
            "distance": None,
        }
    if not vb:
        return {
            "skip_reason": "real 侧 prom 缺 KV 水位指标",
            "mock_value": round(sum(va) / len(va), 1),
            "real_value": None,
            "distance": None,
        }
    return {
        "mock_value": round(sum(va) / len(va), 1),
        "real_value": round(sum(vb) / len(vb), 1),
        "distance": ks_statistic(va, vb),
        "abs_diff": round(sum(vb) / len(vb) - sum(va) / len(va), 1),
        "details": {"mock_n": len(va), "real_n": len(vb)},
    }


def _eval_gini(a, b, lo, hi):
    small = lambda n: n is not None and n <= 1
    if small(a.n_prefill) and small(a.n_decode):
        return {
            "na_reason": "1P1D 无路由选择（%s 拓扑下 gini 恒为 0）" % a.label,
            "mock_value": None,
            "real_value": None,
            "distance": None,
        }
    pairs = {}
    worst = None
    for role in ("prefill", "decode"):
        va = a.summary.get(f"gini_{role}")
        vb = b.summary.get(f"gini_{role}")
        pairs[role] = {"mock": va, "real": vb}
        if va is not None and vb is not None:
            d = abs(vb - va)
            pairs[role]["abs_diff"] = round(d, 4)
            if worst is None or d > worst[0]:
                worst = (d, role)
    if worst is None:
        return {
            "skip_reason": "gini 数据不全（mock=%s real=%s）"
            % (
                {r: pairs[r]["mock"] for r in pairs},
                {r: pairs[r]["real"] for r in pairs},
            ),
            "mock_value": None,
            "real_value": None,
            "distance": None,
            "details": pairs,
        }
    d, role = worst
    return {
        "mock_value": round(pairs[role]["mock"], 4),
        "real_value": round(pairs[role]["real"], 4),
        "distance": d,
        "abs_diff": round(pairs[role]["real"] - pairs[role]["mock"], 4),
        "details": {
            "per_role": pairs,
            "worst_role": role,
            "n_prefill": a.n_prefill,
            "n_decode": a.n_decode,
        },
    }


def _latency_empirical_floor_factory(family):
    """max(5ms, 5% of the mock-side p50) — compare_ab latency calibration."""

    def floor(a, b, lo, hi):
        vals, _ = _side_latency_samples(a, family)
        if not vals:
            return 5.0
        return max(5.0, 0.05 * percentile_nr(vals, 0.50))

    return floor


METRIC_SPECS = [
    {
        "name": "ttft_dist",
        "group": "latency",
        "kind": "w1_ms",
        "evaluate": lambda a, b, lo, hi: _eval_latency_family(a, b, "ttft"),
        "empirical_floor": _latency_empirical_floor_factory("ttft"),
    },
    {
        "name": "e2e_dist",
        "group": "latency",
        "kind": "w1_ms",
        "evaluate": lambda a, b, lo, hi: _eval_latency_family(a, b, "e2e"),
        "empirical_floor": _latency_empirical_floor_factory("e2e"),
    },
    {
        "name": "schedule_latency",
        "group": "latency",
        "kind": "w1_ms",
        "evaluate": lambda a, b, lo, hi: _eval_latency_family(a, b, "schedule"),
        "empirical_floor": _latency_empirical_floor_factory("schedule"),
    },
    {
        "name": "batch_size_dist",
        "group": "batch",
        "kind": "ks",
        "evaluate": _eval_batch_size,
        "empirical_floor": lambda a, b, lo, hi: 0.05,
    },
    {
        "name": "dispatch_reason_share",
        "group": "dispatch",
        "kind": "pp",
        "evaluate": _eval_dispatch_reason_share,
        "empirical_floor": lambda a, b, lo, hi: 1.0,
    },
    {
        "name": "tps_context_with_cache",
        "group": "tps",
        "kind": "rel_pct",
        "evaluate": lambda a, b, lo, hi: _eval_tps(
            a, b, "context_tps_with_cache", lo, hi
        ),
        "empirical_floor": lambda a, b, lo, hi: 2.0,
    },
    {
        "name": "tps_generate",
        "group": "tps",
        "kind": "rel_pct",
        "evaluate": lambda a, b, lo, hi: _eval_tps(a, b, "generate_tps", lo, hi),
        "empirical_floor": lambda a, b, lo, hi: 2.0,
    },
    {
        "name": "cache_hit_token_pct",
        "group": "cache",
        "kind": "pp",
        "evaluate": lambda a, b, lo, hi: _eval_ratio_pp(
            a, b, "cache_hit_token_pct", lo, hi
        ),
        "empirical_floor": lambda a, b, lo, hi: 1.0,
    },
    {
        "name": "success_rate",
        "group": "validity",
        "kind": "pp",
        "evaluate": lambda a, b, lo, hi: _eval_ratio_pp(a, b, "success_rate", lo, hi),
        "empirical_floor": lambda a, b, lo, hi: 1.0,
    },
    {
        "name": "error_rate",
        "group": "validity",
        "kind": "pp",
        "evaluate": lambda a, b, lo, hi: _eval_ratio_pp(a, b, "error_rate", lo, hi),
        "empirical_floor": lambda a, b, lo, hi: 1.0,
    },
    {
        "name": "kv_used_pct",
        "group": "kv",
        "kind": "ks",
        "evaluate": _eval_kv_level,
        "empirical_floor": lambda a, b, lo, hi: 0.05,
    },
    {
        "name": "gini",
        "group": "balance",
        "kind": "gini",
        "evaluate": _eval_gini,
        "empirical_floor": lambda a, b, lo, hi: 0.005,
    },
]


def evaluate_metrics(mock, real, lo, hi):
    out = []
    for spec in METRIC_SPECS:
        res = dict(spec["evaluate"](mock, real, lo, hi))
        res["name"] = spec["name"]
        res["group"] = spec["group"]
        res["kind"] = spec["kind"]
        out.append(res)
    return out


# ---------------------------------------------------------------------------
# Verdict + attribution layers
# ---------------------------------------------------------------------------


def verdict_for(result, floor_value):
    if result.get("skip_reason"):
        return VERDICT_SKIP, None
    if result.get("na_reason"):
        return VERDICT_NA, None
    d = result.get("distance")
    if d is None:
        return VERDICT_SKIP, None
    if floor_value is None or floor_value <= 0:
        return (VERDICT_ALIGNED, 0.0) if d == 0 else (VERDICT_DIVERGED, float("inf"))
    ratio = d / floor_value
    if ratio < ALIGNED_MAX_MULT:
        return VERDICT_ALIGNED, ratio
    if ratio < DIVERGED_MIN_MULT:
        return VERDICT_DEVIATED, ratio
    return VERDICT_DIVERGED, ratio


def attribute_hints(results):
    """Cross-metric attribution rules (task rule table). Returns a list of
    {"hint": str, "metrics": [names]} in rule order; a metric carries the
    first hint that names it."""
    v = {r["name"]: r["verdict"] for r in results}

    def exceeded(name):
        return v.get(name) in (VERDICT_DEVIATED, VERDICT_DIVERGED)

    def quiet(name):
        return v.get(name) in (VERDICT_ALIGNED, VERDICT_SKIP, VERDICT_NA, None)

    hints = []
    e2e_x, ttft_x = exceeded("e2e_dist"), exceeded("ttft_dist")
    if e2e_x and not ttft_x:
        # R2 (more specific than R1): decode tail dominates the e2e gap.
        hints.append(
            {
                "hint": "decode 尾部问题，查 ol 大请求 decode 拟合",
                "metrics": ["e2e_dist"],
            }
        )
    elif (e2e_x or ttft_x) and quiet("dispatch_reason_share"):
        # R1: latency model bias with an unchanged cut-batch behavior.
        names = [n for n in ("ttft_dist", "e2e_dist") if exceeded(n)]
        hints.append(
            {
                "hint": "时间模型偏差，回 L1 误差表对照",
                "metrics": names,
            }
        )
    if exceeded("dispatch_reason_share"):
        hints.append(
            {
                "hint": "切批行为偏，查 predicted_cap 占比与 batch.predict.gap",
                "metrics": ["dispatch_reason_share"],
            }
        )
    if exceeded("cache_hit_token_pct"):
        hints.append(
            {
                "hint": "前缀/KV 机制差，查 spb 与 LRU 语义（A1/A5）",
                "metrics": ["cache_hit_token_pct"],
            }
        )
    if (exceeded("tps_context_with_cache") or exceeded("tps_generate")) and all(
        quiet(n) for n in ("ttft_dist", "e2e_dist", "schedule_latency")
    ):
        hints.append(
            {
                "hint": "记账口径问题非行为问题",
                "metrics": [
                    n for n in ("tps_context_with_cache", "tps_generate") if exceeded(n)
                ],
            }
        )
    return hints


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

_KIND_UNITS = {
    "w1_ms": ("W1", "ms"),
    "ks": ("KS", ""),
    "pp": ("Δpp", "pp"),
    "rel_pct": ("rel", "%"),
    "gini": ("Δgini", ""),
}


def _fmt_side_value(result, key):
    v = result.get(key)
    if isinstance(v, dict):
        return ",".join(f"{r}={val:.1f}" for r, val in v.items())
    if v is None:
        return "—"
    kind = result.get("kind")
    if kind == "w1_ms":
        return f"{v:.0f}ms"
    if isinstance(v, float):
        return f"{v:.2f}"
    return str(v)


def render_stdout(payload):
    lines = []
    lines.append(
        f"L3 twin 分布对比  mock={payload['mock']['label']}  real={payload['real']['label']}"
    )
    win = payload["steady_window"]
    lines.append(
        f"稳态窗口 [{win['lo_s']:.1f}s, {win['hi_s']:.1f}s]（{win['source']}）"
        f"  噪声地板: {payload['floor_source']}"
    )
    for w in payload["warnings"]:
        lines.append(f"WARNING: {w}")
    lines.append("")

    header = (
        f"  {'指标':<24} {'mock':>12} {'real':>12} {'距离':>14} "
        f"{'floor':>10} {'判定':>9}  归因提示"
    )
    lines.append(header)
    lines.append("  " + "-" * (len(header) + 30))
    order = {
        VERDICT_DIVERGED: 0,
        VERDICT_DEVIATED: 1,
        VERDICT_ALIGNED: 2,
        VERDICT_SKIP: 3,
        VERDICT_NA: 4,
    }
    for m in sorted(payload["metrics"], key=lambda m: (order[m["verdict"]], m["name"])):
        kind_label, unit = _KIND_UNITS.get(m["kind"], (m["kind"], ""))
        if m["verdict"] in (VERDICT_SKIP, VERDICT_NA):
            reason = m.get("skip_reason") or m.get("na_reason") or ""
            dist_s = "—"
            floor_s = "—"
            note = reason
        else:
            d = m["distance"]
            dist_s = f"{d:.2f}{unit}" if d is not None else "—"
            fl = m["floor"]
            floor_s = f"{fl:.2f}{unit}" if fl is not None else "—"
            note = m.get("attribution") or ""
        lines.append(
            f"  {m['name']:<24} {_fmt_side_value(m, 'mock_value'):>12} "
            f"{_fmt_side_value(m, 'real_value'):>12} {dist_s:>14} "
            f"{floor_s:>10} {m['verdict']:>9}  {note}"
        )
    lines.append("")

    summary = payload["summary"]
    lines.append(
        f"判定: ALIGNED {summary['aligned']} | DEVIATED {summary['deviated']} | "
        f"DIVERGED {summary['diverged']} | SKIP {summary['skip']} | N/A {summary['na']}"
    )
    if payload["hints"]:
        lines.append("归因提示:")
        for h in payload["hints"]:
            lines.append(f"  - [{', '.join(h['metrics'])}] {h['hint']}")
    gate = payload["gate"]
    lines.append(
        "结论: "
        + (
            "PASS — 无 DIVERGED 指标 (exit 0)"
            if gate["passed"]
            else "FAIL — 存在 DIVERGED 指标 (exit 1)"
        )
    )
    return "\n".join(lines)


_HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="utf-8">
<title>L3 twin 对比 — {label_m} vs {label_r}</title>
<style>
body {{ font-family: 'SF Mono', Menlo, Consolas, monospace; margin: 24px auto;
       max-width: 1180px; color: #1a1a1a; background: #fafafa; }}
h1 {{ font-size: 18px; }}
.meta {{ color: #666; font-size: 13px; margin-bottom: 12px; }}
table {{ border-collapse: collapse; width: 100%; font-size: 13px; margin-bottom: 24px; }}
th, td {{ border: 1px solid #ddd; padding: 5px 10px; text-align: right; }}
th {{ background: #f0f0f0; }}
td.name {{ text-align: left; }}
td.note {{ text-align: left; color: #555; }}
.v-ALIGNED {{ color: #1e7e34; font-weight: bold; }}
.v-DEVIATED {{ color: #b8860b; font-weight: bold; }}
.v-DIVERGED {{ color: #c0392b; font-weight: bold; }}
.v-SKIP {{ color: #888; }}
.v-N/A {{ color: #888; }}
.hints li {{ margin-bottom: 4px; }}
.gate-pass {{ color: #1e7e34; font-weight: bold; }}
.gate-fail {{ color: #c0392b; font-weight: bold; }}
</style>
</head>
<body>
<h1>L3 twin 分布对比（mock vs real）</h1>
<div class="meta">mock = {label_m} &nbsp;|&nbsp; real = {label_r} &nbsp;|&nbsp;
稳态窗口 [{lo:.1f}s, {hi:.1f}s] &nbsp;|&nbsp; 噪声地板: {floor_source}</div>
{warnings}
<table>
<tr><th class="name">指标</th><th>mock</th><th>real</th><th>距离</th>
<th>floor</th><th>判定</th><th class="name">归因提示 / 说明</th></tr>
{rows}
</table>
<h3>归因提示</h3>
<ul class="hints">{hint_items}</ul>
<p>结论: <span class="{gate_cls}">{gate_text}</span></p>
</body>
</html>
"""


def render_html(payload):
    rows = []
    for m in sorted(
        payload["metrics"],
        key=lambda m: (
            m["verdict"] != VERDICT_DIVERGED,
            m["verdict"] != VERDICT_DEVIATED,
            m["name"],
        ),
    ):
        kind_label, unit = _KIND_UNITS.get(m["kind"], (m["kind"], ""))
        if m["verdict"] in (VERDICT_SKIP, VERDICT_NA):
            dist_s = floor_s = "—"
            note = m.get("skip_reason") or m.get("na_reason") or ""
        else:
            d, fl = m["distance"], m["floor"]
            dist_s = f"{d:.2f}{unit}" if d is not None else "—"
            floor_s = f"{fl:.2f}{unit}" if fl is not None else "—"
            note = m.get("attribution") or ""
        rows.append(
            '<tr><td class="name">%s</td><td>%s</td><td>%s</td><td>%s</td>'
            '<td>%s</td><td class="v-%s">%s</td><td class="note">%s</td></tr>'
            % (
                m["name"],
                _fmt_side_value(m, "mock_value"),
                _fmt_side_value(m, "real_value"),
                dist_s,
                floor_s,
                m["verdict"],
                m["verdict"],
                note,
            )
        )
    hint_items = (
        "".join(
            f"<li>[{', '.join(h['metrics'])}] {h['hint']}</li>"
            for h in payload["hints"]
        )
        or "<li>（无）</li>"
    )
    warn_html = ""
    if payload["warnings"]:
        items = "".join(f"<li>{w}</li>" for w in payload["warnings"])
        warn_html = f'<ul style="color:#b8860b">{items}</ul>'
    passed = payload["gate"]["passed"]
    return _HTML_TEMPLATE.format(
        label_m=payload["mock"]["label"],
        label_r=payload["real"]["label"],
        lo=payload["steady_window"]["lo_s"],
        hi=payload["steady_window"]["hi_s"],
        floor_source=payload["floor_source"],
        warnings=warn_html,
        rows="\n".join(rows),
        hint_items=hint_items,
        gate_cls="gate-pass" if passed else "gate-fail",
        gate_text=(
            "PASS — 无 DIVERGED (exit 0)" if passed else "FAIL — 存在 DIVERGED (exit 1)"
        ),
    )


def build_payload(mock, real, lo, hi, win_source, results, floors):
    """Fill verdicts first, THEN attribute (the rule table reads verdicts)."""
    for m in results:
        floor_info = floors.get(m["name"]) or {}
        verdict, ratio = verdict_for(m, floor_info.get("floor"))
        m["verdict"] = verdict
        m["floor"] = floor_info.get("floor")
        m["floor_source"] = floor_info.get("source")
        m["floor_ratio"] = (
            round(ratio, 3)
            if ratio is not None and ratio != float("inf")
            else ("inf" if ratio == float("inf") else None)
        )
    hints = attribute_hints(results)
    hint_by_metric = {}
    for h in hints:
        for name in h["metrics"]:
            hint_by_metric.setdefault(name, h["hint"])
    for m in results:
        m["attribution"] = hint_by_metric.get(m["name"], "")
    counts = {k: 0 for k in ("aligned", "deviated", "diverged", "skip", "na")}
    for m in results:
        counts[
            {
                "ALIGNED": "aligned",
                "DEVIATED": "deviated",
                "DIVERGED": "diverged",
                "SKIP": "skip",
                "N/A": "na",
            }[m["verdict"]]
        ] += 1
    passed = counts["diverged"] == 0
    measured_meta = floors.get("_measured", {}).get("source", "")
    floor_source = (
        "实测噪声地板（" + measured_meta + "）"
        if "measured=True" in measured_meta
        else "经验地板非实测（" + measured_meta + "）"
    )
    return {
        "tool": "compare_twin",
        "mock": {
            "label": mock.label,
            "kind": "mock",
            "per_request_source": mock.per_request_source,
        },
        "real": {
            "label": real.label,
            "kind": "real",
            "per_request_source": real.per_request_source,
        },
        "steady_window": {"lo_s": lo, "hi_s": hi, "source": win_source},
        "floor_source": floor_source,
        "warnings": mock.warnings + real.warnings,
        "summary": counts,
        "hints": hints,
        "gate": {"passed": passed, "exit_code": 0 if passed else 1},
        "metrics": results,
    }


def metrics_to_json(payload):
    doc = {k: v for k, v in payload.items() if k != "metrics"}
    doc["metrics"] = [
        {
            "name": m["name"],
            "group": m["group"],
            "kind": m["kind"],
            "mock_value": m.get("mock_value"),
            "real_value": m.get("real_value"),
            "distance": m.get("distance"),
            "abs_diff": m.get("abs_diff"),
            "floor": m.get("floor"),
            "floor_source": m.get("floor_source"),
            "floor_ratio": m.get("floor_ratio"),
            "verdict": m["verdict"],
            "skip_reason": m.get("skip_reason"),
            "na_reason": m.get("na_reason"),
            "attribution": m.get("attribution"),
            "details": m.get("details"),
        }
        for m in payload["metrics"]
    ]
    return doc


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args(argv=None):
    ap = argparse.ArgumentParser(
        description="L3 twin replay distribution comparison: mock aggregate "
        "vs real client_events/prom export, distance metrics + noise floor + "
        "attribution hints."
    )
    ap.add_argument(
        "--mock-aggregate", required=True, help="mock run dir or aggregate.json path"
    )
    ap.add_argument(
        "--mock-runs",
        default=None,
        help="comma-separated extra mock runs (dirs or aggregate.json "
        "paths) for the measured noise floor (>=2 sides incl. "
        "primary recommended, >=3 per task)",
    )
    ap.add_argument(
        "--real-client-events",
        required=True,
        help="real-side client_events.jsonl (JavaLoadClient format)",
    )
    ap.add_argument(
        "--real-prom",
        default=None,
        help="optional real-side 1s prometheus export "
        "(prom_export.jsonl, long or wide format)",
    )
    ap.add_argument(
        "--out",
        default="twin",
        help="output prefix (default 'twin' -> twin_summary.json "
        "+ twin_report.html; '-' for stdout only)",
    )
    ap.add_argument(
        "--steady-lo",
        type=float,
        default=None,
        help="steady window lower bound seconds (default 0.25*duration)",
    )
    ap.add_argument(
        "--steady-hi",
        type=float,
        default=None,
        help="steady window upper bound seconds (default 0.92*duration)",
    )
    ap.add_argument("--no-html", action="store_true", help="skip the HTML report")
    return ap.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    try:
        mock = load_mock_side(args.mock_aggregate)
        extra = []
        if args.mock_runs:
            for p in args.mock_runs.split(","):
                p = p.strip()
                if p:
                    extra.append(load_mock_side(p))
        real = load_real_side(args.real_client_events, args.real_prom)
        lo, hi = mock.steady_window(args.steady_lo, args.steady_hi)
    except InputError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    win_source = (
        "explicit"
        if args.steady_lo is not None and args.steady_hi is not None
        else f"derived from mock duration_s={mock.duration_s}"
    )
    results = evaluate_metrics(mock, real, lo, hi)
    floors = compute_floors(mock, extra, lo, hi)
    payload = build_payload(mock, real, lo, hi, win_source, results, floors)

    print(render_stdout(payload))

    if args.out != "-":
        out_dir = os.path.dirname(os.path.abspath(args.out))
        os.makedirs(out_dir, exist_ok=True)
        json_path = args.out + "_summary.json"
        with open(json_path, "w", encoding="utf-8") as fh:
            json.dump(metrics_to_json(payload), fh, indent=2, ensure_ascii=False)
        print(f"\nJSON summary -> {json_path}")
        if not args.no_html:
            html_path = args.out + "_report.html"
            with open(html_path, "w", encoding="utf-8") as fh:
                fh.write(render_html(payload))
            print(f"HTML report  -> {html_path}")
    return payload["gate"]["exit_code"]


if __name__ == "__main__":
    sys.exit(main())
