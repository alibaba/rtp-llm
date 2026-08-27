#!/usr/bin/env python3
"""Unified online_eval run analyzer: every dimension, one report, zero deps.

Reads one run directory (legacy layout first, consolidated run-root files as
fallback — the same precedence consolidate_run_outputs.py documents) and
produces <run_dir>/analysis/analysis_report.json plus a self-contained
analysis_report.html (inline CSS/JS, hand-drawn SVG charts, no CDN).

Dimension -> data source map
---------------------------------------------------------------------------
 0  qps_and_errors      per_request.jsonl (per-second send/ok/err, error
                        taxonomy via regex library) + summary totals
 1  latency_layers      server_latency.json (5 stage histograms) + per_request
                        schedule_ms per-second p50/p95/p99 + master.log
                        flexlb_server_schedule_latency 10s rows + e2e
                        ttft/total (auto-disabled when fetch_output_stream=0)
 2  queues              java_mock_stats timeline (mock.json stats[] or legacy
                        mock_engine.log); per-engine panel from mock.json
                        per_engine_timeseries when present, else a
                        decode_run_min/max band as the degraded view
 3  balance             per_request prefill/decode engine counters (Gini/CV/
                        max-min/p90p10) + final_snapshot accepted + mock stats
                        extremes; utilization vs decode capacity from
                        run_meta.params (java_mock_decode_max_concurrency)
 4  inflight            G4 inflight_timeseries (scheduler + per-endpoint
                        prefill/decode curves) as the primary signal; master
                        counters arrival-completion delta (schedule-RPC
                        inflight, auxiliary) + prom inflight_max_age
                        per-second curve + mock.log "LEAK DETECTED" scan +
                        final_snapshot zeros
 5  kv_usage            master prometheus flexlb_app_cache_* kv_cache metrics
                        (final snapshot; needs FLEXLB_MONITOR_MODE=all)
 6  kv_match            master prometheus hit_ratio / theory_hit_ratio /
                        recent_key series (final snapshot) + per-second
                        hit_ratio curves from prometheus_timeseries
 7  cpu_mem             mock stats heap curve + prom jvm_gc_pause_seconds_*
                        + future process_usage_timeseries.txt (per-JVM curves)
 8  pacing              per_request pacing_lag_ms per-second p50/p99 +
                        actual vs target QPS
 9  sla                 per-second SLA violation rate curve + summary totals
10  dispatch            master.log flexlb_batch_dispatch/complete: reason
                        distribution, predicted vs actual gap, the three
                        decision invariants (analyze_slo_batch logic)
11  priority            P30/50/70 facets: ok/err, latency, 8429 counts
12  error_code_matrix   extracted code=N rows crossed with priority
13  length_matrix       input_len/output_len deciles x schedule p99
14  client_concurrency  send_start/total_ms event replay -> per-second
                        in-flight watermark
15  fallback_path       enqueued_by_master=false isolated (and excluded from
                        the primary schedule percentiles)
---------------------------------------------------------------------------
Usage:
  python3 analyze_eval_run.py <run_dir> [--compare <run_dir2> ...]
  python3 analyze_eval_run.py --self-test [--self-test-root /tmp/analyzer_test]

Standard library only; Python 3.9+.
"""

from __future__ import annotations

import argparse
import gzip
import html
import json
import math
import re
import sys
from array import array
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

# ---------------------------------------------------------------------------
# constants / shared regexes
# ---------------------------------------------------------------------------

DIMENSIONS = [
    "qps_and_errors",
    "latency_layers",
    "queues",
    "balance",
    "inflight",
    "kv_usage",
    "kv_match",
    "cpu_mem",
    "pacing",
    "sla",
    "dispatch",
    "priority",
    "error_code_matrix",
    "length_matrix",
    "client_concurrency",
    "fallback_path",
]

STAT_KV_RE = re.compile(r"([a-z_]+)=(-?\d+(?:\.\d+)?)")
SEND_START_RE = re.compile(r'"send_start_epoch_ms"\s*:\s*(-?[\d.eE+]+)')

DISPATCH_RE = re.compile(
    r"flexlb_batch_dispatch batch_id=(?P<batch_id>\d+) "
    r"reason=(?P<reason>\S+) batch_size=(?P<batch_size>\d+) "
    r"wait_ms=(?P<wait_ms>\d+) predicted_ms=(?P<predicted_ms>\d+) "
    r"threshold_ms=(?P<threshold_ms>\d+) fixed_wait_ms=(?P<fixed_wait_ms>\d+) "
    r"batch_size_max=(?P<batch_size_max>\d+) queue_after=(?P<queue_after>\d+) "
    r"worker=(?P<worker>\S*)"
)
COMPLETE_RE = re.compile(
    r"flexlb_batch_complete batch_id=(?P<batch_id>\d+) "
    r"predicted_ms=(?P<predicted_ms>-?\d+) actual_ms=(?P<actual_ms>-?\d+) "
    r"gap_ms=(?P<gap_ms>-?\d+) batch_size=(?P<batch_size>\d+) "
    r"engine=(?P<engine>\S+)"
)
SERVER_LAT_RE = re.compile(
    r"flexlb_server_schedule_latency count=(?P<count>\d+) "
    r"arrival_qps=(?P<arrival_qps>[\d.]+) completion_qps=(?P<completion_qps>[\d.]+) "
    r"server_p50_ms=(?P<server_p50_ms>[\d.]+) server_p95_ms=(?P<server_p95_ms>[\d.]+) "
    r"server_p99_ms=(?P<server_p99_ms>[\d.]+) grpc_queue_p95_ms=(?P<grpc_queue_p95_ms>[\d.]+) "
    r"route_submit_p95_ms=(?P<route_submit_p95_ms>[\d.]+) "
    r"batch_wait_p95_ms=(?P<batch_wait_p95_ms>[\d.]+) "
    r"dispatch_ack_p95_ms=(?P<dispatch_ack_p95_ms>[\d.]+) "
    r"ack_response_p95_ms=(?P<ack_response_p95_ms>[\d.]+)"
)
PRIO_BATCH_WAIT_RE = re.compile(r"batch_wait_p95_prio(\d+)_ms=([\d.]+)")
PROM_SAMPLE_RE = re.compile(
    r"^(?P<name>[a-zA-Z_:][a-zA-Z0-9_:]*)(?P<labels>\{[^}]*\})?\s+"
    r"(?P<value>[-+\deE.]+)(\s+\d+)?\s*$"
)
# G1 per-engine timeline: mock.json per_engine_timeseries is the flat
# [{ts, metrics: {"name{labels}": value}}] layout produced by
# consolidate_run_outputs.py from the mock per-engine poller. The keys of the
# metrics dict are Prometheus samples like
# mock_engine_running{engine_name="decode-0",role="decode",grpc_port="62301",...}.
ENGINE_METRIC_KEY_RE = re.compile(
    r"^(?P<metric>[a-zA-Z_:][a-zA-Z0-9_:]*)\{(?P<labels>[^}]*)\}$"
)
LABEL_PAIR_RE = re.compile(r'([a-zA-Z_][a-zA-Z0-9_]*)="([^"]*)"')
# G5 process usage: run_meta.json process_usage rows / the raw poller file
# (process_usage_timeseries.txt, kv lines written by run_online_eval.sh).
PROCESS_USAGE_KV_RE = re.compile(
    r"^ts_epoch_ms=(?P<ts>\d+)\s+label=(?P<label>\S+)\s+pid=(?P<pid>\d+)\s+"
    r"cpu_pct=(?P<cpu>[-+]?\d+(?:\.\d+)?)\s+rss_kb=(?P<rss>[-+]?\d+)\s+"
    r"etime=(?P<etime>\S+)\s*$"
)
# gRPC-ish error codes: "code=8430", "code:8429", or a bare 8xxx token.
ERR_CODE_RE = re.compile(r"code[=: ]\s*(\d{3,4})")
ERR_BARE_CODE_RE = re.compile(r"\b(8[0-9]{3})\b")

OK_STATUSES = ("ok", "scheduled")
ERR_STATUSES = ("schedule_error", "exception", "timeout", "empty_response")

# Error taxonomy. Order matters: earlier rules win (preempted text beats the
# bare 84xx range, "yielded to higher-priority" beats the no_worker range that
# shares code 8400, backpressure text beats the admission codes that often
# ride along, client_timeout beats network_close because DEADLINE_EXCEEDED
# payloads also contain "closed=[]" fragments).
ERROR_RULES: List[Tuple[str, Any]] = [
    (
        "preempted_8429",
        lambda s, e: "code=8429" in e or "preempted by higher-priority" in e,
    ),
    (
        "yielded_8400",
        lambda s, e: "yielded to higher-priority" in e or "code=8400" in e,
    ),
    ("backpressure", lambda s, e: "active_admissions" in e),
    (
        "admission_8430",
        lambda s, e: "8430" in ERR_CODE_RE.findall(e)
        or "8430" in ERR_BARE_CODE_RE.findall(e),
    ),
    (
        "resource_8431",
        lambda s, e: "8431" in ERR_CODE_RE.findall(e)
        or "8431" in ERR_BARE_CODE_RE.findall(e),
    ),
    (
        "admission_8432",
        lambda s, e: "8432" in ERR_CODE_RE.findall(e)
        or "8432" in ERR_BARE_CODE_RE.findall(e),
    ),
    (
        "queue_full_8502",
        lambda s, e: "8502" in ERR_CODE_RE.findall(e) or "queue full" in e,
    ),
    (
        "slo_expired_8511",
        lambda s, e: "8511" in ERR_CODE_RE.findall(e) or "SLO expired" in e,
    ),
    (
        "dispatch_failed",
        lambda s, e: any(
            c in ERR_CODE_RE.findall(e) or c in ERR_BARE_CODE_RE.findall(e)
            for c in ("8510", "8512", "8514", "8515")
        ),
    ),
    (
        "engine_exec",
        lambda s, e: any(
            c in ERR_CODE_RE.findall(e) or c in ERR_BARE_CODE_RE.findall(e)
            for c in ("8513", "8202", "8203")
        ),
    ),
    (
        "no_worker_8400_8407",
        lambda s, e: "NO_DECODE_WORKER" in e
        or "NO_AVAILABLE_WORKER" in e
        or any(
            "840" == c[:3] and 0 <= int(c[3]) <= 7 for c in ERR_BARE_CODE_RE.findall(e)
        ),
    ),
    (
        "client_timeout",
        lambda s, e: s == "timeout" or "DEADLINE_EXCEEDED" in e or "deadline" in e,
    ),
    (
        "network_close",
        lambda s, e: any(
            t in e for t in ("RST_STREAM", "GOAWAY", "UNAVAILABLE", "reset", "closed")
        ),
    ),
    ("empty_response", lambda s, e: s == "empty_response"),
]
BACKPRESSURE_ACTIVE_RE = re.compile(r"active_admissions=(\d+)")
BACKPRESSURE_LIMIT_RE = re.compile(r"limit=(\d+)")

# ---------------------------------------------------------------------------
# small statistics helpers
# ---------------------------------------------------------------------------


def percentile(values: Sequence[float], quantile: float) -> float:
    """Canvas-aggregator-compatible percentile (sorted on demand)."""
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, int(len(ordered) * quantile))
    return round(float(ordered[idx]), 3)


def dist_stats(values: Sequence[float]) -> Dict[str, float]:
    if not values:
        return {
            "count": 0,
            "mean": 0.0,
            "p50": 0.0,
            "p90": 0.0,
            "p95": 0.0,
            "p99": 0.0,
            "max": 0.0,
        }
    ordered = sorted(values)
    return {
        "count": len(ordered),
        "mean": round(sum(ordered) / len(ordered), 3),
        "p50": percentile(ordered, 0.50),
        "p90": percentile(ordered, 0.90),
        "p95": percentile(ordered, 0.95),
        "p99": percentile(ordered, 0.99),
        "max": round(float(ordered[-1]), 3),
    }


def gini(values: Sequence[float]) -> float:
    """Gini coefficient for non-negative counts (0 = perfectly even)."""
    vals = sorted(float(v) for v in values if v is not None)
    n = len(vals)
    total = sum(vals)
    if n == 0 or total <= 0:
        return 0.0
    cumulative = 0.0
    for rank, v in enumerate(vals, start=1):
        cumulative += rank * v
    return round(max(0.0, (2.0 * cumulative) / (n * total) - (n + 1.0) / n), 4)


def cv(values: Sequence[float]) -> float:
    vals = [float(v) for v in values if v is not None]
    if len(vals) < 2:
        return 0.0
    mean = sum(vals) / len(vals)
    if mean == 0:
        return 0.0
    variance = sum((v - mean) ** 2 for v in vals) / len(vals)
    return round(math.sqrt(variance) / mean, 4)


def spread_ratios(values: Sequence[float]) -> Dict[str, float]:
    vals = sorted(float(v) for v in values if v is not None)
    if not vals:
        return {"max_min": 0.0, "p90_p10": 0.0}
    p10 = percentile(vals, 0.10)
    p90 = percentile(vals, 0.90)
    ratio = round(p90 / p10, 4) if p10 > 0 else 0.0
    mm = round(vals[-1] / vals[0], 4) if vals[0] > 0 else 0.0
    return {"max_min": mm, "p90_p10": ratio}


def balance_grade(gini_value: float) -> str:
    if gini_value < 0.05:
        return "excellent"
    if gini_value < 0.10:
        return "good"
    if gini_value < 0.20:
        return "fair"
    return "poor"


def fmt_num(value: Any) -> str:
    try:
        return f"{float(value):,.1f}" if isinstance(value, float) else f"{int(value):,}"
    except (TypeError, ValueError):
        return str(value)


# ---------------------------------------------------------------------------
# defensive loaders
# ---------------------------------------------------------------------------


def load_json(path: Path) -> Any:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def load_json_dict(path: Path) -> Dict[str, Any]:
    payload = load_json(path)
    return payload if isinstance(payload, dict) else {}


def open_text(path: Path):
    return (
        gzip.open(path, "rt", encoding="utf-8", errors="replace")
        if path.suffix == ".gz"
        else path.open("r", encoding="utf-8", errors="replace")
    )


def parse_grouped_prometheus(path: Path) -> List[Dict[str, Any]]:
    """Grouped prom text (``# ts=<epoch_ms>`` separators) -> [{ts, metrics}].

    Same shape as consolidate_run_outputs.parse_grouped_prometheus_timeseries
    (kept local so the analyzer stays a single stdlib-only file): samples
    inside a group are parsed into a flat ``{name{labels}: value}`` dict;
    HELP/TYPE comments and lines before the first ``# ts=`` marker are
    skipped, so a torn trailing sample simply never lands in a group.
    """
    if not path.is_file():
        return []
    groups: List[Dict[str, Any]] = []
    current: Optional[Dict[str, Any]] = None
    with path.open("r", encoding="utf-8", errors="replace") as stream:
        for line in stream:
            line = line.strip()
            if line.startswith("#") and "ts=" in line:
                match = re.search(r"ts=(\d+)", line)
                if match:
                    current = {"ts": int(match.group(1)), "metrics": {}}
                    groups.append(current)
                continue
            if current is None or not line or line.startswith("#"):
                continue
            sample = PROM_SAMPLE_RE.match(line)
            if not sample:
                continue
            try:
                value = float(sample.group("value"))
            except ValueError:
                continue
            current["metrics"][
                sample.group("name") + (sample.group("labels") or "")
            ] = value
    return groups


def parse_jsonl_rows(path: Path) -> List[Dict[str, Any]]:
    """JSONL file with one JSON object per line -> list of objects.

    Each line is json.loads'd independently (torn trailing lines dropped).
    """
    if not path.is_file():
        return []
    rows: List[Dict[str, Any]] = []
    with open_text(path) as stream:
        for line in stream:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except ValueError:
                continue
            if isinstance(row, dict):
                rows.append(row)
    return rows


def prom_ts_series(
    prom_ts: Sequence[Dict[str, Any]], metric_name: str, base: Optional[float] = None
) -> Dict[str, Any]:
    """Extract one per-second curve from a G3 [{ts, metrics}] timeline.

    All label variants of ``metric_name`` are SUMMED per group (queue gauges
    tagged by priority add up to the total queue depth). ``base`` (the
    per-request send_start floor) re-anchors the t axis so the curve lines
    up with the client-side timelines; groups whose ts predates base by more
    than an hour are treated as a different clock domain and keep their own
    first-sample origin.
    """
    pts: List[Tuple[float, float]] = []
    for group in prom_ts:
        if not isinstance(group, dict):
            continue
        ts = float(group.get("ts", 0) or 0)
        metrics = group.get("metrics")
        if not isinstance(metrics, dict):
            continue
        total: Optional[float] = None
        for key, value in metrics.items():
            if not isinstance(value, (int, float)):
                continue
            if str(key).split("{", 1)[0] == metric_name:
                total = (total or 0.0) + float(value)
        if total is not None:
            pts.append((ts, round(total, 3)))
    if not pts:
        return {"available": False}
    t0 = pts[0][0]
    if base and (t0 - base) > -3600_000:
        t0 = base
    series_t = [int((ts - t0) // 1000) for ts, _ in pts]
    values = [v for _, v in pts]
    return {
        "available": True,
        "t": series_t,
        "series": values,
        "avg": round(sum(values) / len(values), 3),
        "max": max(values),
        "min": min(values),
    }


def prom_ts_ratio_series(
    prom_ts: Sequence[Dict[str, Any]], metric_name: str, base: Optional[float] = None
) -> Dict[str, Any]:
    """Like prom_ts_series but for ratio gauges: label variants are averaged
    (summing hit ratios across engines is meaningless)."""
    pts: List[Tuple[float, float]] = []
    for group in prom_ts:
        if not isinstance(group, dict):
            continue
        ts = float(group.get("ts", 0) or 0)
        metrics = group.get("metrics")
        if not isinstance(metrics, dict):
            continue
        values: List[float] = []
        for key, value in metrics.items():
            if not isinstance(value, (int, float)):
                continue
            if str(key).split("{", 1)[0] == metric_name:
                values.append(float(value))
        if values:
            pts.append((ts, round(sum(values) / len(values), 5)))
    if not pts:
        return {"available": False}
    t0 = pts[0][0]
    if base and (t0 - base) > -3600_000:
        t0 = base
    return {
        "available": True,
        "t": [int((ts - t0) // 1000) for ts, _ in pts],
        "series": [v for _, v in pts],
        "avg": round(sum(v for _, v in pts) / len(pts), 5),
    }


def unified_t(ts_epoch_ms: float, base: Optional[float], series_t0: float) -> int:
    """m4: anchor every timeline on the per-request send_start floor so t
    values are comparable across dimensions. A series whose clock clearly
    disagrees (older than base by >1h, e.g. relative timestamps) keeps its
    own first-sample origin."""
    ts = float(ts_epoch_ms or 0)
    if ts and base and (ts - base) > -3600_000:
        return int((ts - base) // 1000)
    if series_t0:
        return int((ts - series_t0) // 1000)
    return 0


# ---------------------------------------------------------------------------
# resolvers: legacy layout first, consolidated run-root files as fallback.
# Every resolver records the file it used so report.meta.sources can pin the
# provenance of each dimension.
# ---------------------------------------------------------------------------


class RunData:
    """Resolved view of one run directory (legacy layout preferred)."""

    def __init__(self, run_dir: Path) -> None:
        self.run_dir = run_dir
        self.sources: Dict[str, str] = {}
        self.notes: List[str] = []

        # M2: load each component JSON exactly once — on large runs mock.json
        # approaches ~1GB and three resolvers re-parsing it tripled the wall
        # time (and the transient memory peak). After the A-split mock.json
        # stays lightweight, but the cache still guards the legacy embedded
        # per_engine_timeseries layout.
        self._mock_json = load_json_dict(run_dir / "mock.json")
        self._master_json = load_json_dict(run_dir / "master.json")

        self.per_request_paths = self._resolve_per_request()
        self.mock_stats = self._resolve_mock_stats()
        self.mock_per_engine = self._resolve_mock_per_engine()
        self.final_snapshot = self._resolve_final_snapshot()
        self.master_counters = self._resolve_master_counters()
        self.master_prometheus = self._resolve_master_prometheus()
        self.master_prometheus_timeseries = self._resolve_master_prometheus_timeseries()
        self.inflight_timeseries = self._resolve_inflight_timeseries()
        self.master_log_paths = self._resolve_master_logs()
        self.mock_log_paths = self._resolve_mock_logs()
        self.server_latency = self._resolve_server_latency()
        self.summary = self._resolve_summary()
        self.run_meta = load_json_dict(run_dir / "run_meta.json")
        if self.run_meta:
            self.sources["run_meta"] = "run_meta.json"
        self.params = (
            self.run_meta.get("params", {})
            if isinstance(self.run_meta.get("params"), dict)
            else {}
        )
        self.process_usage_rows = self._resolve_process_usage()

    def _record(self, key: str, path: Path, extra: str = "") -> None:
        self.sources[key] = str(path.relative_to(self.run_dir)) + (
            f" ({extra})" if extra else ""
        )

    def _resolve_per_request(self) -> List[Path]:
        lc = self.run_dir / "load_client"
        for pattern in ("shard_*/per_request.jsonl",):
            shards = sorted(lc.glob(pattern)) if lc.is_dir() else []
            if shards:
                for p in shards:
                    self._record("per_request", p, "shard")
                return shards
        single = lc / "per_request.jsonl"
        if single.is_file():
            self._record("per_request", single)
            return [single]
        for name in ("per_request.jsonl", "per_request.jsonl.gz"):
            p = self.run_dir / name
            if p.is_file():
                self._record("per_request", p)
                return [p]
        self.sources["per_request"] = "missing"
        return []

    def _parse_mock_stats_lines(self, paths: List[Path]) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        for path in paths:
            if not path.is_file():
                continue
            with path.open("r", encoding="utf-8", errors="replace") as stream:
                for line in stream:
                    if "java_mock_stats " not in line:
                        continue
                    snap = {
                        k: (float(v) if "." in v else int(v))
                        for k, v in STAT_KV_RE.findall(line)
                    }
                    if snap:
                        rows.append(snap)
        return rows

    def _resolve_mock_stats(self) -> List[Dict[str, Any]]:
        legacy = self.run_dir / "mock_engine.log"
        if legacy.is_file():
            rows = self._parse_mock_stats_lines([legacy])
            if rows:
                self._record("mock_stats", legacy, f"{len(rows)} samples")
                return rows
        stats = self._mock_json.get("stats")
        if isinstance(stats, list) and stats:
            self._record(
                "mock_stats", self.run_dir / "mock.json", f"{len(stats)} samples"
            )
            return [r for r in stats if isinstance(r, dict)]
        # Third fallback: the consolidated mock.log still carries the raw lines.
        rows = self._parse_mock_stats_lines([self.run_dir / "mock.log"])
        if rows:
            self._record(
                "mock_stats", self.run_dir / "mock.log", f"{len(rows)} samples"
            )
            return rows
        self.sources["mock_stats"] = "missing"
        return []

    def _resolve_mock_per_engine(self) -> Dict[str, List[Dict[str, Any]]]:
        """A-split fallback chain: split gzip file first, then the legacy
        mock.json embedded key, then the raw pre-consolidation prom file."""
        gz_path = self.run_dir / "mock_per_engine_timeseries.json.gz"
        if gz_path.is_file():
            try:
                with gzip.open(gz_path, "rt", encoding="utf-8") as stream:
                    payload = json.load(stream)
            except (OSError, ValueError):
                payload = None
            if isinstance(payload, list) and payload:
                grouped = self._regroup_per_engine_timeline(payload)
                if grouped:
                    self.sources["mock_per_engine"] = (
                        "mock_per_engine_timeseries.json.gz (A-split)"
                    )
                    return grouped
        per_engine = self._mock_json.get("per_engine_timeseries")
        if isinstance(per_engine, dict) and per_engine:
            self.sources["mock_per_engine"] = "mock.json per_engine_timeseries"
            return {k: v for k, v in per_engine.items() if isinstance(v, list)}
        if isinstance(per_engine, list) and per_engine:
            grouped = self._regroup_per_engine_timeline(per_engine)
            if grouped:
                self.sources["mock_per_engine"] = (
                    "mock.json per_engine_timeseries (G1 grouped)"
                )
                return grouped
        # m3: raw grouped prom file (legacy, pre-consolidation layout) — the
        # mock poller's on-disk format parses directly.
        prom_path = self.run_dir / "mock_metrics_per_engine.prom"
        if prom_path.is_file():
            groups = parse_grouped_prometheus(prom_path)
            grouped = self._regroup_per_engine_timeline(groups)
            if grouped:
                self._record(
                    "mock_per_engine", prom_path, f"{len(groups)} samples (raw prom)"
                )
                return grouped
        self.sources["mock_per_engine"] = "missing (per-engine collection not enabled)"
        return {}

    @staticmethod
    def _regroup_per_engine_timeline(
        groups: List[Any],
    ) -> Dict[str, List[Dict[str, Any]]]:
        """G1 flat timeline [{ts, metrics}] -> {engine_name: [rows]}.

        The metrics dict is keyed by full Prometheus samples
        (``mock_engine_running{engine_name="decode-0",role="decode",...}``);
        each group is split per engine_name label into rows carrying the flat
        field names the dimension consumers expect. Role-specific fields are
        derived from the role label: a decode engine's running gauge lands in
        decode_running, a prefill engine's in prefill_running (matching the
        java_mock_stats field semantics). kv_cache_ratio is derived as
        active / (active + available).
        """
        grouped: Dict[str, List[Dict[str, Any]]] = {}
        for group in groups:
            if not isinstance(group, dict):
                continue
            ts = group.get("ts", 0)
            metrics = group.get("metrics")
            if not isinstance(metrics, dict):
                continue
            rows: Dict[str, Dict[str, Any]] = {}
            for key, value in metrics.items():
                if not isinstance(value, (int, float)):
                    continue
                match = ENGINE_METRIC_KEY_RE.match(str(key))
                if not match:
                    continue
                labels = dict(LABEL_PAIR_RE.findall(match.group("labels")))
                engine = labels.get("engine_name")
                if not engine:
                    continue
                row = rows.setdefault(
                    engine,
                    {
                        "ts_epoch_ms": ts,
                        "role": labels.get("role", ""),
                    },
                )
                metric = match.group("metric")
                if metric == "mock_engine_running":
                    row["running"] = value
                elif metric == "mock_engine_waiting":
                    row["waiting"] = value
                elif metric == "mock_engine_active_kv_tokens":
                    row["kv_cache_tokens"] = value
                elif metric == "mock_engine_available_kv_tokens":
                    row["kv_cache_available"] = value
            for engine, row in rows.items():
                role = row.pop("role", "")
                if role == "decode":
                    row["decode_running"] = row.get("running", 0)
                    row["decode_waiting"] = row.get("waiting", 0)
                elif role == "prefill":
                    row["prefill_running"] = row.get("running", 0)
                    row["prefill_waiting"] = row.get("waiting", 0)
                active = row.get("kv_cache_tokens")
                available = row.get("kv_cache_available")
                # M4: keep the point even when available==0 — a fully
                # reserved KV pool (active>0, available=0) is ratio 1.0,
                # not a dropped sample.
                if isinstance(active, (int, float)):
                    if available:
                        row["kv_cache_ratio"] = round(active / (active + available), 4)
                    else:
                        row["kv_cache_ratio"] = 1.0 if active > 0 else 0.0
                grouped.setdefault(engine, []).append(row)
        return grouped

    def _resolve_final_snapshot(self) -> Dict[str, Any]:
        snap = self._mock_json.get("final_snapshot")
        if isinstance(snap, dict) and snap:
            self.sources["final_snapshot"] = "mock.json final_snapshot"
            return snap
        self.sources["final_snapshot"] = "missing"
        return {}

    def _resolve_master_counters(self) -> List[Dict[str, Any]]:
        legacy = self.run_dir / "master_counters_timeseries.txt"
        if legacy.is_file():
            rows = []
            with legacy.open("r", encoding="utf-8", errors="replace") as stream:
                for line in stream:
                    row = {
                        k: (float(v) if "." in v else int(v))
                        for k, v in STAT_KV_RE.findall(line)
                    }
                    if row:
                        rows.append(row)
            if rows:
                self._record("master_counters", legacy, f"{len(rows)} samples")
                return rows
        ts = self._master_json.get("counters_timeseries")
        if isinstance(ts, list) and ts:
            self._record(
                "master_counters", self.run_dir / "master.json", f"{len(ts)} samples"
            )
            return [r for r in ts if isinstance(r, dict)]
        self.sources["master_counters"] = "missing"
        return []

    def _resolve_master_prometheus_timeseries(self) -> List[Dict[str, Any]]:
        """B3: G3 per-second master prometheus timeline ([{ts, metrics}]).

        Legacy raw file first (pre-consolidation), then the merged
        master.json prometheus_timeseries key. Consumers: batcher queue
        curves (queues), inflight max age (inflight), hit ratio (kv_match).
        """
        legacy = self.run_dir / "master_prometheus_timeseries.prom"
        if legacy.is_file():
            groups = parse_grouped_prometheus(legacy)
            if groups:
                self._record("master_prometheus_ts", legacy, f"{len(groups)} samples")
                return groups
        ts = self._master_json.get("prometheus_timeseries")
        if isinstance(ts, list) and ts:
            self._record(
                "master_prometheus_ts",
                self.run_dir / "master.json",
                f"{len(ts)} samples",
            )
            return [g for g in ts if isinstance(g, dict)]
        self.sources["master_prometheus_ts"] = "missing"
        return []

    def _resolve_inflight_timeseries(self) -> List[Dict[str, Any]]:
        """M3: G4 per-second /rtp_llm/inflight_status snapshots.

        Rows are {"ts_epoch_ms": ..., "inflight": {...}} with inflight
        carrying scheduler_inflight plus prefill_endpoints / decode_endpoints
        lists (see HttpLoadBalanceServer.inflightStatus).
        """
        legacy = self.run_dir / "master_inflight_timeseries.jsonl"
        if legacy.is_file():
            rows = parse_jsonl_rows(legacy)
            if rows:
                self._record("inflight_ts", legacy, f"{len(rows)} samples")
                return rows
        ts = self._master_json.get("inflight_timeseries")
        if isinstance(ts, list) and ts:
            self._record(
                "inflight_ts", self.run_dir / "master.json", f"{len(ts)} samples"
            )
            return [r for r in ts if isinstance(r, dict)]
        self.sources["inflight_ts"] = "missing"
        return []

    def _resolve_master_prometheus(self) -> Dict[str, float]:
        legacy = self.run_dir / "master_prometheus_after.prom"
        if legacy.is_file():
            metrics: Dict[str, float] = {}
            with legacy.open("r", encoding="utf-8", errors="replace") as stream:
                for line in stream:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    m = PROM_SAMPLE_RE.match(line)
                    if not m:
                        continue
                    try:
                        metrics[m.group("name") + (m.group("labels") or "")] = float(
                            m.group("value")
                        )
                    except ValueError:
                        continue
            if metrics:
                self._record("master_prometheus", legacy, f"{len(metrics)} series")
                return metrics
        prom = self._master_json.get("prometheus_after")
        if isinstance(prom, dict) and prom:
            self._record(
                "master_prometheus", self.run_dir / "master.json", f"{len(prom)} series"
            )
            return {k: float(v) for k, v in prom.items() if isinstance(v, (int, float))}
        self.sources["master_prometheus"] = "missing"
        return {}

    def _resolve_master_logs(self) -> List[Path]:
        log_dir = self.run_dir / "flexlb_logs"
        paths = sorted(log_dir.glob("flexlb.log*")) if log_dir.is_dir() else []
        if paths:
            for p in paths:
                self._record("master_log", p)
            return paths
        for name in ("master.log", "flexlb.log"):
            p = self.run_dir / name
            if p.is_file():
                self._record("master_log", p)
                return [p]
        self.sources["master_log"] = "missing"
        return []

    def _resolve_mock_logs(self) -> List[Path]:
        paths: List[Path] = []
        for name in ("mock_engine.log", "mock.log"):
            p = self.run_dir / name
            if p.is_file():
                paths.append(p)
        if not paths:
            self.sources["mock_log"] = "missing"
        return paths

    def _resolve_server_latency(self) -> Dict[str, Any]:
        legacy = load_json_dict(self.run_dir / "load_client" / "server_latency.json")
        if legacy:
            self.sources["server_latency"] = "load_client/server_latency.json"
            return legacy
        merged = load_json_dict(self.run_dir / "client.json").get("server_latency")
        if isinstance(merged, dict) and merged:
            self.sources["server_latency"] = "client.json server_latency"
            return merged
        self.sources["server_latency"] = "missing"
        return {}

    def _resolve_summary(self) -> Dict[str, Any]:
        legacy = load_json_dict(self.run_dir / "load_client" / "summary.json")
        if legacy:
            self.sources["summary"] = "load_client/summary.json"
            return legacy
        merged = load_json_dict(self.run_dir / "client.json")
        if merged:
            self.sources["summary"] = "client.json"
            return merged
        self.sources["summary"] = "missing"
        return {}

    def _resolve_process_usage(self) -> List[Dict[str, Any]]:
        # G5: consolidation merges process_usage_timeseries.txt into
        # run_meta.json["process_usage"] (kv-line rows) and deletes the txt —
        # read the merged key first. The raw file (pre-consolidation runs)
        # carries the same kv lines as written by the process usage poller;
        # the legacy whitespace 5-column format is kept as a last resort.
        merged = self.run_meta.get("process_usage")
        if isinstance(merged, list) and merged:
            rows = []
            for entry in merged:
                if not isinstance(entry, dict):
                    continue
                try:
                    rows.append(
                        {
                            "ts": entry.get("ts_epoch_ms", 0),
                            "pid": entry.get("pid"),
                            "label": entry.get("label", ""),
                            "cpu": float(entry.get("cpu_pct", 0) or 0),
                            "rss_kb": float(entry.get("rss_kb", 0) or 0),
                            "etime": entry.get("etime", ""),
                        }
                    )
                except (TypeError, ValueError):
                    continue
            if rows:
                self._record(
                    "process_usage",
                    self.run_dir / "run_meta.json",
                    f"{len(rows)} samples (process_usage)",
                )
                return rows
        path = self.run_dir / "process_usage_timeseries.txt"
        if not path.is_file():
            return []
        rows = []
        with path.open("r", encoding="utf-8", errors="replace") as stream:
            for line in stream:
                line = line.strip()
                match = PROCESS_USAGE_KV_RE.match(line)
                if match:
                    try:
                        rows.append(
                            {
                                "ts": int(match.group("ts")),
                                "pid": int(match.group("pid")),
                                "label": match.group("label"),
                                "cpu": float(match.group("cpu")),
                                "rss_kb": float(match.group("rss")),
                                "etime": match.group("etime"),
                            }
                        )
                    except ValueError:
                        continue
                    continue
                # legacy 5-column "<ts> <pid> <cpu>% <rss> <etime>" rows
                parts = line.split()
                if len(parts) < 5 or not parts[0][:1].isdigit():
                    continue
                try:
                    rows.append(
                        {
                            "ts": parts[0],
                            "pid": parts[1],
                            "cpu": float(parts[2].rstrip("%")),
                            "rss_kb": float(parts[3]),
                            "etime": parts[4],
                        }
                    )
                except ValueError:
                    continue
        if rows:
            self._record("process_usage", path, f"{len(rows)} samples")
        return rows

    # ---- derived helpers -------------------------------------------------

    @property
    def fetch_output_stream(self) -> bool:
        """True when the load client read engine output streams in this run.

        False (FETCH_OUTPUT_STREAM=0) means the client skipped the phase-2
        stream read: client-side e2e ttft/total are unavailable, while the
        engine still executed prefill + decode in full.
        """
        value = str(self.params.get("fetch_output_stream", "1")).strip()
        if value in ("0", "false", "False", "no"):
            return False
        if value in ("1", "true", "True", "yes"):
            return True
        # legacy fallback: the env file snapshot in run_meta
        env = self.run_meta.get("flexlb_env", {})
        if isinstance(env, dict):
            return not any("FETCH_OUTPUT_STREAM=0" in str(v) for v in env.values())
        return True

    def decode_capacity(self) -> Optional[int]:
        """Total decode concurrency: per-engine cap x engine count."""
        raw = self.params.get("java_mock_decode_max_concurrency")
        n_decode = self.params.get("n_decode")
        try:
            cap = int(str(raw))
            count = int(str(n_decode))
            return cap * count if cap > 0 and count > 0 else None
        except (TypeError, ValueError):
            return None


# ---------------------------------------------------------------------------
# error taxonomy
# ---------------------------------------------------------------------------


def classify_error(status: str, error: str) -> str:
    for name, rule in ERROR_RULES:
        try:
            if rule(status, error):
                return name
        except Exception:
            continue
    return "other"


def extract_error_codes(error: str) -> List[str]:
    codes = ERR_CODE_RE.findall(error)
    if not codes:
        codes = ERR_BARE_CODE_RE.findall(error)
    return codes


# ---------------------------------------------------------------------------
# per_request streaming scan (two passes: regex floor, then aggregate)
# ---------------------------------------------------------------------------


class PerRequestScan:
    """Streaming aggregation over per_request.jsonl rows.

    Never holds full row dicts: per-second buckets keep scalar arrays for the
    schedule/ttft/total percentiles, and the length matrix / concurrency
    replay use array('l')/array('d') columns. Memory stays bounded by the
    scalar totals, not the raw stream (~500MB JSONL -> ~100MB of floats).
    """

    def __init__(self) -> None:
        self.epoch0: Optional[float] = None
        self.row_count = 0
        self.ok_rows = 0
        self.buckets: Dict[int, Dict[str, Any]] = defaultdict(
            lambda: {
                "send": 0,
                "ok": 0,
                "err": 0,
                "sla_viol": 0,
                "sched": [],
                "ttft": [],
                "total": [],
                "lag": [],
                "err_class": Counter(),
                "bp_active": [],
                "bp_limit": [],
            }
        )
        self.status_counts: Counter = Counter()
        self.route_paths: Counter = Counter()
        self.err_class: Counter = Counter()
        self.err_codes: Counter = Counter()
        self.prefill_counts: Counter = Counter()
        self.decode_counts: Counter = Counter()
        self.priority: Dict[int, Dict[str, Any]] = defaultdict(
            lambda: {
                "send": 0,
                "ok": 0,
                "err": 0,
                "err_8429": 0,
                "sched": [],
                "ttft": [],
                "err_class": Counter(),
                "err_codes": Counter(),
            }
        )
        self.code_x_priority: Dict[str, Counter] = defaultdict(Counter)
        self.fallback: Dict[str, Any] = {
            "count": 0,
            "ok": 0,
            "err": 0,
            "sched": [],
        }
        self.m_il = array("l")
        self.m_ol = array("l")
        self.m_sched = array("d")
        self.m_prio = array("l")
        self.starts = array("d")
        self.ends = array("d")
        self.ok_ttft: List[float] = []
        self.ok_total: List[float] = []
        self.sched_all: List[float] = []
        self.lag_all: List[float] = []
        self.min_ts: Optional[float] = None
        self.max_ts: Optional[float] = None
        # m1: per_second() is consumed by 4+ dimension analyzers — build once.
        self._per_second: Optional[List[Dict[str, Any]]] = None
        # B2: rows with no usable timestamp anchor (no send_start, no
        # wall_clock) skipped per-second bucketing but still counted in totals.
        self.unbucketed_rows = 0
        # M1: send-window statistics for the pacing actual-rate numerator —
        # N(send_start>0 rows) / (max-min send window) is the send-rate the
        # pacing comparison wants (the summary's first-to-last RPC rate
        # undercounts a uniform ramp's tail).
        self.send_started = 0
        self.send_start_min: Optional[float] = None
        self.send_start_max: Optional[float] = None

    # -- pass 1: cheap regex scan for the send_start floor ------------------

    def scan_floor(self, path: Path) -> None:
        with open_text(path) as stream:
            for line in stream:
                m = SEND_START_RE.search(line)
                value = float(m.group(1)) if m else 0.0
                if value and (self.epoch0 is None or value < self.epoch0):
                    self.epoch0 = value

    # -- pass 2: full aggregate ---------------------------------------------

    def scan_rows(self, path: Path) -> None:
        with open_text(path) as stream:
            for line in stream:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except ValueError:
                    continue
                if isinstance(row, dict):
                    self.add_row(row)

    def add_row(self, row: Dict[str, Any]) -> None:
        self.row_count += 1
        status = str(row.get("status", "") or "")
        error = str(row.get("error", "") or "")
        ok = status in OK_STATUSES or (not error and status not in ERR_STATUSES)
        started = float(row.get("send_start_epoch_ms", 0) or 0)
        ts = float(row.get("ts", 0) or 0)
        if self.epoch0 is None and started:
            self.epoch0 = started
        base = self.epoch0 if self.epoch0 is not None else 0.0
        # B2: deadline-timeout rows carry send_start=0 (the RPC never left
        # the client). wall_clock_ts is SECONDS in the Java client output
        # (JavaLoadClient: System.currentTimeMillis() / 1000.0), so it is
        # scaled back to ms before use. Rows without any anchor skip
        # per-second bucketing and only count towards the totals.
        wall = float(row.get("wall_clock_ts", 0) or 0) * 1000.0
        if started > 0:
            second = int((started - base) // 1000)
        elif wall > 0 and base:
            second = max(0, int((wall - base) // 1000))
        else:
            second = None
        if self.min_ts is None or ts < self.min_ts:
            self.min_ts = ts
        if self.max_ts is None or ts > self.max_ts:
            self.max_ts = ts

        bucket = self.buckets[second] if second is not None else None
        if bucket is not None:
            bucket["send"] += 1
        else:
            self.unbucketed_rows += 1
        self.status_counts[status or "unknown"] += 1
        route = str(row.get("route_path", "") or "unknown")
        self.route_paths[route] += 1

        try:
            prio = int(float(row.get("priority", 0) or 0))
        except (TypeError, ValueError):
            prio = 0
        pfacet = self.priority[prio]
        pfacet["send"] += 1

        sched = float(row.get("schedule_ms", 0) or 0)
        lag = float(row.get("pacing_lag_ms", 0) or 0)
        total = float(row.get("total_ms", 0) or 0)
        ttft = float(row.get("ttft_ms", 0) or 0)
        # B2: only rows with a real send_start join the watermark replay —
        # deadline-timeout rows (send_start=0) must never inflate the
        # client concurrency watermark, and no negative-t end events exist.
        if started > 0:
            self.starts.append(started)
            self.send_started += 1
            if self.send_start_min is None or started < self.send_start_min:
                self.send_start_min = started
            if self.send_start_max is None or started > self.send_start_max:
                self.send_start_max = started
            if total > 0:
                self.ends.append(started + total)

        # B1: fallback rows (enqueued_by_master=false) carry client-side
        # fallback measurements — their schedule_ms (usually 0) never enters
        # the primary schedule statistics. Java summary parity: only
        # schedule_ms>0 rows join the master-path percentiles; the fallback
        # track stays isolated in self.fallback / the fallback dimension.
        fallback_row = row.get("enqueued_by_master") is False
        sched_primary = ok and not fallback_row and sched > 0
        if ok:
            self.ok_rows += 1
            if bucket is not None:
                bucket["ok"] += 1
            pfacet["ok"] += 1
            if sched_primary:
                if bucket is not None:
                    bucket["sched"].append(sched)
                self.sched_all.append(sched)
                pfacet["sched"].append(sched)
            self.lag_all.append(lag)
            if bucket is not None:
                bucket["lag"].append(lag)
                bucket["ttft"].append(ttft)
                bucket["total"].append(total)
            self.ok_ttft.append(ttft)
            self.ok_total.append(total)
            pfacet["ttft"].append(ttft)
            prefill = row.get("prefill")
            decode = row.get("decode")
            if prefill:
                self.prefill_counts[str(prefill)] += 1
            if decode:
                self.decode_counts[str(decode)] += 1
            if fallback_row:
                self.fallback["ok"] += 1
                self.fallback["sched"].append(sched)
            # length matrix columns stay aligned: a row excluded from the
            # primary sched statistics skips all four m_* columns.
            if sched_primary:
                self.m_il.append(int(float(row.get("input_len", 0) or 0)))
                self.m_ol.append(int(float(row.get("output_len", 0) or 0)))
                self.m_sched.append(sched)
                self.m_prio.append(prio)
        else:
            if bucket is not None:
                bucket["err"] += 1
            pfacet["err"] += 1
            cls = classify_error(status, error)
            if bucket is not None:
                bucket["err_class"][cls] += 1
            self.err_class[cls] += 1
            pfacet["err_class"][cls] += 1
            for code in extract_error_codes(error):
                self.err_codes[code] += 1
                pfacet["err_codes"][code] += 1
                self.code_x_priority[code][str(prio)] += 1
            if cls == "preempted_8429":
                pfacet["err_8429"] += 1
            if cls == "backpressure" and bucket is not None:
                m = BACKPRESSURE_ACTIVE_RE.search(error)
                if m:
                    bucket["bp_active"].append(float(m.group(1)))
                m = BACKPRESSURE_LIMIT_RE.search(error)
                if m:
                    bucket["bp_limit"].append(float(m.group(1)))
            if fallback_row:
                self.fallback["err"] += 1
        if fallback_row:
            self.fallback["count"] += 1

    def run(self, paths: Sequence[Path]) -> None:
        for path in paths:
            self.scan_floor(path)
        for path in paths:
            self.scan_rows(path)

    # -- derived ------------------------------------------------------------

    def per_second(self) -> List[Dict[str, Any]]:
        if self._per_second is not None:
            return self._per_second
        out: List[Dict[str, Any]] = []
        for second in sorted(self.buckets):
            b = self.buckets[second]
            out.append(
                {
                    "t": second,
                    "send": b["send"],
                    "ok": b["ok"],
                    "err": b["err"],
                    "sched_p50": percentile(b["sched"], 0.50),
                    "sched_p95": percentile(b["sched"], 0.95),
                    "sched_p99": percentile(b["sched"], 0.99),
                    "ttft_p50": percentile(b["ttft"], 0.50),
                    "ttft_p95": percentile(b["ttft"], 0.95),
                    "ttft_p99": percentile(b["ttft"], 0.99),
                    "lag_p50": percentile(b["lag"], 0.50),
                    "lag_p99": percentile(b["lag"], 0.99),
                    "err_class": {k: v for k, v in b["err_class"].items() if v},
                    "bp_active_p50": (
                        percentile(b["bp_active"], 0.50) if b["bp_active"] else None
                    ),
                    "bp_limit_mode": (
                        Counter(b["bp_limit"]).most_common(1)[0][0]
                        if b["bp_limit"]
                        else None
                    ),
                }
            )
        self._per_second = out
        return out

    def concurrency_series(self) -> Dict[str, Any]:
        """Client-side in-flight watermark replay from send/finish events."""
        if not self.starts:
            return {"available": False, "reason": "no per_request rows"}
        events: List[Tuple[float, int]] = []
        events.extend((float(s), 1) for s in self.starts)
        events.extend((float(e), -1) for e in self.ends)
        events.sort(key=lambda ev: (ev[0], ev[1]))
        base = self.epoch0 if self.epoch0 is not None else events[0][0]
        per_second_max: Dict[int, int] = {}
        current = 0
        peak = 0
        for ts, delta in events:
            current += delta
            if current > peak:
                peak = current
            second = int((ts - base) // 1000)
            if current > per_second_max.get(second, 0):
                per_second_max[second] = current
        seconds = sorted(per_second_max)
        return {
            "available": True,
            "peak": peak,
            "p50": percentile([per_second_max[s] for s in seconds], 0.50),
            "series_t": seconds,
            "series_max": [per_second_max[s] for s in seconds],
        }

    def length_matrix(self, slices: int = 10) -> Dict[str, Any]:
        """input/output-length decile buckets x schedule p99 heatmap."""
        n = len(self.m_sched)
        if n == 0:
            return {"available": False, "reason": "no successful rows"}
        ilens = sorted(self.m_il)
        olens = sorted(self.m_ol)

        def decile_edges(values: Sequence[int]) -> List[int]:
            edges = []
            for q in range(1, slices):
                edges.append(
                    values[min(len(values) - 1, int(len(values) * q / slices))]
                )
            return sorted(set(edges))

        i_edges = decile_edges(ilens)
        o_edges = decile_edges(olens)

        def bucket_index(value: int, edges: Sequence[int]) -> int:
            idx = 0
            for e in edges:
                if value > e:
                    idx += 1
                else:
                    break
            return idx

        cells: Dict[Tuple[int, int], List[float]] = defaultdict(list)
        for i in range(n):
            key = (
                bucket_index(self.m_il[i], i_edges),
                bucket_index(self.m_ol[i], o_edges),
            )
            cells[key].append(self.m_sched[i])
        rows: List[Dict[str, Any]] = []
        for (ir, orc), vals in sorted(cells.items()):
            rows.append(
                {
                    "input_bucket": ir,
                    "output_bucket": orc,
                    "count": len(vals),
                    "sched_p99": percentile(vals, 0.99),
                    "sched_p50": percentile(vals, 0.50),
                }
            )
        return {
            "available": True,
            "input_edges": i_edges,
            "output_edges": o_edges,
            "cells": rows,
            "n_rows": n,
        }

    def sla_per_second(self, sla_ttft_ms: float) -> List[Dict[str, Any]]:
        out = []
        for second in sorted(self.buckets):
            b = self.buckets[second]
            ok_count = b["ok"]
            viol = sum(1 for v in b["ttft"] if v > sla_ttft_ms) if ok_count else 0
            out.append(
                {
                    "t": second,
                    "ok": ok_count,
                    "violations": viol,
                    "rate": round(viol / ok_count, 4) if ok_count else 0.0,
                }
            )
        return out


# ---------------------------------------------------------------------------
# master log parsing (dispatch / complete / server_schedule_latency / leaks)
# ---------------------------------------------------------------------------


def parse_master_logs(paths: Sequence[Path]) -> Dict[str, Any]:
    dispatches: List[Dict[str, Any]] = []
    completions: List[Dict[str, Any]] = []
    latency_rows: List[Dict[str, Any]] = []
    leak_lines: List[str] = []
    for path in paths:
        if not path.is_file():
            continue
        with path.open("r", encoding="utf-8", errors="replace") as stream:
            for line in stream:
                if "LEAK DETECTED" in line:
                    leak_lines.append(line.strip()[:300])
                m = DISPATCH_RE.search(line)
                if m:
                    rec = {
                        k: (int(v) if v.isdigit() else v)
                        for k, v in m.groupdict().items()
                    }
                    dispatches.append(rec)
                    continue
                m = COMPLETE_RE.search(line)
                if m:
                    rec = {
                        k: (int(v) if re.match(r"-?\d+$", v) else v)
                        for k, v in m.groupdict().items()
                    }
                    completions.append(rec)
                    continue
                m = SERVER_LAT_RE.search(line)
                if m:
                    rec = {k: float(v) for k, v in m.groupdict().items()}
                    rec["prio_batch_wait_p95"] = {
                        p: float(v) for p, v in PRIO_BATCH_WAIT_RE.findall(line)
                    }
                    latency_rows.append(rec)
    return {
        "dispatches": dispatches,
        "completions": completions,
        "server_latency_rows": latency_rows,
        "leak_lines": leak_lines,
    }


def scan_mock_leak(paths: Sequence[Path]) -> List[str]:
    lines: List[str] = []
    for path in paths:
        if not path.is_file():
            continue
        with path.open("r", encoding="utf-8", errors="replace") as stream:
            for line in stream:
                if "LEAK DETECTED" in line:
                    lines.append(line.strip()[:300])
    return lines


# ---------------------------------------------------------------------------
# dimension analyzers
# ---------------------------------------------------------------------------


def analyze_qps_and_errors(
    scan: PerRequestScan, summary: Dict[str, Any], data: RunData
) -> Dict[str, Any]:
    err_total = sum(scan.err_class.values())
    class_rows = [
        {
            "class": name,
            "count": count,
            "share": round(count / err_total, 4) if err_total else 0.0,
        }
        for name, count in sorted(scan.err_class.items(), key=lambda kv: -kv[1])
    ]
    code_rows = [
        {"code": code, "count": count}
        for code, count in sorted(scan.err_codes.items(), key=lambda kv: -kv[1])
    ]
    bp_actives = [v for b in scan.buckets.values() for v in b["bp_active"]]
    bp_limits = [v for b in scan.buckets.values() for v in b["bp_limit"]]
    return {
        "available": scan.row_count > 0,
        "reason_unavailable": None if scan.row_count else "per_request.jsonl not found",
        "totals": {
            "rows": scan.row_count,
            # bucket-independent totals (B2: rows without a timestamp anchor
            # still count here, they only skip the per-second buckets)
            "ok": scan.ok_rows,
            "err": err_total,
            "unbucketed_rows": scan.unbucketed_rows,
            "error_rate": (
                round(err_total / scan.row_count, 4) if scan.row_count else 0.0
            ),
            "summary_total_requests": summary.get("total_requests"),
            "summary_status_counts": summary.get("status_counts", {}),
        },
        "status_counts": dict(scan.status_counts),
        "route_path_counts": dict(scan.route_paths),
        "error_classes": class_rows,
        "error_codes": code_rows,
        "backpressure_subdimension": {
            "samples": len(bp_actives),
            "active_p50": percentile(bp_actives, 0.50) if bp_actives else None,
            "active_max": max(bp_actives) if bp_actives else None,
            "limit_mode": (
                Counter(bp_limits).most_common(1)[0][0] if bp_limits else None
            ),
        },
        "per_second": scan.per_second(),
    }


def _client_master_net_curve(
    scan: PerRequestScan, log_rows: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """B4: 10s-window aligned client-sched minus server percentile curve.

    The master emits one flexlb_server_schedule_latency row per 10s window;
    row i is matched against the per-second client schedule percentiles of
    seconds [i*10, i*10+10) (averaged inside the window). The per-second
    percentiles are already B1-filtered (no fallback / schedule_ms=0 rows),
    so the difference isolates network client->master from both measured
    sides. Window alignment is approximate: the master 10s rows are indexed
    from the load start with no absolute timestamp on the line itself.
    """
    if not log_rows:
        return []
    ps = scan.per_second()
    out: List[Dict[str, Any]] = []
    for i, row in enumerate(log_rows):
        lo, hi = i * 10, i * 10 + 10
        c_p50 = [r["sched_p50"] for r in ps if lo <= r["t"] < hi and r["sched_p50"] > 0]
        c_p99 = [r["sched_p99"] for r in ps if lo <= r["t"] < hi and r["sched_p99"] > 0]
        if not c_p50 or not c_p99:
            continue
        mean_p50 = round(sum(c_p50) / len(c_p50), 3)
        mean_p99 = round(sum(c_p99) / len(c_p99), 3)
        server_p50 = float(row.get("server_p50_ms", 0) or 0)
        server_p99 = float(row.get("server_p99_ms", 0) or 0)
        out.append(
            {
                "window": lo // 10,
                "client_p50": mean_p50,
                "client_p99": mean_p99,
                "server_p50": server_p50,
                "server_p99": server_p99,
                "net_p50": round(mean_p50 - server_p50, 3),
                "net_p99": round(mean_p99 - server_p99, 3),
            }
        )
    return out


def analyze_latency_layers(
    scan: PerRequestScan,
    server_latency: Dict[str, Any],
    log_rows: List[Dict[str, Any]],
    fetch_output_stream: bool,
) -> Dict[str, Any]:
    stages = {}
    for name in (
        "grpc_queue_ms",
        "route_submit_ms",
        "batch_wait_ms",
        "dispatch_ack_ms",
        "ack_response_ms",
        "server_total_ms",
    ):
        entry = server_latency.get(name)
        stages[name] = entry if isinstance(entry, dict) else None
    # network client->master: mean schedule_ms (client measured) minus mean
    # server_total_ms (server measured) — a coarse one-number estimate kept
    # for quick reads; the 10s-window aligned p50/p99 curve below is the
    # per-window refinement.
    net_client_master = None
    if scan.sched_all and isinstance(stages.get("server_total_ms"), dict):
        mean_client = sum(scan.sched_all) / len(scan.sched_all)
        mean_server = float(stages["server_total_ms"].get("mean", 0) or 0)
        net_client_master = round(mean_client - mean_server, 3)
    net_curve = _client_master_net_curve(scan, log_rows)
    result: Dict[str, Any] = {
        "available": bool(scan.sched_all) or bool(stages) or bool(log_rows),
        "stage_histograms": stages,
        "layers": {
            "decision_route_submit_ms": stages.get("route_submit_ms"),
            "master_wait_batch_wait_ms": stages.get("batch_wait_ms"),
            "network_client_to_master_ms_mean": net_client_master,
            "network_master_to_engine_dispatch_ack_ms": stages.get("dispatch_ack_ms"),
            "e2e_disabled": not fetch_output_stream,
        },
        "schedule_per_second": [
            {
                "t": row["t"],
                "p50": row["sched_p50"],
                "p95": row["sched_p95"],
                "p99": row["sched_p99"],
            }
            for row in scan.per_second()
        ],
        "server_schedule_latency_rows": log_rows,
        "schedule_overall": dist_stats(scan.sched_all),
        "batch_wait_by_priority": server_latency.get("batch_wait_ms_by_priority"),
    }
    if net_curve:
        result["network_client_to_master_curve"] = net_curve
        result["layers"]["network_client_to_master_ms"] = {
            "p50_avg": round(sum(w["net_p50"] for w in net_curve) / len(net_curve), 3),
            "p99_avg": round(sum(w["net_p99"] for w in net_curve) / len(net_curve), 3),
            "windows": len(net_curve),
            "note": (
                "10s-window aligned client sched percentile minus master "
                "server percentile (approximate window indexing)"
            ),
        }
    if not fetch_output_stream:
        result["note"] = (
            "FETCH_OUTPUT_STREAM=0: end-to-end layer (total_ms / ttft_ms) "
            "disabled — the client skipped engine stream reads; engine-side "
            "prefill/decode execution ran to completion in this run."
        )
    else:
        result["e2e"] = {
            "ttft_ms": dist_stats(scan.ok_ttft),
            "total_ms": dist_stats(scan.ok_total),
            "ttft_per_second": [
                {
                    "t": row["t"],
                    "p50": row["ttft_p50"],
                    "p95": row["ttft_p95"],
                    "p99": row["ttft_p99"],
                }
                for row in scan.per_second()
            ],
        }
    return result


BATCHER_QUEUE_METRICS = (
    "flexlb_app_flexlb_batcher_queue_size",
    "flexlb_app_routing_queue_length",
)


def analyze_queues(
    mock_stats: List[Dict[str, Any]],
    per_engine: Dict[str, Any],
    prom_ts: Sequence[Dict[str, Any]] = (),
    base: Optional[float] = None,
) -> Dict[str, Any]:
    if not mock_stats:
        return {
            "available": False,
            "reason_unavailable": "java_mock_stats timeline not found",
        }
    t0 = mock_stats[0].get("ts_epoch_ms", 0)
    series = []
    for row in mock_stats:
        series.append(
            {
                "t": unified_t(row.get("ts_epoch_ms", t0), base, t0),
                "prefill_waiting": row.get("prefill_waiting", 0),
                "prefill_running": row.get("prefill_running", 0),
                "prefill_running_reqs": row.get("prefill_running_reqs", 0),
                "decode_waiting": row.get("decode_waiting", 0),
                "decode_running": row.get("decode_running", 0),
                "avg_batch_size": row.get("avg_batch_size", 0),
                "max_batch_size": row.get("max_batch_size", 0),
                "avg_batch_ms": row.get("avg_batch_ms", 0),
                "max_batch_ms": row.get("max_batch_ms", 0),
                "decode_done": row.get("decode_done", 0),
                "heap_used_mb": row.get("heap_used_mb", 0),
                "decode_run_min": row.get("decode_run_min", 0),
                "decode_run_max": row.get("decode_run_max", 0),
            }
        )

    def agg(field: str) -> Dict[str, float]:
        values = [row.get(field, 0) for row in mock_stats]
        return {
            "avg": round(sum(values) / len(values), 3) if values else 0.0,
            "max": max(values, default=0),
            "min": min(values, default=0),
        }

    result: Dict[str, Any] = {
        "available": True,
        "samples": len(mock_stats),
        "series": series,
        "aggregate": {
            f: agg(f)
            for f in (
                "prefill_waiting",
                "prefill_running",
                "decode_waiting",
                "decode_running",
                "avg_batch_size",
                "max_batch_size",
                "avg_batch_ms",
                "max_batch_ms",
                "max_prefill_waiting",
                "max_decode_waiting",
                "prefill_running_reqs",
            )
        },
        "per_engine_mode": "per_engine_timeseries" if per_engine else "decode_run_band",
    }
    if per_engine:
        result["per_engine"] = {}
        for engine, rows in per_engine.items():
            t0e = rows[0].get("ts_epoch_ms", 0) if rows else 0
            # M6: decode_waiting / prefill_waiting join the running series so
            # the per-engine panels show both depth dimensions; the role label
            # (from the engine_name prefix or the row itself) tags each curve.
            role = (
                "decode"
                if engine.startswith("decode")
                else ("prefill" if engine.startswith("prefill") else "")
            )
            result["per_engine"][engine] = [
                {
                    "t": unified_t(r.get("ts_epoch_ms", t0e), base, t0e),
                    "role": r.get("role") or role,
                    "decode_running": r.get("decode_running", 0),
                    "decode_waiting": r.get("decode_waiting", 0),
                    "prefill_running": r.get("prefill_running", 0),
                    "prefill_waiting": r.get("prefill_waiting", 0),
                }
                for r in rows
                if isinstance(r, dict)
            ]
    else:
        result["decode_run_band"] = {
            "t": [row["t"] for row in series],
            "min": [row["decode_run_min"] for row in series],
            "max": [row["decode_run_max"] for row in series],
            "note": (
                "per-engine collection not present; showing the mock "
                "aggregate decode_run_min/max spread as a degraded view"
            ),
        }
    # B3: batcher queue dimension from the G3 master prometheus timeline.
    # Both gauges are per-second curves (label variants summed); when the
    # timeline is missing or carries neither series the panel explains that
    # FLEXLB_MONITOR_MODE=all is required (critical-only filters them out).
    batcher: Dict[str, Any] = {"available": False}
    if prom_ts:
        curves = {
            metric: prom_ts_series(prom_ts, metric, base)
            for metric in BATCHER_QUEUE_METRICS
        }
        if any(c.get("available") for c in curves.values()):
            batcher = {
                "available": True,
                "source": "master prometheus_timeseries (G3)",
                "series": {m: c for m, c in curves.items() if c.get("available")},
                "aggregate": {
                    m: {"avg": c.get("avg"), "max": c.get("max"), "min": c.get("min")}
                    for m, c in curves.items()
                    if c.get("available")
                },
            }
        else:
            batcher["reason_unavailable"] = (
                "master prometheus_timeseries carries no "
                "flexlb_app_flexlb_batcher_queue_size / "
                "flexlb_app_routing_queue_length series — run with "
                "FLEXLB_MONITOR_MODE=all (the critical-only filter drops "
                "the batcher gauges)"
            )
    else:
        batcher["reason_unavailable"] = (
            "master prometheus_timeseries not found (needs the G3 secondary "
            "poller; also requires FLEXLB_MONITOR_MODE=all)"
        )
    result["batcher"] = batcher
    return result


def _final_snapshot_accepted(snapshot: Dict[str, Any]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    engines = snapshot.get("engines")
    if not isinstance(engines, list):
        return counts
    for engine in engines:
        if not isinstance(engine, dict):
            continue
        key = str(engine.get("grpc_addr") or engine.get("name") or "?")
        accepted = engine.get("accepted")
        if isinstance(accepted, (int, float)):
            counts[key] = int(accepted)
    return counts


def analyze_balance(
    scan: PerRequestScan,
    mock_stats: List[Dict[str, Any]],
    final_snapshot: Dict[str, Any],
    capacity: Optional[int],
    per_engine: Dict[str, Any],
) -> Dict[str, Any]:
    def metrics(counts: Dict[str, Any]) -> Dict[str, Any]:
        values = [float(v) for v in counts.values()]
        if not values:
            return {"available": False}
        return {
            "available": True,
            "engines": dict(sorted(counts.items(), key=lambda kv: -kv[1])),
            "gini": gini(values),
            "cv": cv(values),
            **spread_ratios(values),
        }

    decode_metrics = metrics(scan.decode_counts)
    prefill_metrics = metrics(scan.prefill_counts)
    snap_accepted = _final_snapshot_accepted(final_snapshot)
    snapshot_metrics = metrics(snap_accepted) if snap_accepted else {"available": False}

    decode_running = [row.get("decode_running", 0) for row in mock_stats]
    capacity_note = None
    utilization = None
    if decode_running:
        if capacity:
            utilization = {
                "capacity": capacity,
                "avg_running": round(sum(decode_running) / len(decode_running), 3),
                "peak_running": max(decode_running),
                "avg_utilization": round(
                    sum(decode_running) / len(decode_running) / capacity, 4
                ),
                "peak_utilization": round(max(decode_running) / capacity, 4),
            }
        else:
            capacity_note = "decode capacity unknown (run_meta params missing java_mock_decode_max_concurrency)"
    extrema = {
        "max_engine_prefill_waiting": max(
            (r.get("max_prefill_waiting", 0) for r in mock_stats), default=0
        ),
        "max_engine_decode_waiting": max(
            (r.get("max_decode_waiting", 0) for r in mock_stats), default=0
        ),
    }
    grade = "unavailable"
    if decode_metrics.get("available"):
        grade = balance_grade(decode_metrics["gini"])
    # per-second Gini when per-engine timeline exists
    per_second_gini: List[Dict[str, Any]] = []
    if per_engine:
        lengths = {e: len(rows) for e, rows in per_engine.items()}
        max_len = max(lengths.values()) if lengths else 0
        for i in range(max_len):
            vals = []
            for engine, rows in per_engine.items():
                if i < len(rows) and isinstance(rows[i], dict):
                    vals.append(float(rows[i].get("decode_running", 0)))
            if vals:
                per_second_gini.append({"idx": i, "gini": gini(vals)})
    return {
        "available": decode_metrics.get("available") or bool(mock_stats),
        "decode_from_per_request": decode_metrics,
        "prefill_from_per_request": prefill_metrics,
        "final_snapshot_accepted": snapshot_metrics,
        "mock_extremes": extrema,
        "utilization": utilization,
        "capacity_note": capacity_note,
        "grade": grade,
        "per_second_gini": per_second_gini,
        "summary_prefill_balance": None,  # filled by caller from summary if present
    }


def analyze_inflight(
    master_counters: List[Dict[str, Any]],
    mock_stats: List[Dict[str, Any]],
    prom: Dict[str, float],
    mock_leak_lines: List[str],
    final_snapshot: Dict[str, Any],
    inflight_ts: Sequence[Dict[str, Any]] = (),
    prom_ts: Sequence[Dict[str, Any]] = (),
    base: Optional[float] = None,
) -> Dict[str, Any]:
    result: Dict[str, Any] = {"available": False, "leak_verdict": "insufficient_data"}

    # M3: the G4 /rtp_llm/inflight_status snapshots are the primary inflight
    # signal — scheduler_inflight plus per-endpoint prefill (batches /
    # requests / route_requests) and decode (requests) gauges, one sample per
    # second. The master-counter delta curve below becomes the auxiliary
    # view (renamed schedule_rpc_inflight).
    if inflight_ts:
        t0 = inflight_ts[0].get("ts_epoch_ms", 0)
        scheduler: List[Dict[str, Any]] = []
        prefill: Dict[str, Dict[str, List[float]]] = {}
        decode: Dict[str, Dict[str, List[float]]] = {}
        prefill_total: List[float] = []
        decode_total: List[float] = []
        for row in inflight_ts:
            inflight = row.get("inflight")
            if not isinstance(inflight, dict):
                continue
            t = unified_t(row.get("ts_epoch_ms", t0), base, t0)
            sched_value = inflight.get("scheduler_inflight", 0)
            if not isinstance(sched_value, (int, float)):
                sched_value = 0
            scheduler.append({"t": t, "scheduler_inflight": sched_value})
            p_sum = 0
            for ep in inflight.get("prefill_endpoints") or []:
                if not isinstance(ep, dict):
                    continue
                key = str(ep.get("ip_port", "?"))
                slot = prefill.setdefault(
                    key,
                    {
                        "inflight_batches": [],
                        "inflight_requests": [],
                        "inflight_route_requests": [],
                    },
                )
                for field in slot:
                    value = ep.get(field, 0)
                    slot[field].append(
                        float(value) if isinstance(value, (int, float)) else 0.0
                    )
                p_batches = ep.get("inflight_batches", 0)
                p_sum += (
                    float(p_batches) if isinstance(p_batches, (int, float)) else 0.0
                )
            d_sum = 0
            for ep in inflight.get("decode_endpoints") or []:
                if not isinstance(ep, dict):
                    continue
                key = str(ep.get("ip_port", "?"))
                slot = decode.setdefault(key, {"inflight_requests": []})
                value = ep.get("inflight_requests", 0)
                value = float(value) if isinstance(value, (int, float)) else 0.0
                slot["inflight_requests"].append(value)
                d_sum += value
            prefill_total.append(p_sum)
            decode_total.append(d_sum)
        if scheduler:
            labels = [row["t"] for row in scheduler]
            result["available"] = True
            result["inflight_status"] = {
                "source": "master inflight_timeseries (G4 /rtp_llm/inflight_status)",
                "scheduler_series": scheduler,
                "prefill_endpoints": prefill,
                "decode_endpoints": decode,
                "prefill_total_batches": prefill_total,
                "decode_total_requests": decode_total,
                "labels": labels,
                "peak_scheduler_inflight": max(
                    (row["scheduler_inflight"] for row in scheduler), default=0
                ),
                "peak_prefill_batches": max(prefill_total, default=0),
                "peak_decode_requests": max(decode_total, default=0),
            }
            # G4-driven leak verdict: every engine-level gauge must drain to
            # zero by the tail (last 3 samples) — a non-zero tail on any
            # endpoint marks a suspected leak, a fully drained tail upgrades
            # the verdict to clean.
            tail_values: List[float] = []
            for slot in list(prefill.values()) + list(decode.values()):
                for values in slot.values():
                    tail_values.extend(values[-3:])
            scheduler_tail = [row["scheduler_inflight"] for row in scheduler[-3:]]
            if tail_values or scheduler_tail:
                if any(abs(x) > 0.5 for x in tail_values + scheduler_tail):
                    result["leak_verdict"] = "suspected_leak"
                    result["g4_tail_residual"] = max(
                        (abs(x) for x in tail_values + scheduler_tail), default=0
                    )
                else:
                    result["leak_verdict"] = "clean"
                    result["g4_tail_residual"] = 0

    # auxiliary view: master arrival/completion counter delta (schedule-RPC
    # inflight — includes requests still waiting to be scheduled, which is
    # why G4 above is the authoritative engine-level inflight signal).
    if master_counters:
        t0 = master_counters[0].get("ts_epoch_ms", 0)
        peak = 0
        series = []
        for row in master_counters:
            inflight = row.get("arrival_count", 0) - row.get("completion_count", 0)
            peak = max(peak, inflight)
            series.append(
                {
                    "t": unified_t(row.get("ts_epoch_ms", t0), base, t0),
                    "inflight": inflight,
                }
            )
        result["available"] = True
        result["schedule_rpc_inflight"] = {"series": series, "peak": peak}
        result["peak"] = peak
        # leak verdict from the counter curve only stands when G4 did not
        # already decide (G4 sees engine-level state; the counter delta may
        # legitimately hold scheduling backlog).
        if not result.get("inflight_status"):
            tail = series[-5:]
            residual = tail[-1]["inflight"] if tail else None
            slopes = (
                [
                    tail[i + 1]["inflight"] - tail[i]["inflight"]
                    for i in range(len(tail) - 1)
                ]
                if len(tail) > 1
                else []
            )
            converged = bool(slopes) and all(abs(s) <= 1 for s in slopes)
            residual_ok = residual is not None and residual <= max(2, peak * 0.01)
            result["tail_residual"] = residual
            result["tail_slope_max"] = max((abs(s) for s in slopes), default=None)
            result["leak_verdict"] = (
                "clean" if (converged and residual_ok) else "suspected_leak"
            )
    age = None
    for key, value in prom.items():
        if "inflight_max_age" in key:
            age = value
            break
    if age is not None:
        result["inflight_max_age_ms"] = age
    # S7: inflight max age per-second curve from the G3 timeline (the final
    # snapshot above is only the last sample).
    if prom_ts:
        age_curve = prom_ts_series(
            prom_ts, "flexlb_app_flexlb_inflight_max_age_ms", base
        )
        if age_curve.get("available"):
            result["inflight_max_age_series"] = age_curve
    result["mock_leak_lines"] = mock_leak_lines[:10]
    if mock_leak_lines:
        result["leak_verdict"] = "suspected_leak"
    engines = final_snapshot.get("engines")
    if isinstance(engines, list) and engines:
        nonzero = []
        for engine in engines:
            if not isinstance(engine, dict):
                continue
            inflight = engine.get("inflight", engine.get("inflight_requests"))
            if isinstance(inflight, (int, float)) and inflight:
                nonzero.append(
                    {str(engine.get("name") or engine.get("grpc_addr")): inflight}
                )
        result["final_snapshot_inflight_nonzero"] = nonzero
        if nonzero:
            result["leak_verdict"] = "suspected_leak"
    if not result["available"]:
        result["reason_unavailable"] = (
            "no master inflight_timeseries (G4) and no master counters "
            "timeseries — leak check unavailable"
        )
    return result


def _prom_subset(prom: Dict[str, float], needles: Sequence[str]) -> Dict[str, float]:
    return {k: v for k, v in sorted(prom.items()) if any(n in k for n in needles)}


def analyze_kv_usage(
    prom: Dict[str, float], per_engine: Dict[str, Any]
) -> Dict[str, Any]:
    series = _prom_subset(prom, ("kv_cache",))
    result: Dict[str, Any] = {"available": bool(series)}
    if series:
        result["metrics"] = series
    if per_engine:
        kv_curves: Dict[str, List[Dict[str, Any]]] = {}
        for engine, rows in per_engine.items():
            t0e = rows[0].get("ts_epoch_ms", 0) if rows else 0
            curve = [
                {
                    "t": round((r.get("ts_epoch_ms", t0e) - t0e) / 1000),
                    "kv_tokens": r.get("kv_cache_tokens"),
                    "kv_ratio": r.get("kv_cache_ratio"),
                }
                for r in rows
                if isinstance(r, dict)
                and (
                    r.get("kv_cache_tokens") is not None
                    or r.get("kv_cache_ratio") is not None
                )
            ]
            if curve:
                kv_curves[engine] = curve
        if kv_curves:
            result["per_engine"] = kv_curves
            result["available"] = True
    if not result["available"]:
        result["reason_unavailable"] = (
            "data unavailable (needs FLEXLB_MONITOR_MODE=all + per-engine collection; "
            "the master prometheus snapshot carries no flexlb_app_cache_* kv_cache series)"
        )
    return result


def analyze_kv_match(
    prom: Dict[str, float],
    prom_ts: Sequence[Dict[str, Any]] = (),
    base: Optional[float] = None,
) -> Dict[str, Any]:
    series = _prom_subset(prom, ("hit_ratio", "theory_hit_ratio", "recent_key"))
    result: Dict[str, Any] = {"available": bool(series)}
    if series:
        result["metrics"] = series
    # S7: per-second hit_ratio / theory_hit_ratio curves from the G3
    # timeline (label variants averaged across engines). The final snapshot
    # above is a single last-sample view.
    if prom_ts:
        for metric in (
            "flexlb_app_cache_hit_ratio",
            "flexlb_app_cache_theory_hit_ratio",
        ):
            curve = prom_ts_ratio_series(prom_ts, metric, base)
            if curve.get("available"):
                result.setdefault("per_second", {})[metric] = curve
                result["available"] = True
    if not result["available"]:
        result["reason_unavailable"] = (
            "no cache hit_ratio / theory_hit_ratio / recent_key series in the "
            "master prometheus snapshot and no per-second curves in "
            "prometheus_timeseries (needs FLEXLB_MONITOR_MODE=all)"
        )
    return result


def analyze_cpu_mem(
    mock_stats: List[Dict[str, Any]],
    prom: Dict[str, float],
    process_rows: List[Dict[str, Any]],
) -> Dict[str, Any]:
    result: Dict[str, Any] = {"available": False}
    if mock_stats:
        heap = [row.get("heap_used_mb", 0) for row in mock_stats]
        heap_max = max((row.get("heap_max_mb", 0) for row in mock_stats), default=0)
        t0 = mock_stats[0].get("ts_epoch_ms", 0)
        result["heap"] = {
            "t": [round((r.get("ts_epoch_ms", t0) - t0) / 1000) for r in mock_stats],
            "used_mb": heap,
            "max_mb": heap_max,
            "peak_used": max(heap) if heap else 0,
        }
        result["available"] = True
    gc = _prom_subset(prom, ("jvm_gc_pause",))
    if gc:
        result["gc"] = gc
        result["available"] = True
    if process_rows:
        # m5: group by label (mock / master / client_N), merging same-label
        # pids into one series (multi-pid mock clusters, client shards).
        # Rows without a label fall back to the pid as the group key.
        by_label: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for row in process_rows:
            label = str(row.get("label") or f"pid_{row.get('pid', '?')}")
            by_label[label].append(row)
        usage: Dict[str, Dict[str, Any]] = {}
        for label, rows in sorted(by_label.items()):
            cpu_series = [r["cpu"] for r in rows]
            rss_series = [r["rss_kb"] for r in rows]
            usage[label] = {
                "cpu": cpu_series,
                "rss_kb": rss_series,
                "peak_cpu": max(cpu_series, default=0),
                "peak_rss_mb": round(max(rss_series, default=0) / 1024, 1),
                "pids": sorted({str(r.get("pid")) for r in rows}),
            }
        result["process_usage"] = usage
        result["available"] = True
    if not result["available"]:
        result["reason_unavailable"] = (
            "no mock stats / prometheus / process_usage_timeseries.txt"
        )
    return result


def analyze_pacing(
    scan: PerRequestScan,
    summary: Dict[str, Any],
    params: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    if not scan.lag_all:
        return {
            "available": False,
            "reason_unavailable": "no pacing_lag_ms samples (per_request missing)",
        }
    params = params or {}
    # M1: target resolution follows the send mode. uniform: the Java client
    # writes target_qps into the summary. replay: the user's explicit
    # SEND_MODE_QPS override lives in run_meta.params (send_mode_qps); the
    # summary's offered_qps is the last fallback. The old code read
    # summary.send_mode_qps — a key the Java client never writes — so the
    # ratio was always None and the ratio<0.98 verdict branch was dead.
    send_mode = (
        summary.get("send_mode")
        or str(params.get("send_mode") or "").strip().lower()
        or "replay"
    )
    if send_mode == "uniform":
        target = summary.get("target_qps") or summary.get("offered_qps")
        target_source = "summary.target_qps (uniform)"
    else:
        target = (
            params.get("send_mode_qps")
            or summary.get("send_mode_qps")
            or summary.get("offered_qps")
        )
        target_source = "run_meta.params send_mode_qps (replay override)"
    # M1: actual rate on the send-window口径 — N(send_start>0) / span, the
    # rate at which the client actually released RPCs. summary.actual_send_qps
    # is the (first..last)-RPC rate and undercounts ramp tails.
    actual = None
    actual_source = None
    if (
        scan.send_started
        and scan.send_start_min is not None
        and scan.send_start_max is not None
    ):
        span_s = (scan.send_start_max - scan.send_start_min) / 1000.0
        if span_s > 0:
            actual = round(scan.send_started / span_s, 3)
            actual_source = "send window: N/send_span_s"
    if actual is None:
        actual = summary.get("actual_send_qps") or summary.get("send_qps")
        actual_source = "summary.actual_send_qps"
    ratio = None
    if isinstance(target, (int, float)) and target and isinstance(actual, (int, float)):
        ratio = round(actual / target, 4)
    lag_p99 = percentile(scan.lag_all, 0.99)
    verdict = "good"
    if lag_p99 > 200 or (ratio is not None and ratio < 0.98):
        verdict = (
            "degraded" if lag_p99 <= 500 and (ratio is None or ratio >= 0.95) else "bad"
        )
    return {
        "available": True,
        "send_mode": send_mode,
        "target_qps": target,
        "target_source": target_source,
        "actual_qps": actual,
        "actual_source": actual_source,
        "send_vs_target_ratio": ratio,
        "pacing_lag_ms": dist_stats(scan.lag_all),
        "per_second": [
            {"t": row["t"], "lag_p50": row["lag_p50"], "lag_p99": row["lag_p99"]}
            for row in scan.per_second()
        ],
        "summary_client_pacing_lag_ms": summary.get("client_pacing_lag_ms"),
        "verdict": verdict,
    }


def analyze_sla(scan: PerRequestScan, summary: Dict[str, Any]) -> Dict[str, Any]:
    sla_ms = summary.get("sla_ttft_ms")
    if not isinstance(sla_ms, (int, float)):
        # fall back to the run_meta params snapshot
        sla_ms = None
    result: Dict[str, Any] = {
        "available": bool(scan.ok_ttft),
        "sla_ttft_ms": sla_ms,
        "summary_sla_violations": summary.get("sla_violations"),
        "summary_violation_rate": summary.get("sla_violation_rate")
        or summary.get("violation_rate"),
    }
    if sla_ms and scan.ok_ttft:
        viol = sum(1 for v in scan.ok_ttft if v > sla_ms)
        result["computed_violations"] = viol
        result["computed_rate"] = round(viol / len(scan.ok_ttft), 4)
        result["per_second"] = scan.sla_per_second(float(sla_ms))
    return result


def analyze_dispatch(parsed: Dict[str, Any], prom: Dict[str, float]) -> Dict[str, Any]:
    dispatches = parsed["dispatches"]
    completions = parsed["completions"]
    if not dispatches and not completions:
        # prometheus reason counters may still exist
        reasons = _prom_subset(prom, ("dispatch_reason",))
        if reasons:
            return {
                "available": True,
                "source": "prometheus",
                "reasons": reasons,
                "note": "structured flexlb_batch_dispatch lines not found; reason counters from prometheus",
            }
        return {
            "available": False,
            "reason_unavailable": "no flexlb_batch_dispatch/complete lines and no dispatch_reason prometheus series",
        }
    violations: List[Dict[str, Any]] = []
    for d in dispatches:
        reason = d.get("reason", "")
        invalid = (
            (
                reason == "predicted_execution_cap"
                and d.get("batch_size", 0) > 1
                and d.get("predicted_ms", 0) >= d.get("threshold_ms", 0)
            )
            or (
                reason == "fixed_window_timeout"
                and d.get("wait_ms", 0) + 2 < d.get("fixed_wait_ms", 0)
            )
            or (
                reason == "batch_full"
                and d.get("batch_size", 0) < d.get("batch_size_max", 0)
            )
        )
        if invalid:
            violations.append(d)
    gaps = [c["gap_ms"] for c in completions if "gap_ms" in c]
    return {
        "available": True,
        "source": "structured_log",
        "dispatch_count": len(dispatches),
        "completion_count": len(completions),
        "reasons": dict(Counter(d.get("reason", "?") for d in dispatches)),
        "batch_size": dist_stats([d.get("batch_size", 0) for d in dispatches]),
        "wait_ms": dist_stats([d.get("wait_ms", 0) for d in dispatches]),
        "predicted_ms": dist_stats([d.get("predicted_ms", 0) for d in dispatches]),
        "actual_ms": dist_stats([c.get("actual_ms", 0) for c in completions]),
        "prediction_gap_ms": dist_stats(gaps),
        "scatter": [
            {"predicted": c.get("predicted_ms", 0), "actual": c.get("actual_ms", 0)}
            for c in completions
        ][:2000],
        "invariant_violation_count": len(violations),
        "invariant_violations": violations[:20],
    }


def analyze_priority(scan: PerRequestScan) -> Dict[str, Any]:
    if not scan.priority:
        return {"available": False, "reason_unavailable": "no per_request rows"}
    facets = {}
    for prio in sorted(scan.priority):
        f = scan.priority[prio]
        facets[str(prio)] = {
            "send": f["send"],
            "ok": f["ok"],
            "err": f["err"],
            "ok_rate": round(f["ok"] / f["send"], 4) if f["send"] else 0.0,
            "sched_ms": dist_stats(f["sched"]),
            "ttft_ms": dist_stats(f["ttft"]),
            "err_8429": f["err_8429"],
            "error_classes": dict(
                sorted(f["err_class"].items(), key=lambda kv: -kv[1])
            ),
            "error_codes": dict(sorted(f["err_codes"].items(), key=lambda kv: -kv[1])),
        }
    return {"available": True, "facets": facets}


def analyze_error_code_matrix(scan: PerRequestScan) -> Dict[str, Any]:
    if not scan.code_x_priority:
        return {"available": False, "reason_unavailable": "no error codes extracted"}
    matrix = {
        code: dict(counts) for code, counts in sorted(scan.code_x_priority.items())
    }
    return {
        "available": True,
        "matrix": matrix,
        "codes": dict(sorted(scan.err_codes.items(), key=lambda kv: -kv[1])),
    }


def analyze_concurrency(
    scan: PerRequestScan, summary: Dict[str, Any]
) -> Dict[str, Any]:
    series = scan.concurrency_series()
    series["configured_max_concurrency"] = summary.get("max_concurrency")
    return series


def analyze_fallback(scan: PerRequestScan) -> Dict[str, Any]:
    fb = scan.fallback
    total = scan.row_count
    return {
        "available": total > 0,
        "count": fb["count"],
        "share": round(fb["count"] / total, 6) if total else 0.0,
        "ok": fb["ok"],
        "err": fb["err"],
        "sched_ms": dist_stats(fb["sched"]),
        "note": (
            (
                "fallback (enqueued_by_master=false) rows are excluded from the "
                "primary schedule-latency percentiles and reported separately"
            )
            if fb["count"]
            else "no fallback rows observed"
        ),
    }


# ---------------------------------------------------------------------------
# report assembly
# ---------------------------------------------------------------------------


def build_report(run_dir: Path) -> Dict[str, Any]:
    data = RunData(run_dir)
    scan = PerRequestScan()
    if data.per_request_paths:
        scan.run(data.per_request_paths)
    parsed_logs = parse_master_logs(data.master_log_paths)
    mock_leak_lines = scan_mock_leak(data.mock_log_paths)

    report: Dict[str, Any] = {
        "meta": {
            "run_dir": str(run_dir),
            "run_id": run_dir.name,
            "generated_at_utc": datetime.now(timezone.utc).isoformat(
                timespec="seconds"
            ),
            "fetch_output_stream": data.fetch_output_stream,
            "sources": data.sources,
            "data_notes": [],
        },
        "verdict": {},
        DIMENSIONS[0]: analyze_qps_and_errors(scan, data.summary, data),
        DIMENSIONS[1]: analyze_latency_layers(
            scan,
            data.server_latency,
            parsed_logs["server_latency_rows"],
            data.fetch_output_stream,
        ),
        DIMENSIONS[2]: analyze_queues(
            data.mock_stats,
            data.mock_per_engine,
            data.master_prometheus_timeseries,
            scan.epoch0,
        ),
        DIMENSIONS[3]: analyze_balance(
            scan,
            data.mock_stats,
            data.final_snapshot,
            data.decode_capacity(),
            data.mock_per_engine,
        ),
        DIMENSIONS[4]: analyze_inflight(
            data.master_counters,
            data.mock_stats,
            data.master_prometheus,
            mock_leak_lines,
            data.final_snapshot,
            data.inflight_timeseries,
            data.master_prometheus_timeseries,
            scan.epoch0,
        ),
        DIMENSIONS[5]: analyze_kv_usage(data.master_prometheus, data.mock_per_engine),
        DIMENSIONS[6]: analyze_kv_match(
            data.master_prometheus, data.master_prometheus_timeseries, scan.epoch0
        ),
        DIMENSIONS[7]: analyze_cpu_mem(
            data.mock_stats, data.master_prometheus, data.process_usage_rows
        ),
        DIMENSIONS[8]: analyze_pacing(scan, data.summary, data.params),
        DIMENSIONS[9]: analyze_sla(scan, data.summary),
        DIMENSIONS[10]: analyze_dispatch(parsed_logs, data.master_prometheus),
        DIMENSIONS[11]: analyze_priority(scan),
        DIMENSIONS[12]: analyze_error_code_matrix(scan),
        DIMENSIONS[13]: scan.length_matrix(),
        DIMENSIONS[14]: analyze_concurrency(scan, data.summary),
        DIMENSIONS[15]: analyze_fallback(scan),
        "summary_extras": {
            "send_mode": data.summary.get("send_mode"),
            "offered_qps": data.summary.get("offered_qps"),
            "elapsed_s": data.summary.get("elapsed_s"),
            "shard_summaries": data.summary.get("shard_summaries"),
            "validity_checks": data.summary.get("validity_checks"),
            "prefill_balance": data.summary.get("prefill_balance"),
            "decode_balance": data.summary.get("decode_balance"),
        },
    }
    report["meta"]["data_notes"] = data.notes
    balance = report["balance"]
    balance["summary_prefill_balance"] = data.summary.get("prefill_balance")
    balance["summary_decode_balance"] = data.summary.get("decode_balance")
    err_rate = report["qps_and_errors"]["totals"]["error_rate"]
    report["verdict"] = {
        "test_valid": data.summary.get("test_valid"),
        "validity_checks": data.summary.get("validity_checks"),
        "error_rate": err_rate,
        "leak_verdict": report["inflight"].get("leak_verdict", "insufficient_data"),
        "balance_grade": balance.get("grade", "unavailable"),
        "balance_gini": (balance.get("decode_from_per_request") or {}).get("gini"),
        "pacing_verdict": report["pacing"].get("verdict", "unavailable"),
        "fetch_output_stream": data.fetch_output_stream,
    }
    return report


def key_metrics(report: Dict[str, Any]) -> Dict[str, Any]:
    q = report.get("qps_and_errors", {}).get("totals", {})
    sched = report.get("latency_layers", {}).get("schedule_overall", {})
    e2e = report.get("latency_layers", {}).get("e2e", {})
    balance = report.get("balance", {}).get("decode_from_per_request") or {}
    heap = report.get("cpu_mem", {}).get("heap", {})
    return {
        "total": q.get("rows"),
        "ok": q.get("ok"),
        "err": q.get("err"),
        "error_rate": q.get("error_rate"),
        "sched_p50_ms": sched.get("p50"),
        "sched_p95_ms": sched.get("p95"),
        "sched_p99_ms": sched.get("p99"),
        "ttft_p99_ms": (e2e.get("ttft_ms") or {}).get("p99"),
        "total_p99_ms": (e2e.get("total_ms") or {}).get("p99"),
        "balance_gini": balance.get("gini"),
        "leak_verdict": report.get("verdict", {}).get("leak_verdict"),
        "pacing_verdict": report.get("verdict", {}).get("pacing_verdict"),
        "heap_peak_mb": heap.get("peak_used"),
        "test_valid": report.get("verdict", {}).get("test_valid"),
    }


def build_comparison(
    base: Dict[str, Any], others: List[Dict[str, Any]]
) -> Dict[str, Any]:
    base_keys = key_metrics(base)
    runs = []
    for other in others:
        okm = key_metrics(other)
        rows = []
        for metric, base_value in base_keys.items():
            other_value = okm.get(metric)
            if isinstance(base_value, (int, float)) and isinstance(
                other_value, (int, float)
            ):
                delta = round(other_value - base_value, 4)
                pct = round(delta / base_value * 100, 2) if base_value else None
            else:
                delta = (
                    None
                    if other_value == base_value
                    else f"{base_value!r} -> {other_value!r}"
                )
                pct = None
            rows.append(
                {
                    "metric": metric,
                    "base": base_value,
                    "other": other_value,
                    "delta": delta,
                    "pct": pct,
                }
            )
        runs.append(
            {
                "run_id": other.get("meta", {}).get("run_id"),
                "run_dir": other.get("meta", {}).get("run_dir"),
                "delta_table": rows,
            }
        )
    return {"base_run_id": base.get("meta", {}).get("run_id"), "runs": runs}


# ---------------------------------------------------------------------------
# self-contained HTML rendering (inline CSS + hand-drawn SVG + tiny tooltip JS)
# ---------------------------------------------------------------------------

PALETTE = [
    "#e8a33d",
    "#3ecfb2",
    "#e0637e",
    "#9d8cff",
    "#6cb2ff",
    "#7ee081",
    "#f2c14e",
    "#ef8354",
    "#b8c1ec",
    "#c5a3ff",
    "#5ce1e6",
    "#ff9770",
]
CSS = """
:root{--bg:#0a0e15;--panel:#101623;--panel2:#0d1320;--line:#1c2433;--txt:#c9d4e3;
--dim:#8a93a6;--amber:#e8a33d;--teal:#3ecfb2;--rose:#e0637e;--green:#7ee081;--violet:#9d8cff}
*{box-sizing:border-box;margin:0;padding:0}
body{background:var(--bg);color:var(--txt);font:14px/1.55 ui-monospace,'SF Mono','Cascadia Code',Menlo,Consolas,monospace;
padding:0 0 80px;position:relative}
body::before{content:'';position:fixed;inset:0;pointer-events:none;z-index:0;
background:repeating-linear-gradient(0deg,rgba(255,255,255,.012) 0 1px,transparent 1px 3px)}
.wrap{max-width:1340px;margin:0 auto;padding:0 28px;position:relative;z-index:1}
.hero{padding:34px 0 18px;border-bottom:1px solid var(--line)}
.hero h1{font-size:26px;letter-spacing:.14em;color:var(--amber);font-weight:700}
.hero h1 small{color:var(--dim);font-weight:400;letter-spacing:.08em;font-size:13px}
.meta-bar{display:flex;flex-wrap:wrap;gap:10px 26px;margin-top:12px;color:var(--dim);font-size:12px}
.meta-bar b{color:var(--txt);font-weight:600}
.verdict-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(196px,1fr));gap:12px;margin:20px 0 8px}
.card{background:var(--panel);border:1px solid var(--line);border-left:3px solid var(--amber);
border-radius:6px;padding:12px 14px;transition:border-color .15s}
.card:hover{border-color:#2a3546}
.card .k{color:var(--dim);font-size:11px;letter-spacing:.12em;text-transform:uppercase}
.card .v{font-size:21px;font-weight:700;margin-top:4px;color:var(--txt)}
.card .s{color:var(--dim);font-size:11px;margin-top:2px}
.card.good .v{color:var(--green)} .card.warn .v{color:var(--amber)} .card.bad .v{color:var(--rose)}
.banner{background:linear-gradient(90deg,rgba(232,163,61,.14),transparent);
border:1px solid rgba(232,163,61,.4);color:var(--amber);border-radius:6px;
padding:10px 14px;margin:14px 0;font-size:13px}
.sec{margin-top:34px}
.sec-head{display:flex;align-items:baseline;gap:14px;border-bottom:1px solid var(--line);padding-bottom:8px}
.sec-no{font-size:24px;color:var(--amber);font-weight:700;min-width:44px}
.sec-head h2{font-size:16px;letter-spacing:.1em;color:var(--txt);text-transform:uppercase}
.sec-sub{color:var(--dim);font-size:12px}
.grid2{display:grid;grid-template-columns:1fr 1fr;gap:16px}
@media(max-width:980px){.grid2{grid-template-columns:1fr}}
.panel{background:var(--panel2);border:1px solid var(--line);border-radius:6px;padding:12px 14px;margin-top:14px}
.panel h3{font-size:12px;letter-spacing:.1em;color:var(--teal);text-transform:uppercase;margin-bottom:8px}
.panel h3 .unit{color:var(--dim);text-transform:none;letter-spacing:0}
.chart svg{width:100%;height:auto;display:block}
.legend{display:flex;flex-wrap:wrap;gap:4px 16px;font-size:11px;color:var(--dim);margin-top:6px}
.legend i{display:inline-block;width:10px;height:10px;margin-right:5px;border-radius:2px;vertical-align:-1px}
table.t{width:100%;border-collapse:collapse;font-size:12px;margin-top:8px}
table.t th{color:var(--dim);text-align:right;font-weight:600;padding:4px 8px;border-bottom:1px solid var(--line);letter-spacing:.05em}
table.t td{text-align:right;padding:4px 8px;border-bottom:1px solid #141b29}
table.t th:first-child,table.t td:first-child{text-align:left}
table.t tr:hover td{background:#131a29}
.num{font-variant-numeric:tabular-nums}
.bar-bg{background:#1a2334;border-radius:3px;height:14px;position:relative;min-width:120px}
.bar-fg{position:absolute;inset:0 auto 0 0;border-radius:3px}
.muted{color:var(--dim)} .warn-t{color:var(--amber)} .bad-t{color:var(--rose)} .good-t{color:var(--green)}
#tooltip{display:none;position:fixed;z-index:50;background:#161e2e;border:1px solid #2a3546;
border-radius:6px;padding:8px 10px;font-size:12px;pointer-events:none;max-width:260px;
box-shadow:0 6px 24px rgba(0,0,0,.5)}
#tooltip b{color:var(--amber)}
.unavail{border:1px dashed #2a3546;border-radius:6px;padding:16px;color:var(--dim);margin-top:14px;font-size:13px}
footer{margin-top:48px;color:#3b4557;font-size:11px;text-align:center}
"""

# global chart registry: [(cid, payload)] rendered once into a JSON script tag
_CHARTS: List[Tuple[str, Dict[str, Any]]] = []
_chart_seq = [0]


def _nice_ticks(lo: float, hi: float, count: int = 5) -> List[float]:
    if hi <= lo:
        hi = lo + 1.0
    span = hi - lo
    step = 10 ** math.floor(math.log10(span / count))
    for mult in (1, 2, 5, 10):
        if span / (step * mult) <= count:
            step *= mult
            break
    start = math.floor(lo / step) * step
    ticks = []
    value = start
    while value <= hi + step * 0.001:
        if value >= lo - step * 0.001:
            ticks.append(round(value, 6))
        value += step
    return ticks if len(ticks) >= 2 else [lo, hi]


def line_chart(
    title: str,
    unit: str,
    series: List[Tuple[str, Sequence[Any]]],
    x_labels: Optional[Sequence[Any]] = None,
    height: int = 240,
    stack: bool = False,
    band: Optional[Dict[str, Any]] = None,
    colors: Optional[List[str]] = None,
) -> str:
    """Render one SVG chart (line / stacked-area / band) + register hover data."""
    _chart_seq[0] += 1
    cid = f"c{_chart_seq[0]}"
    n = max((len(s) for _, s in series), default=0)
    if n == 0:
        return f'<div class="panel"><h3>{html.escape(title)}</h3><p class="muted">no data</p></div>'
    W, H = 920, height
    L, R, T, B = 64, 12, 14, 30
    colors = colors or PALETTE

    def _clean(seq: Sequence[Any]) -> List[Optional[float]]:
        out = []
        for v in seq[:n]:
            try:
                out.append(float(v) if v is not None else None)
            except (TypeError, ValueError):
                out.append(None)
        return out

    clean = [(name, _clean(data)) for name, data in series]
    if stack:
        totals = []
        for i in range(n):
            totals.append(sum(v[i] or 0 for _, v in clean))
        ymax = max(totals, default=1.0)
    else:
        flat = [v for _, vals in clean for v in vals if v is not None]
        ymax = max(flat) if flat else 1.0
    ymax = ymax if ymax > 0 else 1.0
    ticks = _nice_ticks(0.0, ymax)
    y_max = ticks[-1]
    if band:
        band_lo = [v if v is not None else 0 for v in _clean(band["lo"])]
        band_hi = [v if v is not None else 0 for v in _clean(band["hi"])]
        y_max = max(y_max, max(band_hi, default=0.0)) or 1.0
        ticks = _nice_ticks(0.0, y_max)
        y_max = ticks[-1]

    def X(i: int) -> float:
        return L + (W - L - R) * (i / (n - 1) if n > 1 else 0)

    def Y(v: float) -> float:
        return T + (H - T - B) * (1 - (min(v, y_max) / y_max if y_max else 0))

    parts = [f'<svg viewBox="0 0 {W} {H}" role="img">']
    for tick in ticks:
        y = Y(tick)
        parts.append(
            f'<line x1="{L}" y1="{y:.1f}" x2="{W - R}" y2="{y:.1f}" stroke="#1c2433" stroke-width="1"/>'
        )
        parts.append(
            f'<text x="{L - 6}" y="{y + 4:.1f}" fill="#8a93a6" font-size="10" text-anchor="end">{_fmt_tick(tick)}</text>'
        )
    # x labels (up to 12)
    if n > 1:
        step_idx = max(1, n // 12)
        for i in range(0, n, step_idx):
            parts.append(
                f'<text x="{X(i):.1f}" y="{H - 8}" fill="#8a93a6" font-size="10" text-anchor="middle">{html.escape(str(x_labels[i] if x_labels else i))}</text>'
            )
    if band:
        band_color = band.get("color", "#3ecfb2")
        pts_lo = " ".join(f"{X(i):.1f},{Y(band_lo[i]):.1f}" for i in range(n))
        pts_hi = " ".join(
            f"{X(i):.1f},{Y(band_hi[i]):.1f}" for i in range(n - 1, -1, -1)
        )
        pts_hi_fwd = " ".join(f"{X(i):.1f},{Y(band_hi[i]):.1f}" for i in range(n))
        parts.append(
            f'<polygon points="{pts_lo} {pts_hi}" fill="{band_color}22" stroke="none"/>'
        )
        parts.append(
            f'<polyline points="{pts_lo}" fill="none" stroke="{band_color}" stroke-width="1" opacity=".55"/>'
        )
        parts.append(
            f'<polyline points="{pts_hi_fwd}" fill="none" stroke="{band_color}" stroke-width="1" opacity=".55"/>'
        )
    if stack:
        cum = [0.0] * n
        for idx, (name, vals) in enumerate(clean):
            color = colors[idx % len(colors)]
            top = [cum[i] + (vals[i] or 0) for i in range(n)]
            pts = " ".join(f"{X(i):.1f},{Y(top[i]):.1f}" for i in range(n))
            pts_back = " ".join(
                f"{X(i):.1f},{Y(cum[i]):.1f}" for i in range(n - 1, -1, -1)
            )
            parts.append(
                f'<polygon points="{pts} {pts_back}" fill="{color}" fill-opacity=".42" stroke="{color}" stroke-width="1"/>'
            )
            cum = top
    else:
        for idx, (name, vals) in enumerate(clean):
            color = colors[idx % len(colors)]
            seg, pen = [], False
            for i, v in enumerate(vals):
                if v is None:
                    pen = False
                    continue
                seg.append(f"{'M' if not pen else 'L'}{X(i):.1f},{Y(v):.1f}")
                pen = True
            if seg:
                parts.append(
                    f'<path d="{" ".join(seg)}" fill="none" stroke="{color}" stroke-width="1.6" stroke-linejoin="round"/>'
                )
    parts.append(
        f'<line x1="{L}" y1="{H - B}" x2="{W - R}" y2="{H - B}" stroke="#2a3546" stroke-width="1"/>'
    )
    parts.append("</svg>")
    legend = "".join(
        f'<span><i style="background:{colors[i % len(colors)]}"></i>{html.escape(name)}</span>'
        for i, (name, _) in enumerate(clean)
    )
    payload = {
        "x": list(range(n)),
        "x_labels": [str(x_labels[i]) if x_labels else str(i) for i in range(n)],
        "unit": unit,
        "series": [
            {"name": name, "color": colors[i % len(colors)], "data": vals}
            for i, (name, vals) in enumerate(clean)
        ],
    }
    if band:
        payload["series"].insert(
            0,
            {
                "name": band.get("name", "band"),
                "color": band.get("color", "#3ecfb2"),
                "data": band_hi,
            },
        )
    _CHARTS.append((cid, payload))
    return (
        f'<div class="panel"><h3>{html.escape(title)} <span class="unit">[{html.escape(unit)}]</span></h3>'
        f'<div class="chart" data-cid="{cid}">{"".join(parts)}</div>'
        f'<div class="legend">{legend}</div></div>'
    )


def _fmt_tick(value: float) -> str:
    if value >= 10000:
        return f"{value / 1000:.0f}k"
    if value >= 100:
        return f"{value:.0f}"
    return f"{value:g}"


def scatter_chart(
    title: str,
    points: Sequence[Dict[str, Any]],
    x_name: str,
    y_name: str,
    unit: str = "ms",
) -> str:
    if not points:
        return f'<div class="panel"><h3>{html.escape(title)}</h3><p class="muted">no data</p></div>'
    _chart_seq[0] += 1
    cid = f"c{_chart_seq[0]}"
    W, H = 920, 320
    L, R, T, B = 64, 16, 14, 34
    xs = [float(p["predicted"]) for p in points]
    ys = [float(p["actual"]) for p in points]
    hi = max(xs + ys) * 1.05 or 1.0

    def X(v: float) -> float:
        return L + (W - L - R) * (v / hi)

    def Y(v: float) -> float:
        return T + (H - T - B) * (1 - v / hi)

    parts = [f'<svg viewBox="0 0 {W} {H}" role="img">']
    for tick in _nice_ticks(0, hi, 5):
        parts.append(
            f'<line x1="{X(tick):.1f}" y1="{Y(tick):.1f}" x2="{W - R}" y2="{Y(tick):.1f}" stroke="#1c2433"/>'
        )
        parts.append(
            f'<text x="{L - 6}" y="{Y(tick) + 4:.1f}" fill="#8a93a6" font-size="10" text-anchor="end">{_fmt_tick(tick)}</text>'
        )
        parts.append(
            f'<text x="{X(tick):.1f}" y="{H - 10}" fill="#8a93a6" font-size="10" text-anchor="middle">{_fmt_tick(tick)}</text>'
        )
    parts.append(
        f'<line x1="{X(0):.1f}" y1="{Y(0):.1f}" x2="{X(hi):.1f}" y2="{Y(hi):.1f}" stroke="#e0637e" stroke-dasharray="4 4" stroke-width="1" opacity=".7"/>'
    )
    parts.append(
        "".join(
            f'<circle cx="{X(x):.1f}" cy="{Y(y):.1f}" r="1.8" fill="#3ecfb2" fill-opacity=".55"/>'
            for x, y in zip(xs, ys)
        )
    )
    parts.append(
        f'<text x="{W - R}" y="{T + 10}" fill="#8a93a6" font-size="10" text-anchor="end">{html.escape(x_name)} →</text>'
    )
    parts.append(
        f'<text x="{L}" y="{H - 2}" fill="#8a93a6" font-size="10">{html.escape(y_name)} ↑</text>'
    )
    parts.append("</svg>")
    _CHARTS.append(
        (
            cid,
            {
                "x": xs,
                "x_labels": [f"{v:.0f}" for v in xs],
                "unit": unit,
                "series": [{"name": y_name, "color": "#3ecfb2", "data": ys}],
            },
        )
    )
    return (
        f'<div class="panel"><h3>{html.escape(title)} <span class="unit">[{unit}, y=x dashed]</span></h3>'
        f'<div class="chart" data-cid="{cid}">{"".join(parts)}</div></div>'
    )


def kv_table(rows: Sequence[Tuple[str, Any]], cls: str = "") -> str:
    if not rows:
        return '<p class="muted">no data</p>'
    body = "".join(
        f'<tr><td>{html.escape(str(k))}</td><td class="num">{html.escape(_val_str(v))}</td></tr>'
        for k, v in rows
    )
    return f'<table class="t {cls}"><thead><tr><th>metric</th><th>value</th></tr></thead><tbody>{body}</tbody></table>'


def _val_str(v: Any) -> str:
    if isinstance(v, dict):
        keep = {
            k: v.get(k) for k in ("count", "mean", "p50", "p95", "p99", "max") if k in v
        }
        return (
            " ".join(f"{k}={fmt_num(val)}" for k, val in keep.items())
            if keep
            else json.dumps(v)[:80]
        )
    if isinstance(v, float):
        return f"{v:,.3f}".rstrip("0").rstrip(".")
    return str(v)


def bar_table(
    title: str, rows: Sequence[Dict[str, Any]], total: Optional[int] = None
) -> str:
    if not rows:
        return f'<div class="panel"><h3>{html.escape(title)}</h3><p class="muted">no data</p></div>'
    total = total or sum(r.get("count", 0) for r in rows) or 1
    body = []
    for i, row in enumerate(rows[:16]):
        count = row.get("count", 0)
        share = row.get("share") or (count / total if total else 0)
        color = PALETTE[i % len(PALETTE)]
        label = (
            row.get("class")
            or row.get("code")
            or row.get("reason")
            or str(row.get("name", "?"))
        )
        pct = share * 100 if share <= 1 else share
        body.append(
            f"<tr><td>{html.escape(str(label))}</td>"
            f'<td class="num">{fmt_num(count)}</td>'
            f'<td><div class="bar-bg"><div class="bar-fg" style="width:{min(100, pct):.1f}%;background:{color}"></div></div></td>'
            f'<td class="num">{pct:.2f}%</td></tr>'
        )
    return (
        f'<div class="panel"><h3>{html.escape(title)}</h3>'
        f'<table class="t"><thead><tr><th>class</th><th>count</th><th style="width:38%">share</th><th>%</th></tr></thead>'
        f'<tbody>{"".join(body)}</tbody></table></div>'
    )


def verdict_card(cls: str, key: str, value: Any, sub: str = "") -> str:
    return (
        f'<div class="card {cls}"><div class="k">{html.escape(key)}</div>'
        f'<div class="v">{html.escape(str(value))}</div>'
        f'<div class="s">{html.escape(sub)}</div></div>'
    )


def unavailable_panel(text: str) -> str:
    return f'<div class="unavail">&#9723; {html.escape(text)}</div>'


# ---- per-dimension section renderers --------------------------------------


def _sec(no: int, title: str, sub: str, body: str) -> str:
    return (
        f'<section class="sec" id="dim{no}"><div class="sec-head">'
        f'<div class="sec-no">{no:02d}</div><h2>{html.escape(title)}</h2>'
        f'<span class="sec-sub">{html.escape(sub)}</span></div>{body}</section>'
    )


def render_qps(report: Dict[str, Any]) -> str:
    dim = report["qps_and_errors"]
    if not dim.get("available"):
        return _sec(
            0,
            "QPS & Failure Taxonomy",
            "per_request.jsonl",
            unavailable_panel(dim.get("reason_unavailable") or "unavailable"),
        )
    t = dim["totals"]
    ps = dim["per_second"]
    body = ['<div class="verdict-grid">']
    body.append(verdict_card("", "sent", fmt_num(t["rows"]), "per_request rows"))
    body.append(verdict_card("good", "ok", fmt_num(t["ok"]), "scheduled"))
    body.append(
        verdict_card(
            "bad", "errors", fmt_num(t["err"]), f"rate {t['error_rate'] * 100:.2f}%"
        )
    )
    body.append(
        verdict_card(
            "",
            "route paths",
            len(dim["route_path_counts"]),
            ", ".join(
                f"{k}={v}" for k, v in list(dim["route_path_counts"].items())[:4]
            ),
        )
    )
    body.append("</div>")
    labels = [row["t"] for row in ps]
    body.append('<div class="grid2">')
    body.append(
        line_chart(
            "per-second send / ok / err",
            "req/s",
            [
                ("send", [r["send"] for r in ps]),
                ("ok", [r["ok"] for r in ps]),
                ("err", [r["err"] for r in ps]),
            ],
            labels,
        )
    )
    classes = sorted({c for row in ps for c in row["err_class"]})
    stacked = [(c, [row["err_class"].get(c, 0) for row in ps]) for c in classes]
    body.append(
        line_chart(
            "error taxonomy per second (stacked)", "err/s", stacked, labels, stack=True
        )
    )
    err_rate = [(r["err"] / r["send"] * 100.0) if r["send"] else 0.0 for r in ps]
    body.append(
        line_chart("per-second error rate", "%", [("err rate %", err_rate)], labels)
    )
    body.append("</div>")
    body.append(bar_table("error classes (total)", dim["error_classes"], t["err"]))
    body.append(bar_table("error codes (extracted code=N)", dim["error_codes"]))
    bp = dim["backpressure_subdimension"]
    body.append(
        '<div class="panel"><h3>backpressure sub-dimension</h3>'
        + kv_table(
            [
                ("samples with active_admissions", bp["samples"]),
                ("active_admissions p50", bp["active_p50"]),
                ("active_admissions max", bp["active_max"]),
                ("limit (mode)", bp["limit_mode"]),
            ]
        )
        + "</div>"
    )
    return _sec(
        0,
        "QPS & Failure Taxonomy",
        "per_request.jsonl + error regex library",
        "".join(body),
    )


def render_latency(report: Dict[str, Any]) -> str:
    dim = report["latency_layers"]
    body = []
    if dim.get("stage_histograms"):
        rows = []
        for name, entry in dim["stage_histograms"].items():
            rows.append((name, entry if entry else "n/a"))
        body.append(
            '<div class="panel"><h3>server stage latency (final snapshot)</h3>'
            + kv_table(rows)
            + "</div>"
        )
    layers = dim.get("layers", {})
    net_layer = layers.get("network_client_to_master_ms")
    l3_value = (
        net_layer if net_layer else layers.get("network_client_to_master_ms_mean")
    )
    l3_label = (
        "L3 net client&#8594;master (10s-window client&#8722;server p50/p99 avg)"
        if net_layer
        else "L3 net client&#8594;master (sched&#8722;server_total, mean-only coarse estimate)"
    )
    body.append(
        '<div class="panel"><h3>four-layer decomposition</h3>'
        + kv_table(
            [
                (
                    "L1 decision (route_submit_ms)",
                    layers.get("decision_route_submit_ms"),
                ),
                (
                    "L2 master wait (batch_wait_ms)",
                    layers.get("master_wait_batch_wait_ms"),
                ),
                (l3_label, l3_value),
                (
                    "L3 net master&#8594;engine (dispatch_ack_ms)",
                    layers.get("network_master_to_engine_dispatch_ack_ms"),
                ),
                (
                    "L4 e2e",
                    (
                        "DISABLED (fetch_output_stream=0)"
                        if dim.get("layers", {}).get("e2e_disabled")
                        else "see below"
                    ),
                ),
            ]
        )
        + "</div>"
    )
    sp = dim.get("schedule_per_second") or []
    if sp:
        labels = [row["t"] for row in sp]
        body.append(
            line_chart(
                "client schedule_ms percentiles per second",
                "ms",
                [
                    ("p50", [r["p50"] for r in sp]),
                    ("p95", [r["p95"] for r in sp]),
                    ("p99", [r["p99"] for r in sp]),
                ],
                labels,
            )
        )
    rows = dim.get("server_schedule_latency_rows") or []
    if rows:
        labels = [f"+{i * 10}s" for i in range(len(rows))]
        body.append(
            line_chart(
                "master flexlb_server_schedule_latency (10s rows)",
                "ms",
                [
                    ("server_p50", [r["server_p50_ms"] for r in rows]),
                    ("server_p95", [r["server_p95_ms"] for r in rows]),
                    ("server_p99", [r["server_p99_ms"] for r in rows]),
                    ("route_submit_p95", [r["route_submit_p95_ms"] for r in rows]),
                    ("batch_wait_p95", [r["batch_wait_p95_ms"] for r in rows]),
                    ("dispatch_ack_p95", [r["dispatch_ack_p95_ms"] for r in rows]),
                ],
                labels,
            )
        )
    # B4: network client->master approximation — 10s-window client sched
    # percentiles minus the master's server percentiles for the same window.
    net_curve = dim.get("network_client_to_master_curve") or []
    if net_curve:
        labels = [f"w{r['window']}" for r in net_curve]
        body.append(
            line_chart(
                "network client&#8594;master (client sched &#8722; server p50/p99, 10s windows)",
                "ms",
                [
                    ("net_p50", [r["net_p50"] for r in net_curve]),
                    ("net_p99", [r["net_p99"] for r in net_curve]),
                ],
                labels,
            )
        )
    if dim.get("e2e"):
        e2e = dim["e2e"]
        body.append(
            '<div class="panel"><h3>end-to-end</h3>'
            + kv_table(
                [
                    ("ttft_ms", e2e.get("ttft_ms")),
                    ("total_ms", e2e.get("total_ms")),
                    ("schedule_ms overall", dim.get("schedule_overall")),
                ]
            )
            + "</div>"
        )
        tp = e2e.get("ttft_per_second") or []
        if tp:
            labels = [row["t"] for row in tp]
            body.append(
                line_chart(
                    "client ttft_ms percentiles per second",
                    "ms",
                    [
                        ("p50", [r["p50"] for r in tp]),
                        ("p95", [r["p95"] for r in tp]),
                        ("p99", [r["p99"] for r in tp]),
                    ],
                    labels,
                )
            )
    if dim.get("note"):
        body.append(f'<div class="banner">{html.escape(dim["note"])}</div>')
    bw = dim.get("batch_wait_by_priority")
    if isinstance(bw, dict) and bw:
        body.append(
            '<div class="panel"><h3>batch_wait_ms by priority</h3>'
            + kv_table([(f"P{k}", v) for k, v in bw.items()])
            + "</div>"
        )
    return _sec(
        1,
        "Latency: Four Layers",
        "server_latency.json + per_request + master.log",
        "".join(body),
    )


def render_queues(report: Dict[str, Any]) -> str:
    dim = report["queues"]
    if not dim.get("available"):
        return _sec(
            2,
            "Queue Depth",
            "java_mock_stats",
            unavailable_panel(dim.get("reason_unavailable") or ""),
        )
    series = dim["series"]
    labels = [row["t"] for row in series]
    body = ['<div class="grid2">']
    body.append(
        line_chart(
            "prefill / decode waiting & running",
            "requests",
            [
                ("prefill_waiting", [r["prefill_waiting"] for r in series]),
                ("prefill_running", [r["prefill_running"] for r in series]),
                ("decode_waiting", [r["decode_waiting"] for r in series]),
                ("decode_running", [r["decode_running"] for r in series]),
            ],
            labels,
        )
    )
    body.append(
        line_chart(
            "avg batch size / batch ms",
            "size / ms",
            [
                ("avg_batch_size", [r["avg_batch_size"] for r in series]),
                ("max_batch_size", [r["max_batch_size"] for r in series]),
            ],
            labels,
        )
    )
    body.append("</div>")
    if dim.get("per_engine"):
        panels = []
        for engine, rows in list(dim["per_engine"].items())[:12]:
            labels_e = [r["t"] for r in rows]
            role = rows[0].get("role", "") if rows else ""
            # M6: waiting joins running — both depth dimensions per engine.
            series = []
            if role != "prefill":
                series += [
                    ("decode_running", [r.get("decode_running", 0) for r in rows]),
                    ("decode_waiting", [r.get("decode_waiting", 0) for r in rows]),
                ]
            if role != "decode":
                series += [
                    ("prefill_running", [r.get("prefill_running", 0) for r in rows]),
                    ("prefill_waiting", [r.get("prefill_waiting", 0) for r in rows]),
                ]
            panels.append(
                line_chart(
                    f"engine {engine} ({role}) running & waiting",
                    "reqs",
                    series,
                    labels_e,
                    height=150,
                )
            )
        body.append('<div class="grid2">' + "".join(panels) + "</div>")
    elif dim.get("decode_run_band"):
        band = dim["decode_run_band"]
        body.append(
            line_chart(
                "decode running spread (degraded: no per-engine data)",
                "engines",
                [("decode_run_max", band["max"])],
                band["t"],
                height=180,
                band={
                    "name": "min..max band",
                    "lo": band["min"],
                    "hi": band["max"],
                    "color": "#6cb2ff",
                },
            )
        )
    # B3: batcher queue dimension — master-side queueing between routing and
    # the batcher, per-second curves from the G3 timeline.
    batcher = dim.get("batcher") or {}
    if batcher.get("available"):
        b_series = batcher.get("series") or {}
        b_labels = None
        b_curves = []
        display = {
            "flexlb_app_flexlb_batcher_queue_size": "batcher_queue_size",
            "flexlb_app_routing_queue_length": "routing_queue_length",
        }
        for metric, curve in b_series.items():
            b_curves.append((display.get(metric, metric), curve["series"]))
            if b_labels is None:
                b_labels = curve["t"]
        if b_curves:
            body.append(
                line_chart(
                    "master batcher / routing queue depth (per second)",
                    "requests",
                    b_curves,
                    b_labels or [],
                )
            )
        agg_rows = [
            (display.get(m, m), f"avg={v['avg']} max={v['max']} min={v['min']}")
            for m, v in (batcher.get("aggregate") or {}).items()
        ]
        body.append(
            '<div class="panel"><h3>batcher aggregate</h3>'
            + kv_table(agg_rows)
            + "</div>"
        )
    elif batcher.get("reason_unavailable"):
        body.append(unavailable_panel(batcher["reason_unavailable"]))
    agg_rows = [
        (k, f"avg={v['avg']} max={v['max']} min={v['min']}")
        for k, v in dim["aggregate"].items()
    ]
    body.append(
        '<div class="panel"><h3>aggregate (full run)</h3>'
        + kv_table(agg_rows)
        + "</div>"
    )
    return _sec(2, "Queue Depth", "java_mock_stats timeline (26 fields)", "".join(body))


def render_balance(report: Dict[str, Any]) -> str:
    dim = report["balance"]
    body = []
    grade = dim.get("grade", "unavailable")
    gini_v = (dim.get("decode_from_per_request") or {}).get("gini")
    body.append('<div class="verdict-grid">')
    cls = {"excellent": "good", "good": "good", "fair": "warn", "poor": "bad"}.get(
        grade, ""
    )
    body.append(verdict_card(cls, "balance grade", grade, f"decode gini={gini_v}"))
    util = dim.get("utilization")
    if util:
        body.append(
            verdict_card(
                "",
                "decode utilization",
                f"{util['avg_utilization'] * 100:.1f}%",
                f"peak {util['peak_utilization'] * 100:.1f}% of cap {util['capacity']}",
            )
        )
    for key, label in (
        ("decode_from_per_request", "decode (per_request)"),
        ("prefill_from_per_request", "prefill (per_request)"),
        ("final_snapshot_accepted", "final snapshot accepted"),
    ):
        metrics = dim.get(key) or {}
        if metrics.get("available"):
            engines = metrics["engines"]
            rows = [(f"{label}: {e}", c) for e, c in engines.items()]
            rows += [
                ("  gini", metrics.get("gini")),
                ("  cv", metrics.get("cv")),
                ("  max/min", metrics.get("max_min")),
                ("  p90/p10", metrics.get("p90_p10")),
            ]
            body.append(
                '<div class="panel"><h3>'
                + html.escape(label)
                + "</h3>"
                + kv_table(rows)
                + "</div>"
            )
    body.append("</div>")
    psg = dim.get("per_second_gini") or []
    if psg:
        body.append(
            line_chart(
                "per-second decode Gini (per-engine)",
                "gini",
                [("gini", [r["gini"] for r in psg])],
                [r["idx"] for r in psg],
            )
        )
    if dim.get("capacity_note"):
        body.append(f'<div class="banner">{html.escape(dim["capacity_note"])}</div>')
    if dim.get("mock_extremes"):
        body.append(
            '<div class="panel"><h3>mock extremes</h3>'
            + kv_table([(k, v) for k, v in dim["mock_extremes"].items()])
            + "</div>"
        )
    return _sec(
        3,
        "Scheduling Balance",
        "per_request engine counters + final_snapshot + mock extremes",
        "".join(body),
    )


def render_inflight(report: Dict[str, Any]) -> str:
    dim = report["inflight"]
    verdict = dim.get("leak_verdict", "insufficient_data")
    body = ['<div class="verdict-grid">']
    cls = (
        "good"
        if verdict == "clean"
        else ("bad" if verdict == "suspected_leak" else "warn")
    )
    body.append(
        verdict_card(
            cls,
            "leak verdict",
            verdict,
            f"tail residual={dim.get('tail_residual') if dim.get('tail_residual') is not None else dim.get('g4_tail_residual')} "
            f"slope_max={dim.get('tail_slope_max')}",
        )
    )
    status = dim.get("inflight_status") or {}
    if status:
        body.append(
            verdict_card(
                "",
                "scheduler inflight peak",
                status.get("peak_scheduler_inflight"),
                "G4 /rtp_llm/inflight_status",
            )
        )
        body.append(
            verdict_card(
                "",
                "prefill batch peak",
                status.get("peak_prefill_batches"),
                "G4 per-endpoint sum",
            )
        )
        body.append(
            verdict_card(
                "",
                "decode request peak",
                status.get("peak_decode_requests"),
                "G4 per-endpoint sum",
            )
        )
    if dim.get("peak") is not None:
        body.append(
            verdict_card(
                "",
                "schedule-RPC inflight peak",
                dim.get("peak"),
                "arrival&#8722;completion counter delta",
            )
        )
    if dim.get("inflight_max_age_ms") is not None:
        body.append(
            verdict_card(
                "",
                "inflight max age",
                f"{dim['inflight_max_age_ms']}ms",
                "master prometheus final",
            )
        )
    body.append("</div>")
    # M3: G4 inflight_status panels — scheduler curve + per-endpoint curves.
    if status:
        sched = status.get("scheduler_series") or []
        if sched:
            body.append(
                line_chart(
                    "master scheduler_inflight (G4)",
                    "requests",
                    [("scheduler_inflight", [r["scheduler_inflight"] for r in sched])],
                    [r["t"] for r in sched],
                )
            )
        labels = status.get("labels") or []
        if labels and (
            status.get("prefill_total_batches") is not None
            or status.get("decode_total_requests") is not None
        ):
            series = []
            if status.get("prefill_total_batches") is not None:
                series.append(
                    ("prefill_batches_total", status["prefill_total_batches"])
                )
            if status.get("decode_total_requests") is not None:
                series.append(
                    ("decode_requests_total", status["decode_total_requests"])
                )
            body.append(
                line_chart(
                    "engine-level in-flight totals (G4, per-endpoint sums)",
                    "batches / requests",
                    series,
                    labels,
                )
            )
        prefill_eps = status.get("prefill_endpoints") or {}
        for ep, slot in sorted(prefill_eps.items())[:8]:
            body.append(
                line_chart(
                    f"prefill endpoint {ep} in-flight (G4)",
                    "requests",
                    [
                        ("inflight_batches", slot["inflight_batches"]),
                        ("inflight_requests", slot["inflight_requests"]),
                        ("inflight_route_requests", slot["inflight_route_requests"]),
                    ],
                    labels,
                    height=150,
                )
            )
        decode_eps = status.get("decode_endpoints") or {}
        for ep, slot in sorted(decode_eps.items())[:8]:
            body.append(
                line_chart(
                    f"decode endpoint {ep} in-flight (G4)",
                    "requests",
                    [("inflight_requests", slot["inflight_requests"])],
                    labels,
                    height=150,
                )
            )
    # auxiliary: master arrival/completion counter delta (schedule-RPC
    # inflight; includes not-yet-scheduled backlog).
    aux = dim.get("schedule_rpc_inflight") or {}
    if aux.get("series"):
        series = aux["series"]
        body.append(
            line_chart(
                "schedule-RPC in-flight (arrival_total &#8722; completion_total, auxiliary)",
                "requests",
                [("inflight", [r["inflight"] for r in series])],
                [r["t"] for r in series],
            )
        )
    # S7: inflight max age per-second curve.
    age_curve = dim.get("inflight_max_age_series") or {}
    if age_curve.get("available"):
        body.append(
            line_chart(
                "inflight max age per second (G3)",
                "ms",
                [("inflight_max_age_ms", age_curve["series"])],
                age_curve["t"],
            )
        )
    if dim.get("mock_leak_lines"):
        items = "".join(f"<li>{html.escape(l)}</li>" for l in dim["mock_leak_lines"])
        body.append(
            f'<div class="banner bad-t"><b>LEAK DETECTED lines in mock log:</b><ul>{items}</ul></div>'
        )
    if dim.get("final_snapshot_inflight_nonzero"):
        body.append(
            '<div class="banner bad-t">final_snapshot non-zero inflight: '
            + html.escape(json.dumps(dim["final_snapshot_inflight_nonzero"]))
            + "</div>"
        )
    if not dim.get("available"):
        body.append(
            unavailable_panel(
                dim.get("reason_unavailable")
                or "master inflight_timeseries / counters not found — leak check unavailable"
            )
        )
    return _sec(
        4,
        "In-flight & Leak Detection",
        "G4 inflight_status + master counters (schedule-RPC aux) + LEAK scan",
        "".join(body),
    )


def render_kv_usage(report: Dict[str, Any]) -> str:
    dim = report["kv_usage"]
    if not dim.get("available"):
        return _sec(
            5,
            "KV Cache Usage",
            "master prometheus flexlb_app_cache_*",
            unavailable_panel(dim.get("reason_unavailable") or "unavailable"),
        )
    # available can be satisfied by the per-engine G1 timeline alone (the
    # final snapshot may carry no flexlb_app_cache_* series), so both panels
    # are conditional.
    body = []
    metrics = dim.get("metrics")
    if metrics:
        body.append(
            '<div class="panel"><h3>final snapshot metrics</h3>'
            + kv_table(list(metrics.items()))
            + "</div>"
        )
    per_engine = dim.get("per_engine") or {}
    for engine, curve in sorted(per_engine.items()):
        if not curve:
            continue
        body.append(
            line_chart(
                f"{engine} kv_cache_tokens (per-engine)",
                "tokens",
                [("kv_tokens", [p["kv_tokens"] for p in curve])],
                [p["t"] for p in curve],
            )
        )
        # M4: ratio curve keeps fully-reserved samples (available==0 with
        # active>0 -> ratio 1.0) instead of dropping them.
        ratios = [p.get("kv_ratio") for p in curve]
        if any(r is not None for r in ratios):
            body.append(
                line_chart(
                    f"{engine} kv_cache_ratio (per-engine)",
                    "ratio",
                    [("kv_ratio", [r if r is not None else 0.0 for r in ratios])],
                    [p["t"] for p in curve],
                    height=150,
                )
            )
    if not body:
        body.append(
            unavailable_panel(
                dim.get("reason_unavailable")
                or "no kv_cache series in the final snapshot"
            )
        )
    return _sec(
        5,
        "KV Cache Usage",
        "master prometheus flexlb_app_cache_* + per-engine timeline",
        "".join(body),
    )


def render_kv_match(report: Dict[str, Any]) -> str:
    dim = report["kv_match"]
    if not dim.get("available"):
        return _sec(
            6,
            "KV Cache Match Rate",
            "hit_ratio / theory_hit_ratio / recent_key",
            unavailable_panel(dim.get("reason_unavailable") or "unavailable"),
        )
    body = []
    if dim.get("metrics"):
        body.append(
            '<div class="panel"><h3>final snapshot</h3>'
            + kv_table(list(dim["metrics"].items()))
            + "</div>"
        )
    # S7: per-second hit_ratio curves (label variants averaged per second).
    per_second = dim.get("per_second") or {}
    display = {
        "flexlb_app_cache_hit_ratio": "hit_ratio",
        "flexlb_app_cache_theory_hit_ratio": "theory_hit_ratio",
    }
    if per_second:
        curves = []
        labels = None
        for metric, curve in per_second.items():
            if curve.get("available"):
                curves.append((display.get(metric, metric), curve["series"]))
                if labels is None:
                    labels = curve["t"]
        if curves:
            body.append(
                line_chart(
                    "cache hit_ratio per second (G3, engine avg)",
                    "ratio",
                    curves,
                    labels or [],
                )
            )
    if not body:
        body.append(unavailable_panel(dim.get("reason_unavailable") or "unavailable"))
    return _sec(
        6,
        "KV Cache Match Rate",
        "master prometheus final snapshot + per-second hit_ratio curves",
        "".join(body),
    )


def render_cpu_mem(report: Dict[str, Any]) -> str:
    dim = report["cpu_mem"]
    if not dim.get("available"):
        return _sec(
            7,
            "CPU / Memory",
            "mock heap + GC + process usage",
            unavailable_panel(dim.get("reason_unavailable") or "unavailable"),
        )
    body = []
    heap = dim.get("heap")
    if heap:
        body.append(
            line_chart(
                "mock engine heap used",
                "MB",
                [("heap_used", heap["used_mb"])],
                heap["t"],
            )
        )
        body.append(
            '<div class="panel"><h3>heap</h3>'
            + kv_table(
                [
                    ("peak used MB", heap.get("peak_used")),
                    ("heap max MB", heap.get("max_mb")),
                ]
            )
            + "</div>"
        )
    gc = dim.get("gc")
    if gc:
        body.append(
            '<div class="panel"><h3>jvm gc (prometheus final)</h3>'
            + kv_table(list(gc.items()))
            + "</div>"
        )
    pu = dim.get("process_usage")
    if pu:
        # m5: one panel per label (mock / master / client_N) with BOTH the
        # cpu% and the RSS curve; pids that shared the label are listed in
        # the chart title (a restarted JVM splits across pids, not labels).
        cpu_panels = []
        rss_panels = []
        for label, series in list(pu.items())[:6]:
            n = len(series["cpu"])
            pid_note = (
                f" pids={','.join(series.get('pids', []))}"
                if series.get("pids")
                else ""
            )
            cpu_panels.append(
                line_chart(
                    f"{label} cpu%{pid_note}",
                    "%",
                    [("cpu", series["cpu"])],
                    [i for i in range(n)],
                    height=150,
                )
            )
            rss_panels.append(
                line_chart(
                    f"{label} rss",
                    "MB",
                    [("rss_mb", [round(v / 1024, 1) for v in series["rss_kb"]])],
                    [i for i in range(n)],
                    height=150,
                )
            )
        body.append('<div class="grid2">' + "".join(cpu_panels) + "</div>")
        body.append('<div class="grid2">' + "".join(rss_panels) + "</div>")
    return _sec(
        7, "CPU / Memory", "mock heap curve + GC pauses + per-JVM usage", "".join(body)
    )


def render_pacing(report: Dict[str, Any]) -> str:
    dim = report["pacing"]
    if not dim.get("available"):
        return _sec(
            8,
            "Pacing Quality",
            "pacing_lag_ms + target vs actual QPS",
            unavailable_panel(dim.get("reason_unavailable") or "unavailable"),
        )
    ps = dim.get("per_second") or []
    body = ['<div class="verdict-grid">']
    cls = {"good": "good", "degraded": "warn", "bad": "bad"}.get(
        dim.get("verdict", ""), ""
    )
    body.append(
        verdict_card(
            cls,
            "pacing verdict",
            dim.get("verdict"),
            f"lag p99={dim['pacing_lag_ms']['p99']:.1f}ms",
        )
    )
    body.append(verdict_card("", "target QPS", dim.get("target_qps") or "n/a"))
    body.append(verdict_card("", "actual QPS", dim.get("actual_qps") or "n/a"))
    if dim.get("send_vs_target_ratio") is not None:
        body.append(verdict_card("", "actual / target", dim["send_vs_target_ratio"]))
    body.append("</div>")
    if ps:
        body.append(
            line_chart(
                "pacing lag percentiles per second",
                "ms",
                [
                    ("lag_p50", [r["lag_p50"] for r in ps]),
                    ("lag_p99", [r["lag_p99"] for r in ps]),
                ],
                [r["t"] for r in ps],
            )
        )
    return _sec(
        8, "Pacing Quality", "pacing_lag_ms per-second + send vs target", "".join(body)
    )


def render_sla(report: Dict[str, Any]) -> str:
    dim = report["sla"]
    ps = dim.get("per_second") or []
    body = ['<div class="verdict-grid">']
    if dim.get("computed_rate") is not None:
        rate = dim["computed_rate"]
        body.append(
            verdict_card(
                "bad" if rate > 0.05 else ("warn" if rate > 0.01 else "good"),
                "SLA violation rate",
                f"{rate * 100:.2f}%",
                f"sla_ttft={dim.get('sla_ttft_ms')}ms",
            )
        )
    body.append(
        verdict_card(
            "", "summary violations", dim.get("summary_sla_violations") or "n/a"
        )
    )
    body.append("</div>")
    if ps:
        body.append(
            line_chart(
                "SLA violation rate per second",
                "ratio",
                [("rate", [r["rate"] for r in ps])],
                [r["t"] for r in ps],
            )
        )
    if not dim.get("available"):
        body.append(unavailable_panel("no successful ttft samples"))
    return _sec(9, "SLA Violations", "per-second ttft vs sla_ttft_ms", "".join(body))


def render_dispatch(report: Dict[str, Any]) -> str:
    dim = report["dispatch"]
    if not dim.get("available"):
        return _sec(
            10,
            "Dispatch Decisions",
            "flexlb_batch_dispatch/complete",
            unavailable_panel(dim.get("reason_unavailable") or "unavailable"),
        )
    body = ['<div class="verdict-grid">']
    body.append(
        verdict_card(
            "good" if dim.get("invariant_violation_count") == 0 else "bad",
            "invariant violations",
            dim.get("invariant_violation_count"),
            "3 decision invariants",
        )
    )
    body.append(
        verdict_card("", "dispatches", dim.get("dispatch_count"), dim.get("source"))
    )
    body.append(verdict_card("", "completions", dim.get("completion_count")))
    gap = dim.get("prediction_gap_ms") or {}
    body.append(
        verdict_card("", "gap p99", f"{gap.get('p99', 0)}ms", "predicted vs actual")
    )
    body.append("</div>")
    reasons = [{"class": k, "count": v} for k, v in dim.get("reasons", {}).items()]
    body.append(
        bar_table("dispatch reasons", sorted(reasons, key=lambda r: -r["count"]))
    )
    body.append('<div class="grid2">')
    body.append(
        scatter_chart(
            "predicted vs actual batch latency",
            dim.get("scatter") or [],
            "predicted_ms",
            "actual_ms",
        )
    )
    body.append(
        '<div class="panel"><h3>distributions</h3>'
        + kv_table(
            [
                ("batch_size", dim.get("batch_size")),
                ("wait_ms", dim.get("wait_ms")),
                ("predicted_ms", dim.get("predicted_ms")),
                ("actual_ms", dim.get("actual_ms")),
                ("gap_ms", dim.get("prediction_gap_ms")),
            ]
        )
        + "</div>"
    )
    body.append("</div>")
    if dim.get("invariant_violations"):
        items = "".join(
            f"<li><code>{html.escape(json.dumps(v))}</code></li>"
            for v in dim["invariant_violations"][:8]
        )
        body.append(
            f'<div class="banner bad-t"><b>invariant violations:</b><ul>{items}</ul></div>'
        )
    return _sec(
        10,
        "Dispatch Decisions",
        "master.log structured lines + 3 invariants",
        "".join(body),
    )


def render_priority(report: Dict[str, Any]) -> str:
    dim = report["priority"]
    if not dim.get("available"):
        return _sec(
            11,
            "Priority Facets",
            "P30/P50/P70 slices",
            unavailable_panel(dim.get("reason_unavailable") or "unavailable"),
        )
    rows = []
    for prio, f in dim["facets"].items():
        sched = f["sched_ms"]
        rows.append(
            (
                f"P{prio}",
                f"send={f['send']} ok={f['ok']} err={f['err']} ok_rate={f['ok_rate']:.3f} "
                f"sched_p50={sched.get('p50', 0)} sched_p99={sched.get('p99', 0)} "
                f"ttft_p99={f['ttft_ms'].get('p99', 0)} err8429={f['err_8429']}",
            )
        )
    body = '<div class="panel"><h3>per-priority</h3>' + kv_table(rows) + "</div>"
    return _sec(11, "Priority Facets", "ok/err, latency, 8429 per priority", body)


def render_error_matrix(report: Dict[str, Any]) -> str:
    dim = report["error_code_matrix"]
    if not dim.get("available"):
        return _sec(
            12,
            "Error Code x Priority",
            "code=N extraction",
            unavailable_panel(dim.get("reason_unavailable") or "unavailable"),
        )
    matrix = dim["matrix"]
    prios = sorted({p for counts in matrix.values() for p in counts})
    head = "".join(f"<th>P{p}</th>" for p in prios)
    rows = []
    for code, counts in sorted(matrix.items()):
        cells = "".join(f'<td class="num">{counts.get(p, "")}</td>' for p in prios)
        rows.append(f"<tr><td>code={code}</td>{cells}</tr>")
    body = (
        f'<div class="panel"><h3>error code &#215; priority matrix</h3>'
        f'<table class="t"><thead><tr><th>code</th>{head}</tr></thead><tbody>{"".join(rows)}</tbody></table></div>'
    )
    return _sec(
        12, "Error Code x Priority", "cross table from per_request errors", body
    )


def render_length_matrix(report: Dict[str, Any]) -> str:
    dim = report["length_matrix"]
    if not dim.get("available"):
        return _sec(
            13,
            "Length-slice Latency Matrix",
            "input/output deciles x schedule p99",
            unavailable_panel(dim.get("reason_unavailable") or "unavailable"),
        )
    i_edges = dim["input_edges"]
    o_edges = dim["output_edges"]
    n_i, n_o = len(i_edges) + 1, len(o_edges) + 1
    grid = {}
    max_v = 0.0
    for cell in dim["cells"]:
        grid[(cell["input_bucket"], cell["output_bucket"])] = cell
        max_v = max(max_v, cell["sched_p99"])

    def bucket_label(idx: int, edges: List[int]) -> str:
        lo = 0 if idx == 0 else edges[idx - 1]
        hi = edges[idx] if idx < len(edges) else "&#8734;"
        return f"{lo}-{hi}"

    head = "".join(f"<th>out {bucket_label(j, o_edges)}</th>" for j in range(n_o))
    rows = []
    for i in range(n_i):
        cells = []
        for j in range(n_o):
            cell = grid.get((i, j))
            if cell:
                alpha = 0.12 + 0.75 * (cell["sched_p99"] / max_v if max_v else 0)
                cells.append(
                    f'<td class="num" style="background:rgba(232,163,61,{alpha:.2f})">'
                    f'{cell["sched_p99"]:.0f}<br><span style="font-size:10px;color:#8a93a6">n={cell["count"]}</span></td>'
                )
            else:
                cells.append('<td class="num muted">&#183;</td>')
        rows.append(
            f"<tr><td><b>in {bucket_label(i, i_edges)}</b></td>{''.join(cells)}</tr>"
        )
    body = (
        f'<div class="panel"><h3>schedule p99 [ms] by input_len &#215; output_len decile '
        f'({dim["n_rows"]} ok rows)</h3>'
        f'<table class="t"><thead><tr><th></th>{head}</tr></thead><tbody>{"".join(rows)}</tbody></table></div>'
    )
    return _sec(
        13, "Length-slice Latency Matrix", "input/output deciles x schedule p99", body
    )


def render_concurrency(report: Dict[str, Any]) -> str:
    dim = report["client_concurrency"]
    if not dim.get("available"):
        return _sec(
            14,
            "Client Concurrency Watermark",
            "send_start/total_ms event replay",
            unavailable_panel(dim.get("reason") or "unavailable"),
        )
    body = ['<div class="verdict-grid">']
    body.append(verdict_card("", "peak in-flight (client)", dim.get("peak")))
    body.append(verdict_card("", "p50 per-second max", dim.get("p50")))
    body.append(
        verdict_card(
            "",
            "configured max_concurrency",
            dim.get("configured_max_concurrency") or "n/a",
        )
    )
    body.append("</div>")
    body.append(
        line_chart(
            "client-side in-flight watermark (per-second max)",
            "requests",
            [("max in-flight", dim["series_max"])],
            dim["series_t"],
        )
    )
    return _sec(
        14,
        "Client Concurrency Watermark",
        "rebuilt from send_start + total_ms",
        "".join(body),
    )


def render_fallback(report: Dict[str, Any]) -> str:
    dim = report["fallback_path"]
    body = ['<div class="verdict-grid">']
    body.append(
        verdict_card(
            "warn" if dim.get("count") else "good",
            "fallback rows",
            dim.get("count"),
            f"share={dim.get('share')}",
        )
    )
    body.append(
        verdict_card("", "fallback ok / err", f"{dim.get('ok')} / {dim.get('err')}")
    )
    body.append("</div>")
    body.append(
        '<div class="panel"><h3>fallback schedule_ms (isolated)</h3>'
        + kv_table([("sched_ms", dim.get("sched_ms"))])
        + "</div>"
    )
    if dim.get("note"):
        body.append(f'<div class="banner">{html.escape(dim["note"])}</div>')
    return _sec(
        15, "Fallback Path Isolation", "enqueued_by_master=false", "".join(body)
    )


def render_comparison(report: Dict[str, Any]) -> str:
    comp = report.get("comparison")
    if not comp:
        return ""
    sections = []
    for run in comp.get("runs", []):
        rows = []
        for r in run["delta_table"]:
            delta = r["delta"]
            if isinstance(delta, (int, float)):
                delta_s = (
                    f"{delta:+,.4f}" if isinstance(delta, float) else f"{delta:+,}"
                )
                if r.get("pct") is not None:
                    delta_s += f" ({r['pct']:+.1f}%)"
                cls = (
                    "bad-t"
                    if isinstance(delta, (int, float))
                    and delta > 0
                    and "rate" in r["metric"]
                    else ""
                )
            else:
                delta_s = str(delta) if delta is not None else "same"
                cls = ""
            rows.append(
                f'<tr><td>{html.escape(r["metric"])}</td>'
                f'<td class="num">{html.escape(_val_str(r["base"]))}</td>'
                f'<td class="num">{html.escape(_val_str(r["other"]))}</td>'
                f'<td class="num {cls}">{html.escape(delta_s)}</td></tr>'
            )
        sections.append(
            f'<div class="panel"><h3>vs {html.escape(str(run.get("run_id")))}</h3>'
            f'<table class="t"><thead><tr><th>metric</th><th>base</th><th>other</th><th>delta</th></tr></thead>'
            f'<tbody>{"".join(rows)}</tbody></table></div>'
        )
    return _sec(16, "Run Comparison", "key metric deltas", "".join(sections))


TOOLTIP_JS = """
const DATA = JSON.parse(document.getElementById('chart-data').textContent);
const TT = document.getElementById('tooltip');
document.querySelectorAll('.chart').forEach(el => {
  el.addEventListener('mousemove', ev => {
    const d = DATA[el.dataset.cid];
    if (!d) return;
    const r = el.getBoundingClientRect();
    const n = d.x.length;
    const i = Math.max(0, Math.min(n - 1, Math.round((ev.clientX - r.left) / r.width * (n - 1))));
    let out = '<b>' + (d.x_labels[i] !== undefined ? d.x_labels[i] : i) + '</b>';
    for (const s of d.series) {
      const v = s.data[i];
      if (v !== null && v !== undefined) out += '<div><span style="color:' + s.color + '">&#9632;</span> ' + s.name + ': ' + v + (d.unit ? ' ' + d.unit : '') + '</div>';
    }
    TT.innerHTML = out;
    TT.style.display = 'block';
    TT.style.left = Math.min(window.innerWidth - 280, ev.clientX + 16) + 'px';
    TT.style.top = (ev.clientY + 16) + 'px';
  });
  el.addEventListener('mouseleave', () => { TT.style.display = 'none'; });
});
"""


def render_html(report: Dict[str, Any]) -> str:
    _CHARTS.clear()
    _chart_seq[0] = 0
    meta = report["meta"]
    verdict = report["verdict"]
    v = verdict
    cards = []
    tv = v.get("test_valid")
    cards.append(
        verdict_card(
            "good" if tv else ("bad" if tv is False else ""),
            "test_valid",
            "PASS" if tv else ("FAIL" if tv is False else "n/a"),
            "summary validity_checks",
        )
    )
    leak = v.get("leak_verdict")
    cards.append(
        verdict_card(
            (
                "good"
                if leak == "clean"
                else ("bad" if leak == "suspected_leak" else "warn")
            ),
            "leak verdict",
            leak,
            "counters tail + LEAK scan",
        )
    )
    grade = v.get("balance_grade")
    cards.append(
        verdict_card(
            {"excellent": "good", "good": "good", "fair": "warn", "poor": "bad"}.get(
                grade, ""
            ),
            "balance grade",
            grade,
            f"decode gini={v.get('balance_gini')}",
        )
    )
    err_rate = v.get("error_rate") or 0
    cards.append(
        verdict_card(
            "bad" if err_rate > 0.2 else ("warn" if err_rate > 0.02 else "good"),
            "error rate",
            f"{err_rate * 100:.2f}%",
            "errors / total",
        )
    )
    pacing = v.get("pacing_verdict")
    cards.append(
        verdict_card(
            {"good": "good", "degraded": "warn", "bad": "bad"}.get(pacing, ""),
            "pacing",
            pacing,
            "lag p99 + send/target",
        )
    )
    sections = [
        render_qps(report),
        render_latency(report),
        render_queues(report),
        render_balance(report),
        render_inflight(report),
        render_kv_usage(report),
        render_kv_match(report),
        render_cpu_mem(report),
        render_pacing(report),
        render_sla(report),
        render_dispatch(report),
        render_priority(report),
        render_error_matrix(report),
        render_length_matrix(report),
        render_concurrency(report),
        render_fallback(report),
        render_comparison(report),
    ]
    banner = ""
    if meta.get("fetch_output_stream") is False:
        banner = (
            '<div class="banner"><b>FETCH_OUTPUT_STREAM=0</b> &mdash; end-to-end latency layer '
            "(total_ms / ttft_ms) is disabled in this run (client skipped engine "
            "stream reads; engine executed fully); schedule latency remains valid.</div>"
        )
    chart_json = json.dumps(
        {cid: payload for cid, payload in _CHARTS}, separators=(",", ":")
    ).replace("</", "<\\/")
    src_rows = "".join(
        f"<tr><td>{html.escape(k)}</td><td>{html.escape(str(s))}</td></tr>"
        for k, s in sorted(meta.get("sources", {}).items())
    )
    return f"""<!DOCTYPE html>
<html lang="zh-cn"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>FlexLB Run Analysis &mdash; {html.escape(meta['run_id'])}</title>
<style>{CSS}</style></head>
<body><div class="wrap">
<div class="hero">
<h1>FLEXLB RUN ANALYSIS <small>// unified online_eval report</small></h1>
<div class="meta-bar">
<span>run <b>{html.escape(meta['run_id'])}</b></span>
<span>dir <b>{html.escape(meta['run_dir'])}</b></span>
<span>generated <b>{html.escape(meta['generated_at_utc'])}</b></span>
<span>dimensions <b>{len([s for s in sections if s])}</b></span>
<span>charts <b>{len(_CHARTS)}</b></span>
</div></div>
{banner}
<div class="verdict-grid">{''.join(cards)}</div>
{''.join(s for s in sections if s)}
<section class="sec"><div class="sec-head"><div class="sec-no">++</div>
<h2>Data Sources</h2><span class="sec-sub">resolver decisions (legacy first, consolidated fallback)</span></div>
<div class="panel"><table class="t"><thead><tr><th>source</th><th>resolved file</th></tr></thead>
<tbody>{src_rows}</tbody></table></div></section>
<footer>analyze_eval_run.py &middot; self-contained report &middot; no external dependencies</footer>
</div>
<div id="tooltip"></div>
<script id="chart-data" type="application/json">{chart_json}</script>
<script>{TOOLTIP_JS}</script>
</body></html>"""


# ---------------------------------------------------------------------------
# --self-test: synthesize minimal-but-legal runs from the aggregation JSONs
# ---------------------------------------------------------------------------

import random  # noqa: E402  (self-test only; keeps the analysis imports lean)

ERR_TEMPLATES_SCHEDULE = [
    (
        0.955,
        "admission rejected: active_admissions={active} limit={limit} (backpressure), retryable",
    ),
    (0.02, "schedule_error: preempted by higher-priority request {req} (code=8429)"),
    (0.01, "schedule_error: code=8431 resource busy: kv blocks exhausted"),
    (
        0.005,
        "schedule_error: yielded to higher-priority request {req} (code=8400 NO_AVAILABLE_WORKER)",
    ),
    (0.005, "schedule_error: code=8502 queue full: decode queue reached capacity"),
    (0.003, "schedule_error: code=8403 NO_DECODE_WORKER available"),
    (0.002, "schedule_error: code=8511 SLO expired before dispatch"),
]
ERR_TEMPLATES_EXCEPTION = [
    (
        0.70,
        "io.grpc.StatusRuntimeException: DEADLINE_EXCEEDED: CallOptions deadline exceeded after 9.997s. [closed=[], open=[[remote_addr=/127.0.0.1:62500]]]",
    ),
    (
        0.15,
        "io.grpc.StatusRuntimeException: INTERNAL: RST_STREAM closed by remote peer",
    ),
    (0.10, "engine error: code=8513 engine execution failed on decode step"),
    (0.05, "dispatch error: code=8510 dispatch failed: worker lost"),
]


def _pick_template(templates: List[Tuple[float, str]], roll: float) -> str:
    """Deterministic weighted pick: roll in [0,1) -> template text."""
    acc = 0.0
    for weight, text in templates:
        acc += weight
        if roll <= acc:
            return text
    return templates[-1][1]


def _sched_error_text(idx: int, active: int, limit: int, req: int) -> str:
    """Deterministic round-robin taxonomy (weights match the audit mix; a
    full 1000-row window covers every class so small samples still hit all)."""
    m = idx % 1000
    if m < 955:
        return f"admission rejected: active_admissions={active} limit={limit} (backpressure), retryable"
    if m < 975:
        return f"schedule_error: preempted by higher-priority request {req} (code=8429)"
    if m < 985:
        return "schedule_error: code=8431 resource busy: kv blocks exhausted"
    if m < 990:
        return f"schedule_error: yielded to higher-priority request {req} (code=8400 NO_AVAILABLE_WORKER)"
    if m < 995:
        return "schedule_error: code=8502 queue full: decode queue reached capacity"
    if m < 998:
        return "schedule_error: code=8403 NO_DECODE_WORKER available"
    return "schedule_error: code=8511 SLO expired before dispatch"


def _exc_error_text(exc_idx: int) -> str:
    """Deterministic round-robin over the 4 exception templates using an
    exception-local counter (not the global row idx): 8 consecutive
    exceptions cover every class even in small-scale self-tests."""
    m = exc_idx % 8
    if m < 5:
        return (
            "io.grpc.StatusRuntimeException: DEADLINE_EXCEEDED: CallOptions deadline "
            "exceeded after 9.997s. [closed=[], open=[[remote_addr=/127.0.0.1:62500]]]"
        )
    if m == 5:
        return (
            "io.grpc.StatusRuntimeException: INTERNAL: RST_STREAM closed by remote peer"
        )
    if m == 6:
        return "engine error: code=8513 engine execution failed on decode step"
    return "dispatch error: code=8510 dispatch failed: worker lost"


def _make_per_request_row(
    rng: random.Random,
    idx: int,
    started_ms: float,
    ok: bool,
    status: str,
    sched: float,
    prio: int,
    prefill: str,
    decode: str,
    fallback: bool,
    exc_idx: int = 0,
) -> Dict[str, Any]:
    if ok:
        ttft = round(sched + rng.uniform(80, 900), 3)
        total = round(ttft + rng.uniform(100, 2500), 3)
        error = ""
    else:
        ttft = 0.0
        total = round(rng.uniform(5, 10500), 3)
        if status == "exception":
            error = _exc_error_text(exc_idx)
        else:
            error = _sched_error_text(idx, 256, 256, 100000 + idx)
    return {
        "rid": f"{9000000000000000000 + idx}",
        "trace_id": f"{9000000000000000000 + idx}",
        "request_id": 8000000000000000000 + idx,
        "ts": round(rng.uniform(0, 150000), 1),
        "input_len": rng.randint(48, 9000),
        "output_len": rng.randint(8, 700),
        "status": status,
        "schedule_ms": round(sched, 3),
        "ttft_ms": ttft,
        "total_ms": total,
        "enqueued_by_master": not fallback,
        "prefill": prefill if ok else "",
        "decode": decode if ok else "",
        "error": error,
        "route_path": "master" if not fallback else "fallback",
        "wall_clock_ts": round(started_ms / 1000.0, 3),
        "send_due_epoch_ms": float(int(started_ms)),
        "send_start_epoch_ms": started_ms,
        "pacing_lag_ms": round(abs(rng.gauss(0.5, 3.0)), 3),
        "priority": prio,
    }


def synthesize_per_request(
    agg: Dict[str, Any], rng_seed: int = 42
) -> List[Dict[str, Any]]:
    """Expand an aggregation JSON's per_second timeline back into rows."""
    rng = random.Random(rng_seed)
    start_epoch = agg.get("start_epoch_ms", 1787623000000)
    rows: List[Dict[str, Any]] = []
    prios = [30, 50, 70]
    prefill_engines = [f"127.0.0.1:6250{i}" for i in range(4)]
    decode_engines = [f"127.0.0.1:6251{i}" for i in range(4)]
    decode_weights = [0.35, 0.30, 0.20, 0.15]  # deliberately uneven
    idx = 0
    exc_count = 0
    for sec in agg.get("per_second", []):
        t, send, ok_n = sec["t"], sec["send"], sec["ok"]
        err_n = send - ok_n
        p50, p95, p99 = (
            sec.get("sched_p50", 1),
            sec.get("sched_p95", 2),
            sec.get("sched_p99", 3),
        )
        entries: List[Tuple[float, bool, str]] = []
        entries += [(float(t) * 1000, True, "ok")] * ok_n
        n_exc = max(1, int(err_n * 0.004)) if err_n else 0
        entries += [(float(t) * 1000, False, "exception")] * n_exc
        entries += [(float(t) * 1000, False, "schedule_error")] * (err_n - n_exc)
        rng.shuffle(entries)
        for i, (offset, ok, status) in enumerate(entries):
            started = start_epoch + t * 1000 + (i * 1000.0 / max(1, send))
            if ok:
                roll = rng.random()
                sched = p50 if roll < 0.6 else (p95 if roll < 0.9 else p99)
                sched *= rng.uniform(0.85, 1.15)
            else:
                sched = round(rng.uniform(0.1, max(p99, 5.0)), 3)
            prio = prios[idx % 3]
            prefill = prefill_engines[idx % 4]
            r = rng.random()
            acc = 0.0
            decode = decode_engines[0]
            for engine, w in zip(decode_engines, decode_weights):
                acc += w
                if r <= acc:
                    decode = engine
                    break
            fallback = rng.random() < 0.02
            if fallback and ok:
                # Real fallback rows (enqueued_by_master=false) carry the
                # client-side fallback path measurement — schedule_ms stays 0
                # (the master never scheduled them), which is exactly the B1
                # pollution shape the self-test asserts against.
                sched = 0.0
            rows.append(
                _make_per_request_row(
                    rng,
                    idx,
                    started,
                    ok,
                    status,
                    sched,
                    prio,
                    prefill,
                    decode,
                    fallback,
                    exc_idx=exc_count,
                )
            )
            if status == "exception":
                exc_count += 1
            idx += 1
    return rows


def _mock_stats_line(start_epoch: int, t: int, agg_row: Dict[str, Any]) -> str:
    return (
        f"java_mock_stats ts_epoch_ms={start_epoch + t * 1000} "
        f"enqueue_rpcs={agg_row.get('dr', 0) * 2} enqueued_requests={agg_row.get('pr', 0) * 3} "
        f"status_rpcs=100 cache_rpcs=50 prefill_batches={agg_row.get('pr', 0) + 1} "
        f"avg_batch_size={agg_row.get('avg_bs', 1.1):.2f} max_batch_size=4 "
        f"avg_batch_ms=300.00 max_batch_ms=324 "
        f"prefill_waiting={agg_row.get('pw', 0)} prefill_running={agg_row.get('pr', 0)} "
        f"prefill_running_reqs={agg_row.get('pr_reqs', 0)} max_prefill_waiting={agg_row.get('pw', 0) + 1} "
        f"decode_waiting={agg_row.get('dw', 0)} decode_running={agg_row.get('dr', 0)} "
        f"decode_run_min={max(0, agg_row.get('dr', 0) - 2)} decode_run_max={agg_row.get('dr', 0) + 2} "
        f"max_decode_waiting={agg_row.get('dw', 0)} decode_done={agg_row.get('dr', 0) * 7} "
        f"decode_exec_p50=188 decode_exec_p95=210 decode_exec_max=260 "
        f"heap_used_mb={agg_row.get('heap_mb', 100)} heap_max_mb=20480 "
        f"generate_stream_rpcs=0 fetch_response_rpcs={agg_row.get('pr', 0)} cancel_rpcs=0"
    )


def _master_log_lines(start_epoch: int, n_lines: int = 60) -> str:
    rng = random.Random(7)
    lines: List[str] = []
    reasons = ["fixed_window_timeout", "predicted_execution_cap", "batch_full"]
    for i in range(n_lines):
        reason = reasons[i % 3]
        batch_size = rng.randint(1, 8)
        wait = rng.randint(0, 40)
        predicted = rng.randint(20, 400)
        # deliberate invariant violations on marked lines; everywhere else keep
        # the three invariants satisfied: batch_full fills the group (size ==
        # max), fixed_window_timeout waited the window (>= fixed_wait-2),
        # predicted_execution_cap multi-member groups stay under budget.
        viol_cap = reason == "predicted_execution_cap" and i % 17 == 0
        viol_wait = reason == "fixed_window_timeout" and i % 23 == 5
        if reason == "batch_full":
            batch_size = 8
        elif reason == "fixed_window_timeout" and not viol_wait:
            wait = rng.randint(48, 60)
        elif reason == "predicted_execution_cap" and not viol_cap and batch_size > 1:
            predicted = min(predicted, 440)
        if viol_cap:
            batch_size, predicted = 3, 500
        lines.append(
            f"2026-08-25 10:00:{i % 60:02d} INFO flexlb_batch_dispatch batch_id={i} "
            f"reason={reason} batch_size={batch_size} wait_ms={wait} predicted_ms={predicted} "
            f"threshold_ms=450 fixed_wait_ms=50 batch_size_max=8 queue_after=2 worker=w0"
        )
        actual = predicted + rng.randint(-40, 60)
        lines.append(
            f"2026-08-25 10:00:{i % 60:02d} INFO flexlb_batch_complete batch_id={i} "
            f"predicted_ms={predicted} actual_ms={actual} gap_ms={actual - predicted} "
            f"batch_size={batch_size} engine=127.0.0.1:62510"
        )
    for sec in range(30, 150, 10):
        lines.append(
            f"2026-08-25 10:{sec // 60:02d}:{sec % 60:02d} INFO flexlb_server_schedule_latency "
            f"count={sec * 37} arrival_qps=7900.0 completion_qps=6100.0 "
            f"server_p50_ms={0.18 + sec * 0.001:.2f} server_p95_ms={1.2 + sec * 0.01:.2f} "
            f"server_p99_ms={155.0 + (sec % 7):.1f} grpc_queue_p95_ms=0.4 route_submit_p95_ms=0.2 "
            f"batch_wait_p95_ms=150.5 dispatch_ack_p95_ms=2.1 ack_response_p95_ms=0.6 "
            f"batch_wait_p95_prio30_ms=180.2 batch_wait_p95_prio50_ms=150.5 batch_wait_p95_prio70_ms=98.1"
        )
    return "\n".join(lines) + "\n"


def _summary_json(agg: Dict[str, Any], rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    totals = agg.get("totals", {})
    ok_n = totals.get("ok", 0)
    err_n = totals.get("err", 0)
    return {
        "total_requests": totals.get("requests", len(rows)),
        "completed": ok_n,
        "scheduled": ok_n,
        "success_count": ok_n,
        "errors": err_n,
        "error_count": err_n,
        "error_rate": round(err_n / max(1, ok_n + err_n), 6),
        "elapsed_s": 150.0,
        "offered_qps": 7930.0,
        "completed_qps": round(ok_n / 150.0, 3),
        "send_qps": 6374.0,
        "actual_send_qps": 6373.2,
        "max_concurrency": 8,
        "send_mode": "replay",
        "send_mode_qps": "",
        "sla_ttft_ms": 500.0,
        "sla_violations": max(1, int(ok_n * 0.3)),
        "sla_violation_rate": 0.3,
        "test_valid": True,
        "validity_checks": {
            "mock_healthy": True,
            "no_oom": True,
            "all_results_recorded": True,
        },
        "client_pacing_lag_ms": {
            "count": len(rows),
            "p50": 0.4,
            "p90": 1.2,
            "p95": 2.0,
            "p99": 8.5,
            "max": 40.0,
            "mean": 0.9,
        },
        "schedule_latency_source": "client",
        "schedule_latency_ms": {
            "count": len(rows),
            "p50": 0.19,
            "p90": 1.0,
            "p95": 3.0,
            "p99": 156.0,
            "max": 2780.0,
            "mean": 12.4,
        },
        "ttft_ms": {
            "count": ok_n,
            "p50": 420.0,
            "p90": 780.0,
            "p95": 900.0,
            "p99": 1400.0,
            "max": 9000.0,
            "mean": 510.0,
        },
        "status_counts": {
            "ok": ok_n,
            "exception": totals.get("status_counts", {}).get("exception", 0),
            "schedule_error": totals.get("status_counts", {}).get("schedule_error", 0),
        },
        "route_path_counts": {"master": len(rows)},
        "priority_stats": {"30": {"send": 1}, "50": {"send": 1}, "70": {"send": 1}},
        "prefill_balance": {
            "counts": {
                "127.0.0.1:62500": 1,
                "127.0.0.1:62501": 1,
                "127.0.0.1:62502": 1,
                "127.0.0.1:62503": 1,
            },
            "stddev": 0.0,
            "max_over_avg": 1.0,
        },
        "decode_balance": {
            "counts": {
                "127.0.0.1:62510": 4,
                "127.0.0.1:62511": 3,
                "127.0.0.1:62512": 2,
                "127.0.0.1:62513": 1,
            },
            "stddev": 1.1,
            "max_over_avg": 1.6,
        },
    }


_SERVER_LATENCY = {
    "arrival_count": 100,
    "arrival_qps": 6373.0,
    "completion_count": 100,
    "completion_qps": 6100.0,
    "server_total_ms": {
        "count": 100,
        "p50": 640,
        "p90": 2994,
        "p95": 6914,
        "p99": 10258,
        "max": 11206,
        "mean": 1516.54,
    },
    "grpc_queue_ms": {
        "count": 100,
        "p50": 0,
        "p90": 5,
        "p95": 14,
        "p99": 29,
        "max": 29,
        "mean": 1.79,
    },
    "route_submit_ms": {
        "count": 100,
        "p50": 0,
        "p90": 0,
        "p95": 9,
        "p99": 9,
        "max": 9,
        "mean": 0.9,
    },
    "batch_wait_ms": {
        "count": 100,
        "p50": 638,
        "p90": 2992,
        "p95": 6912,
        "p99": 10257,
        "max": 11186,
        "mean": 1509.38,
    },
    "dispatch_ack_ms": {
        "count": 100,
        "p50": 0,
        "p90": 2,
        "p95": 23,
        "p99": 26,
        "max": 26,
        "mean": 2.52,
    },
    "ack_response_ms": {
        "count": 100,
        "p50": 0,
        "p90": 0,
        "p95": 6,
        "p99": 6,
        "max": 20,
        "mean": 0.68,
    },
    "batch_wait_ms_by_priority": {
        "30": {
            "count": 22,
            "p50": 2312,
            "p90": 8214,
            "p95": 10257,
            "p99": 11186,
            "max": 11186,
            "mean": 4013.455,
        },
        "50": {
            "count": 59,
            "p50": 658,
            "p90": 2561,
            "p95": 2970,
            "p99": 2992,
            "max": 2992,
            "mean": 996.305,
        },
        "70": {
            "count": 19,
            "p50": 271,
            "p90": 337,
            "p95": 342,
            "p99": 342,
            "max": 342,
            "mean": 203.158,
        },
    },
}

_PROM_LINES = [
    "# HELP flexlb_app_engine_balancing_master_dispatch_reason_total dispatch reasons",
    'flexlb_app_engine_balancing_master_dispatch_reason_total{reason="fixed_window_timeout"} 38',
    'flexlb_app_engine_balancing_master_dispatch_reason_total{reason="batch_full"} 14',
    'flexlb_app_engine_balancing_master_dispatch_reason_total{reason="predicted_execution_cap"} 8',
    'jvm_gc_pause_seconds_count{action="end of minor GC",cause="G1 Evacuation Pause"} 41.0',
    'jvm_gc_pause_seconds_max{action="end of minor GC",cause="G1 Evacuation Pause"} 0.028',
    'jvm_gc_pause_seconds_sum{action="end of minor GC",cause="G1 Evacuation Pause"} 0.412',
    "flexlb_app_flexlb_inflight_max_age_ms 0.0",
    # M10: cache metrics so kv_usage / kv_match have a final-snapshot view.
    "flexlb_app_cache_kv_cache_tokens 8192000",
    "flexlb_app_cache_kv_cache_available_tokens 2048000",
    'flexlb_app_cache_hit_ratio{engine="master"} 0.83',
    'flexlb_app_cache_theory_hit_ratio{engine="master"} 0.91',
    "flexlb_app_cache_recent_key_matched_total 5432",
]


def _per_engine_flat_timeline(
    start_epoch: int, n_samples: int = 8
) -> List[Dict[str, Any]]:
    """M10: minimal G1 flat per-engine timeline ([{ts, metrics}] with full
    Prometheus sample keys). The last decode-0 sample saturates the KV pool
    (available=0, active>0) — the M4 full-load ratio=1.0 assertion point."""
    timeline: List[Dict[str, Any]] = []
    for i in range(n_samples):
        active = 1024.0 * (i + 1)
        available = max(0.0, 8192.0 - active)
        timeline.append(
            {
                "ts": start_epoch + i * 5000,
                "metrics": {
                    'mock_engine_running{engine_name="decode-0",role="decode",grpc_port="62510"}': float(
                        2 + i % 3
                    ),
                    'mock_engine_waiting{engine_name="decode-0",role="decode",grpc_port="62510"}': float(
                        i % 2
                    ),
                    'mock_engine_active_kv_tokens{engine_name="decode-0",role="decode",grpc_port="62510"}': active,
                    'mock_engine_available_kv_tokens{engine_name="decode-0",role="decode",grpc_port="62510"}': available,
                    'mock_engine_running{engine_name="prefill-0",role="prefill",grpc_port="62500"}': float(
                        1 + i % 2
                    ),
                    'mock_engine_waiting{engine_name="prefill-0",role="prefill",grpc_port="62500"}': float(
                        (i + 1) % 3
                    ),
                    'mock_engine_active_kv_tokens{engine_name="prefill-0",role="prefill",grpc_port="62500"}': 512.0,
                    'mock_engine_available_kv_tokens{engine_name="prefill-0",role="prefill",grpc_port="62500"}': 3584.0,
                },
            }
        )
    return timeline


def _master_prom_ts_timeline(
    start_epoch: int, n_samples: int = 12
) -> List[Dict[str, Any]]:
    """M10: minimal G3 master prometheus timeline — batcher queue gauges with
    priority label variants (they must SUM into one curve), the inflight max
    age gauge and cache hit ratios (they must AVERAGE)."""
    timeline: List[Dict[str, Any]] = []
    for i in range(n_samples):
        timeline.append(
            {
                "ts": start_epoch + i * 1000,
                "metrics": {
                    'flexlb_app_flexlb_batcher_queue_size{priority="30"}': float(
                        (2 * i) % 9
                    ),
                    'flexlb_app_flexlb_batcher_queue_size{priority="50"}': float(
                        (3 * i) % 7
                    ),
                    'flexlb_app_routing_queue_length{priority="30"}': float(i % 5),
                    'flexlb_app_routing_queue_length{priority="50"}': float(i % 3),
                    "flexlb_app_flexlb_inflight_max_age_ms": float(120 + i * 37),
                    'flexlb_app_cache_hit_ratio{engine="master"}': round(
                        0.72 + i * 0.01, 4
                    ),
                    'flexlb_app_cache_theory_hit_ratio{engine="master"}': round(
                        0.85 + i * 0.008, 4
                    ),
                },
            }
        )
    return timeline


def _inflight_ts_rows(start_epoch: int, n_samples: int = 12) -> List[Dict[str, Any]]:
    """M10: minimal G4 /rtp_llm/inflight_status snapshots. The tail (last 3
    samples) drains every gauge to zero so the G4 leak verdict is clean."""
    rows: List[Dict[str, Any]] = []
    for i in range(n_samples):
        live = i < n_samples - 3
        rows.append(
            {
                "ts_epoch_ms": start_epoch + i * 1000,
                "inflight": {
                    "scheduler_inflight": float(4 + i) if live else 0.0,
                    "prefill_endpoints": [
                        {
                            "ip_port": "127.0.0.1:62500",
                            "inflight_batches": 1 if live else 0,
                            "inflight_requests": 2 if live else 0,
                            "inflight_route_requests": 1 if live else 0,
                        }
                    ],
                    "decode_endpoints": [
                        {
                            "ip_port": f"127.0.0.1:6251{p}",
                            "inflight_requests": (3 - p) if live else 0,
                        }
                        for p in range(2)
                    ],
                },
            }
        )
    return rows


def _process_usage_rows(start_epoch: int, n_samples: int = 6) -> List[Dict[str, Any]]:
    """M10: G5 process-usage samples grouped by label (mock / master /
    client_0) — exercises the m5 label grouping in analyze_cpu_mem."""
    rows: List[Dict[str, Any]] = []
    specs = (
        ("mock", 4242, 318.5, 524288),
        ("master", 4243, 96.2, 262144),
        ("client_0", 4244, 55.0, 131072),
    )
    for i in range(n_samples):
        for label, pid, cpu, rss in specs:
            rows.append(
                {
                    "ts_epoch_ms": start_epoch + i * 5000,
                    "label": label,
                    "pid": pid + i,
                    "cpu_pct": round(cpu + i * 2.5, 1),
                    "rss_kb": rss + i * 8192,
                    "etime": "01:23:45",
                }
            )
    return rows


def _grouped_prom_text(timeline: Sequence[Dict[str, Any]]) -> str:
    """Grouped prom text (``# ts=<epoch_ms>`` separators) — the on-disk format
    of the G1/G3 pollers (legacy, pre-consolidation layout)."""
    lines: List[str] = []
    for group in timeline:
        lines.append(f"# ts={group.get('ts')}")
        for key, value in (group.get("metrics") or {}).items():
            lines.append(f"{key} {value}")
    return "\n".join(lines) + "\n"


def _synthesize_agg_fixture(
    duration_s: int, send_per_s: int, err_share: float, seed: int
) -> Dict[str, Any]:
    """M10: a minimal, fully consistent aggregation JSON (the shape
    build_synthetic_run consumes) so the self-test runs even without the
    /tmp/flexlb_eval_ts fixtures captured from a live run."""
    rng = random.Random(seed)
    per_second: List[Dict[str, Any]] = []
    mock_stats: List[Dict[str, Any]] = []
    master: List[Dict[str, Any]] = []
    total = ok_total = 0
    for t in range(duration_s):
        send = send_per_s + rng.randint(-5, 5)
        ok_n = max(0, round(send * (1 - err_share)))
        per_second.append(
            {
                "t": t,
                "send": send,
                "ok": ok_n,
                "sched_p50": 0.9,
                "sched_p95": 3.2,
                "sched_p99": 12.5,
            }
        )
        total += send
        ok_total += ok_n
        if t % 5 == 0:
            mock_stats.append(
                {
                    "t": t,
                    "pr": 6,
                    "dr": 12,
                    "dw": 2,
                    "pw": 1,
                    "pr_reqs": 40,
                    "heap_mb": 2048 + t,
                }
            )
            master.append(
                {"t": t, "arrival_total": total, "completion_total": ok_total}
            )
    exc_n = max(1, int((total - ok_total) * 0.004))
    return {
        "start_epoch_ms": 1787623000000 + seed * 1000000,
        "per_second": per_second,
        "mock_stats": mock_stats,
        "master": master,
        "totals": {
            "requests": total,
            "ok": ok_total,
            "err": total - ok_total,
            "status_counts": {
                "exception": exc_n,
                "schedule_error": total - ok_total - exc_n,
            },
        },
    }


def build_synthetic_run(
    agg_path: Path, run_dir: Path, layout: str, rows_limit: Optional[int] = None
) -> Dict[str, Any]:
    """Materialize a minimal legal run dir from an aggregation JSON."""
    agg = json.loads(agg_path.read_text(encoding="utf-8"))
    run_dir.mkdir(parents=True, exist_ok=True)
    start_epoch = agg.get("start_epoch_ms", 1787623000000)
    rows = synthesize_per_request(agg)
    if rows_limit:
        rows = rows[:rows_limit]
    summary = _summary_json(agg, rows)
    mock_lines = [
        _mock_stats_line(start_epoch, row["t"], row)
        for row in agg.get("mock_stats", [])
    ]
    counter_lines = [
        f"ts_epoch_ms={start_epoch + row['t'] * 1000} "
        f"arrival_count={row['arrival_total']} completion_count={row['completion_total']}"
        for row in agg.get("master", [])
    ]
    meta = {
        "generated_at_utc": "2026-08-25T10:30:00+00:00",
        "params": {
            "n_prefill": "4",
            "n_decode": "4",
            "fetch_output_stream": "1",
            "send_mode": "replay",
            "send_mode_qps": "",
            "duration_s": "150",
            "sla_ttft_ms": "500",
            "max_concurrency": "8",
            "java_mock_stats_interval_ms": "5000",
            "java_mock_decode_max_concurrency": "64",
            "performance_file": "dsv4_flash_performance.sample.json",
            "flexlb_config": "{}",
        },
        "flexlb_env": {"FLEXLB_MONITOR_MODE": "basic"},
        "endpoints": {"engine_count": 8},
    }
    # M1: give the synthetic replay run a send_mode_qps target matching the
    # synthesized send window, so the pacing ratio is ~1.0 and the healthy
    # verdict is assertable (real params store the user's SEND_MODE_QPS).
    send_starts = [
        float(r["send_start_epoch_ms"]) for r in rows if r.get("send_start_epoch_ms")
    ]
    if send_starts:
        span_s = (max(send_starts) - min(send_starts)) / 1000.0
        if span_s > 0:
            meta["params"]["send_mode_qps"] = round(len(send_starts) / span_s, 1)
    if layout == "legacy":
        lc = run_dir / "load_client"
        lc.mkdir(parents=True, exist_ok=True)
        half = len(rows) // 2
        for name, part in (("shard_0", rows[:half]), ("shard_1", rows[half:])):
            shard = lc / name
            shard.mkdir(exist_ok=True)
            with (shard / "per_request.jsonl").open("w") as fh:
                for row in part:
                    fh.write(json.dumps(row) + "\n")
        (lc / "summary.json").write_text(json.dumps(summary, indent=1))
        (lc / "server_latency.json").write_text(json.dumps(_SERVER_LATENCY, indent=1))
        (run_dir / "mock_engine.log").write_text("\n".join(mock_lines) + "\n")
        (run_dir / "master_counters_timeseries.txt").write_text(
            "\n".join(counter_lines) + "\n"
        )
        (run_dir / "master_prometheus_after.prom").write_text(
            "\n".join(_PROM_LINES) + "\n"
        )
        # M10: raw collector outputs — the pre-consolidation resolver chain
        # (m3 .prom fallback, G3 prom text, G4 jsonl, G5 kv lines).
        (run_dir / "mock_metrics_per_engine.prom").write_text(
            _grouped_prom_text(_per_engine_flat_timeline(start_epoch))
        )
        (run_dir / "master_prometheus_timeseries.prom").write_text(
            _grouped_prom_text(_master_prom_ts_timeline(start_epoch))
        )
        (run_dir / "master_inflight_timeseries.jsonl").write_text(
            "\n".join(json.dumps(r) for r in _inflight_ts_rows(start_epoch)) + "\n"
        )
        (run_dir / "process_usage_timeseries.txt").write_text(
            "\n".join(
                f"ts_epoch_ms={r['ts_epoch_ms']} label={r['label']} pid={r['pid']} "
                f"cpu_pct={r['cpu_pct']} rss_kb={r['rss_kb']} etime={r['etime']}"
                for r in _process_usage_rows(start_epoch)
            )
            + "\n"
        )
        log_dir = run_dir / "flexlb_logs"
        log_dir.mkdir(exist_ok=True)
        (log_dir / "flexlb.log").write_text(_master_log_lines(start_epoch))
        (run_dir / "run_meta.json").write_text(json.dumps(meta, indent=1))
    else:  # consolidated
        per_engine_timeline = _per_engine_flat_timeline(start_epoch)
        mock_json = {
            "stats_sample_count": len(mock_lines),
            "stats": [
                {
                    k: (float(v) if "." in str(v) else int(v))
                    for k, v in STAT_KV_RE.findall(line)
                }
                for line in mock_lines
            ],
            # A-split shape: per-engine samples live in the split gzip file,
            # mock.json keeps only the pointer (mirrors consolidate_run_outputs).
            "per_engine_file": "mock_per_engine_timeseries.json.gz",
            "per_engine_sample_count": len(per_engine_timeline),
            "final_snapshot": {
                "ts_epoch_ms": start_epoch + 150000,
                "engines": [
                    {
                        "name": f"decode-{i}",
                        "role": "decode",
                        "grpc_addr": f"127.0.0.1:6251{i}",
                        "accepted": [30, 26, 20, 14][i],
                        "inflight": 0,
                    }
                    for i in range(4)
                ]
                + [
                    {
                        "name": f"prefill-{i}",
                        "role": "prefill",
                        "grpc_addr": f"127.0.0.1:6250{i}",
                        "accepted": 25,
                        "inflight": 0,
                    }
                    for i in range(4)
                ],
            },
        }
        (run_dir / "mock.json").write_text(json.dumps(mock_json))
        with gzip.open(run_dir / "mock_per_engine_timeseries.json.gz", "wt") as fh:
            json.dump(per_engine_timeline, fh)
        master_json = {
            "counters_timeseries": [
                {
                    k: (float(v) if "." in v else int(v))
                    for k, v in STAT_KV_RE.findall(line)
                }
                for line in counter_lines
            ],
            "prometheus_after": _prom_dict(_PROM_LINES),
            # M10: G3/G4 timelines in their consolidated in-json forms.
            "prometheus_timeseries": _master_prom_ts_timeline(start_epoch),
            "inflight_timeseries": _inflight_ts_rows(start_epoch),
            "master_info_before": {},
            "master_info_after": {},
            "slo_batch_summary": {},
        }
        (run_dir / "master.json").write_text(json.dumps(master_json))
        client_json = dict(summary)
        client_json["server_latency"] = _SERVER_LATENCY
        (run_dir / "client.json").write_text(json.dumps(client_json))
        (run_dir / "master.log").write_text(_master_log_lines(start_epoch))
        with gzip.open(run_dir / "per_request.jsonl.gz", "wt") as fh:
            for row in rows:
                fh.write(json.dumps(row) + "\n")
        # G5 merged key (consolidation moves the txt into run_meta.json).
        meta["process_usage"] = _process_usage_rows(start_epoch)
        (run_dir / "run_meta.json").write_text(json.dumps(meta, indent=1))
    return {"rows": len(rows), "agg_totals": agg.get("totals", {})}


def _prom_dict(lines: List[str]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for line in lines:
        m = PROM_SAMPLE_RE.match(line.strip())
        if m:
            out[m.group("name") + (m.group("labels") or "")] = float(m.group("value"))
    return out


def run_self_test(args: argparse.Namespace) -> int:
    import shutil

    root = Path(args.self_test_root)
    if root.exists():
        shutil.rmtree(root)
    agg1 = Path("/tmp/flexlb_eval_ts/agg_20260825_095837.json")
    agg2 = Path("/tmp/flexlb_eval_ts/agg_20260825_100910.json")
    if not agg1.is_file() or not agg2.is_file():
        # M10: the self-test must not depend on external fixtures —
        # synthesize equivalent aggregation JSONs locally.
        fixture_dir = root / "fixtures"
        fixture_dir.mkdir(parents=True, exist_ok=True)
        agg1 = fixture_dir / "agg_synth_a.json"
        agg2 = fixture_dir / "agg_synth_b.json"
        # run A carries enough errors (~3k) for the deterministic error-code
        # round-robin (idx%1000) to cover every class, including the rare
        # yielded/no-worker buckets.
        agg1.write_text(
            json.dumps(_synthesize_agg_fixture(100, 120, 0.25, seed=1)),
            encoding="utf-8",
        )
        agg2.write_text(
            json.dumps(_synthesize_agg_fixture(60, 80, 0.10, seed=2)), encoding="utf-8"
        )
        print(
            "[self-test] /tmp/flexlb_eval_ts fixtures not found -> "
            "synthesized local aggregation fixtures"
        )
    scale = args.self_test_scale
    print(
        f"[self-test] synthesizing run A (legacy layout) from {agg1.name} (scale={scale})"
    )
    info_a = build_synthetic_run(
        agg1, root / "run_a_legacy", "legacy", rows_limit=scale or None
    )
    print(f"[self-test]   rows={info_a['rows']}")
    print(f"[self-test] synthesizing run B (consolidated layout) from {agg2.name}")
    info_b = build_synthetic_run(
        agg2, root / "run_b_consolidated", "consolidated", rows_limit=scale or None
    )
    print(f"[self-test]   rows={info_b['rows']}")

    print("[self-test] analyzing run A (+ compare B)")
    report = build_report(root / "run_a_legacy")
    other = build_report(root / "run_b_consolidated")
    report["comparison"] = build_comparison(report, [other])

    out_dir = root / "run_a_legacy" / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "analysis_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    html_text = render_html(report)
    (out_dir / "analysis_report.html").write_text(html_text, encoding="utf-8")

    # ---- verification ----
    failures: List[str] = []
    ok = lambda label: print(f"  [ok] {label}")

    rep = json.loads((out_dir / "analysis_report.json").read_text())
    ok("analysis_report.json generated and json.load passes")

    for field in (
        "test_valid",
        "leak_verdict",
        "balance_grade",
        "error_rate",
        "pacing_verdict",
        "fetch_output_stream",
    ):
        if field not in rep["verdict"]:
            failures.append(f"verdict.{field} missing")
    ok("verdict fields complete" if not failures else "verdict fields PARTIAL")

    missing = [d for d in DIMENSIONS if d not in rep]
    if missing:
        failures.append(f"missing dimension sections: {missing}")
    ok(
        f"all {len(DIMENSIONS)} dimension sections present"
        if not missing
        else "dimensions MISSING"
    )

    n_svg = html_text.count("<svg")
    if n_svg < 15:
        failures.append(f"only {n_svg} <svg> charts in HTML (>=15 required)")
    ok(f"HTML charts (<svg count)={n_svg} (>=15 required)")

    q = rep["qps_and_errors"]
    send_sum = sum(r["send"] for r in q["per_second"])
    expect_send = info_a["agg_totals"].get("requests")
    if scale:
        expect_send = min(scale, expect_send)
        ok(f"per-second send sum={send_sum} (scaled run, expect <= {expect_send})")
    elif send_sum != expect_send:
        failures.append(f"per-second send sum {send_sum} != expected {expect_send}")
    else:
        ok(f"per-second send sum={send_sum} == expected {expect_send}")

    classes = {row["class"]: row["count"] for row in q["error_classes"]}
    err_total = sum(classes.values()) or 1
    print("  error-class hits (synthesized -> detected):")
    expected_hits = {
        "backpressure": 0.5,
        "preempted_8429": 1,
        "resource_8431": 1,
        "client_timeout": 1,
        "queue_full_8502": 1,
        "no_worker_8400_8407": 1,
        "slo_expired_8511": 1,
        "yielded_8400": 1,
        "network_close": 1,
        "engine_exec": 1,
        "dispatch_failed": 1,
    }
    for name, min_share in expected_hits.items():
        count = classes.get(name, 0)
        hit = count > 0 and (count / err_total >= (min_share if min_share < 1 else 0))
        print(
            f"    {name:22s} count={count:>8,} share={count / err_total:6.2%} {'HIT' if hit else 'MISS'}"
        )
        if not hit:
            failures.append(f"error class {name} not detected")
    bp_share = classes.get("backpressure", 0) / err_total
    if bp_share < 0.5:
        failures.append(f"backpressure share {bp_share:.2%} not dominant")
    else:
        ok(f"backpressure dominant: {bp_share:.2%} of errors")

    if not rep.get("comparison", {}).get("runs"):
        failures.append("comparison section missing")
    else:
        n_delta = len(rep["comparison"]["runs"][0]["delta_table"])
        ok(f"comparison delta table present ({n_delta} metrics vs run B)")

    # spot check: dispatch invariants were synthesized and must be detected
    inv = rep["dispatch"].get("invariant_violation_count", 0)
    if inv < 1:
        failures.append("dispatch invariant violations not detected")
    else:
        ok(f"dispatch invariant violations detected: {inv}")

    # ---- M10: the synthetic runs now carry per-engine / batcher / G4
    # inflight / process-usage / cache samples — each enhancement must light
    # up in the report.
    qs = rep["queues"]
    if qs.get("per_engine_mode") != "per_engine_timeseries":
        failures.append(
            f"queues.per_engine_mode={qs.get('per_engine_mode')!r}, "
            "expected per_engine_timeseries"
        )
    else:
        ok("queues.per_engine_mode == per_engine_timeseries (G1 samples parsed)")
    pe_rows = (qs.get("per_engine") or {}).get("decode-0") or [{}]
    if pe_rows and "decode_waiting" in pe_rows[0] and "role" in pe_rows[0]:
        ok("per-engine rows carry decode_waiting/prefill_waiting + role (M6)")
    else:
        failures.append(
            "per-engine rows missing decode_waiting/prefill_waiting/role fields"
        )
    batcher = qs.get("batcher") or {}
    if batcher.get("available"):
        ok(
            f"batcher queue curves available ({len(batcher.get('series') or {})} metrics, B3)"
        )
    else:
        failures.append("queues.batcher not available — G3 timeline not parsed")

    kvu = rep["kv_usage"]
    if not kvu.get("available"):
        failures.append("kv_usage not available")
    else:
        ratios = [
            r["kv_ratio"]
            for r in (kvu.get("per_engine") or {}).get("decode-0", [])
            if r.get("kv_ratio") is not None
        ]
        if 1.0 in ratios:
            ok("kv_usage per-engine curve has the full-load ratio=1.0 point (M4)")
        else:
            failures.append(f"no full-load kv ratio 1.0 point (ratios={ratios})")
    kvm = rep["kv_match"]
    if not kvm.get("available"):
        failures.append("kv_match not available")
    elif "flexlb_app_cache_hit_ratio" not in (kvm.get("per_second") or {}):
        failures.append("kv_match per-second hit_ratio curve missing (S7)")
    else:
        ok("kv_match per-second hit_ratio curve present (S7)")

    inf = rep["inflight"]
    if not inf.get("inflight_status"):
        failures.append("inflight.inflight_status missing — G4 samples not parsed")
    else:
        ok("inflight primary signal = G4 inflight_timeseries (M3)")
    if rep["verdict"].get("leak_verdict") != "clean":
        failures.append(
            f"leak_verdict={rep['verdict'].get('leak_verdict')!r}, expected clean"
        )
    else:
        ok("leak_verdict == clean (G4 tail drains to zero)")

    pu = (rep["cpu_mem"] or {}).get("process_usage") or {}
    if "mock" not in pu:
        failures.append(
            f"cpu_mem.process_usage missing 'mock' label (got {sorted(pu)})"
        )
    else:
        ok(f"cpu_mem.process_usage grouped by label (labels: {sorted(pu)})")

    pacing = rep["pacing"]
    if rep["verdict"].get("pacing_verdict") != "good":
        failures.append(
            f"pacing_verdict={rep['verdict'].get('pacing_verdict')!r}, expected good"
        )
    else:
        ok(
            f"pacing_verdict == good "
            f"(ratio={pacing.get('send_vs_target_ratio')}, "
            f"target from {pacing.get('target_source')})"
        )
    if pacing.get("send_vs_target_ratio") is None:
        failures.append("pacing send_vs_target_ratio missing (M1 target resolution)")
    elif pacing["send_vs_target_ratio"] < 0.98:
        failures.append(
            f"pacing ratio {pacing['send_vs_target_ratio']} < 0.98 on healthy synthetic data"
        )

    # B1: fallback rows exist with schedule_ms=0 and stay OUT of the primary
    # schedule statistics. Completeness identity: primary + fallback == all
    # ok rows — anything else means fallback rows leaked in or were dropped.
    fb = rep["fallback_path"]
    ok_total = rep["qps_and_errors"]["totals"]["ok"]
    sched_count = rep["latency_layers"]["schedule_overall"]["count"]
    if fb["ok"]:
        if sched_count + fb["ok"] != ok_total:
            failures.append(
                f"B1 completeness violated: sched({sched_count}) + "
                f"fallback_ok({fb['ok']}) != ok({ok_total})"
            )
        else:
            ok(
                f"B1 split exact: sched={sched_count} + fallback_ok={fb['ok']} "
                f"== ok={ok_total}"
            )
        sched_p50 = rep["latency_layers"]["schedule_overall"]["p50"]
        if sched_p50 <= 0:
            failures.append(
                "primary schedule p50 <= 0 — fallback zeros leaked into the main track"
            )
        elif fb["sched_ms"]["p50"] != 0:
            failures.append(
                "fallback track p50 != 0 — unexpected fallback schedule values"
            )
        else:
            ok(f"fallback track isolated (sched p50=0); primary p50={sched_p50}")
    elif not scale:
        failures.append("no fallback rows synthesized (expected ~2% of ok rows)")

    # A-split: run B (consolidated) must read per-engine data from the split
    # mock_per_engine_timeseries.json.gz file, not from mock.json.
    if (other.get("queues") or {}).get("per_engine_mode") != "per_engine_timeseries":
        failures.append(
            "run B per_engine_mode != per_engine_timeseries (A-split .gz not read)"
        )
    else:
        ok("run B per-engine via mock_per_engine_timeseries.json.gz (A-split)")

    print()
    if failures:
        print("[self-test] FAILED:")
        for f in failures:
            print(f"  - {f}")
        return 1
    print(f"[self-test] PASSED — report: {out_dir / 'analysis_report.json'}")
    print(f"[self-test]         html:  {out_dir / 'analysis_report.html'}")
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Unified online_eval run analyzer (see module docstring for the dimension map)"
    )
    parser.add_argument(
        "run_dir",
        nargs="?",
        type=Path,
        help="run directory (legacy or consolidated layout)",
    )
    parser.add_argument(
        "--compare",
        action="append",
        type=Path,
        default=[],
        metavar="RUN_DIR",
        help="additional run dirs to compare (repeatable)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="output directory (default <run_dir>/analysis)",
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="synthesize test runs (local fixtures when "
        "/tmp/flexlb_eval_ts is absent) and verify",
    )
    parser.add_argument(
        "--self-test-root", type=Path, default=Path("/tmp/analyzer_test_run")
    )
    parser.add_argument(
        "--self-test-scale",
        type=int,
        default=None,
        help="cap synthesized per_request rows (speeds up smoke runs)",
    )
    parser.add_argument("--no-html", action="store_true")
    args = parser.parse_args(argv)

    if args.self_test:
        return run_self_test(args)
    if not args.run_dir:
        parser.error("run_dir is required (or use --self-test)")
    run_dir: Path = args.run_dir
    if not run_dir.is_dir():
        print(f"error: run dir not found: {run_dir}", file=sys.stderr)
        return 2

    report = build_report(run_dir)
    if args.compare:
        others = []
        for other_dir in args.compare:
            if not other_dir.is_dir():
                print(
                    f"warning: compare dir not found, skipped: {other_dir}",
                    file=sys.stderr,
                )
                continue
            others.append(build_report(other_dir))
        if others:
            report["comparison"] = build_comparison(report, others)

    out_dir = args.output_dir or (run_dir / "analysis")
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "analysis_report.json"
    json_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"report: {json_path}")
    if not args.no_html:
        html_path = out_dir / "analysis_report.html"
        html_path.write_text(render_html(report), encoding="utf-8")
        print(f"html:   {html_path}")
    v = report["verdict"]
    print(
        f"verdict: test_valid={v.get('test_valid')} leak={v.get('leak_verdict')} "
        f"balance={v.get('balance_grade')} err_rate={v.get('error_rate')} "
        f"pacing={v.get('pacing_verdict')} fetch_output_stream={v.get('fetch_output_stream')}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
