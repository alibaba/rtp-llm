#!/usr/bin/env python3
"""Analyze FlexLB burst-traffic test results and generate a Markdown report.

This script analyzes results from multi-speed replay of the real production
trace ``trace_30min.jsonl`` (8332 requests, 13.3 min, avg 10.5 QPS, 13 natural
burst segments). For each replay speed level, it:

  - Parses ``summary.json`` for TTFT percentiles, SLA violations, status counts
  - Parses ``per_request.jsonl`` for per-second QPS/TTFT time series
  - Auto-detects natural burst segments (QPS > 1.5x avg, lasting >= 2s)
  - Parses ``monitor.jsonl`` for inflight/JVM time series
  - Parses ``flexlb.log`` for dispatch events, queue depth, errors

Produces a comprehensive Markdown report comparing all speed levels, plus
per-speed ``analysis.json`` files with structured analysis data.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class SpeedRunData:
    """All loaded data for a single speed-level run."""

    speed: int
    speed_label: str
    run_dir: Path
    summary: Optional[Dict[str, Any]] = None
    per_request: List[Dict[str, Any]] = field(default_factory=list)
    monitor: List[Dict[str, Any]] = field(default_factory=list)
    log_events: Dict[str, Any] = field(default_factory=dict)
    log_found: bool = False
    analysis: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Trace characteristics (real trace)
# ---------------------------------------------------------------------------

TRACE_AVG_QPS = 10.5  # average QPS of trace_30min.jsonl
TRACE_PEAK_QPS = 28  # peak QPS in the trace
TRACE_REQUESTS = 8332
TRACE_DURATION_MIN = 13.3


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dirs",
        nargs="+",
        required=True,
        help="run directories (space-separated, in order)",
    )
    parser.add_argument(
        "--speeds",
        nargs="+",
        type=int,
        required=True,
        help="replay speed levels (e.g. 5 10 15 20 30 50)",
    )
    parser.add_argument(
        "--sla-ttft-ms",
        type=float,
        default=500.0,
        help="SLA TTFT threshold in ms",
    )
    parser.add_argument(
        "--output",
        help="output report path (default: stdout)",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Speed extraction from directory name
# ---------------------------------------------------------------------------


def extract_speed(run_dir: Path) -> int:
    """Extract speed integer from directory name like 'burst_10x_20260101_120000'."""
    name = run_dir.name
    m = re.search(r"burst_(\d+)x", name)
    if m:
        return int(m.group(1))
    # Fallback: try to find any number followed by 'x'
    m = re.search(r"(\d+)x", name)
    if m:
        return int(m.group(1))
    return 0


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_json_safe(path: Path) -> Optional[Dict[str, Any]]:
    """Safely load a JSON file, returning None on error."""
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError, ValueError):
        return None


def load_jsonl_safe(path: Path) -> List[Dict[str, Any]]:
    """Safely load a JSONL file, skipping bad lines."""
    records: List[Dict[str, Any]] = []
    if not path.exists():
        return records
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if isinstance(obj, dict):
                    records.append(obj)
            except (json.JSONDecodeError, ValueError):
                continue
    return records


def load_run_data(run_dir: Path, speed: int) -> SpeedRunData:
    """Load all data sources for a single run directory."""
    speed_label = f"{speed}x"
    data = SpeedRunData(speed=speed, speed_label=speed_label, run_dir=run_dir)

    data.summary = load_json_safe(run_dir / "load_client" / "summary.json")
    data.per_request = load_jsonl_safe(run_dir / "load_client" / "per_request.jsonl")
    data.monitor = load_jsonl_safe(run_dir / "monitor.jsonl")
    data.log_events, data.log_found = parse_flexlb_log(run_dir / "flexlb.log")
    data.analysis = analyze_run(data)
    return data


# ---------------------------------------------------------------------------
# flexlb.log parsing
# ---------------------------------------------------------------------------

_DISPATCH_RE = re.compile(
    r"dispatch.*?batch[_ ]?size[=: ]+(\d+)" r"|batch.*?size[=: ]+(\d+).*?dispatch",
    re.IGNORECASE,
)
_DISPATCH_REASON_RE = re.compile(
    r"(target_batch_size|deadline_guard|window_timeout|batch_ready|"
    r"max_inflight|predict_threshold|queue_overflow|budget_overrun)",
    re.IGNORECASE,
)
_QUEUE_DEPTH_RE = re.compile(
    r"queue[_ ]?depth[=: ]+(\d+)|queue.*?size[=: ]+(\d+)", re.IGNORECASE
)
_BUDGET_OVERRUN_RE = re.compile(r"budget.*?overrun|overrun.*?budget", re.IGNORECASE)
_DROP_RE = re.compile(r"drop|reject|backpressure|rate.?limit", re.IGNORECASE)
_ERROR_RE = re.compile(r"ERROR|FATAL|Exception|OutOfMemoryError", re.IGNORECASE)
_OOM_RE = re.compile(r"OutOfMemoryError", re.IGNORECASE)
_INFO_LINE_RE = re.compile(r"\bINFO\b\s", re.IGNORECASE)


def parse_flexlb_log(path: Path) -> Tuple[Dict[str, Any], bool]:
    """Parse flexlb.log for dispatch events, queue depth, errors."""
    events: Dict[str, Any] = {
        "dispatch_batch_sizes": [],
        "dispatch_reasons": {},
        "queue_depth_values": [],
        "budget_overruns": 0,
        "drops": 0,
        "errors": 0,
        "oom": False,
        "log_lines": 0,
    }
    if not path.exists():
        return events, False
    log_found = True
    try:
        content = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return events, False

    for line in content.splitlines():
        if "JAVA_TOOL_OPTIONS" in line or "Picked up JAVA_TOOL_OPTIONS" in line:
            continue
        events["log_lines"] += 1
        is_info = bool(_INFO_LINE_RE.search(line))
        if _OOM_RE.search(line):
            events["oom"] = True
        if not is_info and _ERROR_RE.search(line):
            events["errors"] += 1
        m = _DISPATCH_RE.search(line)
        if m:
            size_str = m.group(1) or m.group(2)
            if size_str:
                try:
                    events["dispatch_batch_sizes"].append(int(size_str))
                except ValueError:
                    pass
        m = _DISPATCH_REASON_RE.search(line)
        if m:
            reason = m.group(1).lower()
            events["dispatch_reasons"][reason] = (
                events["dispatch_reasons"].get(reason, 0) + 1
            )
        m = _QUEUE_DEPTH_RE.search(line)
        if m:
            depth_str = m.group(1) or m.group(2)
            if depth_str:
                try:
                    events["queue_depth_values"].append(int(depth_str))
                except ValueError:
                    pass
        if _BUDGET_OVERRUN_RE.search(line):
            events["budget_overruns"] += 1
        if _DROP_RE.search(line):
            events["drops"] += 1
    return events, log_found


# ---------------------------------------------------------------------------
# Per-request analysis & natural burst detection
# ---------------------------------------------------------------------------


def analyze_per_request(
    per_request: List[Dict[str, Any]], replay_speed: int
) -> Dict[str, Any]:
    """Analyze per-request data for time-series TTFT and QPS.

    QPS is computed in trace-time (original ts) so the equivalent QPS at a given
    replay speed is ``trace_qps * replay_speed``.
    """
    if not per_request:
        return {
            "per_second": [],
            "trace_avg_qps": 0.0,
            "trace_peak_qps": 0,
            "equivalent_avg_qps": 0.0,
            "equivalent_peak_qps": 0,
            "burst_segments": [],
        }

    first_ts = min(r.get("ts", 0) for r in per_request if r.get("ts", 0) > 0)
    if first_ts == 0:
        first_ts = per_request[0].get("ts", 0)

    per_second_ttft: Dict[int, List[float]] = {}
    per_second_count: Dict[int, int] = {}

    for r in per_request:
        ts = r.get("ts", 0)
        ttft = r.get("ttft_ms", 0)
        if ts > 0:
            second_bucket = (ts - first_ts) // 1000
        else:
            second_bucket = 0
        if ttft > 0:
            per_second_ttft.setdefault(second_bucket, []).append(ttft)
        per_second_count[second_bucket] = per_second_count.get(second_bucket, 0) + 1

    max_second = max(per_second_count.keys()) if per_second_count else 0
    time_series: List[Dict[str, Any]] = []
    for sec in range(max_second + 1):
        ttfts = per_second_ttft.get(sec, [])
        time_series.append(
            {
                "second": sec,
                "trace_qps": per_second_count.get(sec, 0),
                "equivalent_qps": per_second_count.get(sec, 0) * replay_speed,
                "avg_ttft_ms": round(sum(ttfts) / len(ttfts), 1) if ttfts else 0.0,
                "max_ttft_ms": round(max(ttfts), 1) if ttfts else 0.0,
                "count": len(ttfts),
            }
        )

    # Compute trace-level average and peak QPS
    trace_qps_values = [ts["trace_qps"] for ts in time_series]
    trace_avg_qps = (
        sum(trace_qps_values) / len(trace_qps_values) if trace_qps_values else 0.0
    )
    trace_peak_qps = max(trace_qps_values) if trace_qps_values else 0

    # Detect natural burst segments
    burst_segments = detect_burst_segments(time_series, trace_avg_qps)

    return {
        "per_second": time_series,
        "first_ts": first_ts,
        "trace_avg_qps": round(trace_avg_qps, 2),
        "trace_peak_qps": trace_peak_qps,
        "equivalent_avg_qps": round(trace_avg_qps * replay_speed, 1),
        "equivalent_peak_qps": trace_peak_qps * replay_speed,
        "burst_segments": burst_segments,
    }


def detect_burst_segments(
    time_series: List[Dict[str, Any]], avg_qps: float
) -> List[Dict[str, Any]]:
    """Detect natural burst segments where QPS > 1.5x average for >= 2 seconds.

    Returns list of segments with:
      - start_s, end_s: trace-time second boundaries
      - peak_qps: max QPS in the segment
      - avg_qps: average QPS in the segment
      - duration_s: segment length
    """
    if not time_series or avg_qps <= 0:
        return []

    threshold = avg_qps * 1.5
    segments: List[Dict[str, Any]] = []
    in_burst = False
    burst_start = 0
    burst_qps_values: List[int] = []

    for ts in time_series:
        qps = ts["trace_qps"]
        if qps > threshold:
            if not in_burst:
                in_burst = True
                burst_start = ts["second"]
                burst_qps_values = []
            burst_qps_values.append(qps)
        else:
            if in_burst:
                duration = ts["second"] - burst_start
                if duration >= 2:
                    segments.append(
                        {
                            "start_s": burst_start,
                            "end_s": ts["second"],
                            "duration_s": duration,
                            "peak_qps": max(burst_qps_values),
                            "avg_qps": round(
                                sum(burst_qps_values) / len(burst_qps_values), 1
                            ),
                        }
                    )
                in_burst = False
                burst_qps_values = []

    # Handle burst at the end of the time series
    if in_burst and burst_qps_values:
        last_sec = time_series[-1]["second"]
        duration = last_sec - burst_start + 1
        if duration >= 2:
            segments.append(
                {
                    "start_s": burst_start,
                    "end_s": last_sec + 1,
                    "duration_s": duration,
                    "peak_qps": max(burst_qps_values),
                    "avg_qps": round(sum(burst_qps_values) / len(burst_qps_values), 1),
                }
            )

    return segments


def analyze_burst_response(
    time_series: List[Dict[str, Any]],
    burst_segments: List[Dict[str, Any]],
    avg_qps: float,
) -> Dict[str, Any]:
    """Analyze TTFT response to natural burst segments.

    For each detected burst segment, measure:
      - TTFT before burst (baseline)
      - TTFT during burst (peak, avg)
      - TTFT degradation ratio
      - Recovery time (TTFT returns to near baseline after burst ends)
    """
    if not time_series or not burst_segments:
        return {
            "segment_count": 0,
            "avg_degrade_ratio": 0.0,
            "avg_recovery_s": None,
            "segments": [],
        }

    # Compute global baseline (TTFT during non-burst periods)
    burst_seconds = set()
    for seg in burst_segments:
        for s in range(seg["start_s"], seg["end_s"]):
            burst_seconds.add(s)

    baseline_ttfts = [
        ts["avg_ttft_ms"]
        for ts in time_series
        if ts["second"] not in burst_seconds and ts["avg_ttft_ms"] > 0
    ]
    baseline = sum(baseline_ttfts) / len(baseline_ttfts) if baseline_ttfts else 0.0

    segment_analyses: List[Dict[str, Any]] = []
    degrade_ratios: List[float] = []
    recovery_times: List[int] = []

    for seg in burst_segments:
        start = seg["start_s"]
        end = seg["end_s"]

        # TTFT during burst
        during = [
            ts
            for ts in time_series
            if start <= ts["second"] < end and ts["avg_ttft_ms"] > 0
        ]
        burst_ttft_peak = max((ts["max_ttft_ms"] for ts in during), default=0.0)
        burst_ttft_avg = (
            sum(ts["avg_ttft_ms"] for ts in during) / len(during) if during else 0.0
        )

        # Degrade ratio
        degrade_ratio = burst_ttft_avg / baseline if baseline > 0 else 1.0

        # Recovery: find when TTFT returns to < 1.3x baseline after burst ends
        recovery_s = None
        if baseline > 0:
            recovery_threshold = baseline * 1.3
            for ts in time_series:
                if ts["second"] < end:
                    continue
                if ts["avg_ttft_ms"] > 0 and ts["avg_ttft_ms"] <= recovery_threshold:
                    recovery_s = ts["second"] - end
                    break

        segment_analyses.append(
            {
                "start_s": start,
                "end_s": end,
                "duration_s": seg["duration_s"],
                "peak_qps": seg["peak_qps"],
                "avg_qps": seg["avg_qps"],
                "baseline_ttft_ms": round(baseline, 1),
                "burst_ttft_avg_ms": round(burst_ttft_avg, 1),
                "burst_ttft_peak_ms": round(burst_ttft_peak, 1),
                "degrade_ratio": round(degrade_ratio, 2),
                "recovery_s": recovery_s,
            }
        )

        if degrade_ratio > 1.0:
            degrade_ratios.append(degrade_ratio)
        if recovery_s is not None:
            recovery_times.append(recovery_s)

    return {
        "segment_count": len(burst_segments),
        "avg_degrade_ratio": (
            round(sum(degrade_ratios) / len(degrade_ratios), 2)
            if degrade_ratios
            else 1.0
        ),
        "avg_recovery_s": (
            round(sum(recovery_times) / len(recovery_times), 1)
            if recovery_times
            else None
        ),
        "baseline_ttft_ms": round(baseline, 1),
        "segments": segment_analyses,
    }


# ---------------------------------------------------------------------------
# Monitor analysis
# ---------------------------------------------------------------------------


def analyze_monitor(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Extract inflight and JVM time series from monitor records."""
    if not records:
        return {
            "inflight_series": [],
            "inflight_peak": 0,
            "inflight_end": 0,
            "prefill_inflight_peaks": {},
            "decode_inflight_peaks": {},
            "jvm_heap_peak": 0,
            "jvm_heap_end": 0,
        }

    inflight_series: List[Dict[str, Any]] = []
    scheduler_vals: List[float] = []
    prefill_peaks: Dict[str, float] = {}
    decode_peaks: Dict[str, float] = {}
    heap_vals: List[float] = []

    for r in records:
        elapsed = r.get("elapsed_s", 0)
        sched = r.get("scheduler_inflight")
        if isinstance(sched, (int, float)):
            scheduler_vals.append(sched)
            inflight_series.append({"elapsed_s": elapsed, "value": sched})
        pf = r.get("prefill_inflight")
        if isinstance(pf, dict):
            for k, v in pf.items():
                if isinstance(v, (int, float)):
                    prefill_peaks[str(k)] = max(prefill_peaks.get(str(k), 0), v)
        dc = r.get("decode_inflight")
        if isinstance(dc, dict):
            for k, v in dc.items():
                if isinstance(v, (int, float)):
                    decode_peaks[str(k)] = max(decode_peaks.get(str(k), 0), v)
        heap = r.get("jvm_heap_used_mb")
        if isinstance(heap, (int, float)):
            heap_vals.append(heap)

    return {
        "inflight_series": inflight_series,
        "inflight_peak": max(scheduler_vals) if scheduler_vals else 0,
        "inflight_end": scheduler_vals[-1] if scheduler_vals else 0,
        "prefill_inflight_peaks": prefill_peaks,
        "decode_inflight_peaks": decode_peaks,
        "jvm_heap_peak": max(heap_vals) if heap_vals else 0,
        "jvm_heap_end": heap_vals[-1] if heap_vals else 0,
    }


# ---------------------------------------------------------------------------
# Log analysis
# ---------------------------------------------------------------------------


def analyze_log_events(events: Dict[str, Any]) -> Dict[str, Any]:
    """Summarize log events."""
    batch_sizes = events.get("dispatch_batch_sizes", [])
    return {
        "dispatch_count": len(batch_sizes),
        "batch_size_avg": (
            round(sum(batch_sizes) / len(batch_sizes), 1) if batch_sizes else 0.0
        ),
        "batch_size_max": max(batch_sizes) if batch_sizes else 0,
        "dispatch_reasons": events.get("dispatch_reasons", {}),
        "queue_depth_peak": (
            max(events.get("queue_depth_values", [0]))
            if events.get("queue_depth_values")
            else 0
        ),
        "budget_overruns": events.get("budget_overruns", 0),
        "drops": events.get("drops", 0),
        "log_errors": events.get("errors", 0),
        "oom": events.get("oom", False),
    }


# ---------------------------------------------------------------------------
# Combined analysis per run
# ---------------------------------------------------------------------------


def analyze_run(data: SpeedRunData) -> Dict[str, Any]:
    """Perform full analysis for a single run."""
    pr_analysis = analyze_per_request(data.per_request, data.speed)
    monitor_analysis = analyze_monitor(data.monitor)
    log_analysis = analyze_log_events(data.log_events)

    burst_response = analyze_burst_response(
        pr_analysis["per_second"],
        pr_analysis["burst_segments"],
        pr_analysis["trace_avg_qps"],
    )

    inflight_peak = monitor_analysis.get("inflight_peak", 0)
    inflight_hit_limit = inflight_peak >= 2  # MAX_INFLIGHT_BATCHES=2
    inflight_end = monitor_analysis.get("inflight_end", 0)
    inflight_drained = inflight_end is not None and inflight_end <= 0

    return {
        "per_request": pr_analysis,
        "monitor": monitor_analysis,
        "log": log_analysis,
        "burst_response": burst_response,
        "inflight_hit_limit": inflight_hit_limit,
        "inflight_drained": inflight_drained,
    }


# ---------------------------------------------------------------------------
# Helper functions for report generation
# ---------------------------------------------------------------------------


def _fmt_val(val: Any, suffix: str = "") -> str:
    if val is None:
        return "N/A"
    if isinstance(val, bool):
        return "Yes" if val else "No"
    if isinstance(val, float):
        return f"{val:.1f}{suffix}"
    if isinstance(val, int):
        return f"{val}{suffix}"
    return str(val)


def _get_max_concurrency(runs: List[SpeedRunData]) -> str:
    """Get max_concurrency from first available summary.json, or env var, or default."""
    for run in runs:
        if run.summary and isinstance(run.summary.get("max_concurrency"), (int, float)):
            return str(run.summary["max_concurrency"])
    return os.environ.get("MAX_CONCURRENCY", "16384")


def _get_summary_field(summary: Optional[Dict[str, Any]], *keys: str) -> Optional[Any]:
    if summary is None:
        return None
    val: Any = summary
    for key in keys:
        if not isinstance(val, dict):
            return None
        val = val.get(key)
        if val is None:
            return None
    return val


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------


def generate_report(runs: List[SpeedRunData], sla_ms: float, speeds: List[int]) -> str:
    """Build the full Markdown burst-traffic comparison report."""
    L: List[str] = []
    labels = [r.speed_label for r in runs]

    L.append("# FlexLB Mock Engine 尖峰流量压力测试报告（真实 trace 多倍速回放）")
    L.append("")
    L.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    L.append("")

    # --- Section 1: Test info ---
    L.append("## 1. 测试信息")
    L.append(f"- 日期: {datetime.now().strftime('%Y-%m-%d')}")
    L.append("- 环境: Mock engine (Python), FlexLB Master (Java/Spring Boot)")
    L.append("- Trace 数据: `data/online_logs/trace_30min.jsonl`")
    L.append(f"  - {TRACE_REQUESTS} 条真实请求, {TRACE_DURATION_MIN} 分钟")
    L.append(f"  - 平均 {TRACE_AVG_QPS} QPS, 峰值 {TRACE_PEAK_QPS} QPS")
    L.append(f"  - 含 ~13 个自然尖峰段 (QPS > {TRACE_AVG_QPS * 1.5:.1f})")
    L.append("- 配置:")
    L.append("  - N_PREFILL=2, N_DECODE=4")
    L.append("  - SCHEDULE_MODE=batch")
    L.append("  - LOAD_BALANCE_STRATEGY=COST_BASED_PREFILL")
    L.append("  - DECODE_LOAD_BALANCE_STRATEGY=COST_BASED_DECODE")
    L.append(f"  - MAX_CONCURRENCY={_get_max_concurrency(runs)}")
    L.append(f"  - SLA_TTFT_MS={int(sla_ms)}")
    L.append("  - ZERO_OUTPUT_POLICY=one (trace ol=0, set to 1)")
    L.append("  - LIMIT=0 (unlimited, full 8332 requests)")
    L.append("  - DURATION_S=0 (complete trace)")
    L.append("  - MAX_INFLIGHT_BATCHES=2, WAIT_MS=220ms, PREDICT_THRESHOLD_MS=550ms")
    L.append("  - MONITOR_INTERVAL=2s")
    L.append("")

    # --- Section 2: Test plan ---
    L.append("## 2. 测试方案")
    L.append("")
    L.append("### 2.1 压力梯度")
    L.append("使用真实生产 trace 数据，通过 6 档 replay speed 放大压力：")
    L.append("")
    L.append("| Speed | 等效平均 QPS | 等效峰值 QPS | Trace 回放时长 (s) |")
    L.append("|-------|------------|------------|-------------------|")
    for sp in speeds:
        eq_avg = TRACE_AVG_QPS * sp
        eq_peak = TRACE_PEAK_QPS * sp
        duration = round(TRACE_DURATION_MIN * 60 / sp)
        L.append(f"| {sp}x | {eq_avg:.1f} | {eq_peak} | ~{duration} |")
    L.append("")

    L.append("### 2.2 观测指标")
    L.append("| 指标 | 说明 |")
    L.append("|------|------|")
    L.append("| TTFT p50/p90/p99/max | 首字延迟分位数 (ms) |")
    L.append("| SLA 违规率 | TTFT > SLA_TTFT_MS 的请求占比 |")
    L.append("| 错误率 | 非成功请求占比 |")
    L.append("| inflight 峰值 | Scheduler inflight 最高值 |")
    L.append("| 自然尖峰段 | QPS > 1.5x 平均且持续 >= 2s 的区间 |")
    L.append("| 尖峰响应延迟 | 尖峰期间 TTFT 相对基线的恶化幅度 |")
    L.append("| 恢复时间 | 尖峰结束后 TTFT 回归基线的时间 |")
    L.append("| batch size 分布 | FlexLB dispatch 的 batch 大小 |")
    L.append("| dispatch reason | dispatch 触发原因分布 |")
    L.append("")

    # --- Section 3: Overview comparison table ---
    col_count = len(runs)
    L.append("## 3. 总览对比表")
    L.append("| 指标 | " + " | ".join(labels) + " |")
    L.append("|------|" + "|".join(["-----"] * col_count) + "|")

    # Equivalent avg QPS
    vals = [
        str(r.analysis.get("per_request", {}).get("equivalent_avg_qps", "N/A"))
        for r in runs
    ]
    L.append(f"| 等效平均 QPS | " + " | ".join(vals) + " |")

    # Equivalent peak QPS
    vals = [
        str(r.analysis.get("per_request", {}).get("equivalent_peak_qps", "N/A"))
        for r in runs
    ]
    L.append(f"| 等效峰值 QPS | " + " | ".join(vals) + " |")

    # Total requests
    vals = [str(_get_summary_field(r.summary, "total_requests") or "N/A") for r in runs]
    L.append(f"| 总请求数 | " + " | ".join(vals) + " |")

    # TTFT p50
    vals = [
        _fmt_val(_get_summary_field(r.summary, "ttft_ms", "p50"), " ms") for r in runs
    ]
    L.append(f"| TTFT p50 (ms) | " + " | ".join(vals) + " |")

    # TTFT p90
    vals = [
        _fmt_val(_get_summary_field(r.summary, "ttft_ms", "p90"), " ms") for r in runs
    ]
    L.append(f"| TTFT p90 (ms) | " + " | ".join(vals) + " |")

    # TTFT p99
    vals = [
        _fmt_val(_get_summary_field(r.summary, "ttft_ms", "p99"), " ms") for r in runs
    ]
    L.append(f"| TTFT p99 (ms) | " + " | ".join(vals) + " |")

    # TTFT max
    vals = [
        _fmt_val(_get_summary_field(r.summary, "ttft_ms", "max"), " ms") for r in runs
    ]
    L.append(f"| TTFT max (ms) | " + " | ".join(vals) + " |")

    # SLA violation rate
    vals = []
    for r in runs:
        rate = _get_summary_field(r.summary, "sla_violation_rate")
        if rate is not None and isinstance(rate, (int, float)):
            vals.append(f"{rate * 100:.2f}%")
        else:
            vals.append("N/A")
    L.append(f"| SLA 违规率 | " + " | ".join(vals) + " |")

    # Error rate
    vals = []
    for r in runs:
        total = _get_summary_field(r.summary, "total_requests") or 0
        errors = _get_summary_field(r.summary, "errors") or 0
        if total > 0:
            vals.append(f"{errors / total * 100:.2f}%")
        else:
            vals.append("N/A")
    L.append(f"| 错误率 | " + " | ".join(vals) + " |")

    # Inflight peak
    vals = [
        str(r.analysis.get("monitor", {}).get("inflight_peak", "N/A")) for r in runs
    ]
    L.append(f"| inflight 峰值 | " + " | ".join(vals) + " |")

    # Burst segment count
    vals = [
        str(r.analysis.get("burst_response", {}).get("segment_count", 0)) for r in runs
    ]
    L.append(f"| 自然尖峰段数 | " + " | ".join(vals) + " |")

    # Avg degrade ratio
    vals = []
    for r in runs:
        ratio = r.analysis.get("burst_response", {}).get("avg_degrade_ratio", 1.0)
        vals.append(f"{ratio:.2f}x" if ratio > 1.0 else "无恶化")
    L.append(f"| 平均恶化幅度 | " + " | ".join(vals) + " |")

    # Avg recovery
    vals = []
    for r in runs:
        rec = r.analysis.get("burst_response", {}).get("avg_recovery_s")
        if rec is not None:
            vals.append(f"{rec:.1f}s")
        else:
            vals.append("N/A")
    L.append(f"| 平均恢复时间 | " + " | ".join(vals) + " |")

    # Log errors
    vals = [
        (
            str(r.analysis.get("log", {}).get("log_errors", "N/A"))
            if r.log_found
            else "N/A"
        )
        for r in runs
    ]
    L.append(f"| 日志错误数 | " + " | ".join(vals) + " |")

    # OOM
    vals = ["Yes" if r.analysis.get("log", {}).get("oom") else "No" for r in runs]
    L.append(f"| OOM | " + " | ".join(vals) + " |")
    L.append("")

    # --- Section 4: Per-speed analysis ---
    L.append("## 4. 逐 Speed 分析")
    L.append("")

    for idx, run in enumerate(runs):
        L.append(
            f"### 4.{idx + 1} {run.speed_label} (等效 {run.analysis.get('per_request', {}).get('equivalent_avg_qps', 'N/A')} QPS)"
        )
        L.append("")

        # TTFT stats
        ttft = _get_summary_field(run.summary, "ttft_ms")
        if isinstance(ttft, dict):
            L.append("**TTFT 统计**:")
            L.append(f"- p50: {ttft.get('p50', 'N/A')} ms")
            L.append(f"- p90: {ttft.get('p90', 'N/A')} ms")
            L.append(f"- p99: {ttft.get('p99', 'N/A')} ms")
            L.append(f"- max: {ttft.get('max', 'N/A')} ms")
            L.append(f"- mean: {ttft.get('mean', 'N/A')} ms")
            L.append("")

        # SLA
        sla_violations = _get_summary_field(run.summary, "sla_violations")
        sla_rate = _get_summary_field(run.summary, "sla_violation_rate")
        total = _get_summary_field(run.summary, "total_requests") or 0
        if sla_rate is not None:
            L.append(
                f"**SLA 违规**: {sla_violations} 次 "
                f"({sla_rate * 100:.2f}% of {total} requests)"
            )
            L.append("")

        # Status distribution
        status_counts = _get_summary_field(run.summary, "status_counts")
        if isinstance(status_counts, dict) and status_counts:
            L.append("**状态分布**:")
            for status, count in sorted(status_counts.items()):
                L.append(f"- {status}: {count}")
            L.append("")

        # Trace QPS stats
        pr_data = run.analysis.get("per_request", {})
        L.append("**Trace QPS 统计**:")
        L.append(f"- Trace 平均 QPS: {pr_data.get('trace_avg_qps', 'N/A')}")
        L.append(f"- Trace 峰值 QPS: {pr_data.get('trace_peak_qps', 'N/A')}")
        L.append(f"- 等效平均 QPS: {pr_data.get('equivalent_avg_qps', 'N/A')}")
        L.append(f"- 等效峰值 QPS: {pr_data.get('equivalent_peak_qps', 'N/A')}")
        L.append("")

        # Natural burst segments
        burst_resp = run.analysis.get("burst_response", {})
        segments = burst_resp.get("segments", [])
        if segments:
            L.append("**自然尖峰段响应分析**:")
            L.append(f"- 检测到 {len(segments)} 个自然尖峰段")
            L.append(f"- 基线 TTFT: {burst_resp.get('baseline_ttft_ms', 'N/A')} ms")
            L.append(
                f"- 平均恶化幅度: {burst_resp.get('avg_degrade_ratio', 1.0):.2f}x 基线"
            )
            rec = burst_resp.get("avg_recovery_s")
            if rec is not None:
                L.append(f"- 平均恢复时间: {rec:.1f}s")
            else:
                L.append("- 平均恢复时间: N/A (未恢复或在监测窗口外)")
            L.append("")

            L.append(
                "| # | 尖峰区间 (s) | 持续 (s) | 峰值 QPS | 基线 TTFT (ms) | 尖峰 TTFT 均值 (ms) | 尖峰 TTFT 峰值 (ms) | 恶化倍数 | 恢复 (s) |"
            )
            L.append(
                "|---|-------------|---------|---------|---------------|-------------------|-------------------|---------|---------|"
            )
            for i, seg in enumerate(segments):
                L.append(
                    f"| {i+1} | {seg['start_s']}-{seg['end_s']} | "
                    f"{seg['duration_s']} | {seg['peak_qps']} | "
                    f"{seg['baseline_ttft_ms']} | {seg['burst_ttft_avg_ms']} | "
                    f"{seg['burst_ttft_peak_ms']} | {seg['degrade_ratio']}x | "
                    f"{seg['recovery_s'] if seg['recovery_s'] is not None else 'N/A'} |"
                )
            L.append("")
        else:
            L.append("**自然尖峰段**: 未检测到 (或数据不足)")
            L.append("")

        # Inflight trend
        monitor_data = run.analysis.get("monitor", {})
        L.append("**Inflight 趋势**:")
        L.append(
            f"- Scheduler inflight 峰值: {monitor_data.get('inflight_peak', 'N/A')}"
        )
        L.append(
            f"- Scheduler inflight 末值: {monitor_data.get('inflight_end', 'N/A')}"
        )
        L.append(
            f"- 是否触达上限 (MAX_INFLIGHT_BATCHES=2): "
            f"{'是' if run.analysis.get('inflight_hit_limit') else '否'}"
        )
        pf_peaks = monitor_data.get("prefill_inflight_peaks", {})
        if pf_peaks:
            pf_str = ", ".join(f"{k}={v}" for k, v in sorted(pf_peaks.items()))
            L.append(f"- Prefill inflight 峰值: {pf_str}")
        dc_peaks = monitor_data.get("decode_inflight_peaks", {})
        if dc_peaks:
            dc_str = ", ".join(f"{k}={v}" for k, v in sorted(dc_peaks.items()))
            L.append(f"- Decode inflight 峰值: {dc_str}")
        L.append("")

        # JVM
        heap_peak = monitor_data.get("jvm_heap_peak", 0)
        heap_end = monitor_data.get("jvm_heap_end", 0)
        if heap_peak > 0:
            L.append("**JVM 内存**:")
            L.append(f"- Heap 峰值: {heap_peak:.0f} MB")
            L.append(f"- Heap 末值: {heap_end:.0f} MB")
            L.append("")

        # Batch & dispatch
        log_data = run.analysis.get("log", {})
        L.append("**Batch & Dispatch 分析**:")
        L.append(f"- Dispatch 次数: {log_data.get('dispatch_count', 0)}")
        L.append(f"- Batch size 均值: {log_data.get('batch_size_avg', 0)}")
        L.append(f"- Batch size 最大: {log_data.get('batch_size_max', 0)}")
        reasons = log_data.get("dispatch_reasons", {})
        if reasons:
            reason_str = ", ".join(
                f"{k}={v}" for k, v in sorted(reasons.items(), key=lambda x: -x[1])
            )
            L.append(f"- Dispatch reason 分布: {reason_str}")
        else:
            L.append("- Dispatch reason 分布: (未检测到)")
        L.append(f"- Queue depth 峰值: {log_data.get('queue_depth_peak', 0)}")
        L.append(f"- Budget overrun 次数: {log_data.get('budget_overruns', 0)}")
        L.append(f"- Drop/Reject 次数: {log_data.get('drops', 0)}")
        L.append("")

        # TTFT worst 5 seconds
        per_second = pr_data.get("per_second", [])
        if per_second:
            worst = sorted(
                [ts for ts in per_second if ts["avg_ttft_ms"] > 0],
                key=lambda x: -x["avg_ttft_ms"],
            )[:5]
            if worst:
                L.append("**TTFT 最差 5 秒**:")
                L.append(
                    "| 秒 | Trace QPS | 等效 QPS | 平均 TTFT (ms) | 最大 TTFT (ms) |"
                )
                L.append(
                    "|-----|-----------|---------|---------------|---------------|"
                )
                for ts in worst:
                    L.append(
                        f"| {ts['second']} | {ts['trace_qps']} | "
                        f"{ts['equivalent_qps']} | "
                        f"{ts['avg_ttft_ms']} | {ts['max_ttft_ms']} |"
                    )
                L.append("")

        L.append("---")
        L.append("")

    # --- Section 5: Burst response & recovery quantification ---
    L.append("## 5. 尖峰响应与恢复时间量化")
    L.append("")
    L.append(
        "| Speed | 等效峰值 QPS | 尖峰数 | 平均恶化幅度 | 平均恢复时间 (s) | inflight 峰值 |"
    )
    L.append(
        "|-------|------------|--------|------------|----------------|-------------|"
    )
    for run in runs:
        pr_data = run.analysis.get("per_request", {})
        burst_resp = run.analysis.get("burst_response", {})
        eq_peak = pr_data.get("equivalent_peak_qps", "N/A")
        seg_count = burst_resp.get("segment_count", 0)
        avg_ratio = burst_resp.get("avg_degrade_ratio", 1.0)
        avg_rec = burst_resp.get("avg_recovery_s")
        inflight_peak = run.analysis.get("monitor", {}).get("inflight_peak", "N/A")
        ratio_str = f"{avg_ratio:.2f}x" if avg_ratio > 1.0 else "无恶化"
        rec_str = f"{avg_rec:.1f}" if avg_rec is not None else "N/A"
        L.append(
            f"| {run.speed_label} | {eq_peak} | {seg_count} | "
            f"{ratio_str} | {rec_str} | {inflight_peak} |"
        )
    L.append("")

    # --- Section 6: Optimization suggestions ---
    L.append("## 6. 优化建议")
    L.append("")
    suggestions: List[str] = []

    for run in runs:
        burst_resp = run.analysis.get("burst_response", {})
        avg_ratio = burst_resp.get("avg_degrade_ratio", 1.0)
        if avg_ratio > 2.0:
            suggestions.append(
                f"- {run.speed_label}: 尖峰期间 TTFT 恶化达 {avg_ratio:.2f}x 基线，"
                f"建议考虑增大 MAX_INFLIGHT_BATCHES 或缩短 WAIT_MS"
            )
        if run.analysis.get("inflight_hit_limit"):
            suggestions.append(
                f"- {run.speed_label}: inflight 触达 MAX_INFLIGHT_BATCHES=2 上限，"
                f"队列可能积压，建议评估是否调整上限"
            )
        avg_rec = burst_resp.get("avg_recovery_s")
        if avg_rec is not None and avg_rec > 10:
            suggestions.append(
                f"- {run.speed_label}: 平均恢复时间 {avg_rec:.1f}s 较长，"
                f"建议优化 batcher 窗口或预取策略"
            )
        log_data = run.analysis.get("log", {})
        if log_data.get("drops", 0) > 0:
            suggestions.append(
                f"- {run.speed_label}: 检测到 {log_data['drops']} 次 drop/reject，"
                f"建议检查反压策略或队列容量"
            )
        if log_data.get("budget_overruns", 0) > 0:
            suggestions.append(
                f"- {run.speed_label}: 检测到 {log_data['budget_overruns']} 次 "
                f"budget overrun，建议调整 PREDICT_THRESHOLD_MS"
            )

    if not suggestions:
        suggestions.append("- 所有 speed 下系统表现稳定，未检测到明显瓶颈")
        suggestions.append(
            "- 当前配置 (MAX_INFLIGHT_BATCHES=2, WAIT_MS=220ms) "
            "在测试流量范围内表现良好"
        )

    for s in suggestions:
        L.append(s)
    L.append("")

    L.append("---")
    L.append("")
    L.append(
        f"*本报告由 analyze_burst_results.py 自动生成，"
        f"基于 {len(runs)} 档 replay speed ({', '.join(labels)}) 测试结果。*"
    )

    return "\n".join(L)


# ---------------------------------------------------------------------------
# Per-speed analysis.json output
# ---------------------------------------------------------------------------


def save_analysis_json(run: SpeedRunData) -> None:
    """Save structured analysis data for a single run."""
    analysis_path = run.run_dir / "analysis.json"
    output = {
        "speed": run.speed,
        "speed_label": run.speed_label,
        "trace_info": {
            "requests": TRACE_REQUESTS,
            "duration_min": TRACE_DURATION_MIN,
            "avg_qps": TRACE_AVG_QPS,
            "peak_qps": TRACE_PEAK_QPS,
        },
        "summary": run.summary,
        "analysis": run.analysis,
        "log_found": run.log_found,
    }
    try:
        with open(analysis_path, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, ensure_ascii=False, default=str)
    except (OSError, TypeError):
        print(
            f"WARNING: failed to write analysis.json for {run.speed_label}",
            file=sys.stderr,
        )


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def main() -> None:
    args = parse_args()

    if not args.run_dirs:
        print("ERROR: no run directories provided", file=sys.stderr)
        sys.exit(1)

    # Build speed label mapping
    speed_map: Dict[int, int] = {}
    for i, sp in enumerate(args.speeds):
        speed_map[i] = sp

    runs: List[SpeedRunData] = []
    for idx, run_dir_path in enumerate(args.run_dirs):
        run_dir = Path(run_dir_path)
        if not run_dir.exists():
            print(f"WARNING: run directory not found: {run_dir}", file=sys.stderr)
            continue
        # Determine speed: prefer CLI arg, fall back to directory name
        speed = speed_map.get(idx, extract_speed(run_dir))
        if speed == 0:
            print(
                f"WARNING: could not determine speed for {run_dir}, skipping",
                file=sys.stderr,
            )
            continue
        print(f"Loading: {run_dir} (speed={speed}x)", file=sys.stderr)
        run_data = load_run_data(run_dir, speed)
        runs.append(run_data)

    if not runs:
        print("ERROR: no valid run directories found", file=sys.stderr)
        sys.exit(1)

    # Sort runs by speed
    runs.sort(key=lambda r: r.speed)

    # Save per-speed analysis.json
    for run in runs:
        save_analysis_json(run)

    # Generate report
    all_speeds = [r.speed for r in runs]
    report = generate_report(runs, args.sla_ttft_ms, all_speeds)

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(report, encoding="utf-8")
        print(f"Report written to {out}", file=sys.stderr)
    else:
        print(report)


if __name__ == "__main__":
    main()
