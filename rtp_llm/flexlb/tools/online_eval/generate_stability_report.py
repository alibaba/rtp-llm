#!/usr/bin/env python3
"""Generate a Markdown stability-test comparison report from multiple FlexLB pressure runs.

Reads per-run ``summary.json``, ``monitor.jsonl``, and ``flexlb.log`` to produce
a side-by-side comparison with PASS/FAIL/WARN verdicts across increasing load levels.
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
from typing import Mapping, Optional

sys.path.insert(0, os.path.dirname(__file__))
from online_eval.report import load_summary  # noqa: E402

_OOM_RE = re.compile(r"java\.lang\.OutOfMemoryError", re.IGNORECASE)
_ERROR_RE = re.compile(r"java\.lang\.OutOfMemoryError|FATAL|Exception", re.IGNORECASE)
_LABEL_RE = re.compile(r"(\d+)x")
_LOG_INFO_RE = re.compile(r"\bINFO\b\s", re.IGNORECASE)


@dataclass
class RunData:
    """All loaded data for a single pressure-test run."""

    label: str
    run_dir: Path
    summary: Optional[Mapping[str, object]] = None
    monitor: list[dict] = field(default_factory=list)
    log_errors: int = 0
    log_found: bool = False
    has_oom: bool = False


# ---------------------------------------------------------------------------
# Argument parsing & run discovery
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dirs", nargs="+", help="run directories (space-separated)"
    )
    parser.add_argument("--run-root", help="root dir for auto-discovery with --speeds")
    parser.add_argument("--speeds", help="comma-separated speeds, e.g. 10,20,50")
    parser.add_argument("--output", help="output report path (default: stdout)")
    parser.add_argument(
        "--sla-ttft-ms",
        type=float,
        default=500.0,
        help="SLA TTFT threshold in ms (default: 500)",
    )
    parser.add_argument(
        "--mock-baseline-ttft-ms",
        type=float,
        default=300.0,
        help="Mock latency model baseline TTFT in ms (default: 300, "
        "DSV4-Flash prefill formula baseline)",
    )
    return parser.parse_args()


def extract_label(run_dir: Path) -> str:
    """Extract speed label (e.g. ``10x``) from a directory name."""
    m = _LABEL_RE.search(run_dir.name)
    return f"{m.group(1)}x" if m else run_dir.name


def discover_run_dirs(run_root: Path, speeds: list[str]) -> list[tuple[str, Path]]:
    """Auto-discover ``stability_{speed}x_*`` directories under *run_root*."""
    result: list[tuple[str, Path]] = []
    for speed in speeds:
        matches = sorted(run_root.glob(f"stability_{speed}x_*"))
        if not matches:
            print(
                f"WARNING: no dir matching stability_{speed}x_* under {run_root}",
                file=sys.stderr,
            )
        for path in matches:
            result.append((f"{speed}x", path))
    return result


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_monitor_records(path: Path) -> list[dict]:
    """Parse ``monitor.jsonl``, skipping blank or corrupted lines."""
    records: list[dict] = []
    if not path.exists():
        return records
    for line in path.read_text(encoding="utf-8").splitlines():
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


def analyze_log(path: Path) -> tuple[int, bool, bool]:
    """Return ``(error_count, log_found, has_oom)`` from *flexlb.log*.

    Excludes INFO-level lines and JVM startup option lines from the error count
    to avoid false positives from Spring framework logging or JVM flags
    (e.g. ``-XX:+HeapDumpOnOutOfMemoryError``) that mention *Exception*.
    """
    if not path.exists():
        return 0, False, False
    error_count = 0
    oom = False
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if "JAVA_TOOL_OPTIONS" in line or "Picked up JAVA_TOOL_OPTIONS" in line:
            continue
        if _OOM_RE.search(line):
            oom = True
        if _LOG_INFO_RE.search(line):
            continue
        if _ERROR_RE.search(line):
            error_count += 1
    return error_count, True, oom


def load_run_data(run_dir: Path, label: str) -> RunData:
    """Load all data sources for a single run directory."""
    data = RunData(label=label, run_dir=run_dir)
    summary_path = run_dir / "load_client" / "summary.json"
    if summary_path.exists():
        try:
            data.summary = load_summary(summary_path)
        except (ValueError, OSError):
            data.summary = None
    data.monitor = load_monitor_records(run_dir / "monitor.jsonl")
    data.log_errors, data.log_found, data.has_oom = analyze_log(run_dir / "flexlb.log")
    return data


# ---------------------------------------------------------------------------
# Verdict logic
# ---------------------------------------------------------------------------


def verdict_master(data: RunData) -> str:
    """PASS if summary exists and no OOM; WARN if log errors; FAIL otherwise."""
    if data.summary is None or data.has_oom:
        return "FAIL"
    if data.log_errors > 0:
        return "WARN"
    return "PASS"


def verdict_inflight(data: RunData) -> str:
    """Judge inflight boundedness from the last 5 monitor records.

    PASS: avg of last 5 ≤ 5 (fully drained)
    WARN: avg ≤ 20, OR avg > 20 but consistently decreasing (draining)
    FAIL: avg > 20 and not decreasing (accumulating or stuck)
    """
    vals = [
        r["scheduler_inflight"]
        for r in data.monitor
        if isinstance(r.get("scheduler_inflight"), (int, float))
    ]
    if not vals:
        return "FAIL"
    avg = sum(vals[-5:]) / len(vals[-5:])
    if avg <= 5:
        return "PASS"
    # Check if inflight is consistently decreasing in the last 5 records
    last5 = vals[-5:]
    is_decreasing = len(last5) >= 2 and all(
        last5[i] <= last5[i - 1] for i in range(1, len(last5))
    )
    if avg <= 20:
        return "WARN"
    if is_decreasing:
        return "WARN"
    return "FAIL"


def verdict_ttft(data: RunData, sla_ms: float, mock_baseline_ms: float = 0.0) -> str:
    """Judge TTFT p99 against SLA threshold and mock baseline expectation.

    ``mock_baseline_ms`` is the expected TTFT from the mock latency model
    (e.g. DSV4-Flash prefill formula). A 2× tolerance covers scheduling overhead.
    """
    if data.summary is None:
        return "FAIL"
    ttft = data.summary.get("ttft_ms")
    if not isinstance(ttft, Mapping):
        return "FAIL"
    p99 = ttft.get("p99")
    if not isinstance(p99, (int, float)) or isinstance(p99, bool):
        return "FAIL"
    if mock_baseline_ms > 0:
        if p99 <= mock_baseline_ms * 2:
            return "PASS"
        if p99 < sla_ms:
            return "WARN"
    else:
        if p99 < sla_ms:
            return "PASS"
    if p99 < 2 * sla_ms:
        return "WARN"
    return "FAIL"


def overall_verdict(verdicts: list[str]) -> str:
    """Aggregate per-criterion verdicts: any FAIL→FAIL, any WARN→WARN, else PASS."""
    if "FAIL" in verdicts:
        return "FAIL"
    if "WARN" in verdicts:
        return "WARN"
    return "PASS"


# ---------------------------------------------------------------------------
# Monitor stats extraction
# ---------------------------------------------------------------------------


def _nums(records: list[dict], key: str) -> list[float]:
    """Extract numeric values for *key* from monitor records."""
    return [r[key] for r in records if isinstance(r.get(key), (int, float))]


def extract_jvm_stats(records: list[dict]) -> Optional[dict]:
    """Extract JVM heap and GC statistics from monitor records."""
    heap = _nums(records, "jvm_heap_used_mb")
    if not heap:
        return None
    gc_counts = _nums(records, "jvm_gc_pause_count")
    gc_totals = _nums(records, "jvm_gc_pause_total_ms")
    elapsed = _nums(records, "elapsed_s")
    runtime_min = (elapsed[-1] - elapsed[0]) / 60.0 if len(elapsed) >= 2 else 0.0
    growth = (heap[-1] - heap[0]) / runtime_min if runtime_min > 0 else 0.0
    return {
        "heap_peak": max(heap),
        "heap_end": heap[-1],
        "growth_rate": growth,
        "gc_count": int(gc_counts[-1]) if gc_counts else 0,
        "gc_total_ms": int(gc_totals[-1]) if gc_totals else 0,
    }


def extract_inflight_stats(records: list[dict]) -> Optional[dict]:
    """Extract scheduler and per-engine inflight statistics."""
    sched = _nums(records, "scheduler_inflight")
    if not sched:
        return None
    prefill_peaks: dict[str, float] = {}
    decode_peaks: dict[str, float] = {}
    for r in records:
        pf = r.get("prefill_inflight") or r.get("preflight_inflight")
        if isinstance(pf, dict):
            for k, v in pf.items():
                if isinstance(v, (int, float)):
                    prefill_peaks[str(k)] = max(prefill_peaks.get(str(k), 0), v)
        dc = r.get("decode_inflight")
        if isinstance(dc, dict):
            for k, v in dc.items():
                if isinstance(v, (int, float)):
                    decode_peaks[str(k)] = max(decode_peaks.get(str(k), 0), v)
    last5 = sched[-5:]
    avg = sum(last5) / len(last5)
    return {
        "sched_peak": max(sched),
        "sched_avg": avg,
        "fall_back": "YES" if avg <= 5 else ("PARTIAL" if avg <= 20 else "NO"),
        "prefill_peaks": prefill_peaks,
        "decode_peaks": decode_peaks,
    }


def _format_peaks(peaks: dict[str, float]) -> str:
    """Format per-engine peaks as ``val0/val1/...`` sorted by key."""
    if not peaks:
        return "N/A"
    return "/".join(str(peaks[k]) for k in sorted(peaks))


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------


def _get(summary: Optional[Mapping], *keys: str) -> Optional[object]:
    """Safely retrieve a nested value from *summary*; returns *None* on any miss."""
    if summary is None:
        return None
    val: object = summary
    for key in keys:
        if not isinstance(val, Mapping):
            return None
        val = val.get(key)
        if val is None:
            return None
    return val


def _get_max_concurrency(runs: list[RunData]) -> str:
    """Get max_concurrency from first available summary.json, or env var, or default."""
    for run in runs:
        if run.summary and isinstance(run.summary.get("max_concurrency"), (int, float)):
            return str(run.summary["max_concurrency"])
    return os.environ.get("MAX_CONCURRENCY", "16384")


def generate_report(
    runs: list[RunData], sla_ms: float, mock_baseline_ms: float = 0.0
) -> str:
    """Build the full Markdown stability comparison report."""
    labels = [r.label for r in runs]
    L: list[str] = []

    L.append("# FlexLB L1 稳定性测试报告")
    L.append("")
    L.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    L.append("")

    # --- Section 1: Test config ---
    L.append("## 1. 测试配置")
    L.append("- N_PREFILL=2, N_DECODE=4, SCHEDULE_MODE=batch")
    L.append(
        f"- MAX_CONCURRENCY={_get_max_concurrency(runs)}, SLA_TTFT_MS={int(sla_ms)}ms"
    )
    L.append(f"- 递增压力: REPLAY_SPEED = {' → '.join(labels)}")
    L.append("")

    # Compute verdicts
    verdicts = [
        [
            verdict_master(r),
            verdict_inflight(r),
            verdict_ttft(r, sla_ms, mock_baseline_ms),
        ]
        for r in runs
    ]
    crit_names = ["Master 进程存活", "Inflight 有界且回落", "TTFT p99 在预期范围"]

    # --- Section 2: PASS/FAIL ---
    L.append("## 2. PASS/FAIL 判定")
    L.append("| 判定项 | " + " | ".join(labels) + " | 总体 |")
    L.append("|--------|" + "|".join(["-----"] * len(labels)) + "|------|")
    for i, name in enumerate(crit_names):
        col = [v[i] for v in verdicts]
        L.append(f"| {name} | " + " | ".join(col) + f" | {overall_verdict(col)} |")
    flat = [v for row in verdicts for v in row]
    overall = overall_verdict(flat)
    L.append("")
    L.append(f"**总体判定: {overall}**")
    L.append("")

    # --- Section 3: Performance comparison ---
    L.append("## 3. 性能对比表")
    L.append("| 指标 | " + " | ".join(labels) + " |")
    L.append("|------|" + "|".join(["-----"] * len(labels)) + "|")

    def cell(run: RunData, *keys: str, pct: bool = False) -> str:
        if run.summary is None:
            return "N/A (crashed)"
        val = _get(run.summary, *keys)
        if val is None:
            return "N/A"
        if pct and isinstance(val, (int, float)):
            return f"{val * 100:.1f}"
        return str(val)

    perf_rows: list[tuple[str, list[str], bool]] = [
        ("Offered QPS", ["offered_qps"], False),
        ("Completed QPS", ["completed_qps"], False),
        ("Total Requests", ["total_requests"], False),
        ("TTFT p50 (ms)", ["ttft_ms", "p50"], False),
        ("TTFT p90 (ms)", ["ttft_ms", "p90"], False),
        ("TTFT p99 (ms)", ["ttft_ms", "p99"], False),
        ("SLA 违规率 (%)", ["sla_violation_rate"], True),
        ("Prefill 均衡 stddev", ["prefill_balance", "stddev"], False),
        ("Decode 均衡 stddev", ["decode_balance", "stddev"], False),
        ("Errors", ["errors"], False),
    ]
    for name, keys, is_pct in perf_rows:
        vals = [cell(r, *keys, pct=is_pct) for r in runs]
        L.append(f"| {name} | " + " | ".join(vals) + " |")
    log_err_vals = [
        str(r.log_errors) if r.log_found else "N/A (log not found)" for r in runs
    ]
    L.append(f"| 日志错误数 | " + " | ".join(log_err_vals) + " |")
    L.append("")

    # --- Section 4: Inflight trend ---
    L.append("## 4. Inflight 趋势分析")
    L.append("### 4.1 Scheduler inflight")
    L.append("| 压力级别 | 峰值 | 稳态均值(末5条) | 回落至≤5 | 判定 |")
    L.append("|----------|------|---------------|---------|------|")
    for run in runs:
        stats = extract_inflight_stats(run.monitor)
        if stats is None:
            L.append(f"| {run.label} | N/A (no monitor data) | N/A | N/A | FAIL |")
        else:
            fb = stats["fall_back"]
            judge = "PASS" if fb == "YES" else ("WARN" if fb == "PARTIAL" else "FAIL")
            L.append(
                f"| {run.label} | {stats['sched_peak']} | "
                f"{stats['sched_avg']:.1f} | {fb} | {judge} |"
            )
    L.append("")
    L.append("### 4.2 Prefill/Decode inflight 峰值")
    L.append("| 压力级别 | Prefill峰值 | Decode峰值 |")
    L.append("|----------|------------|------------|")
    for run in runs:
        stats = extract_inflight_stats(run.monitor)
        if stats is None:
            L.append(f"| {run.label} | N/A | N/A |")
        else:
            L.append(
                f"| {run.label} | {_format_peaks(stats['prefill_peaks'])} | "
                f"{_format_peaks(stats['decode_peaks'])} |"
            )
    L.append("")

    # --- Section 5: JVM memory/GC trend ---
    L.append("## 5. JVM 内存/GC 趋势")
    L.append(
        "| 压力级别 | 堆峰值(MB) | 堆末值(MB) | 增长率(MB/min) | "
        "GC次数 | GC总耗时(ms) | 判定 |"
    )
    L.append(
        "|----------|-----------|-----------|---------------|"
        "--------|-------------|------|"
    )
    for run in runs:
        jvm = extract_jvm_stats(run.monitor)
        if jvm is None:
            L.append(
                f"| {run.label} | N/A (no monitor data) | N/A | N/A | "
                f"N/A | N/A | FAIL |"
            )
        else:
            judge = "WARN" if jvm["growth_rate"] > 10 else "PASS"
            L.append(
                f"| {run.label} | {jvm['heap_peak']:.0f} | {jvm['heap_end']:.0f} | "
                f"{jvm['growth_rate']:.1f} | {jvm['gc_count']} | "
                f"{jvm['gc_total_ms']} | {judge} |"
            )
    L.append("")

    # --- Section 6: Conclusion & risks ---
    L.append("## 6. 结论与风险")
    L.append(f"- 总体判定: {overall}")
    findings: list[str] = []
    for run, v in zip(runs, verdicts):
        if v[2] == "FAIL":
            findings.append(f"{run.label} 压力下 TTFT p99 超出 SLA 阈值")
        if v[1] in ("WARN", "FAIL"):
            st = extract_inflight_stats(run.monitor)
            if st and st["fall_back"] != "YES":
                findings.append(f"{run.label} 压力下 inflight 未能完全回落")
    L.append(
        f"- 关键发现: {'; '.join(findings) if findings else '所有压力级别均通过稳定性判定'}"
    )
    fail_labels = [r.label for r, v in zip(runs, verdicts) if "FAIL" in v]
    warn_labels = [
        r.label for r, v in zip(runs, verdicts) if "WARN" in v and "FAIL" not in v
    ]
    if fail_labels:
        safe = warn_labels[-1] if warn_labels else "更低"
        L.append(
            f"- 风险点: {'、'.join(fail_labels)} 压力已超出系统稳定范围，"
            f"建议生产环境不超过 {safe} 等效负载"
        )
    elif warn_labels:
        L.append(
            f"- 风险点: {'、'.join(warn_labels)} 压力接近系统容量上限，"
            f"建议生产环境不超过 {warn_labels[0]} 等效负载"
        )
    else:
        L.append("- 风险点: 无明显风险")
    L.append("")

    return "\n".join(L)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Parse arguments, load run data, and emit the stability report."""
    args = parse_args()

    if args.run_dirs:
        run_specs = [(extract_label(Path(d)), Path(d)) for d in args.run_dirs]
    elif args.run_root and args.speeds:
        speeds = [s.strip() for s in args.speeds.split(",") if s.strip()]
        run_specs = discover_run_dirs(Path(args.run_root), speeds)
    else:
        print(
            "ERROR: provide --run-dirs or both --run-root and --speeds", file=sys.stderr
        )
        sys.exit(1)

    if not run_specs:
        print("ERROR: no run directories found", file=sys.stderr)
        sys.exit(1)

    runs = [load_run_data(path, label) for label, path in run_specs]
    report = generate_report(runs, args.sla_ttft_ms, args.mock_baseline_ttft_ms)

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(report, encoding="utf-8")
        print(f"report written to {out}", file=sys.stderr)
    else:
        print(report)


if __name__ == "__main__":
    main()
