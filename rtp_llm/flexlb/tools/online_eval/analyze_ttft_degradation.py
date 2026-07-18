#!/usr/bin/env python3
"""Analyse TTFT time-series around a Master kill-restart cycle.

Loads per_request.jsonl (from flexlb_load_client.py) and monitor.jsonl
(from stability_monitor.py), divides the timeline into baseline /
fallback / recovery phases, computes per-phase TTFT statistics,
classifies the degradation pattern (spike vs sustained), runs an
attribution decision tree, and emits a Markdown report plus a JSON
data file.  An optional matplotlib chart is generated if the library
is available.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PHASES = [
    "BASELINE",
    "FALLBACK",
    "RECOVERY_EARLY",
    "RECOVERY_COLD",
    "RECOVERY_WARM",
    "RECOVERY_STABLE",
]

# Duration (seconds) of each recovery sub-phase after restart.
RECOVERY_EARLY_S = 15  # Spring Boot startup
RECOVERY_COLD_S = 30  # cold start total
RECOVERY_WARM_S = 90  # warm-up total


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_jsonl(path: Path) -> List[dict]:
    """Read a JSONL file, skipping blank lines."""
    records: List[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records


def load_per_requests(path: Path) -> List[dict]:
    """Load per_request.jsonl and filter out ``status == 'exception'`` rows."""
    raw = load_jsonl(path)
    filtered = [r for r in raw if r.get("status") != "exception"]
    return filtered


def load_monitor(path: Path) -> List[dict]:
    """Load monitor.jsonl."""
    if not path.exists():
        return []
    return load_jsonl(path)


def load_timestamps(path: Path) -> Dict[str, float]:
    """Load timestamps.json if it exists (kill_epoch, restart_epoch)."""
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


# ---------------------------------------------------------------------------
# Phase assignment
# ---------------------------------------------------------------------------


def assign_phase(
    wall_ts: float,
    kill_epoch: Optional[float],
    restart_epoch: Optional[float],
) -> str:
    """Return the phase name for a single request based on wall_clock_ts."""
    if kill_epoch is None or restart_epoch is None:
        return "BASELINE"
    if wall_ts < kill_epoch:
        return "BASELINE"
    if wall_ts < restart_epoch:
        return "FALLBACK"
    if wall_ts < restart_epoch + RECOVERY_EARLY_S:
        return "RECOVERY_EARLY"
    if wall_ts < restart_epoch + RECOVERY_COLD_S:
        return "RECOVERY_COLD"
    if wall_ts < restart_epoch + RECOVERY_WARM_S:
        return "RECOVERY_WARM"
    return "RECOVERY_STABLE"


def annotate_phases(
    requests: List[dict],
    kill_epoch: Optional[float],
    restart_epoch: Optional[float],
) -> None:
    """Add a ``phase`` key to each request dict in-place."""
    for r in requests:
        wall_ts = r.get("wall_clock_ts")
        if wall_ts is None:
            # Fall back to the trace ts field if wall_clock_ts is absent.
            wall_ts = float(r.get("ts", 0)) / 1000.0
            r["wall_clock_ts"] = wall_ts
        r["phase"] = assign_phase(wall_ts, kill_epoch, restart_epoch)


# ---------------------------------------------------------------------------
# Percentile helper
# ---------------------------------------------------------------------------


def percentile(values: Sequence[float], pct: float) -> float:
    """Return the *pct*-th percentile (linear interpolation)."""
    if not values:
        return 0.0
    s = sorted(values)
    if len(s) == 1:
        return round(s[0], 3)
    k = (len(s) - 1) * (pct / 100.0)
    lo = int(k)
    hi = min(lo + 1, len(s) - 1)
    frac = k - lo
    return round(s[lo] + (s[hi] - s[lo]) * frac, 3)


# ---------------------------------------------------------------------------
# Phase statistics
# ---------------------------------------------------------------------------


def compute_phase_stats(
    requests: List[dict], baseline_p50: Optional[float] = None
) -> Dict[str, Any]:
    """Compute per-phase summary statistics."""
    stats: Dict[str, Any] = {}
    for phase in PHASES:
        phase_reqs = [r for r in requests if r.get("phase") == phase]
        if not phase_reqs:
            stats[phase] = {"count": 0}
            continue

        ok_reqs = [r for r in phase_reqs if r.get("status") == "ok"]
        ttft_vals = [r["ttft_ms"] for r in ok_reqs if r.get("ttft_ms", 0) > 0]
        sched_vals = [
            r["schedule_ms"] for r in phase_reqs if r.get("schedule_ms", 0) > 0
        ]

        route_counts: Dict[str, int] = {}
        for r in phase_reqs:
            rp = r.get("route_path", "unknown")
            route_counts[rp] = route_counts.get(rp, 0) + 1

        success_rate = round(len(ok_reqs) / len(phase_reqs), 6) if phase_reqs else 0.0

        ttft_p50 = percentile(ttft_vals, 50)
        ttft_p90 = percentile(ttft_vals, 90)
        ttft_p99 = percentile(ttft_vals, 99)
        ttft_max = round(max(ttft_vals), 3) if ttft_vals else 0.0

        degradation = None
        if baseline_p50 is not None and baseline_p50 > 0 and ttft_p50 > 0:
            degradation = round(ttft_p50 / baseline_p50, 3)

        stats[phase] = {
            "count": len(phase_reqs),
            "ok_count": len(ok_reqs),
            "success_rate": success_rate,
            "ttft_p50_ms": ttft_p50,
            "ttft_p90_ms": ttft_p90,
            "ttft_p99_ms": ttft_p99,
            "ttft_max_ms": ttft_max,
            "schedule_p50_ms": percentile(sched_vals, 50),
            "schedule_p90_ms": percentile(sched_vals, 90),
            "route_path_distribution": route_counts,
            "ttft_degradation_vs_baseline": degradation,
        }
    return stats


# ---------------------------------------------------------------------------
# Second-level timeline
# ---------------------------------------------------------------------------


def compute_second_timeline(
    requests: List[dict],
    kill_epoch: Optional[float],
    restart_epoch: Optional[float],
) -> List[Dict[str, Any]]:
    """Aggregate requests into 1-second buckets."""
    if not requests:
        return []

    min_ts = min(r.get("wall_clock_ts", 0) for r in requests)
    max_ts = max(r.get("wall_clock_ts", 0) for r in requests)
    if min_ts >= max_ts:
        return []

    buckets: Dict[int, List[dict]] = {}
    for r in requests:
        bucket_key = int(r.get("wall_clock_ts", 0))
        buckets.setdefault(bucket_key, []).append(r)

    timeline: List[Dict[str, Any]] = []
    for sec in sorted(buckets.keys()):
        reqs = buckets[sec]
        ok_reqs = [r for r in reqs if r.get("status") == "ok"]
        ttft_vals = [r["ttft_ms"] for r in ok_reqs if r.get("ttft_ms", 0) > 0]
        fallback_count = sum(1 for r in reqs if r.get("route_path") == "fallback")
        timeline.append(
            {
                "second": sec,
                "offset_from_kill_s": (
                    round(sec - kill_epoch, 1) if kill_epoch else None
                ),
                "offset_from_restart_s": (
                    round(sec - restart_epoch, 1) if restart_epoch else None
                ),
                "request_count": len(reqs),
                "ttft_p50_ms": percentile(ttft_vals, 50),
                "ttft_p90_ms": percentile(ttft_vals, 90),
                "ttft_max_ms": round(max(ttft_vals), 3) if ttft_vals else 0.0,
                "fallback_ratio": round(fallback_count / len(reqs), 4),
            }
        )
    return timeline


# ---------------------------------------------------------------------------
# Degradation pattern classification
# ---------------------------------------------------------------------------


def classify_degradation(phase_stats: Dict[str, Any]) -> Dict[str, Any]:
    """Classify TTFT degradation as spike or sustained."""
    baseline = phase_stats.get("BASELINE", {})
    early = phase_stats.get("RECOVERY_EARLY", {})
    stable = phase_stats.get("RECOVERY_STABLE", {})

    baseline_p50 = baseline.get("ttft_p50_ms", 0) or 0
    early_p50 = early.get("ttft_p50_ms", 0) or 0
    stable_p50 = stable.get("ttft_p50_ms", 0) or 0

    if baseline_p50 == 0:
        return {
            "pattern": "unknown",
            "reason": "Baseline TTFT p50 is zero — insufficient baseline data.",
        }

    early_ratio = early_p50 / baseline_p50 if early_p50 > 0 else 0
    stable_count = stable.get("count", 0)
    stable_has_data = stable_count > 0 and stable_p50 > 0
    stable_ratio = stable_p50 / baseline_p50 if stable_has_data else 0

    early_degraded = early_ratio > 1.5
    stable_degraded = stable_has_data and stable_ratio > 1.2

    if early_degraded and not stable_degraded:
        pattern = "spike"
        if not stable_has_data:
            reason = (
                f"RECOVERY_EARLY TTFT p50 ({early_p50:.1f}ms) is "
                f"{early_ratio:.2f}x baseline ({baseline_p50:.1f}ms). "
                f"RECOVERY_STABLE has no data yet, but RECOVERY_WARM "
                f"({phase_stats.get('RECOVERY_WARM', {}).get('ttft_p50_ms', 0):.1f}ms) "
                f"shows recovery — degradation appears transient."
            )
        else:
            reason = (
                f"RECOVERY_EARLY TTFT p50 ({early_p50:.1f}ms) is "
                f"{early_ratio:.2f}x baseline ({baseline_p50:.1f}ms), "
                f"but RECOVERY_STABLE TTFT p50 ({stable_p50:.1f}ms) is only "
                f"{stable_ratio:.2f}x baseline — degradation is transient."
            )
    elif stable_degraded:
        pattern = "sustained"
        reason = (
            f"RECOVERY_STABLE TTFT p50 ({stable_p50:.1f}ms) is "
            f"{stable_ratio:.2f}x baseline ({baseline_p50:.1f}ms) — "
            f"degradation persists into steady state."
        )
    elif early_degraded and stable_degraded:
        pattern = "sustained"
        reason = (
            f"Both RECOVERY_EARLY ({early_ratio:.2f}x) and RECOVERY_STABLE "
            f"({stable_ratio:.2f}x) show degradation above threshold."
        )
    else:
        pattern = "none"
        if not stable_has_data:
            reason = (
                f"No significant degradation detected. "
                f"RECOVERY_EARLY ratio={early_ratio:.2f}. "
                f"RECOVERY_STABLE has no data yet."
            )
        else:
            reason = (
                f"No significant degradation detected. "
                f"RECOVERY_EARLY ratio={early_ratio:.2f}, "
                f"RECOVERY_STABLE ratio={stable_ratio:.2f}."
            )

    return {
        "pattern": pattern,
        "baseline_p50_ms": baseline_p50,
        "early_p50_ms": early_p50,
        "stable_p50_ms": stable_p50,
        "early_degradation_ratio": round(early_ratio, 3),
        "stable_degradation_ratio": round(stable_ratio, 3),
        "reason": reason,
    }


# ---------------------------------------------------------------------------
# Attribution decision tree
# ---------------------------------------------------------------------------


def attribute_degradation(
    phase_stats: Dict[str, Any],
    requests: List[dict],
    monitor: List[dict],
    restart_epoch: Optional[float],
) -> List[Dict[str, Any]]:
    """Run a heuristic attribution decision tree and return findings."""
    findings: List[Dict[str, Any]] = []

    # 1. Check schedule_ms anomaly in RECOVERY_EARLY
    early = phase_stats.get("RECOVERY_EARLY", {})
    baseline = phase_stats.get("BASELINE", {})
    if early.get("count", 0) > 0 and baseline.get("count", 0) > 0:
        early_sched_p50 = early.get("schedule_p50_ms", 0) or 0
        base_sched_p50 = baseline.get("schedule_p50_ms", 0) or 0
        if base_sched_p50 > 0 and early_sched_p50 > base_sched_p50 * 3:
            findings.append(
                {
                    "finding": "schedule_backlog",
                    "severity": "high",
                    "detail": (
                        f"RECOVERY_EARLY schedule_ms p50 ({early_sched_p50:.1f}ms) "
                        f"is {early_sched_p50 / base_sched_p50:.1f}x baseline "
                        f"({base_sched_p50:.1f}ms) — points to request backlog "
                        f"during Master restart."
                    ),
                }
            )
        elif base_sched_p50 > 0 and early_sched_p50 > base_sched_p50 * 1.5:
            findings.append(
                {
                    "finding": "schedule_backlog",
                    "severity": "medium",
                    "detail": (
                        f"RECOVERY_EARLY schedule_ms p50 ({early_sched_p50:.1f}ms) "
                        f"is {early_sched_p50 / base_sched_p50:.1f}x baseline "
                        f"({base_sched_p50:.1f}ms) — moderate schedule latency increase."
                    ),
                }
            )

    # 2. Check route_path transition timing
    if restart_epoch is not None:
        post_restart = [
            r for r in requests if r.get("wall_clock_ts", 0) >= restart_epoch
        ]
        if post_restart:
            first_master = None
            for r in sorted(post_restart, key=lambda x: x.get("wall_clock_ts", 0)):
                if r.get("route_path") == "master":
                    first_master = r.get("wall_clock_ts")
                    break
            if first_master is not None:
                delay = first_master - restart_epoch
                findings.append(
                    {
                        "finding": "route_path_transition",
                        "severity": "info",
                        "detail": (
                            f"Route path switched from fallback to master "
                            f"{delay:.1f}s after restart (at wall_clock_ts="
                            f"{first_master:.1f})."
                        ),
                    }
                )
            else:
                findings.append(
                    {
                        "finding": "route_path_no_master",
                        "severity": "warning",
                        "detail": (
                            "No requests routed via master after restart — "
                            "Master may not have fully recovered or all requests "
                            "used fallback path."
                        ),
                    }
                )

    # 3. Check monitor inflight during recovery
    if monitor and restart_epoch is not None:
        recovery_end = restart_epoch + RECOVERY_WARM_S
        recovery_monitor = [
            m for m in monitor if restart_epoch <= m.get("ts", 0) <= recovery_end
        ]
        if recovery_monitor:
            non_zero_inflight = [
                m
                for m in recovery_monitor
                if m.get("scheduler_inflight") is not None
                and m.get("scheduler_inflight", 0) > 0
            ]
            if len(non_zero_inflight) > len(recovery_monitor) * 0.5:
                findings.append(
                    {
                        "finding": "inflight_persistent",
                        "severity": "medium",
                        "detail": (
                            f"Scheduler inflight remained non-zero in "
                            f"{len(non_zero_inflight)}/{len(recovery_monitor)} "
                            f"monitor samples during recovery — possible TTL "
                            f"cleanup lag for orphaned requests."
                        ),
                    }
                )

            # 4. Check GC pauses correlation
            gc_samples = [
                m for m in recovery_monitor if m.get("jvm_gc_pause_count") is not None
            ]
            if len(gc_samples) >= 2:
                gc_delta = gc_samples[-1].get("jvm_gc_pause_count", 0) - gc_samples[
                    0
                ].get("jvm_gc_pause_count", 0)
                gc_time_delta = gc_samples[-1].get(
                    "jvm_gc_pause_total_ms", 0
                ) - gc_samples[0].get("jvm_gc_pause_total_ms", 0)
                if gc_time_delta > 500:
                    findings.append(
                        {
                            "finding": "gc_pressure",
                            "severity": "medium",
                            "detail": (
                                f"JVM GC pauses during recovery: {gc_delta} events, "
                                f"{gc_time_delta:.0f}ms total — may contribute to "
                                f"TTFT spikes."
                            ),
                        }
                    )

    if not findings:
        findings.append(
            {
                "finding": "no_anomaly",
                "severity": "info",
                "detail": "No specific anomalies detected by the attribution tree.",
            }
        )

    return findings


# ---------------------------------------------------------------------------
# Markdown report generation
# ---------------------------------------------------------------------------


def fmt_ms(val: Any) -> str:
    """Format a millisecond value for table display."""
    if val is None or val == 0:
        return "-"
    return f"{val:.1f}"


def fmt_ratio(val: Any) -> str:
    if val is None:
        return "-"
    return f"{val:.2f}x"


def generate_markdown_report(
    phase_stats: Dict[str, Any],
    classification: Dict[str, Any],
    findings: List[Dict[str, Any]],
    timeline: List[Dict[str, Any]],
    args: argparse.Namespace,
    chart_generated: bool,
) -> str:
    """Build the Markdown report string."""
    lines: List[str] = []

    # Header
    lines.append("# TTFT Degradation Analysis Report")
    lines.append("")
    lines.append("## 1. Background")
    lines.append("")
    lines.append(
        "This report analyses TTFT time-series data around a Master kill-restart "
        "cycle to determine whether observed degradation is a transient spike or "
        "a sustained regression, and to attribute root causes."
    )
    lines.append("")

    # Test config
    lines.append("## 2. Test Configuration")
    lines.append("")
    lines.append(f"- **per_request.jsonl**: `{args.per_requests}`")
    if args.monitor:
        lines.append(f"- **monitor.jsonl**: `{args.monitor}`")
    if args.kill_epoch is not None:
        lines.append(f"- **kill_epoch**: `{args.kill_epoch:.1f}`")
    if args.restart_epoch is not None:
        lines.append(f"- **restart_epoch**: `{args.restart_epoch:.1f}`")
    if args.kill_epoch and args.restart_epoch:
        downtime = args.restart_epoch - args.kill_epoch
        lines.append(f"- **Master downtime**: `{downtime:.1f}s`")
    lines.append("")

    # Phase statistics table
    lines.append("## 3. Phase Statistics")
    lines.append("")
    lines.append(
        "| Phase | Count | Success Rate | TTFT p50 (ms) | TTFT p90 (ms) | "
        "TTFT p99 (ms) | TTFT max (ms) | Sched p50 (ms) | Sched p90 (ms) | "
        "Degradation |"
    )
    lines.append(
        "|-------|-------|-------------|---------------|---------------|"
        "---------------|---------------|---------------|---------------|"
        "-------------|"
    )
    for phase in PHASES:
        s = phase_stats.get(phase, {})
        if s.get("count", 0) == 0:
            lines.append(f"| {phase} | 0 | - | - | - | - | - | - | - | - |")
            continue
        route_str = ""
        rp_dist = s.get("route_path_distribution", {})
        if rp_dist:
            parts = [f"{k}:{v}" for k, v in sorted(rp_dist.items())]
            route_str = ", ".join(parts)
        lines.append(
            f"| {phase} | {s['count']} | {s['success_rate']:.1%} | "
            f"{fmt_ms(s.get('ttft_p50_ms'))} | {fmt_ms(s.get('ttft_p90_ms'))} | "
            f"{fmt_ms(s.get('ttft_p99_ms'))} | {fmt_ms(s.get('ttft_max_ms'))} | "
            f"{fmt_ms(s.get('schedule_p50_ms'))} | "
            f"{fmt_ms(s.get('schedule_p90_ms'))} | "
            f"{fmt_ratio(s.get('ttft_degradation_vs_baseline'))} |"
        )
    lines.append("")

    # Degradation pattern
    lines.append("## 4. Degradation Pattern Classification")
    lines.append("")
    pattern = classification.get("pattern", "unknown")
    emoji_map = {
        "spike": "Spike",
        "sustained": "Sustained",
        "none": "None",
        "unknown": "Unknown",
    }
    lines.append(f"**Pattern**: {emoji_map.get(pattern, pattern)}")
    lines.append("")
    lines.append(f"**Reasoning**: {classification.get('reason', '')}")
    lines.append("")
    if pattern == "spike":
        lines.append(
            "The degradation is transient — TTFT returns to near-baseline "
            "levels after the recovery period. Focus on reducing the spike "
            "amplitude and duration."
        )
    elif pattern == "sustained":
        lines.append(
            "The degradation persists into steady state — investigate "
            "systemic issues such as configuration changes, resource "
            "contention, or accumulated state."
        )
    elif pattern == "none":
        lines.append("No significant degradation detected.")
    lines.append("")

    # Attribution analysis
    lines.append("## 5. Attribution Analysis")
    lines.append("")
    if findings:
        for f in findings:
            severity_emoji = {
                "high": "**[HIGH]**",
                "medium": "**[MEDIUM]**",
                "warning": "**[WARN]**",
                "info": "**[INFO]**",
            }.get(f.get("severity", "info"), "**[INFO]**")
            lines.append(f"- {severity_emoji} **{f['finding']}**: {f['detail']}")
    else:
        lines.append("No attribution findings.")
    lines.append("")

    # Recommendations
    lines.append("## 6. Recommendations")
    lines.append("")
    if pattern == "spike":
        lines.append(
            "1. Consider pre-warming the Master process or reducing Spring Boot "
            "startup time to shorten the RECOVERY_EARLY window."
        )
        lines.append(
            "2. If schedule_backlog is detected, investigate queue draining "
            "strategy during fallback-to-master transition."
        )
        lines.append(
            "3. Monitor for orphaned requests that rely on TTL cleanup — "
            "consider explicit cancellation on Master restart."
        )
    elif pattern == "sustained":
        lines.append(
            "1. Investigate persistent configuration or state changes that "
            "may have been introduced during the restart."
        )
        lines.append(
            "2. Check for accumulated inflight requests or memory pressure "
            "that does not resolve after restart."
        )
        lines.append(
            "3. Compare baseline and stable-phase request distributions to "
            "identify workload shifts."
        )
    else:
        lines.append("1. No specific recommendations — system recovered normally.")
    lines.append("")

    # Chart
    lines.append("## 7. Timeline Chart")
    lines.append("")
    if chart_generated:
        lines.append("![TTFT Timeline](ttft_timeline.png)")
    else:
        lines.append(
            "Chart generation skipped (matplotlib not available). "
            "See `ttft_timeline.json` for raw time-series data."
        )
    lines.append("")

    # Timeline summary
    if timeline:
        lines.append("## 8. Second-Level Timeline Summary")
        lines.append("")
        lines.append(
            f"Total time buckets: **{len(timeline)}** | "
            f"Total requests in timeline: **{sum(b['request_count'] for b in timeline)}**"
        )
        lines.append("")

    lines.append("---")
    lines.append("*Generated by `analyze_ttft_degradation.py`*")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Optional matplotlib chart
# ---------------------------------------------------------------------------


def try_generate_chart(
    timeline: List[Dict[str, Any]],
    requests: List[dict],
    kill_epoch: Optional[float],
    restart_epoch: Optional[float],
    output_path: Path,
) -> bool:
    """Attempt to generate a PNG chart. Return False if matplotlib is missing."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return False

    if not requests:
        return False

    # Scatter plot of individual request TTFT
    xs = [r.get("wall_clock_ts", 0) for r in requests]
    ys = [r.get("ttft_ms", 0) for r in requests]
    colors = []
    for r in requests:
        rp = r.get("route_path", "unknown")
        if rp == "master":
            colors.append("tab:blue")
        elif rp == "fallback":
            colors.append("tab:orange")
        else:
            colors.append("tab:gray")

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.scatter(xs, ys, c=colors, s=8, alpha=0.5, label="Individual requests")

    # Moving average from timeline buckets
    if timeline:
        tx = [b["second"] for b in timeline]
        ty = [b["ttft_p50_ms"] for b in timeline]
        ax.plot(tx, ty, color="red", linewidth=1.5, label="TTFT p50 (1s bucket)")

    # Vertical lines for kill / restart
    if kill_epoch is not None:
        ax.axvline(
            x=kill_epoch, color="red", linestyle="--", linewidth=1.5, label="Kill"
        )
    if restart_epoch is not None:
        ax.axvline(
            x=restart_epoch,
            color="green",
            linestyle="--",
            linewidth=1.5,
            label="Restart",
        )

    ax.set_xlabel("Wall-clock time (epoch seconds)")
    ax.set_ylabel("TTFT (ms)")
    ax.set_title("TTFT Timeline — Master Kill-Restart Cycle")
    ax.legend(loc="upper right", fontsize=8)

    # Add phase shading
    if kill_epoch and restart_epoch:
        ax.axvspan(kill_epoch, restart_epoch, alpha=0.1, color="red")
        ax.axvspan(
            restart_epoch,
            restart_epoch + RECOVERY_EARLY_S,
            alpha=0.1,
            color="orange",
        )
        ax.axvspan(
            restart_epoch + RECOVERY_EARLY_S,
            restart_epoch + RECOVERY_WARM_S,
            alpha=0.05,
            color="yellow",
        )

    fig.tight_layout()
    fig.savefig(str(output_path), dpi=150)
    plt.close(fig)
    return True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyse TTFT degradation around a Master kill-restart cycle."
    )
    parser.add_argument(
        "--per-requests",
        required=True,
        help="Path to per_request.jsonl from flexlb_load_client.py",
    )
    parser.add_argument(
        "--monitor",
        default="",
        help="Path to monitor.jsonl from stability_monitor.py",
    )
    parser.add_argument(
        "--kill-epoch",
        type=float,
        default=None,
        help="Wall-clock epoch (seconds) when Master was killed",
    )
    parser.add_argument(
        "--restart-epoch",
        type=float,
        default=None,
        help="Wall-clock epoch (seconds) when Master restarted",
    )
    parser.add_argument(
        "--timestamps-json",
        default="",
        help="Path to timestamps.json (fallback for kill/restart epoch)",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Path for the Markdown report output",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    per_request_path = Path(args.per_requests)
    if not per_request_path.exists():
        print(f"ERROR: {per_request_path} not found", file=sys.stderr)
        sys.exit(1)

    # Load data
    requests = load_per_requests(per_request_path)
    if not requests:
        print("ERROR: no valid requests loaded", file=sys.stderr)
        sys.exit(1)

    monitor_path = Path(args.monitor) if args.monitor else Path()
    monitor = load_monitor(monitor_path)

    # Resolve kill/restart epochs
    kill_epoch = args.kill_epoch
    restart_epoch = args.restart_epoch
    if (kill_epoch is None or restart_epoch is None) and args.timestamps_json:
        ts_data = load_timestamps(Path(args.timestamps_json))
        if kill_epoch is None:
            kill_epoch = ts_data.get("kill_epoch")
        if restart_epoch is None:
            restart_epoch = ts_data.get("restart_epoch")

    if kill_epoch is None or restart_epoch is None:
        print(
            "WARNING: kill_epoch or restart_epoch not provided — "
            "all requests assigned to BASELINE phase.",
            file=sys.stderr,
        )

    # Annotate phases
    annotate_phases(requests, kill_epoch, restart_epoch)

    # Compute baseline p50 first (needed for degradation ratios)
    baseline_reqs = [r for r in requests if r.get("phase") == "BASELINE"]
    baseline_ok = [
        r for r in baseline_reqs if r.get("status") == "ok" and r.get("ttft_ms", 0) > 0
    ]
    baseline_p50 = (
        percentile([r["ttft_ms"] for r in baseline_ok], 50) if baseline_ok else None
    )

    # Compute phase statistics
    phase_stats = compute_phase_stats(requests, baseline_p50)

    # Compute second-level timeline
    timeline = compute_second_timeline(requests, kill_epoch, restart_epoch)

    # Classify degradation pattern
    classification = classify_degradation(phase_stats)

    # Run attribution decision tree
    findings = attribute_degradation(phase_stats, requests, monitor, restart_epoch)

    # Generate chart (optional)
    output_path = Path(args.output)
    output_dir = output_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    chart_path = output_dir / "ttft_timeline.png"
    chart_generated = try_generate_chart(
        timeline, requests, kill_epoch, restart_epoch, chart_path
    )

    # Generate Markdown report
    report_md = generate_markdown_report(
        phase_stats,
        classification,
        findings,
        timeline,
        args,
        chart_generated,
    )
    output_path.write_text(report_md, encoding="utf-8")

    # Generate JSON data file
    json_path = output_dir / "ttft_timeline.json"
    json_data = {
        "kill_epoch": kill_epoch,
        "restart_epoch": restart_epoch,
        "phase_stats": phase_stats,
        "classification": classification,
        "findings": findings,
        "second_level_timeline": timeline,
    }
    json_path.write_text(
        json.dumps(json_data, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(f"Report: {output_path}", file=sys.stderr)
    print(f"JSON data: {json_path}", file=sys.stderr)
    if chart_generated:
        print(f"Chart: {chart_path}", file=sys.stderr)
    else:
        print("Chart: skipped (matplotlib not available)", file=sys.stderr)


if __name__ == "__main__":
    main()
