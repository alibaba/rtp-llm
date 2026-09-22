"""Descriptive A/B comparison for valid Prometheus stress archives.

This deliberately makes no performance-gate claim. The older aggregate format
has a separate, calibrated regression gate in compare_ab.py.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

from reporting import run_meta, write_bundle


class MonitorComparisonError(ValueError):
    pass


def _aggregate_path(path):
    path = Path(path)
    return path / "aggregate.json" if path.is_dir() else path


def _read(path):
    aggregate_path = _aggregate_path(path)
    try:
        data = json.loads(aggregate_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise MonitorComparisonError(f"{aggregate_path}: {exc}") from exc
    if not isinstance(data, dict):
        raise MonitorComparisonError(f"{aggregate_path}: aggregate must be an object")
    return aggregate_path, data


def is_monitor_run(path):
    try:
        return _read(path)[1].get("monitor_backend") == "prometheus"
    except MonitorComparisonError:
        return False


def _controls(data):
    meta = data.get("run_meta") or {}
    workload = meta.get("workload") or {}
    config = meta.get("configuration") or {}
    env = dict(config.get("client_env.json") or {})
    for transient in ("OUTPUT_DIR", "TRACE_FILE", "START_AT_EPOCH_MS", "SHARD_INDEX", "API_KEY"):
        env.pop(transient, None)
    inputs = workload.get("inputs") or []
    hashes = [item.get("sha256") for item in inputs if isinstance(item, dict)]
    if not hashes or any(not value for value in hashes):
        raise MonitorComparisonError("traffic input SHA256 is missing")
    for name in ("master_config.json", "mode_plan.json"):
        if name not in config:
            raise MonitorComparisonError(f"{name} is missing from run metadata")
    if not env:
        raise MonitorComparisonError("client environment is missing from run metadata")
    return dict(traffic_sha256=hashes, master_config=config["master_config.json"],
                mode_plan=config["mode_plan.json"], client_env=env)


def _mean_in_window(points, lo, hi):
    values = []
    for point in points:
        if not isinstance(point, (list, tuple)) or len(point) != 2:
            continue
        t, value = point
        if isinstance(t, (int, float)) and isinstance(value, (int, float)) \
                and math.isfinite(t) and math.isfinite(value) and lo <= t <= hi:
            values.append(float(value))
    return sum(values) / len(values) if values else None


def compare_monitor_runs(args):
    try:
        if args.archive and args.out == "-":
            raise MonitorComparisonError("--archive requires a file --out")
        if args.noise_floor != .02:
            raise MonitorComparisonError("--noise-floor applies only to legacy stress gates")
        paths, runs = zip(*(_read(path) for path in (args.run_a, args.run_b)))
        for path, data in zip(paths, runs):
            if data.get("monitor_backend") != "prometheus":
                raise MonitorComparisonError(f"{path}: not a Prometheus archive")
            if data.get("summary", {}).get("test_valid") is not True \
                    or data.get("errors") or data.get("collection_gaps"):
                raise MonitorComparisonError(
                    f"{path}: invalid collection; errors={data.get('errors')}; "
                    f"gaps={data.get('collection_gaps')}")
        controls = [_controls(data) for data in runs]
        if controls[0] != controls[1]:
            differing = [key for key in controls[0] if controls[0][key] != controls[1][key]]
            raise MonitorComparisonError("experiment controls differ: " + ", ".join(differing))
        duration = float(controls[0]["client_env"].get("DURATION_S", 0))
        lo = args.steady_lo if args.steady_lo is not None else duration * .25
        hi = args.steady_hi if args.steady_hi is not None else duration * .92
        if not (math.isfinite(lo) and math.isfinite(hi) and 0 <= lo < hi):
            raise MonitorComparisonError("invalid steady window")
        series_a, series_b = (run.get("series") or {} for run in runs)
        common = sorted(set(series_a) & set(series_b))
        if not common:
            raise MonitorComparisonError("no common monitor series")
        sources_a, sources_b = (run.get("statistic_sources") or {} for run in runs)
        for key in common:
            if (sources_a.get(key) or {}).get("promql") != (sources_b.get(key) or {}).get("promql"):
                raise MonitorComparisonError(f"PromQL differs for {key}")
        changes = []
        for key in common:
            a, b = (_mean_in_window(series[key], lo, hi) for series in (series_a, series_b))
            changes.append(dict(metric=key, baseline=a, candidate=b,
                                delta=None if a is None or b is None else b - a,
                                relative_pct=None if a in (None, 0) or b is None else (b / a - 1) * 100))
        if not any(row["baseline"] is not None and row["candidate"] is not None for row in changes):
            raise MonitorComparisonError("no paired samples in steady window")
        result = dict(schema_version=1, verdict="DESCRIPTIVE_ONLY",
                      baseline=str(paths[0].resolve()), candidate=str(paths[1].resolve()),
                      window=dict(lo_s=lo, hi_s=hi), controls=controls[0],
                      baseline_only=sorted(set(series_a) - set(series_b)),
                      candidate_only=sorted(set(series_b) - set(series_a)), changes=changes)
        if args.out != "-":
            out = Path(args.out)
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
            print(f"JSON summary -> {out}")
        if args.html or args.archive:
            root = Path.cwd() if args.out == "-" else Path(args.out).resolve().parent
            panels = []
            for index, key in enumerate(common):
                if changes[index]["baseline"] is None or changes[index]["candidate"] is None:
                    continue
                points = [series_a[key], series_b[key]]
                axis = sorted({float(t) for rows in points for t, _ in rows if lo <= t <= hi})
                if not axis:
                    continue
                panels.append(dict(id=f"metric-{index}", title=key, type="line", timeX=True,
                                   x=[str(t) for t in axis], xNums=axis, unit="value",
                                   series=[dict(name=label, data=[dict(rows).get(t) for t in axis])
                                           for label, rows in zip(("baseline", "candidate"), points)]))
            spec = dict(run_id="stress-monitor-ab", title="Stress monitor A/B",
                        subtitle="Descriptive comparison; no performance gate verdict",
                        timeAxis=dict(min=lo, max=hi), panels=panels)
            bundle = write_bundle(root, "comparison", "stress-monitor-ab", result, spec,
                                  meta=run_meta(dict(id="stress-monitor-ab", kind="comparison"),
                                                configuration=controls[0], evidence=[str(path) for path in paths]),
                                  producer="stress-monitor-ab")
            print(f"HTML report -> {bundle / 'report.html'}")
            if args.archive:
                from artifacts.archive import create_archive
                create_archive(Path(args.archive),
                               {"run_a": paths[0].parent, "run_b": paths[1].parent,
                                "comparison": bundle, "summary": Path(args.out)},
                               kind="stress-monitor-ab")
        print(f"DESCRIPTIVE_ONLY: {len(common)} common monitor series; no regression verdict")
        return 0
    except (MonitorComparisonError, TypeError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
