#!/usr/bin/env python3
"""Compare two exported KMonitor experiments without pretending they are request traces.

The collector boundary is a versioned JSON export, so private query credentials
remain outside the mock framework. No production regression gate is inferred.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

from experiment_archive import create_archive

STRESS_DIR = Path(__file__).with_name("stress")
sys.path.insert(0, str(STRESS_DIR))
from canvas_report_render_html import render

REQUIRED_PROVENANCE = (
    "deployment", "hippo_app", "traffic_fingerprint", "master_mode",
    "fetch_output_stream", "prefill_engines", "decode_engines",
    "start_ms", "end_ms", "granularity_ms", "collection_source",
)
MATCHING_CONDITIONS = (
    "traffic_fingerprint", "master_mode", "fetch_output_stream",
    "prefill_engines", "decode_engines", "granularity_ms",
)
SERIES_FIELDS = ("metric", "role", "unit", "spatial_aggregation", "temporal_aggregation")


class Incomparable(ValueError):
    pass


def load_export(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or data.get("schema_version") != 1 or not isinstance(data.get("series"), dict):
        raise Incomparable(f"{path}: unsupported export schema")
    provenance = data.get("provenance") or {}
    if not isinstance(provenance, dict):
        raise Incomparable(f"{path}: invalid provenance")
    missing = [key for key in REQUIRED_PROVENANCE if provenance.get(key) is None]
    if missing:
        raise Incomparable(f"{path}: missing provenance {missing}")
    if (any(type(provenance[key]) is not int for key in
            ("start_ms", "end_ms", "granularity_ms"))
            or provenance["end_ms"] <= provenance["start_ms"]
            or provenance["granularity_ms"] <= 0):
        raise Incomparable(f"{path}: invalid time range/granularity")
    for alias, series in data["series"].items():
        if not isinstance(series, dict) or any(series.get(key) is None for key in SERIES_FIELDS):
            raise Incomparable(f"{path}: {alias} lacks metric semantics")
        points = series.get("points")
        if not isinstance(points, list):
            raise Incomparable(f"{path}: {alias} points must be a list")
        seen = set()
        for point in points:
            if not isinstance(point, dict):
                raise Incomparable(f"{path}: {alias} has invalid samples")
            t, value = point.get("t_ms"), point.get("value")
            if (type(t) is not int or type(value) not in (int, float)
                    or not math.isfinite(value) or t in seen
                    or t < provenance["start_ms"] or t > provenance["end_ms"]):
                raise Incomparable(f"{path}: {alias} has invalid/duplicate samples")
            seen.add(t)
    return data


def compare_exports(a: dict, b: dict, *, steady_lo_ms: int, steady_hi_ms: int,
                    min_common: int = 3) -> tuple[dict, dict]:
    pa, pb = a["provenance"], b["provenance"]
    mismatches = [key for key in MATCHING_CONDITIONS if pa[key] != pb[key]]
    if mismatches:
        raise Incomparable(f"experiment conditions differ: {mismatches}")
    if not (max(pa["start_ms"], pb["start_ms"]) <= steady_lo_ms
            < steady_hi_ms <= min(pa["end_ms"], pb["end_ms"])):
        raise Incomparable("steady window lies outside the common deployment window")
    common_aliases = sorted(set(a["series"]) & set(b["series"]))
    if not common_aliases:
        raise Incomparable("no common metric aliases")
    panels, rows = [], []
    t0 = max(pa["start_ms"], pb["start_ms"])
    for alias in common_aliases:
        sa, sb = a["series"][alias], b["series"][alias]
        if any(sa[key] != sb[key] for key in SERIES_FIELDS):
            raise Incomparable(f"{alias}: metric/role/unit/aggregation mismatch")
        series_maps = [
            {p["t_ms"]: p["value"] for p in side["points"]
             if steady_lo_ms <= p["t_ms"] <= steady_hi_ms}
            for side in (sa, sb)
        ]
        times = sorted(set(series_maps[0]) | set(series_maps[1]))
        common = sorted(set(series_maps[0]) & set(series_maps[1]))
        if len(common) < min_common:
            raise Incomparable(f"{alias}: only {len(common)} paired samples; need {min_common}")
        mean_a = sum(series_maps[0][t] for t in common) / len(common)
        mean_b = sum(series_maps[1][t] for t in common) / len(common)
        rows.append({
            "alias": alias, "metric": sa["metric"], "role": sa["role"],
            "unit": sa["unit"], "spatial_aggregation": sa["spatial_aggregation"],
            "temporal_aggregation": sa["temporal_aggregation"],
            "paired_samples": len(common), "a_samples": len(series_maps[0]),
            "b_samples": len(series_maps[1]), "a_mean": mean_a, "b_mean": mean_b,
            "delta_mean": mean_b - mean_a,
            "relative_delta_pct": (mean_b - mean_a) / mean_a * 100 if mean_a else None,
        })
        x = [(t - t0) / 1000 for t in times]
        panels.append({
            "id": f"remote_{len(panels)}", "title": f"{alias} · {sa['role']}",
            "caption": (f"{sa['metric']} · {sa['spatial_aggregation']} / "
                        f"{sa['temporal_aggregation']}；仅共同采样点参与均值"),
            "timeX": True, "type": "line", "x": [str(t) for t in x],
            "xNums": x, "unit": sa["unit"],
            "series": [
                {"name": "A · real", "color": "#1677ff",
                 "data": [series_maps[0].get(t) for t in times]},
                {"name": "B · mock", "color": "#f5222d",
                 "data": [series_maps[1].get(t) for t in times]},
            ],
        })
    report = {
        "schema_version": 1, "classification": "descriptive_only",
        "reason": "KMonitor aggregates cannot establish request-level equivalence",
        "conditions": {key: pa[key] for key in MATCHING_CONDITIONS},
        "window_ms": {"lo": steady_lo_ms, "hi": steady_hi_ms},
        "a": pa, "b": pb, "metrics": rows,
    }
    chart = {
        "run_id": "remote-compare", "title": "远端 real / mock 对比",
        "subtitle": f"A: {pa['deployment']} · B: {pb['deployment']}",
        "meta": {"sampling": "KMonitor 导出值；曲线空窗不补零；结果仅作描述性比较"},
        "timeAxis": {"min": 0, "max": (steady_hi_ms - t0) / 1000},
        "panels": panels,
    }
    return report, chart


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--real", type=Path, required=True)
    parser.add_argument("--mock", type=Path, required=True)
    parser.add_argument("--steady-lo-ms", type=int, required=True)
    parser.add_argument("--steady-hi-ms", type=int, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--archive", type=Path)
    args = parser.parse_args(argv)
    try:
        a, b = load_export(args.real), load_export(args.mock)
        report, chart = compare_exports(a, b, steady_lo_ms=args.steady_lo_ms,
                                        steady_hi_ms=args.steady_hi_ms)
    except (Incomparable, OSError, ValueError) as exc:
        print(f"INCOMPARABLE: {exc}", file=sys.stderr)
        return 2
    args.out_dir.mkdir(parents=True, exist_ok=True)
    result = args.out_dir / "remote_comparison.json"
    page = args.out_dir / "remote_comparison.html"
    result.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    page.write_text(render(chart), encoding="utf-8")
    if args.archive:
        create_archive(args.archive, {"real": args.real, "mock": args.mock,
                                      "comparison": args.out_dir}, kind="ab",
                       metadata={"classification": "descriptive_only"})
    print(f"comparison={result} chart={page}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
