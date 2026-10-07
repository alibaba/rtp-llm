"""Display frozen run bundles together, without running domain analyzers."""

import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import sys

from reporting import (
    bundle_path, compare_controls, details, links, load_analysis,
    read_bundle, run_meta, write_bundle,
)
from reporting.pairing import event_anchor, paired_overlay, shifted_panel


def _load(path):
    if not Path(path).is_dir():
        raise ValueError("comparison input must be a run bundle directory: " + str(path))
    directory = read_bundle(path)
    manifest = json.loads((directory / "manifest.json").read_text())
    if manifest.get("kind") != "run":
        raise ValueError("comparison input must be a run bundle: " + str(path))
    required = {"analysis.json", "report-spec.json", "report.html"}
    if not required <= manifest["files"].keys() or any(
        manifest["files"][name]["path"] != name for name in required
    ):
        raise ValueError("run bundle must verify analysis.json, report-spec.json and report.html")
    analysis = load_analysis(directory)
    spec = json.loads((directory / "report-spec.json").read_text())
    if not isinstance(analysis, dict) or not isinstance(spec, dict):
        raise ValueError("run analysis and report spec must be objects")
    return directory.resolve(), analysis, spec


def _controls(analysis, spec):
    meta = spec.get("run_meta") or {}
    # Missing metadata stays unknown; no case-specific reconstruction from evidence.
    return dict(
        configuration=meta.get("configuration"),
        workload=meta.get("workload"),
        environment=meta.get("environment"),
        criteria=analysis.get("criteria", (spec.get("meta") or {}).get("params")),
        statistic_sources=analysis.get("statistic_sources"),
    )


def _signature(panel):
    """Pair only charts with the same declared axes and series units."""
    return {
        key: panel.get(key)
        for key in ("type", "unit", "axes", "timeX", "overlay")
    } | {"series": sorted(
        (s["name"], str(s.get("axis")), str(s.get("unit")))
        for s in panel.get("series", [])
    )}


def compare(paths, output, *, alignment_event=None):
    if len(paths) < 2:
        raise ValueError("at least two run bundles are required")
    if alignment_event is not None and (
        not isinstance(alignment_event, str) or not alignment_event.strip()
    ):
        raise ValueError("alignment_event must be a nonempty event name")
    loaded = [_load(path) for path in paths]
    destination = bundle_path(output, "comparison", "runs").resolve()
    for directory, _, _ in loaded:
        if directory == destination or destination in directory.parents:
            raise ValueError("comparison output must not contain an input bundle")
    labels = [chr(65 + i) if i < 26 else f"run-{i + 1}" for i in range(len(loaded))]
    anchors = [event_anchor(spec.get("events", []), alignment_event) for _, _, spec in loaded]
    aligned = alignment_event is not None and all(t is not None for t in anchors)
    time_alignment = dict(
        event=alignment_event,
        status="ALIGNED" if aligned else "UNAVAILABLE" if alignment_event else "NOT_REQUESTED",
        observed_times=dict(zip(labels, anchors)),
        missing_or_ambiguous_runs=[label for label, t in zip(labels, anchors)
                                   if alignment_event and t is None],
    )
    caption = (f"按事件 {alignment_event} 对齐到 0 秒。" if aligned else
               f"对齐事件 {alignment_event} 缺失、重复或时间无效，保留各 run 原时间轴。"
               if alignment_event else "保留各 run 原时间轴；不自动选择统计窗口。")
    controls = [_controls(result, spec) for _, result, spec in loaded]
    comparisons = {
        label: compare_controls(controls[0], control, required=(
            "/configuration", "/workload", "/environment", "/criteria",
        )) for label, control in zip(labels[1:], controls[1:])
    }
    sources, panels, groups, kpis, sections = {}, [], {}, [], []
    bounds = []
    for label, (directory, result, spec), anchor in zip(labels, loaded, anchors):
        shift = anchor if aligned else None
        sources[label] = dict(
            path=str(directory),
            manifest_sha256=hashlib.sha256((directory / "manifest.json").read_bytes()).hexdigest(),
        )
        sections.extend([
            details(label + " · 冻结分析", result),
            details(label + " · 归档报告说明与缺采标注", dict(
                title=spec.get("title"), subtitle=spec.get("subtitle"),
                timeOriginLabel=spec.get("timeOriginLabel"), events=spec.get("events", []),
            )),
            links(label + " · 原始报告", [dict(
                label="打开已归档的单 run 报告",
                href=os.path.relpath(directory / "report.html", destination),
            )]),
        ])
        for section in copy.deepcopy(spec.get("sections", [])):
            section["title"] = label + " · " + section.get("title", "")
            if section["type"] == "links":
                for item in section["items"]:
                    item["href"] = os.path.relpath(directory / item["href"], destination)
            sections.append(section)
        for kpi in spec.get("kpis", []):
            item = copy.deepcopy(kpi)
            item["label"] = label + " · " + item["label"]
            kpis.append(item)
        axis = spec.get("timeAxis") or {}
        for value in (axis.get("min"), axis.get("max")):
            if type(value) in (int, float) and math.isfinite(value):
                bounds.append(value - (shift or 0))
        for original in spec.get("panels", []):
            panel = shifted_panel(original, shift)
            key = str(original["id"])
            panel["id"] = label + ":" + key
            panel["title"] = label + " · " + panel.get("title", key)
            panel["caption"] = panel.get("caption", "") + " " + caption
            panels.append(panel)
            groups.setdefault(key, []).append((label, panel, spec.get("timeOriginLabel")))
            if panel.get("timeX") or panel.get("overlay"):
                bounds.extend(p["x"] for s in panel.get("series", [])
                              for p in s.get("points", [])
                              if type(p.get("x")) in (int, float) and math.isfinite(p["x"]))
    overlays, pairing = [], []
    for key, group in groups.items():
        reasons = []
        if len(group) != len(loaded):
            reasons.append("部分 run 缺少此面板")
        first = group[0][1]
        if first.get("type", "line") != "line" or not (first.get("timeX") or first.get("overlay")):
            reasons.append("非时间曲线，保留独立面板")
        if any(_signature(p) != _signature(first) for _, p, _ in group):
            reasons.append("指标集合、单位或坐标轴不同")
        if not aligned and any(origin != group[0][2] for _, _, origin in group):
            reasons.append("时间原点说明不同")
        if reasons:
            pairing.append(dict(panel=key, status="SEPARATE", reasons=reasons))
            continue
        overlay = copy.deepcopy(first)
        overlay.update(id="overlay:" + key, title="合图 · " + key, overlay=True)
        # The shared multi-curve renderer preserves line styles and presets.
        overlay["axes"] = overlay.get("axes") or {"y": {"title": overlay.get("unit", "")}}
        overlay["series"], overlay["presets"] = paired_overlay(
            [p for _, p, _ in group], labels=[label for label, _, _ in group],
        )
        for series in overlay["series"]:
            series.setdefault("unit", overlay.get("unit", ""))
        for index, (label, _, _) in enumerate(group):
            if index >= 2:
                for series in overlay["series"]:
                    if series["name"].startswith(label + " · "):
                        series["dash"] = [2 * index, 3, 2, 3]
        overlays.append(overlay)
        pairing.append(dict(panel=key, status="PAIRED"))
    result = dict(
        purpose="OBSERVATION_ONLY", sources=sources,
        runs={label: analysis for label, (_, analysis, _) in zip(labels, loaded)},
        controls=dict(reference=labels[0], values=dict(zip(labels, controls)), comparisons=comparisons),
        time_alignment=time_alignment, pairing=pairing,
    )
    spec = dict(
        title="运行报告对照", subtitle="展示归档时冻结的数据与各 run 独立结论。",
        timeOriginLabel=caption, timeAxis=dict(min=min(bounds, default=0), max=max(bounds, default=1)),
        events=[dict(name=alignment_event, t=0)] if aligned else [],
        panels=overlays + panels, kpis=kpis,
        sections=[details("控制变量（以 A 为参照）", result["controls"]),
                  details("时间轴对齐", time_alignment), details("图表配对", pairing), *sections],
    )
    write_bundle(output, "comparison", "runs", result, spec,
                 meta=run_meta(dict(id="runs", kind="comparison"),
                               runs={label: s.get("run_meta") for label, (_, _, s) in zip(labels, loaded)},
                               evidence=sources), producer="report-comparison")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+", type=Path, help="two or more frozen run bundle directories")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--alignment-event", help="explicit event name used as the common time origin")
    args = parser.parse_args(argv)
    try:
        compare(args.runs, args.output, alignment_event=args.alignment_event)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(bundle_path(args.output, "comparison", "runs") / "report.html")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
