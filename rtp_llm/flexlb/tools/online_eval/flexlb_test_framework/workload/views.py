"""Offline scenario views over existing reports; never acquire fresh telemetry.

Timeline comparisons translate real seconds. Parameter sweeps allow only named
configuration leaves to differ and expose feasible candidates, not an optimum.
"""

import argparse
import copy
import hashlib
import json
import math
import statistics
from pathlib import Path

from ..scenario.loader import load_document


def digest(value):
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def pointer(document, path):
    if not isinstance(path, str) or not path.startswith("/"):
        raise ValueError("configuration paths must be JSON pointers")
    value = document
    for part in path[1:].split("/"):
        key = part.replace("~1", "/").replace("~0", "~")
        value = value[int(key)] if isinstance(value, list) else value[key]
    return value


def masked(document, paths):
    result = copy.deepcopy(document)
    for path in paths:
        parts = path[1:].split("/")
        parent = result
        for key in parts[:-1]:
            key = key.replace("~1", "/").replace("~0", "~")
            parent = parent[int(key)] if isinstance(parent, list) else parent[key]
        key = parts[-1].replace("~1", "/").replace("~0", "~")
        value = pointer(document, path)
        if isinstance(value, (dict, list)):
            raise ValueError("sweep axes must name scalar configuration leaves")
        parent[int(key) if isinstance(parent, list) else key] = {"varied": path}
    return result


def boundary(report, spec):
    if set(spec) != {"stage", "edge"} or spec["edge"] not in {"start", "end"}:
        raise ValueError("boundary requires stage and start/end edge")
    stages = [
        s
        for s in report["stages"]
        if s["id"] == spec["stage"] and s["status"] != "BLOCKED"
    ]
    if len(stages) != 1:
        raise ValueError("missing or ambiguous stage boundary: " + spec["stage"])
    value = stages[0].get("started_s" if spec["edge"] == "start" else "finished_s")
    if value is None:
        raise ValueError("stage boundary not recorded")
    return value - report["clock_anchor"]["monotonic_s"]


def metric(report, selection, lower, upper):
    if "requests" in selection:
        from .window_statistics import measure

        return measure(report, selection, lower, upper)
    if set(selection) != {"series", "reducer"} or selection["reducer"] not in {
        "mean",
        "min",
        "max",
        "delta",
    }:
        raise ValueError("metric requires an existing series and supported reducer")
    key = selection["series"]
    values = [v for t, v in report["series"].get(key, []) if lower <= t < upper]
    source = report.get("statistic_sources", {}).get(key)
    result = dict(
        selection=selection,
        source=source or {"series": key},
        window=[lower, upper],
        samples=len(values),
        value=None,
    )
    if report["workload"]["runtime_validity"] != "VALID":
        return dict(result, status="INVALID_EVIDENCE")
    if not values or any(
        type(v) not in (int, float) or not math.isfinite(v) for v in values
    ):
        return dict(result, status="MISSING_DATA")
    reducer = selection["reducer"]
    if reducer == "delta":
        if len(values) < 2 or any(b < a for a, b in zip(values, values[1:])):
            return dict(result, status="COUNTER_RESET_OR_INSUFFICIENT_DATA")
        value = values[-1] - values[0]
    else:
        value = {"mean": statistics.fmean, "min": min, "max": max}[reducer](values)
    return dict(
        result,
        status="AVAILABLE",
        value=value,
        semantics="reduction of sampled series; per-second percentiles are not pooled request percentiles",
    )


def timeline(reports, spec):
    if set(spec) != {"kind", "title", "align", "events", "panels"}:
        raise ValueError("invalid timeline view fields")
    if (
        len({r["id"] for r in reports}) != 1
        or len({r.get("configuration_sha256") for r in reports}) != 1
        or not reports[0].get("configuration_sha256")
    ):
        raise ValueError(
            "version timeline requires the same instance and configuration"
        )
    if len({digest(r["workload"].get("runtime_configuration")) for r in reports}) != 1:
        raise ValueError("runtime configuration changed")
    runs = []
    for report in reports:
        try:
            offset = boundary(report, spec["align"])
        except ValueError:
            runs.append(
                dict(
                    identity=report.get("provenance"),
                    validity="MISSING_ALIGNMENT",
                    events=[],
                    panels=[
                        dict(
                            title=p["title"],
                            series={k: [] for k in p["metrics"]},
                            sources={},
                        )
                        for p in spec["panels"]
                    ],
                )
            )
            continue
        events = []
        for event in spec["events"]:
            if set(event) != {"stage", "edge", "label"}:
                raise ValueError("invalid timeline event")
            try:
                events.append(
                    dict(
                        label=event["label"],
                        t=boundary(report, {k: event[k] for k in ("stage", "edge")})
                        - offset,
                        status="RECORDED",
                    )
                )
            except ValueError:
                events.append(
                    dict(label=event["label"], t=None, status="MISSING_EVENT")
                )
        panels = []
        for panel in spec["panels"]:
            if set(panel) != {"title", "metrics"}:
                raise ValueError("invalid timeline panel")
            series = {
                key: [[t - offset, v] for t, v in report["series"].get(key, [])]
                for key in panel["metrics"]
            }
            panels.append(
                dict(
                    title=panel["title"],
                    series=series,
                    sources={
                        key: report.get("statistic_sources", {}).get(
                            key, {"series": key}
                        )
                        for key in series
                    },
                )
            )
        runs.append(
            dict(
                identity=report.get("provenance"),
                validity=report["workload"]["runtime_validity"],
                events=events,
                panels=panels,
            )
        )
    return dict(
        kind="timeline",
        title=spec["title"],
        runs=runs,
        time_semantics="seconds from recorded boundary; no duration normalization",
        verdict="DESCRIPTIVE_ONLY",
    )


def sweep(reports, spec):
    fields = {
        "kind",
        "title",
        "vary",
        "window",
        "metrics",
        "x",
        "y",
        "color",
        "facet",
        "size",
        "constraints",
    }
    if (
        set(spec) != fields
        or not spec["vary"]
        or len(set(spec["vary"])) != len(spec["vary"])
    ):
        raise ValueError("invalid parameter sweep specification")
    if spec["color"] not in spec["vary"] or spec["facet"] not in spec["vary"]:
        raise ValueError("color and facet must be declared parameter axes")
    points, fixed = [], None
    programs = {r.get("implementation", {}).get("sha256") for r in reports}
    if len(programs) != 1:
        raise ValueError("case program changed across parameter sweep")
    runtimes = {digest(r["workload"].get("runtime_configuration")) for r in reports}
    if len(runtimes) != 1:
        raise ValueError("runtime configuration differs across sweep")
    for report in reports:
        config = report.get("configuration")
        if config is None or digest(config) != report.get("configuration_sha256"):
            raise ValueError(
                "sweep needs the exact archived configuration and its hash"
            )
        invariant = digest(masked(config, spec["vary"]))
        if fixed is not None and fixed != invariant:
            raise ValueError("configuration differs outside declared sweep axes")
        fixed = invariant
        try:
            lo, hi = (boundary(report, spec["window"][k]) for k in ("start", "end"))
            if hi <= lo:
                raise ValueError("empty or reversed sweep window")
        except ValueError:
            lo = hi = None
        measures = {
            name: (
                metric(report, selection, lo, hi)
                if lo is not None
                else dict(
                    status="MISSING_WINDOW",
                    value=None,
                    selection=selection,
                    window=None,
                    samples=0,
                )
            )
            for name, selection in spec["metrics"].items()
        }
        for name in (spec["x"], spec["y"], spec["size"]):
            if name not in measures:
                raise ValueError("plot metric is not defined")
        checks = []
        for c in spec["constraints"]:
            if (
                set(c) != {"metric", "op", "value"}
                or c["op"] not in {"le", "ge"}
                or type(c["value"]) not in (int, float)
                or not math.isfinite(c["value"])
            ):
                raise ValueError("invalid feasibility constraint")
            observed = measures[c["metric"]]
            passed = (
                None
                if observed["status"] != "AVAILABLE"
                else (
                    observed["value"] <= c["value"]
                    if c["op"] == "le"
                    else observed["value"] >= c["value"]
                )
            )
            checks.append(dict(**c, observed=observed["value"], passed=passed))
        available = all(m["status"] == "AVAILABLE" for m in measures.values())
        feasible = available and bool(checks) and all(c["passed"] for c in checks)
        points.append(
            dict(
                identity=report.get("provenance"),
                parameters={p: pointer(config, p) for p in spec["vary"]},
                metrics=measures,
                checks=checks,
                status=(
                    "MISSING_OR_INVALID"
                    if not available
                    else (
                        "FEASIBLE"
                        if feasible
                        else "INFEASIBLE" if checks else "UNCONSTRAINED"
                    )
                ),
            )
        )
    return dict(
        kind="sweep",
        title=spec["title"],
        specification=spec,
        points=points,
        verdict="CANDIDATE_SET_ONLY",
        fixed_configuration_sha256=fixed,
    )


def build(reports, spec):
    if not reports:
        raise ValueError("at least one report required")
    return {"timeline": timeline, "sweep": sweep}[spec["kind"]](reports, spec)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--view", type=Path, required=True)
    parser.add_argument("--reports", type=Path, nargs="+", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    reports = []
    for path in args.reports:
        raw = path.read_bytes()
        report = json.loads(raw)
        report["provenance"] = dict(
            path=str(path.resolve()),
            sha256=hashlib.sha256(raw).hexdigest(),
            implementation=report.get("implementation"),
        )
        report["_request_rows"] = []
        report["request_source_state"] = (
            "VERIFIED" if report.get("request_sources") else "MISSING"
        )
        for source in report.get("request_sources", []):
            try:
                data = (path.parent / source["path"]).read_bytes()
                if hashlib.sha256(data).hexdigest() != source["sha256"]:
                    raise ValueError("request evidence checksum mismatch")
                report["_request_rows"].extend(
                    json.loads(line)
                    for line in data.decode().splitlines()
                    if line.strip()
                )
            except (OSError, ValueError):
                report["request_source_state"] = "INVALID"
        reports.append(report)
    result = build(reports, load_document(args.view))
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "view.json").write_text(json.dumps(result, indent=2, allow_nan=False))
    from .view_render import render

    (args.out / "view.html").write_text(render(result))


if __name__ == "__main__":
    main()
