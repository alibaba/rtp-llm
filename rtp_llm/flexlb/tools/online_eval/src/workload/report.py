"""Presentation of already analyzed workload evidence; no status mutation."""

import copy
import hashlib
import os
from pathlib import Path
from reporting import (
    bundle_path,
    details,
    links,
    table,
    run_meta,
    write_bundle,
)


def build_spec(payload, directory):
    series = payload["series"]
    panels = [
        dict(
            id="metric-" + hashlib.sha256(key.encode()).hexdigest(),
            title=key,
            caption="Seconds since workload start; gaps are not zero.",
            type="line",
            timeX=True,
            x=[str(p[0]) for p in points],
            xNums=[p[0] for p in points],
            series=[dict(name=key, data=[p[1] for p in points], color="#2563eb")],
        )
        for key, points in series.items()
    ]
    target = bundle_path(directory, "run", payload["id"])
    items = []
    for aggregate in payload["workload"].get("stress_aggregates", []):
        if aggregate["status"] != "GENERATED":
            continue
        reports = [("Environment " + aggregate["env_epoch"], aggregate["report"])]
        reports += [
            (name + " metrics", entry["report"])
            for name, entry in aggregate.get("master_aggregates", {}).items()
        ]
        items += [
            dict(label=title, href=os.path.relpath(path, target))
            for title, path in reports
        ]
    gates = payload.get("gate_reports")
    if gates is None:
        gates = [payload["gate_report"]] if payload.get("gate_report") else []
    for gate in gates:
        items.append(
            dict(
                label="Gate · " + Path(gate).parent.name,
                href=os.path.relpath(gate, target),
            )
        )
    return dict(
        run_id=payload["id"],
        title=payload["id"],
        timeOriginLabel="t=0 = workload 运行开始",
        subtitle="Independent checks and continuous telemetry",
        kpis=[
            dict(label="Execution", value=payload["status"]),
            dict(label="Validity", value=payload["workload"]["runtime_validity"]),
            dict(label="Performance", value="NOT_EVALUATED"),
        ],
        panels=panels,
        timeAxis=dict(
            min=0, max=max((p[0] for v in series.values() for p in v), default=1) or 1
        ),
        sections=[
            table(
                "Independent checks",
                ["Stage / check", "Status", "Evidence"],
                [
                    [r["stage"] + "/" + r["id"], r["status"], r]
                    for r in payload["checks"]
                ],
            ),
            details("Playback iterations", payload["iterations"]),
            details("Traffic semantics", payload["traffic_manifests"]),
            links("Statistical reports", items),
        ],
    )


def write_report(directory, analysis):
    payload = copy.deepcopy(analysis)
    target = bundle_path(directory, "run", payload["id"])
    for source in payload.get("request_sources", []):
        source["path"] = os.path.relpath(source["path"], target)
    meta = run_meta(
        dict(id=payload["id"], kind="run"),
        implementation=payload["implementation"],
        workload=payload["traffic_manifests"],
        configuration=dict(
            declared=payload["configuration"],
            sha256=payload["configuration_sha256"],
            runtime=payload["workload"].get("runtime_configuration"),
        ),
        clock=payload["clock_anchor"],
        evidence=payload["request_sources"],
    )
    return write_bundle(
        directory,
        "run",
        payload["id"],
        payload,
        build_spec(analysis, directory),
        meta=meta,
        producer="workload",
    )
