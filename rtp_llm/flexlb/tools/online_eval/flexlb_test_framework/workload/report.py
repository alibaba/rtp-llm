"""Adapt workload evidence to the existing stress chart renderer."""

import hashlib
import html
import json
import math
from pathlib import Path

from online_eval.metrics import parse_prometheus_samples
from stress.canvas_report_render_html import render


def read_series(root, epoch_s):
    series = {}
    for path in sorted(Path(root).glob("telemetry/*/*.prom")):
        timestamp = None
        for line in path.read_text().splitlines():
            if line.startswith("# ts="):
                timestamp = int(line[5:]) / 1000 - epoch_s
                continue
            if timestamp is None:
                continue
            for name, labels, value in parse_prometheus_samples(line, ""):
                if not math.isfinite(value):
                    continue
                key = "/".join(
                    (
                        path.parent.name,
                        path.stem,
                        name,
                        json.dumps(labels, sort_keys=True),
                    )
                )
                series.setdefault(key, []).append([timestamp, value])
    return series


def write_report(directory, result, evidence):
    directory = Path(directory)
    series = read_series(directory, evidence["clock_anchor"]["epoch_s"])
    checks = [
        dict(stage=s["id"], **check) for s in result["stages"] for check in s["checks"]
    ]
    missing = []
    if result.get("workload", {}).get("capture_metrics"):
        for source in ("mock", "master"):
            if not any(key.split("/")[1].startswith(source) for key in series):
                missing.append(source)
    if missing:
        result["workload"]["runtime_validity"] = "INVALID"
        result["workload"]["missing_telemetry"] = missing
        result["workload"]["prior_status"] = result["status"]
        result["status"] = "ERROR"
        result["error"] = result["error"] or "missing workload telemetry: " + ", ".join(
            missing
        )
    payload = dict(
        schema_version=1,
        id=result["id"],
        status=result["status"],
        workload=result["workload"],
        checks=checks,
        stages=result["stages"],
        clock_anchor=evidence["clock_anchor"],
        phases=evidence["phases"],
        series=series,
        configuration_sha256=result.get("implementation", {}).get(
            "configuration_sha256"
        ),
    )
    (directory / "workload-report.json").write_text(
        json.dumps(payload, indent=2, allow_nan=False) + "\n"
    )
    panels = []
    for key, points in series.items():
        panels.append(
            dict(
                id="metric-" + hashlib.sha256(key.encode()).hexdigest(),
                title=key,
                caption="Raw samples, seconds since workload start; gaps are not zero.",
                type="line",
                timeX=True,
                x=[str(p[0]) for p in points],
                xNums=[p[0] for p in points],
                series=[dict(name=key, data=[p[1] for p in points], color="#2563eb")],
            )
        )
    spec = dict(
        run_id=result["id"],
        title=result["id"],
        subtitle="Workload evidence: independent checks and continuous stress telemetry",
        kpis=[
            dict(label="Execution", value=result["status"]),
            dict(label="Validity", value=result["workload"]["runtime_validity"]),
            dict(label="Performance", value="NOT_EVALUATED"),
        ],
        panels=panels,
    )
    table = "<h2>Independent checks</h2><table><tr><th>Stage / check</th><th>Status</th><th>Evidence</th></tr>"
    for row in checks:
        table += (
            "<tr><td>"
            + html.escape(row["stage"] + "/" + row["id"])
            + "</td><td>"
            + html.escape(row["status"])
            + "</td><td><pre>"
            + html.escape(json.dumps(row, ensure_ascii=False, indent=2))
            + "</pre></td></tr>"
        )
    table += "</table>"
    (directory / "workload-report.html").write_text(
        render(spec).replace("</body>", table + "</body>")
    )
    result["workload"]["report"] = str(directory / "workload-report.html")
