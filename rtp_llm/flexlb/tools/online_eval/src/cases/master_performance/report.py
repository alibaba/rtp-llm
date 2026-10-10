"""Present an explicit performance verdict using already published metric data."""

from pathlib import Path

from reporting import write_bundle, run_meta
from reporting.run_context import title
from reporting.view_sections import view_details, view_table


def write_report(directory, evidence, result, telemetry_directory=None):
    from reporting.view_config import view

    presentation = view("master_performance.yaml")
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    p = evidence.get("provenance", {})
    from cases.master_performance.panels import panel, report_panels

    metric_directory = Path(telemetry_directory or directory)
    chart, monitoring = panel(metric_directory, evidence, result, presentation)
    panels = report_panels(chart["series"], evidence.get("criteria", {}), presentation)
    spec = dict(
        run_id=p["instance"],
        title=title(p["instance"]),
        subtitle=presentation["report"]["subtitle"].format(verdict=result["verdict"]),
        timeAxis=dict(min=0, max=evidence.get("criteria", {}).get("measure_s", 1)),
        panels=panels,
        sections=[
            view_table(
                presentation, "checks",
                [
                    [
                        x["metric"],
                        x["actual"],
                        str(x["direction"]) + " " + str(x["bound"]),
                        x["status"],
                    ]
                    for x in result["checks"]
                ],
            ),
            view_details(presentation, "monitoring", monitoring),
            view_details(presentation, "validity", result["errors"]),
            view_details(presentation, "metrics", result["metrics"]),
        ],
    )
    from reporting.events import attach_events
    attach_events(spec, presentation, origin=evidence["window"]["start_epoch_ms"] / 1000,
                  phases=evidence.get("phases", []), events=evidence.get("events", []))
    return write_bundle(
        directory,
        "run",
        "master-performance",
        result,
        spec,
        meta=run_meta(
            dict(id=p["instance"], verdict=result["verdict"]),
            implementation=p.get("master_artifact"),
            workload=p.get("trace"),
            configuration=p,
            evidence=dict(file="performance-gate-evidence.json"),
        ),
        producer="performance-gate",
        role="gate",
    )


def refresh_report(directory):
    """Refresh final archived curves using the already published gate result."""
    import json
    from reporting import bundle_path, load_analysis

    directory = Path(directory)
    evidence = directory / "performance-gate-evidence.json"
    bundle = bundle_path(directory, "run", "master-performance")
    if evidence.is_file() and bundle.exists():
        frozen = load_analysis(bundle)
        write_report(directory, json.loads(evidence.read_text()), frozen)


def validate_view(path, data, fail):
    from reporting.view_schema import validate_section_contract

    if "diagnostic_only" not in data["metrics"]:
        fail(path, "selected gate views require diagnostic_only classification")
    validate_section_contract(path, data, {
        "checks": 4, "monitoring": None, "validity": None, "metrics": None,
    }, fail)
