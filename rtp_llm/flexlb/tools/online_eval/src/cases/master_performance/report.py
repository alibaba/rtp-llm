"""Present an explicit performance verdict using already published metric data."""

from pathlib import Path

from reporting import write_bundle, run_meta
from reporting.run_context import title
from reporting.view_sections import view_details, view_table


def write_report(directory, evidence, result, *, run=None):
    from reporting.view_config import view

    presentation = view("master_performance.yaml")
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    p = evidence.get("provenance", {})
    from cases.master_performance.panels import prepare_curves, report_panels

    curves, monitoring = prepare_curves(directory, evidence, presentation)
    panels = report_panels(curves, evidence.get("criteria", {}), presentation)
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
                        x["id"],
                        x["actual"],
                        str(x["evidence"].get("op", "")) + " " + str(x["expected"]),
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
    if run is not None:
        from reporting.run_context import selected_spec
        spec = selected_spec(spec, run, presentation)
        result = dict(result, run=run)
    meta = run_meta(
        dict(id=p["instance"], verdict=result["verdict"]),
        implementation=p.get("master_artifact"), workload=p.get("trace"), configuration=p,
        evidence=dict(file="performance-gate-evidence.json"),
    )
    if run is not None:
        from reporting.run_context import provenance_from
        meta = provenance_from(run, meta)
    return write_bundle(
        directory,
        "run",
        "master-performance",
        result,
        spec,
        meta=meta,
        producer="performance-gate",
        role="gate",
    )


def validate_view(path, data, fail):
    from reporting.view_schema import validate_section_contract

    validate_section_contract(path, data, {
        "checks": 4, "monitoring": None, "validity": None, "metrics": None,
    }, fail)


def render_view(directory, run, presentation):
    from workload.gate_result import load_gate
    evidence, result = load_gate(directory, "performance")
    return write_report(directory, evidence, result, run=run) / "report.html"
