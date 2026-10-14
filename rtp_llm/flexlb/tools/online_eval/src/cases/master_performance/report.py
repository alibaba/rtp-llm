"""Present an explicit performance verdict using already published metric data."""

from pathlib import Path

from reporting import run_meta
from reporting.gate_report import write_gate_report, write_archived_gate_report
from reporting.run_context import title
from reporting.view_sections import view_details, view_table


def write_report(directory, evidence, result, *, run=None, presentation=None):
    from reporting.view_config import view

    presentation = presentation if presentation is not None else view("master_performance.yaml")
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    p = evidence.get("provenance", {})
    from cases.master_performance.panels import prepare_curves, report_panels

    curves, monitoring = prepare_curves(directory, evidence, presentation)
    panels = report_panels(curves, result, presentation, origin=evidence["window"]["start_epoch_ms"] / 1000)
    spec = dict(
        run_id=p["instance"],
        title=title(p["instance"]),
        subtitle=presentation.subtitle.render(verdict=result["verdict"]),
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
    meta = run_meta(
        dict(id=p["instance"], verdict=result["verdict"]),
        implementation=p.get("master_artifact"), workload=p.get("trace"), configuration=p,
        evidence=dict(file="performance-gate-evidence.json"),
    )
    if run is None:
        return write_archived_gate_report(directory, result, presentation, spec,
            identity="master-performance", producer="performance-gate", meta=meta)
    return write_gate_report(directory, result, presentation, spec,
        identity="master-performance", producer="performance-gate", meta=meta, run=run)


def validate_view(path, data, fail):
    from reporting.view_schema import validate_section_contract

    from reporting.gate_lines import validate_bindings
    from cases.master_performance.analysis import GATE_REFERENCES
    validate_bindings(data, GATE_REFERENCES)
    validate_section_contract(path, data, {
        "checks": 4, "monitoring": None, "validity": None, "metrics": None,
    }, fail)
