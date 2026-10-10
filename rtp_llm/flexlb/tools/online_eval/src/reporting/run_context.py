"""One run identity and provenance contract for all case report producers."""

import copy

from reporting.core import run_meta, table, details


KPI_LABELS = {"execution": "Execution", "validity": "Validity", "request_count": "请求数",
              "verdict": "Gate verdict", "monitoring": "监控与请求诊断"}


def title(identity):
    return " : ".join(str(identity).split("::"))


def provenance_from(analysis, inherited=None):
    meta = copy.deepcopy(inherited or {})
    configuration = copy.deepcopy(meta.get("configuration") or {})
    configuration.update(declared=analysis.get("configuration"),
                         sha256=analysis.get("configuration_sha256"),
                         runtime=analysis.get("workload", {}).get("runtime_configuration"))
    implementation = copy.deepcopy(analysis.get("implementation") or {})
    # Configuration and metric definitions already have their own frozen artifacts.
    implementation.pop("configuration", None)
    if isinstance(implementation.get("monitoring_query_plan"), dict):
        implementation["monitoring_query_plan"].pop("definition", None)
    return run_meta(
        dict(meta.get("identity") or {}, id=analysis["id"], kind="run"),
        implementation=dict(case=implementation,
                            artifacts=meta.get("implementation")),
        workload=analysis.get("traffic_manifests") or meta.get("workload"),
        configuration=configuration,
        environment=analysis.get("runtime_provenance") or meta.get("environment"),
        clock=analysis.get("clock_anchor") or meta.get("clock"),
        evidence=dict(gate=meta.get("evidence"), requests=analysis.get("request_sources")),
    )


def checks_section(analysis):
    rows = []
    for row in analysis.get("checks", []):
        nested = row.get("evidence", {}).get("checks")
        for check in nested if nested is not None else [row]:
            identity = row["stage"] + "/" + (row["id"] + "/" if nested is not None else "") + check["id"]
            rows.append([identity, check["status"], check.get("actual"), check.get("expected")])
    return table("门禁检查", ["阶段 / 检查", "状态", "实际值", "门槛"],
                 rows, opened=True, identity="run.checks")


def validity_section(analysis):
    workload = analysis.get("workload", {})
    return details("有效性与证据完整性", {key: workload.get(key) for key in (
        "runtime_validity", "telemetry_completeness", "missing_telemetry",
        "telemetry_integrity_errors", "telemetry_diagnostics", "telemetry_warnings")}, identity="run.validity")


def canonical_spec(spec, analysis):
    result = copy.deepcopy(spec)
    result.update(run_id=analysis["id"], title=title(analysis["id"]))
    common = {"execution": analysis["status"], "validity": analysis["workload"]["runtime_validity"]}
    result["kpis"] = [dict(id="run." + key, label=KPI_LABELS[key], value=value)
                      for key, value in common.items()] + [
        item for item in result.get("kpis", []) if item.get("id") not in {"run." + key for key in common}]
    result["sections"] = [checks_section(analysis), validity_section(analysis)] + [
        section for section in result.get("sections", [])
        if section.get("id") not in {"run.checks", "run.validity", "case.checks"}]
    return result

def selected_spec(spec, analysis, presentation):
    from reporting.events import attach_events
    return attach_events(canonical_spec(spec, analysis), presentation,
                         origin=spec["timeOriginEpochS"],
                         phases=analysis.get("phases", []), events=analysis.get("events", []))
