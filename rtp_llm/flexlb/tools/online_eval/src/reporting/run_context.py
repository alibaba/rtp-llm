"""One run identity and provenance contract for all case report producers."""

import copy

from reporting.core import run_meta, table, details


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
    return table("门禁检查", ["阶段 / 检查", "状态", "实际值", "门槛"], [
        [row["stage"] + "/" + row["id"], row["status"], row.get("actual"), row.get("expected")]
        for row in analysis.get("checks", [])
    ], opened=True)


def validity_section(analysis):
    workload = analysis.get("workload", {})
    return details("有效性与证据完整性", {key: workload.get(key) for key in (
        "runtime_validity", "telemetry_completeness", "missing_telemetry",
        "telemetry_integrity_errors", "telemetry_diagnostics", "telemetry_warnings")})


def canonical_spec(spec, analysis):
    result = copy.deepcopy(spec)
    result.update(run_id=analysis["id"], title=title(analysis["id"]))
    kpis = result.setdefault("kpis", [])
    result["kpis"] = [dict(label=label, value=value)
                      for label, value in (("Execution", analysis["status"]),
                                           ("Validity", analysis["workload"]["runtime_validity"]))
                      if not any(kpi["label"] == label for kpi in kpis)] + kpis
    sections = [checks_section(analysis), validity_section(analysis)]
    sections.extend(section for section in result.get("sections", [])
                    if section.get("title") not in {"门禁检查", "有效性与证据完整性", "其他报告视角", "门禁详细结果"})
    result["sections"] = sections
    return result
