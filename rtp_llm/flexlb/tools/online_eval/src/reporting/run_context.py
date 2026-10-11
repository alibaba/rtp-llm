"""One run identity and provenance contract for all case report producers."""

import copy

from reporting.core import run_meta, table, details
from reporting.run_model import RunPresentation


KPI_LABELS = {"execution": "Execution", "validity": "Validity", "request_count": "请求数",
              "verdict": "Gate verdict", "monitoring": "监控与请求诊断"}


def title(identity):
    return " : ".join(str(identity).split("::"))


def provenance_from(analysis, inherited=None):
    run = RunPresentation.read(analysis)
    meta = copy.deepcopy(inherited or {})
    configuration = copy.deepcopy(meta.get("configuration") or {})
    for name, value in (("declared", run.configuration), ("sha256", run.configuration_sha256),
                        ("runtime", run.runtime_configuration)):
        if value is not None:
            configuration[name] = value
        elif name not in configuration:
            configuration[name] = None
    implementation = copy.deepcopy(run.implementation)
    # Configuration and metric definitions already have their own frozen artifacts.
    implementation.pop("configuration", None)
    if isinstance(implementation.get("monitoring_query_plan"), dict):
        implementation["monitoring_query_plan"].pop("definition", None)
    return run_meta(
        dict(meta.get("identity") or {}, id=run.identity, kind="run"),
        implementation=dict(case=implementation,
                            artifacts=meta.get("implementation")),
        workload=run.traffic_manifests if run.traffic_manifests is not None else meta.get("workload"),
        configuration=configuration,
        environment=run.runtime_provenance if run.runtime_provenance is not None else meta.get("environment"),
        clock=dict(acquisition=run.clock_anchor if run.clock_anchor is not None else meta.get("clock"),
                   report=run.report_timeline) if run.report_timeline is not None else
              run.clock_anchor if run.clock_anchor is not None else meta.get("clock"),
        evidence=dict(gate=meta.get("evidence"), requests=run.request_sources),
    )


def checks_section(analysis):
    run = analysis if isinstance(analysis, RunPresentation) else RunPresentation.read(analysis)
    rows = [row for check in run.checks for row in check.rows()]
    return table("门禁检查", ["阶段 / 检查", "状态", "实际值", "门槛"],
                 rows, opened=True, identity="run.checks")


def validity_section(analysis):
    run = analysis if isinstance(analysis, RunPresentation) else RunPresentation.read(analysis)
    return details("有效性与证据完整性", run.validity.to_dict(), identity="run.validity")


def canonical_spec(spec, analysis):
    run = RunPresentation.read(analysis)
    result = copy.deepcopy(spec)
    result.update(run_id=run.identity, title=title(run.identity))
    common = {"execution": run.status, "validity": run.validity.runtime_validity}
    result["kpis"] = [dict(id="run." + key, label=KPI_LABELS[key], value=value)
                      for key, value in common.items()] + [
        item for item in result.get("kpis", []) if item.get("id") not in {"run." + key for key in common}]
    result["sections"] = [checks_section(run), validity_section(run)] + [
        section for section in result.get("sections", [])
        if section.get("id") not in {"run.checks", "run.validity", "case.checks"}]
    from reporting.timeline import apply
    return apply(result, analysis)

def selected_spec(spec, analysis, presentation):
    from reporting.events import attach_events
    result = canonical_spec(spec, analysis)
    if (analysis.get('report_timeline') or {}).get('status') == 'UNAVAILABLE':
        return result
    return attach_events(result, presentation,
                         origin=result["timeOriginEpochS"],
                         phases=analysis.get("phases", []), events=analysis.get("events", []))
