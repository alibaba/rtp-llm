"""Case metadata and collection policy validation."""
import copy
import math
from scenario.loader import ScenarioError

KINDS = ("functional", "workload")
DEFAULT_MONITORING = {
    "capture_metrics": True,
    "sample_interval_s": 1.0,
    "max_sample_gap_s": 5.0,
    "collector_shutdown_s": 10.0,
}


def normalize_metadata(value):
    from scenario.validation import mapping

    mapping(value, "metadata", {
        "kind", "description", "category", "tags", "requires", "findings",
        "estimated_duration_s",
    }, {"kind", "description"})
    if value["kind"] not in KINDS:
        raise ScenarioError("metadata.kind must be functional or workload")
    if not isinstance(value["description"], str) or not value["description"].strip():
        raise ScenarioError("metadata.description is required")
    return copy.deepcopy(value)


EXECUTION_BUDGETS = {"timeout_s", "stage_timeout_s", "cleanup_timeout_s", "finalize_timeout_s", "report_timeout_s", "archive_timeout_s"}
EXECUTION_FIELDS = EXECUTION_BUDGETS | {"collection", "monitoring"}


def normalize_execution(value, *, kind):
    from scenario.validation import mapping

    mapping(value, "execution", EXECUTION_FIELDS, {"collection"})
    if value["collection"] not in ("aggregate", "request", "diagnostic"):
        raise ScenarioError("execution.collection must be aggregate, request or diagnostic")
    patch = value.get("monitoring", {})
    mapping(patch, "execution.monitoring", set(DEFAULT_MONITORING) | {"query_plan"})
    monitoring = {**DEFAULT_MONITORING, **patch}
    if type(monitoring["capture_metrics"]) is not bool:
        raise ScenarioError("execution.monitoring.capture_metrics must be boolean")
    for key in ("sample_interval_s", "collector_shutdown_s", "max_sample_gap_s"):
        v = monitoring[key]
        if type(v) not in (int, float) or not math.isfinite(v) or v <= 0:
            raise ScenarioError("invalid execution.monitoring budget: " + key)
    if monitoring["max_sample_gap_s"] < monitoring["sample_interval_s"]:
        raise ScenarioError("maximum sample gap is shorter than sampling interval")
    if "query_plan" in monitoring:
        if kind != "workload":
            raise ScenarioError("execution.monitoring.query_plan requires a workload case")
        from monitoring.query_plan import load_plan

        load_plan(monitoring["query_plan"])
    return {**copy.deepcopy(value), "monitoring": monitoring}
