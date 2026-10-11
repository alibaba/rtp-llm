"""Case classification, collection policy and explicit CI selection."""

import copy
import math
import re
from pathlib import Path

from schema_contract import matches_schema

from scenario.loader import ScenarioError, load_document

CATALOG = Path(__file__).resolve().parents[2] / "config/suites.yaml"
KINDS = ("functional", "workload")
FILTERS = (*KINDS, "all")
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


EXECUTION_BUDGETS = {"timeout_s", "stage_timeout_s", "cleanup_timeout_s", "finalize_timeout_s", "report_timeout_s"}
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


def _catalog(path=CATALOG):
    data = load_document(path)
    if set(data) != {"suite_schema_version", "default_suite", "ci_suites"} or not matches_schema(data, "suite_schema_version", 2):
        raise ScenarioError("invalid CI suite catalog: expected suite_schema_version 2")
    suites = data["ci_suites"]
    if not isinstance(suites, dict) or not suites:
        raise ScenarioError("ci_suites must be a nonempty mapping")
    for name, members in suites.items():
        if not isinstance(name, str) or not re.fullmatch(r"[a-z][a-z0-9_]*", name) or name in FILTERS:
            raise ScenarioError("invalid or reserved CI suite name: " + str(name))
        if not isinstance(members, list) or not members or any(
            not isinstance(member, str) or not re.fullmatch(r"[A-Za-z0-9_-]+::[A-Za-z0-9_-]+", member)
            for member in members
        ) or len(set(members)) != len(members):
            raise ScenarioError("CI suite requires unique case::variant identities: " + name)
    if data["default_suite"] not in suites:
        raise ScenarioError("default_suite must name a declared CI suite")
    return data


def suite_names(catalog=CATALOG):
    return (*_catalog(catalog)["ci_suites"], *FILTERS)


def default_suite(catalog=CATALOG):
    return _catalog(catalog)["default_suite"]


def _matches(key, kind, suite, suites):
    if suite not in (*suites, *FILTERS):
        raise ScenarioError("unknown suite: " + str(suite))
    return suite == "all" or suite == kind or key in suites.get(suite, ())


def preselect_documents(documents, suite="all", catalog=CATALOG):
    """Select variants before resource checks; directory names never select tests."""
    suites = _catalog(catalog)["ci_suites"]
    selected = []
    for path, document in documents:
        variants = []
        for variant in document["variants"]:
            key = document["id"] + "::" + variant["id"]
            metadata = normalize_metadata(variant.get("metadata"))
            if _matches(key, metadata["kind"], suite, suites):
                variants.append(variant)
        if variants:
            doc = copy.deepcopy(document)
            doc["variants"] = copy.deepcopy(variants)
            selected.append((path, doc))
    return selected


def classify(plans, suite="all", catalog=CATALOG):
    suites = _catalog(catalog)["ci_suites"]
    selected = []
    for plan in plans:
        key = plan["scenario_id"] + "::" + plan["variant_id"]
        metadata = normalize_metadata(plan.get("metadata"))
        kind = metadata["kind"]
        execution = normalize_execution(plan.get("execution"), kind=kind)
        if _matches(key, kind, suite, suites):
            selected.append(dict(
                plan, metadata=metadata, execution=execution, test_kind=kind,
                collection_profile=execution["collection"],
                workload_runtime=execution["monitoring"] if kind == "workload" else {},
            ))
    if suite in suites:
        actual = {p["scenario_id"] + "::" + p["variant_id"] for p in selected}
        missing = set(suites[suite]) - actual
        if missing:
            raise ScenarioError("CI suite is unsupported by the selected source/profile; missing: " + ", ".join(sorted(missing)))
    elif suite not in FILTERS:
        raise ScenarioError("unknown suite: " + str(suite))
    return selected
