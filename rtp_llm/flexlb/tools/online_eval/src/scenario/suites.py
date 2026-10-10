"""Instance-owned test metadata and explicit CI selection, independent of paths."""

import copy
import math
import re
from pathlib import Path

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


def normalize_test(value):
    if not isinstance(value, dict) or set(value) - {"kind", "description", "collection", "monitoring", "reports"}:
        raise ScenarioError("test must declare kind, description and collection; optional monitoring")
    if value.get("kind") not in KINDS:
        raise ScenarioError("test.kind must be functional or workload")
    if not isinstance(value.get("description"), str) or not value["description"].strip():
        raise ScenarioError("test.description is required")
    if value.get("collection") not in ("aggregate", "request", "diagnostic"):
        raise ScenarioError("test.collection must be aggregate, request or diagnostic")
    patch = value.get("monitoring", {})
    if not isinstance(patch, dict) or set(patch) - set(DEFAULT_MONITORING):
        raise ScenarioError("invalid test.monitoring fields")
    monitoring = {**DEFAULT_MONITORING, **patch}
    if type(monitoring["capture_metrics"]) is not bool:
        raise ScenarioError("test.monitoring.capture_metrics must be boolean")
    for key in ("sample_interval_s", "collector_shutdown_s", "max_sample_gap_s"):
        v = monitoring[key]
        if type(v) not in (int, float) or not math.isfinite(v) or v <= 0:
            raise ScenarioError("invalid test.monitoring budget: " + key)
    if monitoring["max_sample_gap_s"] < monitoring["sample_interval_s"]:
        raise ScenarioError("maximum sample gap is shorter than sampling interval")
    if "reports" in value:
        from reporting.view_config import declaration

        declaration(value["reports"], kind=value["kind"], path="test.reports")
    return {**value, "monitoring": monitoring}


def _catalog(path=CATALOG):
    data = load_document(path)
    if set(data) != {"schema_version", "default_suite", "ci_suites"} or data["schema_version"] != 2:
        raise ScenarioError("invalid CI suite catalog: expected schema_version 2")
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
            test = normalize_test(variant.get("test"))
            if _matches(key, test["kind"], suite, suites):
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
        test = normalize_test(plan.get("test"))
        kind = test["kind"]
        if _matches(key, kind, suite, suites):
            selected.append(dict(
                plan, test_kind=kind, test_description=test["description"],
                collection_profile=test["collection"],
                workload_runtime=test["monitoring"] if kind == "workload" else {},
            ))
    if suite in suites:
        actual = {p["scenario_id"] + "::" + p["variant_id"] for p in selected}
        missing = set(suites[suite]) - actual
        if missing:
            raise ScenarioError("CI suite is unsupported by the selected source/profile; missing: " + ", ".join(sorted(missing)))
    elif suite not in FILTERS:
        raise ScenarioError("unknown suite: " + str(suite))
    return selected
