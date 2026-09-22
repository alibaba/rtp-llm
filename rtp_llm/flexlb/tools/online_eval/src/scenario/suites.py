"""Explicit test classification, independent of categories and execution profiles."""

import math
from pathlib import Path

from scenario.loader import ScenarioError, load_document

CATALOG = Path(__file__).resolve().parents[2] / "config/suites.yaml"
KINDS = ("functional", "workload")
SUITES = ("core", *KINDS, "all")


def preselect_documents(documents, suite="all", catalog=CATALOG):
    """Avoid compiling unrelated large workloads for functional listings/runs."""
    if suite not in SUITES:
        raise ScenarioError("unknown suite: " + suite)
    if suite in ("all", "workload"):
        return documents
    data = load_document(catalog)
    if data.get("schema_version") != 1 or not isinstance(data.get("cases"), dict) or not isinstance(data.get("core_cases"), dict):
        raise ScenarioError("invalid suite catalog")
    selected = []
    for path, document in documents:
        keys = [document["id"] + "::" + variant["id"] for variant in document["variants"]]
        if any(
            key not in data["cases"]
            or (suite == "core" and key in data["core_cases"])
            or (suite == "functional" and data["cases"][key].get("kind") == "functional")
            for key in keys
        ):
            selected.append((path, document))
    return selected


def classify(plans, suite="all", catalog=CATALOG):
    if suite not in SUITES:
        raise ScenarioError("unknown suite: " + suite)
    data = load_document(catalog)
    if data.get("schema_version") != 1 or not isinstance(data.get("cases"), dict):
        raise ScenarioError("invalid suite catalog")
    runtime = data.get("workload_runtime", {})
    if (
        set(runtime)
        != {
            "capture_metrics",
            "sample_interval_s",
            "collector_shutdown_s",
            "max_sample_gap_s",
        }
        or type(runtime["capture_metrics"]) is not bool
    ):
        raise ScenarioError("invalid workload runtime configuration")
    for key in ("sample_interval_s", "collector_shutdown_s", "max_sample_gap_s"):
        value = runtime[key]
        if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
            raise ScenarioError("invalid workload budget: " + key)
    if runtime["max_sample_gap_s"] < runtime["sample_interval_s"]:
        raise ScenarioError("maximum sample gap is shorter than sampling interval")
    entries = data["cases"]
    core = data.get("core_cases")
    if (
        not isinstance(core, dict)
        or len(core) != 5
        or any(not isinstance(key, str) or not reason for key, reason in core.items())
    ):
        raise ScenarioError("core_cases must define exactly five documented variants")
    for key, entry in entries.items():
        if (
            not isinstance(entry, dict)
            or entry.get("kind") not in KINDS
            or entry.get("collection", "aggregate") not in ("aggregate", "request", "diagnostic")
            or not entry.get("reason")
        ):
            raise ScenarioError("invalid suite entry: " + key)
    selected = []
    for plan in plans:
        key = plan["scenario_id"] + "::" + plan["variant_id"]
        entry = entries.get(key)
        # External extension plans retain functional execution until classified.
        # Bundled configurations must never silently escape the coverage ledger.
        if entry is None and Path(plan["source_path"]).resolve().is_relative_to(
            CATALOG.parent / "scenarios"
        ):
            raise ScenarioError("unclassified bundled case: " + key)
        kind = entry["kind"] if entry else "functional"
        selected_by_suite = (
            suite == "all"
            or suite == kind
            or (suite == "core" and kind == "functional" and key in core)
        )
        if selected_by_suite:
            selected.append(
                dict(
                    plan,
                    test_kind=kind,
                    collection_profile=(entry or {}).get("collection", "aggregate" if kind == "workload" else "diagnostic"),
                    workload_runtime=(
                        dict(data["workload_runtime"]) if kind == "workload" else {}
                    ),
                )
            )
    if suite == "core":
        selected_keys = {
            plan["scenario_id"] + "::" + plan["variant_id"] for plan in selected
        }
        missing = set(core) - selected_keys
        if missing:
            raise ScenarioError(
                "core suite is unsupported by the selected profile: "
                + ", ".join(sorted(missing))
            )
        if len(selected) != 5:
            raise ScenarioError("core suite must compile to exactly five instances")
    return selected
