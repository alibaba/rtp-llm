"""Explicit test classification, independent of categories and execution profiles."""

import math
from pathlib import Path

from .scenario.loader import ScenarioError, load_document

CATALOG = Path(__file__).resolve().parents[1] / "suites.yaml"
KINDS = ("functional", "workload")


def classify(plans, suite="all", catalog=CATALOG):
    if suite not in (*KINDS, "all"):
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
            "sample_history_limit",
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
    if (
        type(runtime["sample_history_limit"]) is not int
        or runtime["sample_history_limit"] < 1
    ):
        raise ScenarioError("sample_history_limit must be a positive integer")
    entries = data["cases"]
    for key, entry in entries.items():
        if (
            not isinstance(entry, dict)
            or entry.get("kind") not in KINDS
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
        if suite in ("all", kind):
            selected.append(
                dict(
                    plan,
                    test_kind=kind,
                    workload_runtime=(
                        dict(data["workload_runtime"]) if kind == "workload" else {}
                    ),
                )
            )
    return selected
