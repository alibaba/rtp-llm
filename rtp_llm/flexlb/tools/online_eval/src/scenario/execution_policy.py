"""Case metadata and collection policy validation."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Optional, cast

from document_io import DocumentError as ScenarioError

KINDS = ("functional", "workload")
DEFAULT_MONITORING = {"capture_metrics": True, "sample_interval_s": 1.0, "max_sample_gap_s": 5.0, "collector_shutdown_s": 10.0}


def normalize_metadata(value: object) -> dict:
    from scenario.validation import mapping

    data = mapping(
        value,
        "metadata",
        {"kind", "description", "category", "tags", "requires", "findings", "estimated_duration_s"},
        {"kind", "description"},
    )
    if data["kind"] not in KINDS:
        raise ScenarioError("metadata.kind must be functional or workload")
    if not isinstance(data["description"], str) or not data["description"].strip():
        raise ScenarioError("metadata.description is required")
    return copy.deepcopy(data)


DEFAULT_BUDGETS = dict(
    timeout_s=600, stage_timeout_s=60, cleanup_timeout_s=120, finalize_timeout_s=120, report_timeout_s=60, archive_timeout_s=120
)
EXECUTION_BUDGETS = frozenset(DEFAULT_BUDGETS)
EXECUTION_FIELDS = EXECUTION_BUDGETS | {"collection", "monitoring"}


def validate_budgets(value: dict, *, required: bool = False, minimum: float = 0.001) -> None:
    from scenario.validation import number

    if required and not EXECUTION_BUDGETS <= value.keys():
        raise ScenarioError("execution requires all time budgets")
    for name in EXECUTION_BUDGETS & value.keys():
        number(value[name], "execution." + name, minimum=minimum)
        if value[name] <= 0:
            raise ScenarioError("invalid execution time budget: " + name)


@dataclass(frozen=True)
class MonitoringPolicy:
    capture_metrics: bool
    sample_interval_s: float
    max_sample_gap_s: float
    collector_shutdown_s: float
    query_plan: Optional[str]

    @classmethod
    def read(cls, value: dict) -> MonitoringPolicy:
        from scenario.validation import mapping

        mapping(value, "execution.monitoring", set(DEFAULT_MONITORING) | {"query_plan"})
        normalized = {**DEFAULT_MONITORING, **value}
        return cls(
            normalized["capture_metrics"],
            normalized["sample_interval_s"],
            normalized["max_sample_gap_s"],
            normalized["collector_shutdown_s"],
            normalized.get("query_plan"),
        )

    def __post_init__(self) -> None:
        from scenario.validation import number

        if type(self.capture_metrics) is not bool:
            raise ScenarioError("execution.monitoring.capture_metrics must be boolean")
        for name in ("sample_interval_s", "max_sample_gap_s", "collector_shutdown_s"):
            value = number(getattr(self, name), "execution.monitoring." + name)
            if value <= 0:
                raise ScenarioError("invalid execution.monitoring budget: " + name)
        if self.max_sample_gap_s < self.sample_interval_s:
            raise ScenarioError("maximum sample gap is shorter than sampling interval")
        if self.query_plan is not None and (type(self.query_plan) is not str or not self.query_plan):
            raise ScenarioError("monitoring query plan must be a nonempty name")

    def to_dict(self) -> dict:
        result = {name: getattr(self, name) for name in self.__dataclass_fields__ if name != "query_plan"}
        if self.query_plan is not None:
            result["query_plan"] = self.query_plan
        return result


@dataclass(frozen=True)
class ExecutionPolicy:
    collection: str
    monitoring: MonitoringPolicy
    timeout_s: Optional[float]
    stage_timeout_s: Optional[float]
    cleanup_timeout_s: Optional[float]
    finalize_timeout_s: Optional[float]
    report_timeout_s: Optional[float]
    archive_timeout_s: Optional[float]

    @classmethod
    def read(cls, value: object, kind: str) -> ExecutionPolicy:
        from scenario.validation import mapping

        data = mapping(value, "execution", EXECUTION_FIELDS, {"collection"})
        if data["collection"] not in ("aggregate", "request", "diagnostic"):
            raise ScenarioError("execution.collection must be aggregate, request or diagnostic")
        raw_monitoring = data.get("monitoring", {})
        monitoring = MonitoringPolicy.read(raw_monitoring)
        if "query_plan" in raw_monitoring:
            if kind != "workload":
                raise ScenarioError("execution.monitoring.query_plan requires a workload case")
            from scenario.registry import load_metric_plan

            # Omission alone chooses the default. Explicit null is rejected.
            load_metric_plan(raw_monitoring["query_plan"])
        return cls(data["collection"], monitoring, *(data.get(name) for name in DEFAULT_BUDGETS))

    def to_dict(self):
        return {
            "collection": self.collection,
            "monitoring": self.monitoring.to_dict(),
            **{
                name: getattr(self, name)
                for name in self.__dataclass_fields__
                if name not in {"collection", "monitoring"} and getattr(self, name) is not None
            },
        }


def normalize_execution(value: object, *, kind: str) -> dict:
    """Compatibility projection; declaration parsing retains the typed policy."""
    policy = ExecutionPolicy.read(value, kind)
    return {**copy.deepcopy(cast(dict, value)), "monitoring": policy.monitoring.to_dict()}
