"""Immutable execution contracts; JSON belongs to admission and artifact codecs."""
from __future__ import annotations

from dataclasses import dataclass, fields
import json
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from cases.declarations import MonitoringPolicy
    from scenario.environment_config import EnvironmentDeclaration

from flexlb_cfg import PROFILES
from scenario.environment_snapshot import EnvironmentSnapshot
from scenario.loader import ScenarioError
from scenario.validation import mapping, number, identifier
from scenario.stage_parameters import StageParameters, decode_parameters
from runtime.resource_plan import JavaMockBudget


def json_document(value):
    return json.dumps(value, separators=(",", ":"), allow_nan=False)


@dataclass(frozen=True)
class ExecutionBudgets:
    timeout_s: float
    stage_timeout_s: float
    cleanup_timeout_s: float
    finalize_timeout_s: float
    report_timeout_s: float
    archive_timeout_s: float
    collection: Optional[str] = None
    monitoring: Optional[MonitoringPolicy] = None

    def __post_init__(self):
        for name in ("timeout_s", "stage_timeout_s", "cleanup_timeout_s", "finalize_timeout_s", "report_timeout_s", "archive_timeout_s"):
            number(getattr(self, name), "execution." + name, minimum=.001)

    @classmethod
    def read(cls, value):
        required = {item.name for item in fields(cls)} - {"collection", "monitoring"}
        mapping(value, "execution", required | {"collection", "monitoring"}, required)
        budgets = {key: number(value[key], "execution." + key, minimum=.001) for key in required}
        collection = value.get("collection")
        if collection is not None and collection not in {"aggregate", "request", "diagnostic"}:
            raise ScenarioError("invalid execution collection")
        monitoring = value.get("monitoring")
        if monitoring is not None and not isinstance(monitoring, dict):
            raise ScenarioError("execution monitoring requires a mapping")
        from cases.declarations import MonitoringPolicy
        return cls(**budgets, collection=collection,
                   monitoring=MonitoringPolicy.read(monitoring) if monitoring is not None else None)

    def to_dict(self):
        result = {item.name: getattr(self, item.name) for item in fields(self)
                  if item.name not in {"collection", "monitoring"}}
        if self.collection is not None:
            result["collection"] = self.collection
        if self.monitoring is not None:
            result["monitoring"] = self.monitoring.to_dict()
        return result


@dataclass(frozen=True)
class StagePlan:
    id: str
    action: str
    timeout_s: float
    parameters: StageParameters
    check_ids: tuple[str, ...]
    purpose: Optional[str] = None

    @classmethod
    def read(cls, value):
        mapping(value, "stage", {"id", "action", "timeout_s", "params", "check_ids", "purpose"},
                {"id", "action", "timeout_s", "params", "check_ids"})
        sid = identifier(value["id"], "stage.id")
        action = identifier(value["action"], "stage.action")
        checks = value["check_ids"]
        if (not isinstance(checks, list) or any(type(check) is not str for check in checks)
                or len(set(checks)) != len(checks)):
            raise ScenarioError("invalid stage check identities")
        purpose = value.get("purpose")
        if purpose is not None and purpose not in {"operation", "observation"}:
            raise ScenarioError("invalid stage purpose")
        return cls(sid, action, number(value["timeout_s"], "stage.timeout_s", minimum=.001),
                   decode_parameters(action, value["params"]), tuple(checks), purpose)

    def to_dict(self):
        result = dict(id=self.id, action=self.action, timeout_s=self.timeout_s,
                      params=self.parameters.to_dict(), check_ids=list(self.check_ids))
        if self.purpose is not None:
            result["purpose"] = self.purpose
        return result


@dataclass(frozen=True)
class EnvironmentPlan:
    """Validated launch declaration and effective, frozen configuration."""
    declaration: "EnvironmentDeclaration"
    snapshot: EnvironmentSnapshot
    patch_json: str
    axes: tuple[tuple[str, str], ...]
    capabilities: tuple[str, ...]

    @classmethod
    def read(cls, value):
        from scenario.environment_config import EnvironmentDeclaration, MasterPatch, effective_capabilities
        extras = {"resolved_config", "rendered", "config_overrides", "effective_axes", "effective_capabilities"}
        allowed = {item.name for item in fields(EnvironmentDeclaration)} | extras
        mapping(value, "environment", allowed, extras | {
            "backend", "n_prefill", "n_decode", "discovery", "perf_preset", "master_layout",
            "master_stable_window_s", "debug_enabled", "master_sync_log"})
        snapshot = EnvironmentSnapshot.read(value["rendered"])
        if json.loads(snapshot.master_json) != value["resolved_config"]:
            raise ScenarioError("environment configuration disagrees with frozen snapshot")
        axes, capabilities = effective_capabilities(value["resolved_config"])
        if axes != value["effective_axes"] or capabilities != value["effective_capabilities"]:
            raise ScenarioError("environment capabilities disagree with frozen configuration")
        MasterPatch.read(value["config_overrides"], "environment.config_overrides")
        declaration = EnvironmentDeclaration.read_normalized({key: val for key, val in value.items() if key not in extras})
        if declaration.backend != "java_mock" or any(type(count) is not int or count < 1
                for count in (declaration.n_prefill, declaration.n_decode)):
            raise ScenarioError("invalid compiled environment topology")
        return cls(declaration, snapshot, json_document(value["config_overrides"]),
                   tuple(axes.items()), tuple(capabilities))

    def to_dict(self):
        return dict(self.declaration.to_dict(), resolved_config=json.loads(self.snapshot.master_json),
                    rendered=self.snapshot.to_dict(), config_overrides=json.loads(self.patch_json),
                    effective_axes=dict(self.axes), effective_capabilities=list(self.capabilities))


@dataclass(frozen=True)
class CompiledInstance:
    id: str
    scenario_id: str
    variant_id: str
    profile: str
    category: str
    description: str
    source: str
    source_path: str
    grade: str
    estimated_duration_s: float
    environment: EnvironmentPlan
    execution: ExecutionBudgets
    stages: tuple[StagePlan, ...]
    findings: tuple[str, ...]
    requires: tuple[str, ...]
    tags: tuple[str, ...]
    resource_budget: JavaMockBudget
    attachments_json: str = "{}"

    @classmethod
    def read(cls, value):
        if isinstance(value, cls):
            return value
        required = {item.name for item in fields(cls)} - {"attachments_json"}
        attachments = {"metadata", "reports", "reporting", "implementation", "test_kind",
                       "collection_profile", "workload_runtime"}
        mapping(value, "instance", required | {"resource_budget", "scenario", "variant",
                "effective_axes", "effective_capabilities"} | attachments, required | {"resource_budget"})
        sid, vid = value["scenario_id"], value["variant_id"]
        profile = value["profile"]
        if profile not in PROFILES or value["id"] != f"{sid}::{vid}::{profile}":
            raise ScenarioError("invalid frozen instance identity")
        for alias, expected in (("scenario", sid), ("variant", vid)):
            if alias in value and value[alias] != expected:
                raise ScenarioError("inconsistent frozen instance identity")
        if value["grade"] not in {"strict", "normal", "loose"}:
            raise ScenarioError("invalid frozen grade")
        if not isinstance(value["stages"], list) or not value["stages"]:
            raise ScenarioError("compiled instance requires stages")
        stages = tuple(StagePlan.read(stage) for stage in value["stages"])
        if len({stage.id for stage in stages}) != len(stages):
            raise ScenarioError("duplicate compiled stage identity")
        # Frozen references are structural dependencies, never arbitrary JSON paths.
        from scenario.stage_compiler import OUTPUTS
        seen = {}
        def references(payload):
            if isinstance(payload, dict):
                if "$ref" in payload:
                    from scenario.stage_parameters import OutputReference
                    reference = OutputReference.read(payload)
                    if reference.stage_id not in seen:
                        raise ScenarioError("unknown or forward frozen stage reference")
                    schema = seen[reference.stage_id]
                    if schema is not None and reference.field not in schema:
                        raise ScenarioError("unknown frozen output field")
                else:
                    for child in payload.values():
                        references(child)
            elif isinstance(payload, list):
                for child in payload:
                    references(child)
        setup_seen, torn_down = False, False
        for index, stage in enumerate(stages):
            if torn_down or (stage.action == "setup" and index != 0):
                raise ScenarioError("invalid frozen stage lifecycle")
            if stage.action == "setup":
                setup_seen = True
            elif not setup_seen:
                raise ScenarioError("frozen setup must precede actions")
            references(stage.parameters.to_dict())
            seen[stage.id] = OUTPUTS.get(stage.action)
            torn_down = stage.action == "teardown"
        for name in ("findings", "requires", "tags"):
            items = value[name]
            if (not isinstance(items, list) or any(type(item) is not str for item in items)
                    or len(set(items)) != len(items)):
                raise ScenarioError("invalid frozen " + name)
        check_ids = {stage.id + "." + check for stage in stages for check in stage.check_ids}
        if not check_ids or set(value["findings"]) - check_ids:
            raise ScenarioError("compiled instance requires declared checks/findings")
        environment = EnvironmentPlan.read(value["environment"])
        for name, expected in (("effective_axes", dict(environment.axes)),
                               ("effective_capabilities", list(environment.capabilities))):
            if name in value and value[name] != expected:
                raise ScenarioError("instance capability projection disagrees with environment")
        resource_budget = JavaMockBudget.from_metadata(value["resource_budget"])
        scalar = {key: value[key] for key in ("id", "scenario_id", "variant_id", "profile", "category",
                  "description", "source", "source_path", "grade", "estimated_duration_s")}
        return cls(**scalar, environment=environment, execution=ExecutionBudgets.read(value["execution"]),
                   stages=stages, findings=tuple(value["findings"]), requires=tuple(value["requires"]),
                   tags=tuple(value["tags"]), resource_budget=resource_budget,
                   attachments_json=json_document({key: value[key] for key in attachments if key in value}))

    def to_dict(self):
        result = {key: getattr(self, key) for key in ("id", "scenario_id", "variant_id", "profile", "category",
                  "description", "source", "source_path", "grade", "estimated_duration_s")}
        result.update(scenario=self.scenario_id, variant=self.variant_id,
            effective_axes=dict(self.environment.axes), effective_capabilities=list(self.environment.capabilities),
            environment=self.environment.to_dict(), execution=self.execution.to_dict(),
            stages=[stage.to_dict() for stage in self.stages], findings=list(self.findings),
            requires=list(self.requires), tags=list(self.tags), resource_budget=self.resource_budget.to_metadata())
        result.update(json.loads(self.attachments_json))
        return result


@dataclass(frozen=True)
class ExecutionPlan:
    instances: tuple[CompiledInstance, ...]
    dependencies: tuple[tuple[str, str], ...]

    def __post_init__(self):
        if not self.instances or len({instance.id for instance in self.instances}) != len(self.instances):
            raise ScenarioError("execution plan requires unique compiled instances")
