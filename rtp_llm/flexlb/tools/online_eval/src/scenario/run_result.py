"""Execution and delivery state, projected to JSON only at external boundaries."""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from enum import Enum
import json

from analysis.checks import CheckResult, EFFECTIVE_CHECK_STATUSES
from runtime.outcome import RunOutcome, OutcomeEvidence
from scenario.plan import CompiledInstance, json_document


class StageStatus(str, Enum):
    BLOCKED = "BLOCKED"
    PASS = "PASS"
    FAIL = "FAIL"
    ERROR = "ERROR"
    TIMEOUT = "TIMEOUT"


class FinalizationPhase(str, Enum):
    EVIDENCE = "evidence"
    REPORT = "report"


class DeliveryStatus(str, Enum):
    RUNNING = "RUNNING"
    PASS = "PASS"
    ERROR = "ERROR"
    TIMEOUT = "TIMEOUT"
    BLOCKED = "BLOCKED"


@dataclass
class StageResult:
    id: str
    action: str
    started_s: float
    status: StageStatus = StageStatus.BLOCKED
    output: dict = field(default_factory=dict)
    checks: list[CheckResult] = field(default_factory=list)
    artifacts: list[str] = field(default_factory=list)
    error: str | None = None
    duration_ms: int = 0
    finished_s: float | None = None

    def to_dict(self):
        result = asdict(self)
        result["status"] = self.status.value
        if self.finished_s is None:
            result.pop("finished_s")
        return result


@dataclass(frozen=True)
class CleanupResult:
    id: str
    status: StageStatus
    error: str | None
    duration_ms: int

    @classmethod
    def read(cls, value):
        return cls(value["id"], StageStatus(value["status"]), value["error"], value["duration_ms"])

    def to_dict(self):
        return dict(id=self.id, status=self.status.value, error=self.error, duration_ms=self.duration_ms)


@dataclass(frozen=True)
class StageExecutionSummary:
    stages: tuple[StageResult, ...]
    status: str
    error: str | None
    finding_confirmed: tuple[str, ...]
    finding_resolved: tuple[str, ...]
    cleanup: tuple[CleanupResult, ...]


@dataclass
class FinalizationResult:
    phase: FinalizationPhase
    budget_s: float
    status: DeliveryStatus = DeliveryStatus.RUNNING
    error: str | None = None
    duration_ms: int = 0

    def to_dict(self):
        return dict(phase=self.phase.value, status=self.status.value, budget_s=self.budget_s,
                    error=self.error, duration_ms=self.duration_ms)


@dataclass
class WorkloadResult:
    runtime_validity: str = "VALID"
    report_status: str | None = None
    evidence_json: str = "{}"

    @classmethod
    def read(cls, value):
        validity = value.get("runtime_validity", "VALID")
        if validity not in {"VALID", "INVALID"}:
            raise ValueError("invalid workload runtime validity")
        return cls(validity, value.get("report_status"), json_document({key: val for key, val in value.items()
                   if key not in {"runtime_validity", "report_status"}}))

    def to_dict(self):
        result = dict(json.loads(self.evidence_json), runtime_validity=self.runtime_validity)
        if self.report_status is not None:
            result["report_status"] = self.report_status
        return result


@dataclass
class RunResult:
    plan: CompiledInstance
    execution: StageExecutionSummary
    duration_ms: int
    error: str | None = None
    workload: WorkloadResult | None = None
    finalization: list[FinalizationResult] = field(default_factory=list)
    outcome: RunOutcome | None = None
    finalization_started: bool = False

    def adjudicate(self):
        stages, cleanup = self.execution.stages, self.execution.cleanup
        checks = [check for stage in stages for check in stage.checks]
        self.outcome = RunOutcome.evaluate(OutcomeEvidence(
            check_statuses=tuple(check.status for check in checks),
            failed_checks=frozenset(stage.id + "." + check.id for stage in stages
                                    for check in stage.checks if check.status == "FAIL"),
            finding_confirmed=frozenset(self.execution.finding_confirmed),
            finding_resolved=bool(self.execution.finding_resolved),
            error_statuses=tuple(stage.status.value for stage in stages if stage.error),
            cleanup_ok=all(row.status == StageStatus.PASS for row in cleanup),
            empty_execution_status=self.execution.status if not stages else "PASS",
            runtime_validity=self.workload.runtime_validity if self.workload else "VALID",
            delivery_statuses=tuple(row.status.value for row in self.finalization),
        ))
        gate, validity = self.outcome.gate, self.outcome.validity
        if gate == "INVALID" and not self.error:
            self.error = ("no effective checks executed" if not any(check.status in EFFECTIVE_CHECK_STATUSES
                          for check in checks) else "check evidence is invalid")
        if validity == "INVALID" and not self.error:
            self.error = "workload evidence is invalid; inspect workload diagnostics"
        return self.outcome

    def absorb_policy_result(self, value):
        # Policy adapters may add evidence/delivery metadata; frozen checks stay owned by execution.
        if "workload" in value:
            self.workload = WorkloadResult.read(value["workload"])

    def to_dict(self):
        outcome = self.adjudicate()
        plan = self.plan
        attachments = json.loads(plan.attachments_json)
        result = dict(id=plan.id, scenario_id=plan.scenario_id, variant_id=plan.variant_id,
            profile=plan.profile, category=plan.category, source=plan.source, grade=plan.grade,
            effective_axes=dict(plan.environment.axes), effective_capabilities=list(plan.environment.capabilities),
            status=outcome.status, stages=[stage.to_dict() for stage in self.execution.stages],
            cleanup=[row.to_dict() for row in self.execution.cleanup], error=self.error,
            duration_ms=self.duration_ms, finding_confirmed=list(self.execution.finding_confirmed),
            finding_resolved=list(self.execution.finding_resolved), source_path=plan.source_path,
            resource_budget=plan.resource_budget.to_metadata(),
            resolved_config=json.loads(plan.environment.snapshot.master_json), clock_domain="monotonic",
            test_kind=attachments.get("test_kind", "functional"), outcome=asdict(outcome))
        if "implementation" in attachments:
            result["implementation"] = attachments["implementation"]
        if self.finalization_started:
            result["execution_status"] = self.execution.status
            result["finalization"] = [phase.to_dict() for phase in self.finalization]
        if self.workload is not None:
            result["workload"] = self.workload.to_dict()
        return result
