"""One final-status rule over execution, adjudication, validity and delivery."""

from dataclasses import asdict, dataclass
from typing import Tuple, FrozenSet

from analysis.checks import check_verdict, EFFECTIVE_CHECK_STATUSES

SUCCESS_STATUSES = frozenset({"PASS", "FINDING-CONFIRMED", "FINDING-RESOLVED"})
TERMINAL_STATUSES = SUCCESS_STATUSES | {"FAIL", "ERROR", "TIMEOUT"}


@dataclass(frozen=True)
class OutcomeEvidence:
    check_statuses: Tuple[str, ...]
    failed_checks: FrozenSet[str]
    finding_confirmed: FrozenSet[str]
    finding_resolved: bool
    error_statuses: Tuple[str, ...]
    cleanup_ok: bool
    empty_execution_status: str
    runtime_validity: str
    delivery_statuses: Tuple[str, ...]


@dataclass(frozen=True)
class RunOutcome:
    execution: str
    gate: str
    validity: str
    delivery: str

    def __post_init__(self):
        allowed = dict(execution={"PASS", "ERROR", "TIMEOUT"},
                       gate={"PASS", "FAIL", "INVALID", "FINDING-CONFIRMED", "FINDING-RESOLVED"},
                       validity={"VALID", "INVALID"}, delivery={"PASS", "RUNNING", "ERROR", "TIMEOUT"})
        for field, values in allowed.items():
            if getattr(self, field) not in values:
                raise ValueError("invalid run outcome " + field)

    @property
    def status(self):
        if self.delivery in {"ERROR", "TIMEOUT"}:
            return self.delivery
        if self.execution in {"ERROR", "TIMEOUT"}:
            return self.execution
        if self.validity == "INVALID" or self.gate == "INVALID":
            return "ERROR"
        if self.delivery == "RUNNING":
            return "FINALIZING"
        return self.gate

    @classmethod
    def evaluate(cls, evidence: OutcomeEvidence):
        execution = ("TIMEOUT" if "TIMEOUT" in evidence.error_statuses
                     else "ERROR" if evidence.error_statuses or not evidence.cleanup_ok
                     else "PASS")
        if evidence.empty_execution_status in {"ERROR", "TIMEOUT"}:
            execution = evidence.empty_execution_status
        gate = check_verdict(evidence.check_statuses)
        if (gate == "FAIL" and evidence.finding_confirmed
                and evidence.finding_confirmed == evidence.failed_checks):
            gate = "FINDING-CONFIRMED"
        elif gate == "PASS" and evidence.finding_resolved:
            gate = "FINDING-RESOLVED"
        phases = evidence.delivery_statuses
        delivery = ("TIMEOUT" if "TIMEOUT" in phases
                    else "ERROR" if any(status in {"ERROR", "BLOCKED"} for status in phases)
                    else "RUNNING" if "RUNNING" in phases else "PASS")
        return cls(execution, gate, evidence.runtime_validity, delivery)

    @classmethod
    def from_result(cls, result):
        """Decode artifact evidence; execution objects call evaluate directly."""
        stages = result["stages"]
        return cls.evaluate(OutcomeEvidence(
            check_statuses=tuple(check["status"] for stage in stages for check in stage["checks"]),
            failed_checks=frozenset(stage["id"] + "." + check["id"] for stage in stages
                                    for check in stage["checks"] if check["status"] == "FAIL"),
            finding_confirmed=frozenset(result.get("finding_confirmed", [])),
            finding_resolved=bool(result.get("finding_resolved")),
            error_statuses=tuple(stage["status"] for stage in stages if stage.get("error")),
            cleanup_ok=all(row["status"] == "PASS" for row in result["cleanup"]),
            empty_execution_status=result.get("execution_status", result["status"]) if not stages else "PASS",
            runtime_validity=result.get("workload", {}).get("runtime_validity", "VALID"),
            delivery_statuses=tuple(row["status"] for row in result.get("finalization", [])),
        ))


def apply_outcome(result):
    outcome = RunOutcome.from_result(result)
    result["outcome"] = asdict(outcome)
    result["status"] = outcome.status
    if outcome.gate == "INVALID" and not result.get("error"):
        result["error"] = "no effective checks executed" if not any(
            check["status"] in EFFECTIVE_CHECK_STATUSES
            for stage in result["stages"] for check in stage["checks"]) else "check evidence is invalid"
    if outcome.validity == "INVALID" and not result.get("error"):
        result["error"] = "workload evidence is invalid; inspect workload diagnostics"
    return outcome
