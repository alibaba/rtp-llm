"""One final-status rule over execution, adjudication, validity and delivery."""

from dataclasses import asdict, dataclass

from analysis.checks import check_verdict, EFFECTIVE_CHECK_STATUSES

SUCCESS_STATUSES = frozenset({"PASS", "FINDING-CONFIRMED", "FINDING-RESOLVED"})
TERMINAL_STATUSES = SUCCESS_STATUSES | {"FAIL", "ERROR", "TIMEOUT"}


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
    def from_result(cls, result):
        stages = result["stages"]
        cleanup = result["cleanup"]
        checks = [check for stage in stages for check in stage["checks"]]
        errors = [stage for stage in stages if stage.get("error")]
        execution = ("TIMEOUT" if any(stage["status"] == "TIMEOUT" for stage in errors)
                     else "ERROR" if errors or any(row["status"] != "PASS" for row in cleanup)
                     else "PASS")
        if not stages and result.get("execution_status", result["status"]) in {"ERROR", "TIMEOUT"}:
            execution = result.get("execution_status", result["status"])
        gate = check_verdict(check["status"] for check in checks)
        known = set(result.get("finding_confirmed", []))
        failed = {stage["id"] + "." + check["id"] for stage in stages
                  for check in stage["checks"] if check["status"] == "FAIL"}
        if gate == "FAIL" and known and known == failed:
            gate = "FINDING-CONFIRMED"
        elif gate == "PASS" and result.get("finding_resolved"):
            gate = "FINDING-RESOLVED"
        validity = result.get("workload", {}).get("runtime_validity", "VALID")
        phases = result.get("finalization", [])
        delivery = ("TIMEOUT" if any(row["status"] == "TIMEOUT" for row in phases)
                    else "ERROR" if any(row["status"] in {"ERROR", "BLOCKED"} for row in phases)
                    else "RUNNING" if any(row["status"] == "RUNNING" for row in phases)
                    else "PASS")
        return cls(execution, gate, validity, delivery)


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
