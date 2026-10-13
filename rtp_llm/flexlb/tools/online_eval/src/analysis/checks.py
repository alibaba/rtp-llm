"""Pure scalar checks and verdict aggregation; producers own measurement semantics."""

import math
import operator
from dataclasses import dataclass, field


CHECK_STATUSES = frozenset({"PASS", "FAIL", "ERROR", "SKIP", "WARNING"})
EFFECTIVE_CHECK_STATUSES = CHECK_STATUSES - {"SKIP"}


def check_verdict(statuses):
    """One aggregation rule: evidence errors dominate, skipped checks cannot pass."""
    values = set(statuses)
    if values - CHECK_STATUSES:
        raise ValueError("invalid check status")
    if not values & EFFECTIVE_CHECK_STATUSES or "ERROR" in values:
        return "INVALID"
    return "FAIL" if "FAIL" in values else "PASS"


@dataclass(frozen=True)
class CheckResult:
    """Shared decision data; importing numerical analysis never loads execution."""

    id: str
    status: str  # PASS | FAIL | ERROR | SKIP | WARNING
    detail: str = ""
    actual: object = None
    expected: object = None
    evidence: dict = field(default_factory=dict)


COMPARISONS = {"eq": operator.eq, "le": operator.le, "ge": operator.ge}


def validate_comparison(op, expected):
    if type(op) is not str or op not in COMPARISONS:
        raise ValueError("unknown comparison")
    if type(expected) in (bool, str):
        if op != "eq":
            raise ValueError("boolean/string checks require eq")
    elif type(expected) not in (int, float) or not math.isfinite(expected):
        raise ValueError("check requires a finite scalar expected value")


def compare(actual, op, expected):
    """No coercion, missing-as-zero, or boolean/numeric equivalence."""
    validate_comparison(op, expected)
    if type(expected) in (bool, str):
        if type(actual) is not type(expected):
            raise ValueError("check scalar types do not match")
    elif type(actual) not in (int, float) or not math.isfinite(actual):
        raise ValueError("check requires a finite numeric measurement")
    return COMPARISONS[op](actual, expected)


def evaluate(identity, actual, op, expected, *, evidence=None, advisory=False):
    passed = compare(actual, op, expected)
    return CheckResult(identity, "PASS" if passed else "WARNING" if advisory else "FAIL",
                       actual=actual, expected=expected, evidence=evidence or {})



def invalid_check(identity, *, detail="", actual=None, expected=None, evidence=None):
    """Insufficient observation cannot become a threshold failure or an advisory pass."""
    return CheckResult(identity, "ERROR", detail=detail, actual=actual, expected=expected,
                       evidence=dict(evidence or {}, validity="INVALID"))
