"""Read-only scalar and frozen-metric checks; producers own measurement semantics."""

import math
import operator
import copy
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


def check_metric(store, identity, metric_id, *, reduction, op, expected,
                 labels=None, start=None, end=None, source=None, epoch=None,
                 min_samples=1, max_gap_s=None, evidence=None, advisory=False):
    """Use frozen definitions and one explicitly selected series, without acquisition.

    MetricStore's inclusive sample window is distinct from request-cohort windows.
    Invalid observations become ERROR; definition/selector contract errors raise.
    """
    from monitoring.metric_store import MetricUnavailable

    validate_comparison(op, expected)
    selection = dict(labels=labels, start=start, end=end, source=source, epoch=epoch,
                     min_samples=min_samples, max_gap_s=max_gap_s)
    provenance = dict(evidence or {}, metric=metric_id, selection=selection, reduction=reduction)
    try:
        actual = store.reduce(metric_id, op=reduction, **selection)
    except MetricUnavailable as exc:
        return invalid_check(identity, detail=str(exc), expected=expected, evidence=provenance)
    rows = store.select(metric_id, **selection)
    provenance["definition"] = store.document["definitions"][metric_id]
    provenance["observations"] = [dict(source=row["source"], epoch=row["epoch"],
        labels=row["labels"], provenance=row["provenance"], sample_count=len(row["points"]),
        observed_bounds=[row["points"][0][0], row["points"][-1][0]]) for row in rows]
    return evaluate(identity, actual, op, expected, evidence=copy.deepcopy(provenance), advisory=advisory)
