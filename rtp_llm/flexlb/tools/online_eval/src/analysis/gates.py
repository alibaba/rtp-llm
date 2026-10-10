"""Aggregate typed checks without acquiring evidence or presenting reports."""

from dataclasses import asdict
from analysis.checks import CheckResult, invalid_check


def gate_checks(checks, errors):
    """Evidence errors dominate thresholds; SKIP/WARNING never imply a failure."""
    rows = list(checks)
    rows.extend(invalid_check(f"evidence_{i + 1}", detail=str(error))
                for i, error in enumerate(errors))
    if len({row.id for row in rows}) != len(rows):
        raise ValueError("duplicate gate check identity")
    if any(row.status not in {"PASS", "FAIL", "ERROR", "SKIP", "WARNING"} for row in rows):
        raise ValueError("invalid gate check status")
    verdict = ("INVALID" if not rows or any(row.status == "ERROR" for row in rows)
               else "FAIL" if any(row.status == "FAIL" for row in rows) else "PASS")
    return verdict, [asdict(row) for row in rows]
