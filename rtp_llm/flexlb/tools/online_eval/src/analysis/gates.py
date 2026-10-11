"""Aggregate typed checks without acquiring evidence or presenting reports."""

from dataclasses import asdict
from analysis.checks import CheckResult, invalid_check, check_verdict


def gate_checks(checks, errors):
    """Evidence errors dominate thresholds; SKIP/WARNING never imply a failure."""
    rows = list(checks)
    rows.extend(invalid_check(f"evidence_{i + 1}", detail=str(error))
                for i, error in enumerate(errors))
    if len({row.id for row in rows}) != len(rows):
        raise ValueError("duplicate gate check identity")
    return check_verdict(row.status for row in rows), [asdict(row) for row in rows]
