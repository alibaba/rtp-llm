"""Aggregate typed checks without acquiring evidence or presenting reports."""

from dataclasses import asdict
from analysis.checks import invalid_check, check_verdict


def gate_checks(checks, errors):
    """Evidence errors dominate thresholds; SKIP/WARNING never imply a failure."""
    rows = list(checks)
    rows.extend(invalid_check(f"evidence_{i + 1}", detail=str(error))
                for i, error in enumerate(errors))
    if len({row.id for row in rows}) != len(rows):
        raise ValueError("duplicate gate check identity")
    return check_verdict(row.status for row in rows), [asdict(row) for row in rows]


def gate_result(checks, errors=(), *, requirements=()):
    """Derive validity and verdict once; unmet experiment requirements are invalid."""
    checks = [invalid_check(c.id, detail="workload evidence requirement not satisfied",
                            actual=c.actual, expected=c.expected, evidence=c.evidence)
              if c.id in requirements and c.status == "FAIL" else c for c in checks]
    verdict, rows = gate_checks(checks, errors)
    return dict(verdict=verdict, validity="INVALID" if verdict == "INVALID" else "VALID", checks=rows)
