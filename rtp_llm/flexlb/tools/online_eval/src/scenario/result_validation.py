"""Revalidate child result identities and nested evidence before aggregation."""
import json
from schema_contract import matches_schema
from runtime.outcome import TERMINAL_STATUSES as STATUSES, RunOutcome

def error_result(instance, message):
    return {
        **instance.metadata,
        "status": "ERROR",
        "duration_ms": 0,
        "stages": [],
        "cleanup": [],
        "error": message,
    }


def execution_issues(row):
    """Validate nested evidence before accepting a child's success classification."""
    issues = []
    (stages, cleanup) = (row.get("stages", []), row.get("cleanup", []))
    if not isinstance(stages, list) or not isinstance(cleanup, list):
        return ["stages and cleanup must be arrays"]
    for item in cleanup:
        if (
            not isinstance(item, dict)
            or item.get("status") != "PASS"
            or item.get("error")
        ):
            issues.append("cleanup did not complete successfully")
    for phase in row.get("finalization", []):
        if not isinstance(phase, dict) or phase.get("status") != "PASS" or phase.get("error"):
            issues.append("finalization did not complete successfully")
    failed_checks = set()
    green = row["status"] in {"PASS", "FINDING-CONFIRMED", "FINDING-RESOLVED"}
    for stage in stages:
        if not isinstance(stage, dict):
            issues.append("stage must be an object")
            continue
        status = stage.get("status")
        if status not in {"PASS", "FAIL", "BLOCKED", "SKIP"} or stage.get("error"):
            issues.append("stage has an execution error or unknown status")
        checks = stage.get("checks", [])
        if not isinstance(checks, list):
            issues.append("stage checks must be an array")
            continue
        failures = []
        for check in checks:
            if (
                not isinstance(check, dict)
                or check.get("status") not in {"PASS", "FAIL", "SKIP", "WARNING"}
                or check.get("error")
            ):
                issues.append("check has an execution error or unknown status")
                continue
            if check["status"] == "FAIL":
                failures.append(check)
                failed_checks.add(f"{stage.get('id')}.{check.get('id')}")
        if status == "PASS" and failures:
            issues.append("PASS stage contains a failed check")
        if status in {"BLOCKED", "SKIP"} and (checks or green):
            issues.append("unexecuted stage contradicts successful instance")
        if green and status == "FAIL" and (not failures):
            issues.append("failed stage has no failed check evidence")
    if row["status"] in {"PASS", "FINDING-RESOLVED"} and failed_checks:
        issues.append("successful instance contains failed checks")
    if row["status"] == "FINDING-CONFIRMED":
        confirmed = row.get("finding_confirmed", [])
        if (
            not isinstance(confirmed, list)
            or any((not isinstance(x, str) for x in confirmed))
            or (not failed_checks)
            or (set(confirmed) != failed_checks)
        ):
            issues.append(
                "finding classification does not match explicit failed checks"
            )
    if "outcome" in row:
        from dataclasses import asdict
        try:
            outcome = RunOutcome.from_result(row)
            if row["outcome"] != asdict(outcome) or row["status"] != outcome.status:
                issues.append("result outcome disagrees with execution evidence")
        except (ValueError, TypeError, KeyError) as exc:
            issues.append("invalid result outcome: " + str(exc))
    return issues


def read_results(path, group):
    """Missing, duplicated, unexpected or malformed rows cannot become a green run."""
    expected = {instance.id: instance for instance in group}
    try:
        data = json.loads(path.read_text())
        if (
            not matches_schema(data, "scenario_results_schema_version", 1)
            or (not isinstance(data.get("instances"), list))
        ):
            raise ValueError("unsupported scenario result schema")
        rows = data["instances"]
        if not isinstance(rows, list):
            raise ValueError("result rows must be an array")
        by_id = {}
        for row in rows:
            if not isinstance(row, dict):
                raise ValueError("result row must be an object")
            identity = row.get("id")
            if identity not in expected or identity in by_id:
                raise ValueError(f"unexpected or duplicate result: {identity!r}")
            if row.get("status") not in STATUSES:
                raise ValueError(f"unsupported result status: {row.get('status')!r}")
            if row.get("profile") != expected[identity].profile:
                raise ValueError(f"result profile mismatch: {identity}")
            if expected[identity].metadata.get("grade") is not None:
                if row.get("grade") != expected[identity].metadata["grade"]:
                    raise ValueError(f"result grade mismatch: {identity}")
            row = dict(row)
            failures = execution_issues(row)
            if (row.get("error") or failures) and row["status"] not in {
                "ERROR",
                "TIMEOUT",
            }:
                row["original_status"] = row["status"]
                row["status"] = "ERROR"
                row["error"] = (
                    row.get("error")
                    or "; ".join(failures)
                    or "stage/cleanup error cannot be classified as a finding"
                )
            if row["status"] in {"PASS", "FINDING-CONFIRMED", "FINDING-RESOLVED"}:
                stages = row.get("stages", [])
                executed_checks = [
                    check
                    for stage in stages
                    if isinstance(stage, dict)
                    and stage.get("status") in {"PASS", "FAIL"}
                    for check in stage.get("checks", [])
                    if isinstance(check, dict)
                    and check.get("status") in {"PASS", "FAIL", "WARNING"}
                ]
                if not executed_checks:
                    row["original_status"] = row["status"]
                    row["status"] = "ERROR"
                    row["error"] = "successful instance has no executed checks"
            by_id[identity] = {**row, **expected[identity].metadata, "id": identity}
        return [
            by_id.get(identity) or error_result(instance, "missing instance result")
            for (identity, instance) in expected.items()
        ]
    except (OSError, ValueError, TypeError, KeyError, AttributeError) as exc:
        return [
            error_result(instance, f"invalid child result {path}: {exc}")
            for instance in group
        ]
