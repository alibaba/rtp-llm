"""Normalize scenario core v1 plans/results for the independent acceptance oracle.

No runtime import is required. Metrics must supply their reviewed check contracts;
an empty evidence dictionary never proves a sampled observation is complete.
"""

import math

from .acceptance import _index


def _scalar_comparison_complete(actual, expected, op):
    numeric = (int, float)
    if type(actual) in numeric and type(expected) in numeric:
        return op in ("eq", "le", "ge") and all(
            type(value) is int or math.isfinite(value) for value in (actual, expected)
        )
    return type(actual) in (bool, str) and type(expected) is type(actual) and op == "eq"


def normalize_plans(plans, check_contracts=None):
    """check_contracts is keyed by instance ID, then qualified stage.check ID."""
    check_contracts = check_contracts or {}
    indexed = _index(plans, "id", "scenario plan")
    if set(check_contracts) - indexed.keys():
        raise ValueError("check contracts refer to unknown instances")
    rows = []
    for iid, plan in indexed.items():
        checks = []
        for stage in plan["stages"]:
            for local in stage["check_ids"]:
                cid = stage["id"] + "." + local
                contract = dict(check_contracts.get(iid, {}).get(cid, {}))
                if "id" in contract or "finding_id" in contract:
                    raise ValueError(
                        "reviewed evidence cannot override check identity/finding"
                    )
                contract["id"] = cid
                if cid in plan["findings"]:
                    contract["finding_id"] = cid
                checks.append(contract)
        if set(check_contracts.get(iid, {})) - {c["id"] for c in checks}:
            raise ValueError(f"{iid}: evidence contract is not a declared check")
        rows.append(
            {
                "id": iid,
                "scenario_id": plan["scenario_id"],
                "variant_id": plan["variant_id"],
                "profile": plan["profile"],
                "grade": plan.get("grade"),
                "effective_axes": plan.get(
                    "effective_axes", plan["environment"].get("effective_axes")
                ),
                "effective_capabilities": plan.get(
                    "effective_capabilities",
                    plan["environment"].get("effective_capabilities"),
                ),
                "category": plan["category"],
                "requires": plan["requires"],
                "source": plan["source"],
                "source_path": plan["source_path"],
                "legacy_case_ids": plan["legacy_case_ids"],
                "resolved_config": plan["environment"]["resolved_config"],
                "resource_budget": plan["resource_budget"],
                "stages": [s["id"] for s in plan["stages"]],
                "checks": checks,
            }
        )
    return {"schema_version": 1, "instances": rows}


def normalize_results(plans, results):
    """Preserve stage errors and cleanup failures; qualify each declared check.

    Successful cleanup rows establish callback completion only, not an OS-level
    leak check. Keep independently observed leaked_resources when supplied.
    """
    indexed = _index(plans, "id", "scenario plan")
    _index(results, "id", "scenario result")
    rows = []
    for result in results:
        plan = indexed[result["id"]]
        actions = {s["id"]: s["action"] for s in plan["stages"]}
        params = {s["id"]: s.get("params", {}) for s in plan["stages"]}
        checks, errors = [], []
        if result.get("error"):
            errors.append(result["error"])
        if result["status"] in ("ERROR", "TIMEOUT", "BLOCKED", "SKIP"):
            errors.append("instance status " + result["status"])
        for stage in result["stages"]:
            if stage.get("error"):
                errors.append(stage["id"] + ": " + stage["error"])
            for check in stage["checks"]:
                evidence = check.get("evidence", {})
                complete = evidence.get("complete") is True
                if actions.get(stage["id"]) == "check":
                    # Core comparisons have a compiled primitive type contract.
                    # They are not metric aggregates requiring sampled windows.
                    complete = _scalar_comparison_complete(
                        check.get("actual"),
                        check.get("expected"),
                        params[stage["id"]].get("op", "eq"),
                    )
                checks.append(
                    {
                        "id": stage["id"] + "." + check["id"],
                        "status": check["status"],
                        "value": check.get("actual"),
                        "sample_count": evidence.get("sample_count"),
                        "evidence_complete": complete,
                        "failure_kind": (
                            "contract" if check["status"] == "FAIL" else None
                        ),
                    }
                )
        failed_cleanup = [r for r in result["cleanup"] if r["status"] != "PASS"]
        rows.append(
            {
                "id": result["id"],
                "stages": [
                    {"id": s["id"], "status": s["status"]} for s in result["stages"]
                ],
                "checks": checks,
                "errors": errors,
                "cleanup_errors": [
                    r.get("error") or r["status"] for r in failed_cleanup
                ],
                "leaked_resources": result.get(
                    "leaked_resources", [r["id"] for r in failed_cleanup]
                ),
                "cleanup_evidence_scope": "reported_callbacks_only",
            }
        )
    return {"schema_version": 1, "instances": rows}


def expand_coverage(document, inventory):
    """Expand dictionary-form scenario/variants/profiles targets to concrete IDs.

    preservation.contract_digest must be explicitly frozen by the mapping owner;
    this function never copies it from the current source to conceal a stale map.
    """
    rows = []
    for lid, mapping in document["legacy_cases"].items():
        targets = []
        for target in mapping.get("targets", []):
            matches = [
                i
                for i in inventory["instances"]
                if i["scenario_id"] == target["scenario"]
                and i["variant_id"] in target["variants"]
                and i["profile"] in target["profiles"]
            ]
            expected = {(v, p) for v in target["variants"] for p in target["profiles"]}
            if (
                not expected
                or {(i["variant_id"], i["profile"]) for i in matches} != expected
            ):
                raise ValueError(
                    f"{lid}: coverage target has missing variants/profiles"
                )
            for instance in matches:
                targets.append(
                    {
                        "instance_id": instance["id"],
                        "check_ids": target["checks"],
                        "contract_digest": mapping.get("preservation", {}).get(
                            "contract_digest"
                        ),
                    }
                )
        rows.append(
            {
                "id": lid,
                "disposition": mapping["disposition"],
                "rationale": mapping.get("rationale"),
                "targets": targets,
            }
        )
    return {"schema_version": 1, "legacy_cases": rows}
