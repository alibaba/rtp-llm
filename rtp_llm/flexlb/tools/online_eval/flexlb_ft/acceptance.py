"""Independent migration accounting; no environment or scenario execution."""

import argparse
import ast
import hashlib
import inspect
import json
import math
import textwrap
from collections import Counter
from pathlib import Path


def _index(rows, key, label):
    result = {}
    for row in rows:
        value = row[key]
        if not isinstance(value, str) or not value.strip() or value in result:
            raise ValueError(f"{label}: empty or duplicate {key}: {value!r}")
        result[value] = row
    return result


def snapshot(cases, profile_caps, revision):
    """Freeze ordered registration and source evidence, without running cases.

    A source digest is an audit anchor, not proof of YAML equivalence. Dynamic
    thresholds, effective configurations and evidence windows need a separate
    reviewed check mapping and paired execution artifacts.
    """
    rows = []
    for case in cases:
        node = ast.parse(textwrap.dedent(inspect.getsource(case.fn))).body[0]
        node.decorator_list = []
        metadata = {
            "id": case.name,
            "category": case.category,
            "profiles": [
                profile
                for profile, caps in profile_caps.items()
                if (case.profiles is None or profile in case.profiles)
                and set(case.requires or ()) <= set(caps)
            ],
            "declared_profiles": case.profiles,
            "requires": list(case.requires or ()),
            "expected_fail": case.expected_fail,
            "source": case.source,
        }
        digest_input = [metadata, ast.dump(node, include_attributes=False)]
        metadata["contract_digest"] = hashlib.sha256(
            json.dumps(digest_input, sort_keys=True).encode()
        ).hexdigest()
        rows.append(metadata)
    _index(rows, "id", "legacy")
    return {"schema_version": 1, "revision": revision, "legacy_cases": rows}


def audit_coverage(baseline, inventory, coverage):
    """Check explicit instance/check references and every legacy profile.

    New instances are allowed. Retained legacy instances are accounted for but
    never counted as migrated. A matching digest records the claimed source
    contract; it cannot certify the implementation of a check.
    """
    legacy = _index(baseline["legacy_cases"], "id", "legacy")
    instances = _index(inventory["instances"], "id", "instance")
    mappings = _index(coverage["legacy_cases"], "id", "coverage")
    errors = []
    checks = {}
    for iid, instance in instances.items():
        checks[iid] = _index(instance["checks"], "id", f"{iid} checks")
        for cid, check in checks[iid].items():
            if "." not in cid or cid.split(".", 1)[0] not in instance["stages"]:
                errors.append(f"{iid}/{cid}: unknown stage or non-qualified check")
            if check.get("numeric"):
                n = check.get("min_samples")
                if type(n) is not int or n < 1:
                    errors.append(
                        f"{iid}/{cid}: numeric check needs positive min_samples"
                    )
    for lid in mappings.keys() - legacy.keys():
        errors.append(f"unknown legacy case: {lid}")
    migrated = retained = 0
    for lid, old in legacy.items():
        mapping = mappings.get(lid)
        if mapping is None:
            errors.append(f"{lid}: missing coverage")
            continue
        disposition = mapping.get("disposition")
        if disposition not in ("migrate", "retain_legacy"):
            errors.append(f"{lid}: invalid disposition")
        if disposition == "retain_legacy" and not mapping.get("rationale"):
            errors.append(f"{lid}: retained case needs rationale")
        seen = set()
        for target in mapping.get("targets", []):
            iid = target["instance_id"]
            instance = instances.get(iid)
            if instance is None:
                errors.append(f"{lid}: unknown instance {iid}")
                continue
            profile = instance["profile"]
            if profile not in old["profiles"]:
                errors.append(f"{lid}/{iid}: profile outside frozen legacy contract")
            seen.add(profile)
            if target.get("contract_digest") != old["contract_digest"]:
                errors.append(f"{lid}/{iid}: source contract digest mismatch")
            if not set(old["requires"]) <= set(instance.get("requires", [])):
                errors.append(f"{lid}/{iid}: legacy capabilities dropped")
            if disposition == "retain_legacy" and (
                instance.get("backend") != "legacy"
                or instance.get("legacy_case_id") != lid
            ):
                errors.append(
                    f"{lid}/{iid}: retained coverage is not a legacy invocation"
                )
            ids = target.get("check_ids", [])
            if not ids or len(ids) != len(set(ids)):
                errors.append(f"{lid}/{iid}: empty or duplicate check references")
            selected = [checks[iid][cid] for cid in ids if cid in checks[iid]]
            if len(selected) != len(ids):
                errors.append(f"{lid}/{iid}: unknown check reference")
            findings = [check for check in selected if check.get("finding_id")]
            if bool(findings) != old["expected_fail"]:
                errors.append(f"{lid}/{iid}: expected-fail mapping changed")
        missing = set(old["profiles"]) - seen
        if missing:
            errors.append(f"{lid}: missing profiles {sorted(missing)}")
        migrated += disposition == "migrate"
        retained += disposition == "retain_legacy"
    return {
        "ok": not errors,
        "errors": errors,
        "counts": {
            "legacy_cases": len(legacy),
            "scenario_definitions": len({i["scenario_id"] for i in instances.values()}),
            "instances": len(instances),
            "checks": sum(map(len, checks.values())),
            "claimed_migrated_cases": migrated,
            "retained_legacy_cases": retained,
        },
    }


def audit_results(inventory, results):
    """Validate execution evidence without allowing a finding to hide errors."""
    planned = _index(inventory["instances"], "id", "instance")
    actual = _index(results["instances"], "id", "result")
    errors = [f"unexpected instance: {iid}" for iid in actual.keys() - planned.keys()]
    counts = Counter()
    for iid, instance in planned.items():
        result = actual.get(iid)
        if result is None:
            errors.append(f"{iid}: missing execution result")
            continue
        for field in ("errors", "cleanup_errors", "leaked_resources"):
            if not isinstance(result.get(field), list) or result[field]:
                errors.append(f"{iid}: {field} missing or nonempty")
        stages = _index(result["stages"], "id", f"{iid} stage result")
        for stage in stages.keys() - set(instance["stages"]):
            errors.append(f"{iid}/{stage}: unplanned stage result")
        for stage in instance["stages"]:
            if stage not in stages or stages[stage].get("status") not in (
                "PASS",
                "FAIL",
            ):
                errors.append(f"{iid}/{stage}: stage did not complete successfully")
        observed = _index(result["checks"], "id", f"{iid} check result")
        check_plan = _index(instance["checks"], "id", f"{iid} check")
        for cid in observed.keys() - check_plan.keys():
            errors.append(f"{iid}/{cid}: unplanned check")
        for stage, row in stages.items():
            if row.get("status") == "FAIL" and not any(
                cid.startswith(stage + ".") and item.get("status") == "FAIL"
                for cid, item in observed.items()
            ):
                errors.append(f"{iid}/{stage}: unexplained stage failure")
        for cid, check in check_plan.items():
            row = observed.get(cid)
            if row is None:
                errors.append(f"{iid}/{cid}: missing check evidence")
                counts["BLOCKED"] += 1
                continue
            status = row.get("status")
            evidence = row.get("evidence_complete") is True
            if check.get("numeric"):
                n, required, value = (
                    row.get("sample_count"),
                    check.get("min_samples"),
                    row.get("value"),
                )
                evidence = (
                    evidence
                    and type(n) is int
                    and type(required) is int
                    and required > 0
                    and n >= required
                    and type(value) in (int, float)
                    and math.isfinite(value)
                )
            if status not in ("PASS", "FAIL") or not evidence:
                errors.append(
                    f"{iid}/{cid}: incomplete evidence or non-contract outcome {status}"
                )
                counts[status if isinstance(status, str) else "ERROR"] += 1
            elif status == "FAIL":
                if check.get("finding_id") and row.get("failure_kind") == "contract":
                    counts["FINDING-CONFIRMED"] += 1
                else:
                    errors.append(
                        f"{iid}/{cid}: failure is not an expected contract finding"
                    )
                    counts["FAIL"] += 1
            else:
                counts["FINDING-RESOLVED" if check.get("finding_id") else "PASS"] += 1
    if not planned or not sum(counts.values()):
        errors.append("zero executed checks")
    return {"ok": not errors, "errors": errors, "counts": dict(counts)}


def audit_selection(serial, parallel):
    """Compare expanded instance contracts, allowing only lane assignment to vary."""

    def rows(document):
        return _index(
            [
                {k: v for k, v in row.items() if k != "lane"}
                for row in document["instances"]
            ],
            "id",
            "selection",
        )

    left, right = rows(serial), rows(parallel)
    errors = [
        f"instance differs: {iid}"
        for iid in left.keys() | right.keys()
        if left.get(iid) != right.get(iid)
    ]
    if not left:
        errors.append("empty selection")
    return {"ok": not errors, "errors": errors}


def audit_lifecycle(events):
    """Audit an executor's resource event trace, including partial setup.

    Keys include an epoch so an old-generation release cannot clean a new
    resource. Stage changes do not imply release. This checks reported events;
    it does not independently inspect processes or sockets.
    """
    owned = set()
    errors = []
    for event in events:
        key = (event["resource_id"], event["epoch"])
        action = event["action"]
        if action == "acquire":
            if key in owned:
                errors.append(f"duplicate acquire: {key}")
            owned.add(key)
        elif action in ("use", "release", "cancel"):
            if key not in owned:
                errors.append(f"{action} without ownership: {key}")
            if action == "release":
                owned.discard(key)
        else:
            errors.append(f"unknown resource action: {action}")
    if owned:
        errors.append(f"unreleased resources: {sorted(owned)}")
    return {"ok": not errors, "errors": errors}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    freeze = sub.add_parser("snapshot")
    freeze.add_argument("--revision", required=True)
    coverage = sub.add_parser("coverage")
    coverage.add_argument("baseline")
    coverage.add_argument("inventory")
    coverage.add_argument("coverage")
    results = sub.add_parser("results")
    results.add_argument("inventory")
    results.add_argument("results")
    lifecycle = sub.add_parser("lifecycle")
    lifecycle.add_argument("events")
    args = parser.parse_args()

    def read(path):
        return json.loads(Path(path).read_text())

    if args.command == "snapshot":
        from flexlb_functional_tests import ALL_CASES, PROFILE_CAPS

        report = snapshot(ALL_CASES, PROFILE_CAPS, args.revision)
    elif args.command == "coverage":
        report = audit_coverage(
            read(args.baseline), read(args.inventory), read(args.coverage)
        )
    elif args.command == "results":
        report = audit_results(read(args.inventory), read(args.results))
    else:
        report = audit_lifecycle(read(args.events))
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report.get("ok", True) else 1


if __name__ == "__main__":
    raise SystemExit(main())
