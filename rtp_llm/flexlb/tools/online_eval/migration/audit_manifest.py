"""Static migration bookkeeping; never substitutes for paired execution evidence."""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import yaml

TOOLS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOOLS))
from flexlb_ft.acceptance import audit_coverage, snapshot
from flexlb_ft.harness import PROFILE_CAPS
from flexlb_ft.scenario import compile_scenarios, load_scenarios
from flexlb_ft.scenario.catalog import handlers
from flexlb_ft.scenario.compiler import plan_counts
from flexlb_ft.scenario_acceptance import normalize_plans
from flexlb_functional_tests import ALL_CASES


def audit_candidates(families, plans):
    """Check progress claims against programs, never infer static equivalence."""
    errors, covered_all, statuses = [], set(), Counter()
    for family in families:
        fid, assigned = family["id"], set(family["legacy_case_ids"])
        selected = [p for p in plans if assigned.intersection(p["legacy_case_ids"])]
        covered = {lid for p in selected for lid in p["legacy_case_ids"]}
        covered_all.update(covered & assigned)
        if covered - assigned:
            errors.append(f"{fid}: candidate program crosses target family ownership")
        state = (
            ("candidate_complete" if covered == assigned else "candidate_partial")
            if selected
            else "planned"
        )
        statuses[state] += 1
        expected = {
            "implementation_status": state,
            "candidate_contracts": sorted(covered),
            "pending_contracts": sorted(assigned - covered),
            "candidate_scenario_ids": sorted({p["scenario_id"] for p in selected}),
            "candidate_counts": plan_counts(selected),
        }
        try:
            expected["source_files"] = sorted(
                {
                    str(Path(p["source_path"]).resolve().relative_to(TOOLS))
                    for p in selected
                }
            )
        except ValueError:
            errors.append(f"{fid}: candidate source is outside the audited tools tree")
        for field, value in expected.items():
            if family.get(field) != value:
                errors.append(
                    f"{fid}: {field} differs from compiled candidate inventory"
                )
        review = family.get("static_review", {})
        if review.get("status") == "passed_at_revision":
            reviewed = review.get("reviewed_contracts", [])
            if (
                not isinstance(reviewed, list)
                or not reviewed
                or not all(isinstance(lid, str) for lid in reviewed)
                or len(set(reviewed)) != len(reviewed)
                or not set(reviewed) <= covered
            ):
                errors.append(
                    f"{fid}: static review includes duplicate or absent candidates"
                )
            elif review.get("unreviewed_candidate_contracts") != sorted(
                covered - set(reviewed)
            ):
                errors.append(
                    f"{fid}: unreviewed candidates do not match reviewed scope"
                )
            revision = review.get("revision", "")
            if (
                not isinstance(revision, str)
                or len(revision) != 40
                or any(char not in "0123456789abcdef" for char in revision)
            ):
                errors.append(f"{fid}: static review needs full commit-hash syntax")
        elif review.get("status") != "pending":
            errors.append(f"{fid}: missing or unsupported static review status")
    return errors, {
        "candidate_contracts": len(covered_all),
        "target_families_by_implementation_status": dict(statuses),
        "scope": "Candidate existence only; static signatures and runtime acceptance remain separate.",
    }


def audit(baseline, targets, coverage, plans, current):
    errors = []
    frozen = {row["id"]: row for row in baseline["legacy_cases"]}
    actual = {row["id"]: row for row in current["legacy_cases"]}
    rows = coverage["legacy_cases"]
    definitions = {row["id"]: row for row in targets["definitions"]}
    if (
        targets["current_contracts"] != len(frozen)
        or targets["document_baseline_contracts"] != 131
        or len(frozen) - len(targets["supplemental_contracts"]) != 131
    ):
        errors.append("document/current/supplemental contract counts do not reconcile")
    if (
        len(definitions) != 29
        or len(targets["definitions"]) != 29
        or targets["target_logical_scenarios"] != 29
    ):
        errors.append(
            "target manifest must account for the 29 reviewed logical families"
        )
    if set(rows) != set(frozen):
        errors.append("coverage set differs from frozen contracts")
    if set(actual) - set(frozen):
        errors.append("new registry contracts need an explicit baseline/target review")
    assignments = {}
    for family in targets["definitions"]:
        for lid in family["legacy_case_ids"]:
            if lid in assignments:
                errors.append(f"{lid}: assigned to multiple target families")
            assignments[lid] = family["id"]
            if lid in frozen and family["category"] != frozen[lid]["category"]:
                errors.append(f"{lid}: category changed")
    if set(assignments) != set(frozen):
        errors.append("target families omit or invent contracts")
    candidate_errors, candidate_progress = audit_candidates(
        targets["definitions"], plans
    )
    errors.extend(candidate_errors)
    inventory = normalize_plans(plans)
    normalized = []
    for lid, old in frozen.items():
        row = rows.get(lid)
        if row is None:
            continue
        if row.get("planned_scenario") != assignments.get(lid):
            errors.append(f"{lid}: planned family differs from manifest")
        entry = dict(
            id=lid,
            disposition=row["disposition"],
            rationale=row.get("rationale"),
            targets=[],
        )
        if row["disposition"] == "retain_legacy":
            live = actual.get(lid)
            if live is None or live["contract_digest"] != old["contract_digest"]:
                errors.append(
                    f"{lid}: retained implementation/metadata drift requires review"
                )
            profiles = row.get("retained_profiles", [])
            if set(profiles) != set(old["profiles"]) or len(profiles) != len(
                set(profiles)
            ):
                errors.append(
                    f"{lid}: retained profile set differs from frozen contract"
                )
            for profile in profiles:
                iid = f"legacy::{lid}::{profile}"
                check = dict(id="legacy.contract")
                if old["expected_fail"]:
                    check["finding_id"] = "legacy.declared_finding"
                inventory["instances"].append(
                    dict(
                        id=iid,
                        scenario_id=lid,
                        profile=profile,
                        category=old["category"],
                        requires=old["requires"],
                        source="legacy",
                        backend="legacy",
                        legacy_case_id=lid,
                        stages=["legacy"],
                        checks=[check],
                    )
                )
                entry["targets"].append(
                    dict(
                        instance_id=iid,
                        check_ids=["legacy.contract"],
                        contract_digest=old["contract_digest"],
                    )
                )
        elif row["disposition"] == "migrate":
            # These targets and their preserved digest must be explicitly supplied
            # by the migration owner; never fill them from the current code.
            entry["targets"] = row.get("targets", [])
        normalized.append(entry)
    report = audit_coverage(
        baseline, inventory, dict(schema_version=1, legacy_cases=normalized)
    )
    report["errors"][:0] = errors
    report["ok"] = not report["errors"]
    report["bookkeeping_counts"] = report.pop("counts")
    report["bookkeeping_check_scope"] = (
        "legacy callable result boundaries plus declared YAML checks; not a count of original assertions"
    )
    report["scope"] = (
        "static coverage bookkeeping; paired runtime evidence is still required"
    )
    report["planning_counts"] = dict(
        target_logical_scenarios=29,
        document_baseline_contracts=131,
        current_contracts=len(frozen),
        supplemental_contracts=len(targets["supplemental_contracts"]),
    )
    report["candidate_progress"] = candidate_progress
    report["yaml_counts"] = plan_counts(plans)
    report["legacy_instances"] = sum(
        len(row["profiles"]) for row in baseline["legacy_cases"]
    )
    return report


def load_inputs():
    root = Path(__file__).resolve().parent
    return (
        json.loads((root / "baseline.json").read_text()),
        json.loads((root / "target_manifest.json").read_text()),
        yaml.safe_load((root / "coverage.yaml").read_text()),
        compile_scenarios(load_scenarios(TOOLS / "scenarios"), handlers=handlers()),
        snapshot(ALL_CASES, PROFILE_CAPS, "current working tree"),
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    result = audit(*load_inputs())
    text = json.dumps(result, indent=2) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text)
    print(text, end="")
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
