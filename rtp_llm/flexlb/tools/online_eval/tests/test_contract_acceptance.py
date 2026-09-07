"""Mutation tests for the independent acceptance oracle; no backend starts."""

import copy
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from flexlb_ft.acceptance import (
    audit_coverage,
    audit_lifecycle,
    audit_results,
    audit_selection,
)


def documents():
    baseline = {
        "legacy_cases": [
            {
                "id": "old",
                "profiles": ["p", "q"],
                "requires": ["batch"],
                "expected_fail": False,
                "contract_digest": "frozen",
            }
        ]
    }
    instances = [
        {
            "id": f"scenario::v::{p}",
            "scenario_id": "scenario",
            "variant_id": "v",
            "profile": p,
            "requires": ["batch"],
            "stages": ["setup", "measure"],
            "checks": [{"id": "measure.latency", "numeric": True, "min_samples": 2}],
        }
        for p in ["p", "q"]
    ]
    inventory = {"instances": instances}
    coverage = {
        "legacy_cases": [
            {
                "id": "old",
                "disposition": "migrate",
                "targets": [
                    {
                        "instance_id": i["id"],
                        "check_ids": ["measure.latency"],
                        "contract_digest": "frozen",
                    }
                    for i in instances
                ],
            }
        ]
    }
    results = {
        "instances": [
            {
                "id": i["id"],
                "errors": [],
                "cleanup_errors": [],
                "leaked_resources": [],
                "stages": [{"id": s, "status": "PASS"} for s in i["stages"]],
                "checks": [
                    {
                        "id": "measure.latency",
                        "status": "PASS",
                        "value": 0,
                        "sample_count": 2,
                        "evidence_complete": True,
                    }
                ],
            }
            for i in instances
        ]
    }
    return baseline, inventory, coverage, results


class CoverageTest(unittest.TestCase):
    def test_definitions_instances_checks_are_distinct(self):
        b, i, c, _ = documents()
        result = audit_coverage(b, i, c)
        self.assertTrue(result["ok"])
        self.assertEqual(result["counts"]["legacy_cases"], 1)
        self.assertEqual(result["counts"]["scenario_definitions"], 1)
        self.assertEqual(result["counts"]["instances"], 2)
        self.assertEqual(result["counts"]["checks"], 2)

    def test_missing_case_profile_check_capability_or_digest_fails(self):
        for mutation in ["case", "profile", "check", "capability", "digest"]:
            with self.subTest(mutation=mutation):
                b, i, c, _ = documents()
                target = c["legacy_cases"][0]["targets"][0]
                if mutation == "case":
                    c["legacy_cases"] = []
                if mutation == "profile":
                    c["legacy_cases"][0]["targets"].pop()
                if mutation == "check":
                    target["check_ids"] = ["setup.action"]
                if mutation == "capability":
                    i["instances"][0]["requires"] = []
                if mutation == "digest":
                    target["contract_digest"] = "changed"
                self.assertFalse(audit_coverage(b, i, c)["ok"])

    def test_duplicate_case_and_instance_are_rejected(self):
        b, i, c, _ = documents()
        b["legacy_cases"] *= 2
        with self.assertRaises(ValueError):
            audit_coverage(b, i, c)
        b, i, c, _ = documents()
        i["instances"] *= 2
        with self.assertRaises(ValueError):
            audit_coverage(b, i, c)

    def test_retained_is_not_migrated_and_requires_real_legacy_binding(self):
        b, i, c, _ = documents()
        c["legacy_cases"][0].update(
            disposition="retain_legacy", rationale="backend missing"
        )
        self.assertFalse(audit_coverage(b, i, c)["ok"])
        for row in i["instances"]:
            row.update(backend="legacy", legacy_case_id="old")
        result = audit_coverage(b, i, c)
        self.assertTrue(result["ok"])
        self.assertEqual(result["counts"]["claimed_migrated_cases"], 0)
        self.assertEqual(result["counts"]["retained_legacy_cases"], 1)

    def test_expected_failure_cannot_be_silently_removed(self):
        b, i, c, _ = documents()
        b["legacy_cases"][0]["expected_fail"] = True
        self.assertFalse(audit_coverage(b, i, c)["ok"])
        for row in i["instances"]:
            row["checks"][0]["finding_id"] = "known"
        self.assertTrue(audit_coverage(b, i, c)["ok"])

    def test_numeric_checks_require_sample_contract(self):
        b, i, c, _ = documents()
        i["instances"][0]["checks"][0]["min_samples"] = 0
        self.assertFalse(audit_coverage(b, i, c)["ok"])


class EvidenceTest(unittest.TestCase):
    def test_zero_is_valid_only_with_real_samples(self):
        _, i, _, r = documents()
        self.assertTrue(audit_results(i, r)["ok"])
        for value in [None, float("nan"), True]:
            bad = copy.deepcopy(r)
            bad["instances"][0]["checks"][0]["value"] = value
            self.assertFalse(audit_results(i, bad)["ok"])
        r["instances"][0]["checks"][0]["sample_count"] = 0
        self.assertFalse(audit_results(i, r)["ok"])

    def test_finding_cannot_hide_execution_cleanup_timeout_or_missing_data(self):
        for mutation in [
            "contract",
            "error",
            "timeout",
            "cleanup",
            "missing",
            "blocked",
        ]:
            with self.subTest(mutation=mutation):
                _, i, _, r = documents()
                i["instances"][0]["checks"][0]["finding_id"] = "known"
                row = r["instances"][0]
                row["checks"][0].update(status="FAIL", failure_kind="contract")
                row["stages"][1]["status"] = "FAIL"
                if mutation == "error":
                    row["checks"][0]["failure_kind"] = "setup"
                if mutation == "timeout":
                    row["stages"][1]["status"] = "TIMEOUT"
                if mutation == "cleanup":
                    row["cleanup_errors"] = ["could not cancel"]
                if mutation == "missing":
                    row["checks"][0]["evidence_complete"] = False
                if mutation == "blocked":
                    row["checks"][0]["status"] = "BLOCKED"
                self.assertEqual(audit_results(i, r)["ok"], mutation == "contract")

    def test_finding_and_unrelated_failure_coexist(self):
        _, i, _, r = documents()
        i["instances"][0]["checks"][0]["finding_id"] = "known"
        for row in r["instances"]:
            row["checks"][0].update(status="FAIL", failure_kind="contract")
        result = audit_results(i, r)
        self.assertFalse(result["ok"])
        self.assertEqual(result["counts"]["FINDING-CONFIRMED"], 1)
        self.assertEqual(result["counts"]["FAIL"], 1)

    def test_empty_missing_or_blocked_never_counts_as_pass(self):
        self.assertFalse(audit_results({"instances": []}, {"instances": []})["ok"])
        _, i, _, r = documents()
        r["instances"][0]["checks"] = []
        result = audit_results(i, r)
        self.assertFalse(result["ok"])
        self.assertEqual(result["counts"]["BLOCKED"], 1)

    def test_cross_stage_resource_lifetime_and_cancel_not_release(self):
        events = [
            {"action": action, "resource_id": "flow", "epoch": 1, "stage": stage}
            for action, stage in [
                ("acquire", "setup"),
                ("use", "measure"),
                ("cancel", "cleanup"),
            ]
        ]
        self.assertFalse(audit_lifecycle(events)["ok"])
        events.append({"action": "release", "resource_id": "flow", "epoch": 1})
        self.assertTrue(audit_lifecycle(events)["ok"])
        events[-1]["epoch"] = 2
        self.assertFalse(audit_lifecycle(events)["ok"])

    def test_partial_setup_must_release_acquired_resources(self):
        events = [{"action": "acquire", "resource_id": "env", "epoch": 1}]
        self.assertFalse(audit_lifecycle(events)["ok"])
        events.append({"action": "release", "resource_id": "env", "epoch": 1})
        self.assertTrue(audit_lifecycle(events)["ok"])

    def test_serial_parallel_contract_set_equal_and_duplicates_rejected(self):
        _, serial, _, _ = documents()
        parallel = copy.deepcopy(serial)
        parallel["instances"].reverse()
        for n, row in enumerate(parallel["instances"]):
            row["lane"] = n
        self.assertTrue(audit_selection(serial, parallel)["ok"])
        parallel["instances"][0]["profile"] = "wrong"
        self.assertFalse(audit_selection(serial, parallel)["ok"])
        parallel["instances"] *= 2
        with self.assertRaises(ValueError):
            audit_selection(serial, parallel)


if __name__ == "__main__":
    unittest.main()
