"""Scenario v1 adapter tests use result fixtures, not a backend."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from flexlb_ft.acceptance import audit_results
from flexlb_ft.scenario_acceptance import (
    expand_coverage,
    normalize_plans,
    normalize_results,
)


def fixture():
    plan = {
        "id": "s::v::p",
        "scenario_id": "s",
        "variant_id": "v",
        "profile": "p",
        "category": "status",
        "requires": [],
        "source": "yaml",
        "source_path": "s.yaml",
        "legacy_case_ids": ["old"],
        "environment": {"resolved_config": {"axis": "p"}},
        "resource_budget": {"bounded": True},
        "findings": [],
        "stages": [
            {"id": "setup", "action": "setup", "check_ids": []},
            {"id": "verify", "action": "check", "check_ids": ["comparison"]},
        ],
    }
    result = {
        "id": plan["id"],
        "status": "PASS",
        "error": None,
        "cleanup": [{"id": "environment", "status": "PASS", "error": None}],
        "stages": [
            {"id": "setup", "status": "PASS", "checks": []},
            {
                "id": "verify",
                "status": "PASS",
                "checks": [
                    {
                        "id": "comparison",
                        "status": "PASS",
                        "actual": True,
                        "expected": True,
                        "evidence": {},
                    }
                ],
            },
        ],
    }
    return plan, result


class ScenarioAcceptanceTest(unittest.TestCase):
    def test_scalar_comparisons_preserve_core_type_boundaries(self):
        for actual, expected, op, complete in [
            (0.5, 1, "le", True),
            (1, 0.5, "ge", True),
            ("ready", "ready", "eq", True),
            ("a", "b", "le", False),
            (True, 1, "eq", False),
            (1, True, "eq", False),
            (float("nan"), 1, "le", False),
            (1, float("inf"), "le", False),
            (None, None, "eq", False),
        ]:
            with self.subTest(actual=actual, expected=expected, op=op):
                p, r = fixture()
                p["stages"][1]["params"] = {"op": op}
                r["stages"][1]["checks"][0].update(actual=actual, expected=expected)
                normalized = normalize_results([p], [r])
                self.assertEqual(
                    normalized["instances"][0]["checks"][0]["evidence_complete"],
                    complete,
                )
                self.assertEqual(
                    audit_results(normalize_plans([p]), normalized)["ok"], complete
                )

    def test_effective_configuration_is_not_inferred_from_profile_name(self):
        p, _ = fixture()
        p.update(
            profile="batch-window",
            grade="strict",
            effective_axes={"dispatcher": "NON_BATCH", "scheduler": "SINGLE"},
            effective_capabilities=["single_scheduler"],
        )
        row = normalize_plans([p])["instances"][0]
        self.assertEqual(row["grade"], "strict")
        self.assertEqual(row["effective_axes"], p["effective_axes"])
        self.assertEqual(row["effective_capabilities"], p["effective_capabilities"])
        p, _ = fixture()
        row = normalize_plans([p])["instances"][0]
        self.assertIsNone(row["grade"])
        self.assertIsNone(row["effective_axes"])
        self.assertIsNone(row["effective_capabilities"])

    def test_core_boolean_comparison_has_typed_evidence(self):
        p, r = fixture()
        inventory = normalize_plans([p])
        self.assertEqual(inventory["instances"][0]["resolved_config"], {"axis": "p"})
        self.assertTrue(audit_results(inventory, normalize_results([p], [r]))["ok"])

    def test_adapter_empty_evidence_never_becomes_measured_zero(self):
        p, r = fixture()
        p["stages"][1]["action"] = "observe"
        row = r["stages"][1]["checks"][0]
        row.update(actual=0, expected=1)
        contracts = {
            p["id"]: {"verify.comparison": {"numeric": True, "min_samples": 2}}
        }
        i = normalize_plans([p], contracts)
        self.assertFalse(audit_results(i, normalize_results([p], [r]))["ok"])
        row["evidence"] = {"complete": True, "sample_count": 2}
        self.assertTrue(audit_results(i, normalize_results([p], [r]))["ok"])

    def test_blocked_and_cleanup_errors_survive_normalization(self):
        p, r = fixture()
        r["status"] = "TIMEOUT"
        r["stages"][1]["status"] = "BLOCKED"
        r["stages"][1]["checks"] = []
        r["cleanup"][0]["status"] = "ERROR"
        normalized = normalize_results([p], [r])
        self.assertEqual(
            normalized["instances"][0]["leaked_resources"], ["environment"]
        )
        self.assertFalse(audit_results(normalize_plans([p]), normalized)["ok"])

    def test_setup_only_is_not_validation_success(self):
        p, r = fixture()
        p["stages"] = p["stages"][:1]
        r["stages"] = r["stages"][:1]
        self.assertFalse(
            audit_results(normalize_plans([p]), normalize_results([p], [r]))["ok"]
        )

    def test_coverage_expands_only_real_declared_instances(self):
        p, _ = fixture()
        i = normalize_plans([p])
        doc = {
            "legacy_cases": {
                "old": {
                    "disposition": "migrate",
                    "preservation": {"contract_digest": "frozen"},
                    "targets": [
                        {
                            "scenario": "s",
                            "variants": ["v"],
                            "profiles": ["p"],
                            "checks": ["verify.comparison"],
                        }
                    ],
                }
            }
        }
        result = expand_coverage(doc, i)
        self.assertEqual(
            result["legacy_cases"][0]["targets"][0]["instance_id"], p["id"]
        )
        doc["legacy_cases"]["old"]["targets"][0]["profiles"].append("missing")
        with self.assertRaises(ValueError):
            expand_coverage(doc, i)

    def test_contract_override_cannot_create_check_or_change_finding(self):
        p, _ = fixture()
        for contract in [
            {"missing.check": {}},
            {"verify.comparison": {"finding_id": "fake"}},
        ]:
            with self.assertRaises(ValueError):
                normalize_plans([p], {p["id"]: contract})

    def test_finding_failure_does_not_hide_top_level_error(self):
        p, r = fixture()
        p["findings"] = ["verify.comparison"]
        r["status"] = "ERROR"
        r["error"] = "cleanup failed"
        r["stages"][1]["status"] = "FAIL"
        r["stages"][1]["checks"][0].update(status="FAIL", actual=False)
        self.assertFalse(
            audit_results(normalize_plans([p]), normalize_results([p], [r]))["ok"]
        )


if __name__ == "__main__":
    unittest.main()
