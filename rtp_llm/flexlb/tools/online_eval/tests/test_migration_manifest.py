"""Coverage bookkeeping must expose dropped contracts and profile drift."""

import copy
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from migration.audit_manifest import audit, load_inputs


class MigrationManifestTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.inputs = load_inputs()

    def test_all_contracts_retained_without_claiming_yaml_migration(self):
        result = audit(*copy.deepcopy(self.inputs))
        self.assertTrue(result["ok"], result["errors"])
        self.assertEqual(result["bookkeeping_counts"]["retained_legacy_cases"], 139)
        self.assertEqual(result["bookkeeping_counts"]["claimed_migrated_cases"], 0)
        self.assertEqual(result["planning_counts"]["target_logical_scenarios"], 29)
        self.assertEqual(result["legacy_instances"], 371)

    def test_missing_contract_and_profile_are_errors(self):
        for change in ("missing", "profile"):
            inputs = copy.deepcopy(self.inputs)
            rows = inputs[2]["legacy_cases"]
            key = next(iter(rows))
            if change == "missing":
                del rows[key]
            else:
                rows[key]["retained_profiles"] = []
            self.assertFalse(audit(*inputs)["ok"])

    def test_code_drift_and_duplicate_target_are_errors(self):
        inputs = copy.deepcopy(self.inputs)
        inputs[-1]["legacy_cases"][0]["contract_digest"] = "changed"
        self.assertFalse(audit(*inputs)["ok"])
        inputs = copy.deepcopy(self.inputs)
        families = inputs[1]["definitions"]
        families[1]["legacy_case_ids"].append(families[0]["legacy_case_ids"][0])
        self.assertFalse(audit(*inputs)["ok"])

    def test_unreviewed_migration_cannot_drop_legacy_invocation(self):
        inputs = copy.deepcopy(self.inputs)
        row = next(iter(inputs[2]["legacy_cases"].values()))
        row["disposition"] = "migrate"
        row["targets"] = []
        self.assertFalse(audit(*inputs)["ok"])


if __name__ == "__main__":
    unittest.main()
