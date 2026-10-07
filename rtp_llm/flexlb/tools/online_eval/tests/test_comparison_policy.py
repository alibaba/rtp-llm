import sys
import yaml
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


class ComparisonPolicyTest(unittest.TestCase):
    def test_scale_in_analysis_policy_from_runnable_scenario(self):
        policy = yaml.safe_load((ROOT / "config/scenarios/cache_scale_in.yaml").read_text())["analysis"]
        self.assertEqual(policy, {"alignment_event": "withdraw_start"})
        from workload.cache_comparison_config import validate_policy
        from cases.config import configure_program
        original = yaml.safe_load((ROOT / "config/scenarios/cache_scale_in.yaml").read_text())
        for field, value in (("expected_verdicts", {"old": "FAIL", "new": "PASS"}), ("mode", "strong")):
            bad = dict(policy, **{field: value})
            with self.assertRaisesRegex(ValueError, "unknown cache comparison fields"):
                validate_policy(bad)
            original["analysis"] = bad
            with self.assertRaisesRegex(Exception, "unknown cache comparison fields"):
                configure_program(original, "test")
        self.assertEqual(validate_policy({}), {})
        with self.assertRaisesRegex(ValueError, "unknown cache comparison fields"):
            validate_policy({"comparison": "cache_scale_in"})


if __name__ == "__main__":
    unittest.main()
