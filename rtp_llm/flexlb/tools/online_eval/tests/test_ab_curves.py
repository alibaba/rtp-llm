import sys
import tempfile
import yaml
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from analysis.compare_ab import build_curve_spec
from reporting.renderer import render
from workload.cache_gate_ab import load_comparison_policy


class AbCurvesTest(unittest.TestCase):
    def test_scale_in_analysis_policy_from_runnable_scenario(self):
        policy = load_comparison_policy(ROOT / "config/scenarios/cache_scale_in.yaml")
        self.assertEqual(policy, {"comparison": "cache_scale_in", "alignment_event": "withdraw_start"})
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
            with tempfile.TemporaryDirectory() as d:
                path = Path(d) / "policy.yaml"
                path.write_text(yaml.safe_dump(bad))
                with self.assertRaisesRegex(ValueError, "unknown cache comparison fields"):
                    load_comparison_policy(path)
        self.assertEqual(validate_policy({"comparison": "cache_scale_in"}), {"comparison": "cache_scale_in"})

    def test_shared_axis_missing_sample_is_gap(self):
        a = {"label": "a", "aggregate": {"per_second": [
            {"t": 1, "success": 2}, {"t": 3, "success": 4}]}}
        b = {"label": "b", "aggregate": {"per_second": [
            {"t": 2, "success": 8}, {"t": 3, "success": 9}]}}
        spec = build_curve_spec(a, b, 1, 3)
        panel = next(p for p in spec["panels"] if p["id"] == "ab_per_second_success")
        self.assertEqual(panel["xNums"], [1, 2, 3])
        self.assertEqual(panel["series"][0]["data"], [2, None, 4])
        self.assertEqual(panel["series"][1]["data"], [None, 8, 9])
        page = render(spec)
        self.assertIn("FlexLegend.controller", page)
        self.assertIn("A/B 时序对比", page)
        self.assertNotIn("cdn.jsdelivr.net", page)
        self.assertIn("Chart.js v4.4.7", page)


if __name__ == "__main__":
    unittest.main()
