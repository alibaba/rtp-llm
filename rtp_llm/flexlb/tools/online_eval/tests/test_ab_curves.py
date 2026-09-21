import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from stress.reporting.compare_ab import build_curve_spec
from stress.reporting.renderer import render


class AbCurvesTest(unittest.TestCase):
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
