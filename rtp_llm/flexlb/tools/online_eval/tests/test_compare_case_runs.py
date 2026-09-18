import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from compare_case_runs import CaseComparisonError, chart_spec, compare, render_table


def case(status, check_status="PASS", duration=10):
    return {"id": "case::sb", "profile": "single-batch", "grade": "normal",
            "test_kind": "functional", "status": status, "duration_ms": duration,
            "stages": [{"id": "verify", "checks": [{"id": "result", "status": check_status}]}]}


class CompareCasesTest(unittest.TestCase):
    def test_finding_is_not_ordinary_fail(self):
        result = compare({"instances": [case("FINDING-CONFIRMED", "FAIL")]},
                         {"instances": [case("FINDING-RESOLVED", "PASS", 20)]})
        self.assertEqual(result["summary"]["changed"], 1)
        self.assertEqual(result["instances"][0]["failed_checks_a"], ["verify.result"])
        self.assertEqual(result["instances"][0]["duration_delta_ms"], 10)
        self.assertIn("FINDING-CONFIRMED", render_table(result))
        self.assertEqual(chart_spec(result)["panels"][0]["series"][1]["data"], [20])

    def test_different_instance_set_rejected(self):
        with self.assertRaisesRegex(CaseComparisonError, "instance sets differ"):
            compare({"instances": [case("PASS")]}, {"instances": []})


if __name__ == "__main__":
    unittest.main()
