import copy
import unittest
from flexlb_test_framework.workload.compare import compare


class WorkloadCompareTest(unittest.TestCase):
    def report(self, offset, points):
        return dict(
            id="same",
            configuration_sha256="abc",
            workload={"runtime_validity": "VALID"},
            clock_anchor={"monotonic_s": offset},
            stages=[
                dict(
                    id="load",
                    status="PASS",
                    started_s=offset + 2,
                    finished_s=offset + 5,
                )
            ],
            series={"metric": points},
        )

    def test_clock_origins_are_not_compared_directly(self):
        a = self.report(100, [[2, 10], [3, 10]])
        b = self.report(999, [[2, 20], [3, 20]])
        r = compare(a, b)["changes"][0]
        self.assertEqual(r["relative_delta"], 1)
        self.assertEqual(r["baseline"], [[0, 10], [1, 10]])
        self.assertEqual(r["verdict"], "DESCRIPTIVE_ONLY")

    def test_missing_window_is_not_unchanged(self):
        a = self.report(1, [[2, 10]])
        b = self.report(2, [[6, 10]])
        r = compare(a, b)["changes"][0]
        self.assertEqual(r["status"], "MISSING_DATA")
        self.assertIsNone(r["rank_score"])

    def test_zero_baseline_and_changed_configuration_are_explicit(self):
        a = self.report(1, [[2, 0]])
        b = self.report(2, [[2, 10]])
        self.assertIsNone(compare(a, b)["changes"][0]["relative_delta"])
        b["configuration_sha256"] = "other"
        with self.assertRaises(ValueError):
            compare(a, b)
