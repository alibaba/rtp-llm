#!/usr/bin/env python3

import unittest

from EvaluateLatencyControlPhases import (
    classify, evaluate, parse_control_update, qps_at,
)


class EvaluateLatencyControlPhasesTest(unittest.TestCase):
    def test_parse_control_update(self):
        line = ("rebased fixed replay pacing epoch, old qps [20.0], "
                "new qps [42.25], update time [100000000]")
        self.assertEqual((100000000, 42.25), parse_control_update(line))
        self.assertIsNone(parse_control_update("unrelated log"))
        with self.assertRaisesRegex(ValueError, "invalid replay QPS"):
            parse_control_update("new qps [nan], update time [100000000]")

    def test_phase_uses_admission_time_and_rejects_stale_control(self):
        times = [10_000_000, 40_000_000, 70_000_000]
        qps = [20.0, 40.0, 10.0]
        self.assertEqual("rising", classify(45_000_000, 5000.0, times, qps,
                                            30_000_000, 10.0, 60_000_000))
        self.assertEqual("falling", classify(75_000_000, 5000.0, times, qps,
                                             30_000_000, 10.0, 60_000_000))
        self.assertEqual("unknown", classify(200_000_000, 5000.0, times, qps,
                                             30_000_000, 10.0, 60_000_000))
        self.assertIsNone(qps_at(5_000_000, times, qps, 60_000_000))
        self.assertIsNone(qps_at(101_000_000, times, qps, 30_000_000))

    def test_evaluate_requires_logged_baseline_match(self):
        sample = (100.0, 120.0, 130.0, 120.0, 130.0, 0, 45_000_000)
        baseline = {("mi-1", 1): sample,
                    ("mi-1", 2): (100.0, 120.0, 130.0, 120.0, 131.0, 0,
                                  45_000_000)}
        candidate = {("mi-1", 1): (100.0, 120.0, 110.0, 120.0, 130.0, 0,
                                    45_000_000)}
        result = evaluate(baseline, candidate,
                          [(10_000_000, 20.0), (40_000_000, 40.0)],
                          30_000_000, 10.0, 60_000_000)
        self.assertEqual(1, result["coverage"]["MI_matched"])
        self.assertEqual(1, result["coverage"]["MI_baseline_mismatch"])
        rising = result["phases"]["MI"]["rising"]
        self.assertEqual(1, rising["count"])
        self.assertEqual(30.0, rising["fixed"]["mae_ms"])
        self.assertEqual(10.0, rising["candidate"]["mae_ms"])


if __name__ == "__main__":
    unittest.main()
