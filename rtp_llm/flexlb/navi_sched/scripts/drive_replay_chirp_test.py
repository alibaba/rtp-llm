import math
import unittest

from drive_replay_chirp import ZERO_QPS_FLOOR, control_qps, history_arguments, wave


class DriveReplayChirpTest(unittest.TestCase):
    def test_explicit_audited_range(self):
        self.assertEqual(history_arguments(100, 200, None),
                         ["--history-mode", "time_range", "--start-time-us", "100",
                          "--end-time-us", "200"])

    def test_default_recent_count_unchanged(self):
        self.assertEqual(history_arguments(None, None, None),
                         ["--history-mode", "recent_count", "--recent-query-count", "1000"])
        self.assertEqual(history_arguments(None, None, 4)[-1], "4")

    def test_reject_ambiguous_or_invalid_history(self):
        for args in ((100, 200, 4), (100, None, None), (None, 200, None),
                     (200, 100, None), (0, 200, None), (None, None, 0)):
            with self.subTest(args=args), self.assertRaises(ValueError):
                history_arguments(*args)

    def test_warmup_and_sweep_endpoints(self):
        arguments = (0.0, 400.0, 300.0, 45.0, 3, 3600.0)
        for elapsed, expected_period, expected_stage in (
                (0, 300, "warmup"), (900, 300, "warmup"),
                (2700, 45, "frequency_up"),
                (4500, 300, "frequency_down")):
            qps, period, stage = wave(elapsed, *arguments)
            self.assertAlmostEqual(0.0, qps, places=6)
            self.assertAlmostEqual(expected_period, period, places=6)
            self.assertEqual(expected_stage, stage)

    def test_phase_and_qps_are_continuous_at_boundaries(self):
        arguments = (0.0, 400.0, 300.0, 45.0, 3, 3600.0)
        for boundary in (900.0, 2700.0):
            before = wave(boundary - 0.001, *arguments)[0]
            after = wave(boundary + 0.001, *arguments)[0]
            self.assertLess(abs(after - before), 0.01)
            before_slope = (wave(boundary, *arguments)[0] -
                            wave(boundary - 0.001, *arguments)[0]) / 0.001
            after_slope = (wave(boundary + 0.001, *arguments)[0] -
                           wave(boundary, *arguments)[0]) / 0.001
            self.assertLess(abs(after_slope - before_slope), 0.01)

    def test_full_sweep_covers_qps_range_without_nonfinite_values(self):
        arguments = (0.0, 400.0, 300.0, 45.0, 3, 3600.0)
        values = [wave(float(tick), *arguments)[0] for tick in range(4501)]
        self.assertTrue(all(math.isfinite(value) and 0 <= value <= 400
                            for value in values))
        self.assertLess(min(values), 0.01)
        self.assertGreater(max(values), 399.99)

    def test_zero_target_uses_control_floor(self):
        self.assertEqual(ZERO_QPS_FLOOR, control_qps(0.0))
        self.assertEqual(100.0, control_qps(100.0))


if __name__ == "__main__":
    unittest.main()
