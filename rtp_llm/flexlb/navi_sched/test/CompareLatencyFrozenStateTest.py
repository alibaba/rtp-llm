#!/usr/bin/env python3
"""Tests for frozen-state replay anchor selection."""

import unittest

from CompareLatencyFrozenState import select_anchors, select_sample_pressure_anchor


class CompareLatencyFrozenStateTest(unittest.TestCase):
    def test_selects_updates_on_both_sides_of_first_high_sample(self):
        events = [
            {"type": "update", "sequence": 2},
            {"type": "sample", "sequence": 3, "event_time_us": 10,
             "target_ms": 100},
            {"type": "sample", "sequence": 4, "event_time_us": 20,
             "target_ms": 30000},
            {"type": "update", "sequence": 4},
        ]
        high, before, after = select_anchors(events, 10)
        self.assertEqual(4, high["sequence"])
        self.assertEqual(2, before)
        self.assertEqual(4, after)

    def test_missing_preceding_update_does_not_invent_anchor(self):
        events = [
            {"type": "sample", "sequence": 10, "event_time_us": 20,
             "target_ms": 30000},
            {"type": "update", "sequence": 12},
        ]
        high, before, after = select_anchors(events, 20)
        self.assertEqual(10, high["sequence"])
        self.assertIsNone(before)
        self.assertEqual(12, after)

    def test_selects_latest_fitting_sample_within_time_and_pressure(self):
        def sample(sequence, time_us, running, waiting):
            return {"type": "sample", "sequence": sequence,
                    "history_sequence": sequence, "event_time_us": time_us,
                    "features": [0, running, 0, 0, 0, 0, waiting]}

        events = [sample(1, 10, 5, 2),
                  {"type": "update", "sequence": 1, "fit_batches": [[1]]},
                  sample(2, 20, 8, 3),
                  {"type": "update", "sequence": 2, "fit_batches": [[2]]},
                  sample(3, 30, 3, 4),
                  {"type": "update", "sequence": 3, "fit_batches": [[3]]},
                  sample(4, 15, 1, 1),
                  {"type": "update", "sequence": 4, "fit_batches": [[4]]}]
        self.assertEqual(3, select_sample_pressure_anchor(events, 30, 10))
        self.assertEqual(4, select_sample_pressure_anchor(events, 25, 10))
        with self.assertRaisesRegex(ValueError, "no fitting update"):
            select_sample_pressure_anchor(events, 5, 10)
        with self.assertRaisesRegex(ValueError, "no fitting update"):
            select_sample_pressure_anchor(events, 30, 1)


if __name__ == "__main__":
    unittest.main()
