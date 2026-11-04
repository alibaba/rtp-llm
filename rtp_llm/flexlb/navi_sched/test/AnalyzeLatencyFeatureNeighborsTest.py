#!/usr/bin/env python3
"""Tests for same-host input-vector comparisons."""

import unittest

import numpy as np

from AnalyzeLatencyFeatureNeighbors import (
    FEATURES, analyze, parse_time, pressure_phases, relative_differences,
)


def sample(sequence, seconds, target_ms, **features):
    row = dict.fromkeys(FEATURES, 0.0)
    row.update(features)
    row.update(sequence=sequence, batch_id=sequence, event_time_us=int(seconds * 1e6),
               target_ms=target_ms)
    return row


class FeatureNeighborsTest(unittest.TestCase):
    def test_identical_features_different_latency(self):
        rows = [sample(1, 100, 10000), sample(2, 300, 20000)]
        result = analyze(rows, 0, 400_000_000)
        group = result["thresholds"]["0.02"]
        self.assertEqual(group["pair_count"], 1)
        self.assertEqual(group["focus_samples_with_severe_neighbor"], 2)
        self.assertEqual(group["examples"][0]["vector_distance"], 0)
        self.assertEqual(group["examples"][0]["target_ratio"], 2)
        self.assertEqual(group["nearest_neighbor_samples"], 2)
        self.assertEqual(group["nearest_neighbor_severe_samples"], 2)

    def test_examples_do_not_reuse_samples(self):
        rows = [sample(i + 1, 100 + i * 300, 10000 * 2**i) for i in range(4)]
        result = analyze(rows, 0, 2_000_000_000)
        group = result["thresholds"]["0.02"]
        self.assertEqual(group["pair_count"], 6)
        self.assertEqual(group["greedy_disjoint_severe_pair_count"], 2)
        ids = [e[side]["sequence"] for e in group["examples"] for side in ("left", "right")]
        self.assertEqual(len(ids), len(set(ids)))

    def test_relative_guard_rejects_outlier_scaled_neighbor(self):
        rows = [sample(1, 100, 10000, running_miss_tokens=1),
                sample(2, 400, 20000, running_miss_tokens=2),
                sample(3, 700, 20000, running_miss_tokens=10000)]
        unguarded = analyze(rows, 0, 200_000_000)
        guarded = analyze(rows, 0, 200_000_000, max_relative_difference=0.05)
        self.assertEqual(unguarded["thresholds"]["0.02"]["pair_count"], 1)
        self.assertEqual(guarded["thresholds"]["0.02"]["pair_count"], 0)

    def test_nearby_completions_excluded(self):
        result = analyze([sample(1, 100, 10000), sample(2, 110, 20000)], 0, 400_000_000)
        self.assertEqual(result["thresholds"]["0.2"]["pair_count"], 0)

    def test_nearby_admissions_excluded(self):
        result = analyze([sample(1, 100, 10000), sample(2, 200, 110000)], 0, 400_000_000)
        self.assertEqual(result["thresholds"]["0.2"]["pair_count"], 0)

    def test_all_components_compared(self):
        for feature in FEATURES:
            with self.subTest(feature=feature):
                rows = [sample(1, 100, 10000), sample(2, 300, 20000, **{feature: 100})]
                result = analyze(rows, 0, 400_000_000)
                self.assertEqual(result["nearest_distance_median"], 2)
                self.assertEqual(result["thresholds"]["0.2"]["pair_count"], 0)

    def test_unit_rescaling_preserves_distance(self):
        rows = [sample(i + 1, i * 200, 10000, running_miss_tokens=value)
                for i, value in enumerate((100, 101, 1000))]
        original = analyze(rows, 0, 500_000_000)
        for row in rows:
            row["running_miss_tokens"] *= 1024
        scaled = analyze(rows, 0, 500_000_000)
        self.assertAlmostEqual(original["nearest_distance_median"], scaled["nearest_distance_median"])
        self.assertEqual(original["thresholds"], scaled["thresholds"])

    def test_relative_guard_handles_zero(self):
        result = relative_differences(np.array([0, 1, 100]), np.array([0, 0, 95]))
        np.testing.assert_allclose(result, [0, 1, 0.05])

    def test_focus_end_is_exclusive(self):
        rows = [sample(1, 100, 10000), sample(2, 300, 20000)]
        result = analyze(rows, 100_000_000, 300_000_000)
        self.assertEqual(result["focus_count"], 1)
        self.assertEqual(result["thresholds"]["0.02"]["focus_samples_with_neighbor"], 1)

    def test_time_requires_timezone(self):
        with self.assertRaises(ValueError):
            parse_time("2026-10-09T17:30:00")
        self.assertEqual(parse_time("2026-10-09T17:30:00+08:00"),
                         parse_time("2026-10-09T09:30:00+00:00"))

    def test_opposite_phases_use_local_pressure_at_admission(self):
        rows = [sample(i, 90 + i, 10000, preceding_request_count=100)
                for i in range(3)]
        rows += [sample(10, 120, 10000, preceding_request_count=200)]
        rows += [sample(20 + i, 700 + i, 20000, preceding_request_count=300)
                 for i in range(3)]
        rows += [sample(30, 730, 20000, preceding_request_count=200)]
        phases = pressure_phases(rows)
        self.assertEqual(phases[3]["phase"], "rising")
        self.assertEqual(phases[-1]["phase"], "falling")
        result = analyze(rows, 0, 800_000_000, min_separation_s=300,
                         max_relative_difference=0.05, opposite_pressure_phase=True)
        group = result["thresholds"]["0.2"]
        self.assertEqual(group["pair_count"], 1)
        self.assertEqual(group["examples"][0]["inferred_admission_separation_s"], 600)
        self.assertEqual(group["examples"][0]["vector_distance"], 0)
        # Same pressure and latency difference do not override same-phase rejection.
        for row in rows[4:7]:
            row["preceding_request_count"] = 100
        same = analyze(rows, 0, 800_000_000, min_separation_s=300,
                       max_relative_difference=0.05, opposite_pressure_phase=True)
        self.assertEqual(same["thresholds"]["0.2"]["pair_count"], 0)

    def test_pressure_phase_unknown_and_flat_are_not_rising_or_falling(self):
        rows = [sample(i, 90 + i, 10000, preceding_request_count=100)
                for i in range(3)]
        rows += [sample(10, 120, 10000, preceding_request_count=100)]
        self.assertEqual(pressure_phases(rows)[0]["phase"], "unknown")
        self.assertEqual(pressure_phases(rows)[-1]["phase"], "flat")
        self.assertEqual(analyze(rows, 0, 200_000_000, min_separation_s=1,
                                 opposite_pressure_phase=True)
                         ["thresholds"]["0.2"]["pair_count"], 0)

    def test_minimum_separation_uses_admission_and_completion(self):
        rows = [sample(1, 100, 10000), sample(2, 700, 20000)]
        self.assertEqual(analyze(rows, 0, 800_000_000, min_separation_s=600)
                         ["thresholds"]["0.2"]["pair_count"], 0)
        self.assertEqual(analyze(rows, 0, 800_000_000, min_separation_s=590)
                         ["thresholds"]["0.2"]["pair_count"], 1)


if __name__ == "__main__":
    unittest.main()
