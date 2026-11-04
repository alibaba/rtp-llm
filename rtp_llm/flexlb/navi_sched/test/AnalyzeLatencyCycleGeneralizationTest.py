#!/usr/bin/env python3

import unittest

import numpy as np

from AnalyzeLatencyCycleGeneralization import (
    aggregate_parameter_shifts,
    feature_neighbor_support,
)


class AnalyzeLatencyCycleGeneralizationTest(unittest.TestCase):
    def test_feature_neighbors_use_only_normalized_inputs(self):
        inputs = np.asarray([
            [0.0, 0.0],
            [1.0, 1.0],
            [0.1, 0.1],
            [10.0, 10.0],
        ])
        target = np.asarray([100.0, 200.0, 100.0, 10_000.0])
        ready_time = np.asarray([0, 1, 2, 3], dtype=np.int64)
        cycle = np.asarray([0, 0, 1, 1])

        result = feature_neighbor_support(
            inputs, target, ready_time, cycle, anomaly_cycle=1)

        self.assertEqual(2, result["active_feature_count"])
        self.assertAlmostEqual(
            2.0, result["normal_leave_one_out_distance_p95"])
        self.assertEqual(1, result["anomaly_outside_normal_p95_count"])
        self.assertEqual(1, result["supported_anomaly_count"])
        self.assertEqual(1, result["severe_target_conflict_count"])
        self.assertEqual(0, result["supported_severe_target_conflict_count"])
        self.assertFalse(result["examples"][0][
            "inside_normal_p95_support"])
        self.assertEqual([], result["supported_severe_examples"])
        self.assertAlmostEqual(9310.0, result["target_gap_absolute_p95_ms"])

    def test_feature_neighbors_report_supported_target_conflict(self):
        inputs = np.asarray([
            [0.0, 0.0],
            [1.0, 1.0],
            [0.1, 0.1],
        ])
        target = np.asarray([100.0, 200.0, 10_000.0])
        ready_time = np.asarray([0, 1, 2], dtype=np.int64)
        cycle = np.asarray([0, 0, 1])

        result = feature_neighbor_support(
            inputs, target, ready_time, cycle, anomaly_cycle=1)

        self.assertEqual(1, result["supported_anomaly_count"])
        self.assertEqual(1, result["severe_target_conflict_count"])
        self.assertEqual(1, result["supported_severe_target_conflict_count"])
        self.assertEqual(1, len(result["supported_severe_examples"]))
        example = result["supported_severe_examples"][0]
        self.assertAlmostEqual(0.2, example["distance"])
        self.assertEqual(2, example["anomaly_ready_time_us"])
        self.assertEqual(0, example["normal_ready_time_us"])

    def test_parameter_shift_uses_absolute_baseline_mean(self):
        hosts = []
        for normal, anomaly in ((-2.0, -1.0), (4.0, 8.0)):
            parameters_normal = [0.0] * 17
            parameters_anomaly = [0.0] * 17
            parameters_normal[0] = normal
            parameters_anomaly[0] = anomaly
            hosts.append({
                "architecture": "MI308",
                "fits": {"raw": {
                    "normal_only": {"parameters": parameters_normal},
                    "anomaly_only": {"parameters": parameters_anomaly},
                }},
            })
        result = aggregate_parameter_shifts(hosts)["MI308"]["raw"][0]
        self.assertAlmostEqual(3.0 * 0.878862443703,
                               result["baseline_absolute_mean"])
        self.assertAlmostEqual(2.5 * 0.878862443703,
                               result["absolute_delta_mean"])
        self.assertAlmostEqual(2.5 / 3.0,
                               result["absolute_delta_over_baseline_absolute_mean"])


if __name__ == "__main__":
    unittest.main()
