#!/usr/bin/env python3

import unittest

import numpy as np

from CompareLatencyGlobalLeastSquares import SEGMENTS, aggregate_metrics, segment_masks


class CompareLatencyGlobalLeastSquaresTest(unittest.TestCase):
    def test_pressure_and_short_latency_are_distinct(self):
        inputs = np.zeros((4, 14))
        inputs[:, 1] = [0, 4, 5, 0]
        inputs[:, 7] = [0, 4, 0, 33]
        target = np.array([100.0, 300.0, 200.0, 1000.0])
        masks = segment_masks(inputs, target)
        np.testing.assert_array_equal(masks["low"], [True, True, False, False])
        np.testing.assert_array_equal(masks["high"], [False, False, False, True])
        np.testing.assert_array_equal(masks["middle"], [False, False, True, False])
        np.testing.assert_array_equal(masks["short"], [True, False, True, False])
        np.testing.assert_array_equal(masks["waiting_0"], [True, False, True, False])
        np.testing.assert_array_equal(masks["waiting_1_4"], [False, True, False, False])
        np.testing.assert_array_equal(masks["waiting_33_64"],
                                      [False, False, False, True])

    def test_waiting_bins_cover_every_count_once(self):
        waiting = [0, 1, 4, 5, 16, 17, 32, 33, 64, 65, 128, 129, 256, 257]
        inputs = np.zeros((len(waiting), 14))
        inputs[:, 7] = waiting
        masks = segment_masks(inputs, np.ones(len(waiting)))
        bins = [mask for name, mask in masks.items() if name.startswith("waiting_")]
        np.testing.assert_array_equal(np.sum(bins, axis=0), np.ones(len(waiting)))
        np.testing.assert_array_equal(masks["low"] | masks["middle"] | masks["high"],
                                      np.ones(len(waiting), dtype=bool))
        self.assertTrue(masks["waiting_257_plus"][-1])

    def test_weighted_mae_across_all_samples(self):
        hosts = []
        for count, static, base, final in ((1, 30.0, 20.0, 10.0),
                                           (3, 50.0, 40.0, 30.0)):
            segment = {name: {"n": count, "mae_ms": value,
                              "bias_ms": -5.0 if name == "static" else 0.0}
                       for name, value in (("static", static),
                                           ("online_base", base),
                                           ("online_final", final))}
            hosts.append({"architecture": "MI", "fit_success": True,
                          "full_window": {name: {**segment, "mean_running": 2.0,
                                                 "mean_waiting": 3.0}
                                          for name in SEGMENTS}})
        result = aggregate_metrics(hosts, "MI")
        self.assertEqual(45.0, result["high"]["static"]["mae_ms"])
        self.assertEqual(35.0, result["high"]["online_base"]["mae_ms"])
        self.assertEqual(25.0, result["high"]["online_final"]["mae_ms"])
        self.assertEqual(4, result["high"]["static"]["n"])
        self.assertEqual(5.0, result["high"]["static"]["signed_error_ms"])
        self.assertEqual(25.0, result["high"]["static"]
                         ["overprediction_contribution_ms"])
        self.assertEqual(20.0, result["high"]["static"]
                         ["underprediction_contribution_ms"])
        self.assertEqual(2.0, result["high"]["mean_running"])
        self.assertEqual(3.0, result["high"]["mean_waiting"])

    def test_empty_waiting_bin(self):
        empty = {name: None for name in ("static", "online_base", "online_final")}
        hosts = [{"architecture": "MI", "fit_success": True,
                  "full_window": {name: {**empty, "mean_running": None,
                                         "mean_waiting": None}
                                  for name in SEGMENTS}}]
        result = aggregate_metrics(hosts, "MI")
        self.assertEqual(0, result["waiting_257_plus"]["static"]["n"])
        self.assertIsNone(result["waiting_257_plus"]["static"]["mae_ms"])
        self.assertIsNone(result["waiting_257_plus"]["mean_running"])


if __name__ == "__main__":
    unittest.main()
