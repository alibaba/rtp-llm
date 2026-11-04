#!/usr/bin/env python3

import unittest

import numpy as np

from BootstrapLatencyHeldComparison import block_comparison


class BootstrapLatencyHeldComparisonTest(unittest.TestCase):
    def test_paired_blocks_keep_shared_sample_weights(self):
        blocks = {20: np.array([20, 40, 20, 60]),
                  10: np.array([10, 20, 10, 30])}
        result = block_comparison(blocks, repetitions=100, seed=1)
        self.assertEqual(2, result["block_count"])
        self.assertEqual(30, result["sample_count"])
        self.assertEqual(1.0, result["online_base"]["static_minus_online_mae_ms"])
        self.assertEqual(-1.0, result["online_final"]["static_minus_online_mae_ms"])
        self.assertEqual([1.0, 1.0], result["online_base"]["confidence_95_ms"])
        self.assertEqual([-1.0, -1.0], result["online_final"]["confidence_95_ms"])
        self.assertEqual(2, result["online_base"]["static_worse_blocks"])
        self.assertEqual(2, result["online_final"]["static_better_blocks"])


if __name__ == "__main__":
    unittest.main()
