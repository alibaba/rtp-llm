#!/usr/bin/env python3

import unittest

import numpy as np

from CompareLatencyLinearLeastSquares import design_matrix, fit_linear


class CompareLatencyLinearLeastSquaresTest(unittest.TestCase):
    def test_design_excludes_inactive_batch_count(self):
        inputs = np.arange(28, dtype=float).reshape(2, 14)
        design = design_matrix(inputs)
        self.assertEqual((2, 13), design.shape)
        np.testing.assert_array_equal(design[0],
                                      np.r_[inputs[0, :6], inputs[0, 7:14]])

    def test_exact_linear_target_for_both_losses(self):
        random = np.random.default_rng(7)
        inputs = np.column_stack((np.ones(80), random.uniform(0, 1, (80, 13))))
        design = design_matrix(inputs)
        target = 100.0 + 20.0 * inputs[:, 1] + 40.0 * inputs[:, 7]
        for loss in ("raw", "raw_ols", "log1p"):
            with self.subTest(loss=loss):
                coefficients, success, _, rank = fit_linear(inputs, target, loss)
                self.assertTrue(success)
                self.assertEqual(13, rank)
                np.testing.assert_allclose(design @ coefficients, target,
                                           atol=1e-5, rtol=0)


if __name__ == "__main__":
    unittest.main()
