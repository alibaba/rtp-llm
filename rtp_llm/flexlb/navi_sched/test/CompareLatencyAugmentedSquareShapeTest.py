#!/usr/bin/env python3

import unittest

import numpy as np

from AnalyzeLatencyPhaseLeastSquares import ACTIVE, PARAMETER_SCALES, prediction_and_jacobian
from CompareLatencyAugmentedSquareShape import (
    augmented_prediction_and_jacobian, fit_augmented,
)


class CompareLatencyAugmentedSquareShapeTest(unittest.TestCase):
    def setUp(self):
        random = np.random.default_rng(11)
        self.inputs = random.uniform(0.0, 2.0, (80, 14))
        self.inputs[:, 0] = 1.0
        self.parameters = np.zeros(17)
        self.parameters[0] = 4.0
        self.parameters[1:6] = [20.0, 4.0, 3.0, 2.0, 1.0]
        self.parameters[7] = 100.0
        self.parameters[8] = 10.0
        self.parameters[9] = 1.0
        self.parameters[10:15] = [12.0, 5.0, 3.0, 2.0, 1.0]
        self.parameters[15] = 1.0 / PARAMETER_SCALES[15]
        self.parameters[16] = 2.0

    def test_zero_weight_is_exact_original_formula(self):
        values = np.r_[self.parameters[ACTIVE], 0.0]
        predicted, gradient = augmented_prediction_and_jacobian(
            self.inputs, values, 3.0)
        original, original_gradient = prediction_and_jacobian(
            self.inputs, self.parameters[ACTIVE], "none")
        np.testing.assert_allclose(predicted, original, rtol=0, atol=1e-10)
        np.testing.assert_allclose(gradient[:, :-1], original_gradient,
                                   rtol=1e-10, atol=1e-10)

    def test_analytic_gradient_and_nested_fit(self):
        values = np.r_[self.parameters[ACTIVE], 20.0]
        predicted, gradient = augmented_prediction_and_jacobian(
            self.inputs, values, 3.0)
        for index in range(len(values)):
            delta = np.zeros(len(values))
            delta[index] = 1e-4
            forward = augmented_prediction_and_jacobian(
                self.inputs, values + delta, 3.0)[0]
            backward = augmented_prediction_and_jacobian(
                self.inputs, values - delta, 3.0)[0]
            np.testing.assert_allclose(gradient[:, index],
                                       (forward - backward) / (2e-4),
                                       rtol=1e-5, atol=1e-5)
        result, scale, baseline_cost, _ = fit_augmented(
            self.inputs, predicted, self.parameters, "log1p")
        self.assertTrue(result.success)
        self.assertLess(result.cost, baseline_cost)
        np.testing.assert_allclose(
            augmented_prediction_and_jacobian(self.inputs, result.x, scale)[0],
            predicted, rtol=0, atol=1e-4)

    def test_nonnegative_weight_keeps_original_baseline_feasible(self):
        target = augmented_prediction_and_jacobian(
            self.inputs, np.r_[self.parameters[ACTIVE], -20.0], 3.0)[0]
        result, _, baseline_cost, _ = fit_augmented(
            self.inputs, target, self.parameters, "log1p", nonnegative_weight=True)
        self.assertGreaterEqual(result.x[-1], 0.0)
        self.assertLessEqual(result.cost, baseline_cost + 1e-8)


if __name__ == "__main__":
    unittest.main()
