#!/usr/bin/env python3

import unittest

import numpy as np

from AnalyzeLatencyPhaseLeastSquares import ACTIVE, PARAMETER_SCALES, prediction_and_jacobian
from CompareLatencyQuarticRootShape import (
    fit_quartic, quartic_prediction_and_jacobian,
)


class CompareLatencyQuarticRootShapeTest(unittest.TestCase):
    def setUp(self):
        random = np.random.default_rng(19)
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

    def test_zero_quartic_is_original_formula(self):
        values = np.r_[self.parameters[ACTIVE], 0.0, 1.0]
        predicted, gradient = quartic_prediction_and_jacobian(self.inputs, values, 3.0)
        original, original_gradient = prediction_and_jacobian(
            self.inputs, self.parameters[ACTIVE], "none")
        np.testing.assert_allclose(predicted, original, rtol=0, atol=1e-10)
        np.testing.assert_allclose(gradient[:, :-2], original_gradient,
                                   rtol=1e-10, atol=1e-10)

    def test_gradient_and_fit(self):
        values = np.r_[self.parameters[ACTIVE], 0.3, 1.2]
        predicted, gradient = quartic_prediction_and_jacobian(self.inputs, values, 3.0)
        for index in range(len(values)):
            delta = np.zeros(len(values))
            delta[index] = 1e-4
            forward = quartic_prediction_and_jacobian(self.inputs, values + delta, 3.0)[0]
            backward = quartic_prediction_and_jacobian(self.inputs, values - delta, 3.0)[0]
            np.testing.assert_allclose(gradient[:, index],
                                       (forward - backward) / (2e-4),
                                       rtol=1e-5, atol=1e-5)
        result, scale, baseline_cost, _ = fit_quartic(
            self.inputs, predicted, self.parameters, "log1p")
        self.assertTrue(result.success)
        self.assertLess(result.cost, baseline_cost)
        np.testing.assert_allclose(
            quartic_prediction_and_jacobian(self.inputs, result.x, scale)[0],
            predicted, rtol=0, atol=1e-4)

    def test_quadratic_only_ablation_fixes_quartic_to_zero(self):
        target = quartic_prediction_and_jacobian(
            self.inputs, np.r_[self.parameters[ACTIVE], 0.0, 1.3], 3.0)[0]
        result, _, baseline_cost, _ = fit_quartic(
            self.inputs, target, self.parameters, "log1p", disable_quartic=True)
        self.assertEqual(len(result.x), len(ACTIVE) + 1)
        self.assertLess(result.cost, baseline_cost)

    def test_zero_shift_bound_is_enforced(self):
        target = quartic_prediction_and_jacobian(
            self.inputs, np.r_[self.parameters[ACTIVE], 0.1, 1.3], 3.0)[0]
        result, _, _, _ = fit_quartic(
            self.inputs, target, self.parameters, "log1p", max_zero_shift_ms=5.0)
        zero_index = int(np.where(ACTIVE == 7)[0][0])
        self.assertGreaterEqual(result.x[zero_index], 95.0)
        self.assertLessEqual(result.x[zero_index], 105.0)


if __name__ == "__main__":
    unittest.main()
