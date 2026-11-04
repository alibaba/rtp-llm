#!/usr/bin/env python3

import unittest

import numpy as np

from AnalyzeLatencyPhaseLeastSquares import ACTIVE, PARAMETER_SCALES, prediction_and_jacobian
from CompareLatencyQuarticPolynomialShape import (
    fit_polynomial, physical_coefficients, polynomial_prediction_and_jacobian,
    polynomial_shape_and_derivatives,
)
from CompareLatencyQuarticRootShape import quartic_prediction_and_jacobian


class CompareLatencyQuarticPolynomialShapeTest(unittest.TestCase):
    def setUp(self):
        random = np.random.default_rng(23)
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

    def test_original_and_even_model_are_nested(self):
        original = np.r_[self.parameters[ACTIVE], 0.0, 1.0, 0.0, 0.0, 0.0]
        predicted, gradient = polynomial_prediction_and_jacobian(self.inputs, original, 3.0)
        expected, expected_gradient = prediction_and_jacobian(
            self.inputs, self.parameters[ACTIVE], "none")
        np.testing.assert_allclose(predicted, expected, rtol=0, atol=1e-10)
        np.testing.assert_allclose(gradient[:, :len(ACTIVE)], expected_gradient,
                                   rtol=1e-10, atol=1e-10)
        even = np.r_[self.parameters[ACTIVE], 0.5, 0.0, 0.0, 1.2, 0.0]
        even_prediction = quartic_prediction_and_jacobian(
            self.inputs, np.r_[self.parameters[ACTIVE], 0.25, 1.44], 3.0)[0]
        np.testing.assert_allclose(
            polynomial_prediction_and_jacobian(self.inputs, even, 3.0)[0],
            even_prediction, rtol=0, atol=1e-10)

    def test_polynomial_is_nonnegative_and_gradient_matches_difference(self):
        coefficients = [0.5, 1.2, -0.3, 0.4, 0.2]
        polynomial = physical_coefficients(coefficients, 10.0, 3.0)
        values = np.linspace(-100.0, 100.0, 1001)
        radicand = sum(weight * values ** power for weight, power in
                        zip(polynomial, (4, 3, 2, 1, 0)))
        self.assertGreaterEqual(float(np.min(radicand)), 10.0 - 1e-8)
        values = np.r_[self.parameters[ACTIVE], coefficients]
        predicted, gradient = polynomial_prediction_and_jacobian(self.inputs, values, 3.0)
        for index in range(len(values)):
            delta = np.zeros(len(values))
            delta[index] = 1e-4
            forward = polynomial_prediction_and_jacobian(self.inputs, values + delta, 3.0)[0]
            backward = polynomial_prediction_and_jacobian(self.inputs, values - delta, 3.0)[0]
            np.testing.assert_allclose(gradient[:, index],
                                       (forward - backward) / (2e-4),
                                       rtol=1e-5, atol=1e-5)
        output, _, _, _ = polynomial_shape_and_derivatives(
            np.asarray([-2.0, 0.0, 2.0]), 10.0, coefficients, 3.0)
        self.assertTrue(np.all(np.isfinite(output)))
        for value in (-2.0, -0.5, 0.5, 2.0):
            delta = 1e-5
            plus = polynomial_shape_and_derivatives(
                np.asarray([value + delta]), 10.0, coefficients, 3.0)[0][0]
            minus = polynomial_shape_and_derivatives(
                np.asarray([value - delta]), 10.0, coefficients, 3.0)[0][0]
            slope = polynomial_shape_and_derivatives(
                np.asarray([value]), 10.0, coefficients, 3.0)[1][0]
            self.assertAlmostEqual(slope, (plus - minus) / (2.0 * delta), places=6)

    def test_fit_preserves_zero_bound_and_even_baseline(self):
        target = polynomial_prediction_and_jacobian(
            self.inputs, np.r_[self.parameters[ACTIVE], 0.2, 1.0, 0.1, 0.1, 0.0], 3.0)[0]
        even = {"fit_parameters": self.parameters,
                "quartic_weight_per_u4": 0.04 / 9.0, "quadratic_weight": 1.01}
        result, _, _, even_cost, _ = fit_polynomial(
            self.inputs, target, self.parameters, even, "log1p", max_zero_shift_ms=5.0)
        zero_index = int(np.where(ACTIVE == 7)[0][0])
        self.assertLessEqual(abs(result.x[zero_index] - 100.0), 5.0)
        self.assertLessEqual(result.cost, even_cost + 1e-8)


if __name__ == "__main__":
    unittest.main()
