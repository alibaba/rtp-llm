#!/usr/bin/env python3

import unittest

import numpy as np
from scipy.optimize._numdiff import approx_derivative

from AnalyzeLatencyPhaseLeastSquares import (
    ACTIVE, PARAMETER_SCALES, input_matrix, prediction_and_jacobian,
)
from AnalyzeLatencyPressureFormula import (
    VARIANTS, forward_and_jacobian, initial_values,
)


class AnalyzeLatencyPressureFormulaTest(unittest.TestCase):
    def setUp(self):
        self.parameters = np.array([
            0.1, 1.3, 0.2, 0.8, 0.02, 0.03, 0.0, 80.0, 100.0,
            0.002, 1.4, 0.3, 0.9, 0.02, 0.03, 2.0, 100.0,
        ])
        raw = np.array([
            [0, 2, 50, 150, 6000, 8000, 3, 40, 80, 100, 2000, 3500, 2000],
            [0, 5, 10, 30, 100, 200, 40, 300, 20, 50, 500, 600, 30000],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        ], dtype=np.float64)
        self.inputs = input_matrix(raw)

    def test_base_matches_deployed_formula(self):
        expected, expected_jac = prediction_and_jacobian(
            self.inputs, self.parameters[ACTIVE])
        for variant in VARIANTS:
            with self.subTest(variant=variant):
                actual = forward_and_jacobian(
                    self.inputs, initial_values(self.parameters, variant), variant)[0]
                np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-9)
        _, actual_jac = forward_and_jacobian(
            self.inputs, initial_values(self.parameters, "base"), "base")
        np.testing.assert_allclose(actual_jac, expected_jac, rtol=1e-10, atol=1e-7)

    def test_all_variant_gradients_and_zero_anchor(self):
        for variant in VARIANTS:
            with self.subTest(variant=variant):
                values = initial_values(self.parameters, variant)
                if variant == "direct":
                    values[-2:] = [3.0, -0.5]
                elif variant == "curvature":
                    values[-2:] = [0.4, -0.2]
                elif variant == "both":
                    values[-4:] = [3.0, -0.5, 0.4, -0.2]
                elif variant == "curvature_waiting":
                    values[-1] = -0.2
                elif variant == "direct_curvature_waiting":
                    values[-3:] = [3.0, -0.5, -0.2]
                predicted, analytic = forward_and_jacobian(self.inputs, values, variant)
                numerical = approx_derivative(
                    lambda candidate: forward_and_jacobian(
                        self.inputs, candidate, variant)[0],
                    values, method="3-point")
                np.testing.assert_allclose(analytic, numerical, rtol=1e-4, atol=1e-4)
        self.assertAlmostEqual(self.parameters[7] * PARAMETER_SCALES[7],
                               predicted[-1], places=8)


if __name__ == "__main__":
    unittest.main()
