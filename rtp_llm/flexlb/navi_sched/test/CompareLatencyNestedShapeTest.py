#!/usr/bin/env python3

import unittest

import numpy as np

from AnalyzeLatencyPhaseLeastSquares import ACTIVE, PARAMETER_SCALES, shape
from CompareLatencyNestedShape import (
    fit_nested, nested_prediction_and_jacobian, nested_shape_and_derivatives,
)


class CompareLatencyNestedShapeTest(unittest.TestCase):
    def setUp(self):
        random = np.random.default_rng(17)
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

    def test_nested_formula_and_zero_point(self):
        values = np.asarray([-10.0, -1.0, 0.0, 1.0, 10.0])
        nested, _, _ = nested_shape_and_derivatives(values, 10.0)
        np.testing.assert_allclose(nested, shape(shape(values, 10.0), 10.0))
        zero = np.zeros((1, 14))
        zero[0, 0] = 1.0
        prediction, _ = nested_prediction_and_jacobian(
            zero, self.parameters[ACTIVE])
        self.assertAlmostEqual(prediction[0],
                               self.parameters[7] * PARAMETER_SCALES[7])

    def test_gradient_and_fit(self):
        values = self.parameters[ACTIVE]
        prediction, gradient = nested_prediction_and_jacobian(self.inputs, values)
        for index in range(len(values)):
            delta = np.zeros(len(values))
            delta[index] = 1e-4
            forward = nested_prediction_and_jacobian(self.inputs, values + delta)[0]
            backward = nested_prediction_and_jacobian(self.inputs, values - delta)[0]
            np.testing.assert_allclose(gradient[:, index],
                                       (forward - backward) / (2e-4),
                                       rtol=1e-5, atol=1e-5)
        baseline = self.parameters.copy()
        baseline[1] *= 0.6
        result, _ = fit_nested(self.inputs, prediction, baseline, "log1p")
        self.assertTrue(result.success)
        np.testing.assert_allclose(
            nested_prediction_and_jacobian(self.inputs, result.x)[0],
            prediction, rtol=0, atol=1e-4)


if __name__ == "__main__":
    unittest.main()
