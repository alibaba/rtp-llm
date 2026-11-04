#!/usr/bin/env python3

import unittest

import numpy as np

from AnalyzeLatencyPhaseLeastSquares import PARAMETER_SCALES
from CompareLatencySquareShape import ACTIVE, fit_square, square_prediction_and_jacobian


class CompareLatencySquareShapeTest(unittest.TestCase):
    def test_square_replaces_shape_and_keeps_zero_anchor(self):
        inputs = np.zeros((2, 14))
        inputs[:, 0] = 1.0
        inputs[:, 1] = [0.0, 4.0]
        parameters = np.zeros(17)
        parameters[0] = 32.0
        parameters[1] = 80.0
        parameters[7] = 100.0
        parameters[15] = 1.0
        parameters /= PARAMETER_SCALES[:17]
        predicted, _ = square_prediction_and_jacobian(inputs, parameters[ACTIVE])
        expected = 100.0 + 1500.0 * ((1.1 ** 2) - (0.1 ** 2))
        np.testing.assert_allclose(predicted, [100.0, expected], rtol=0, atol=1e-9)

    def test_analytic_gradient_and_fit(self):
        random = np.random.default_rng(9)
        inputs = random.uniform(0.0, 2.0, (60, 14))
        inputs[:, 0] = 1.0
        parameters = np.zeros(17)
        parameters[0] = 4.0
        parameters[1:6] = [20.0, 4.0, 3.0, 2.0, 1.0]
        parameters[7] = 100.0
        parameters[9] = 1.0
        parameters[10:15] = [12.0, 5.0, 3.0, 2.0, 1.0]
        parameters[15] = 1.0
        parameters[16] = 2.0
        parameters /= PARAMETER_SCALES[:17]
        active = parameters[ACTIVE]
        predicted, gradient = square_prediction_and_jacobian(inputs, active)
        for index in range(len(active)):
            delta = np.zeros(len(active))
            delta[index] = 1e-4
            forward = square_prediction_and_jacobian(inputs, active + delta)[0]
            backward = square_prediction_and_jacobian(inputs, active - delta)[0]
            np.testing.assert_allclose(gradient[:, index],
                                       (forward - backward) / (2e-4),
                                       rtol=1e-5, atol=1e-5)
        result, _ = fit_square(inputs, predicted, [parameters], "log1p")
        self.assertTrue(result.success)
        np.testing.assert_allclose(square_prediction_and_jacobian(inputs, result.x)[0],
                                   predicted, rtol=0, atol=1e-4)


if __name__ == "__main__":
    unittest.main()
