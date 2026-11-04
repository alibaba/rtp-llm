#!/usr/bin/env python3
"""Checks for the two-width MI latency formula and optimizer."""

import unittest

import numpy as np

from AnalyzeLatencyPhaseLeastSquares import (
    ACTIVE, input_matrix, prediction_and_jacobian as baseline_prediction,
)
from CompareLatencyTransitionWidth import fit_model, prediction_and_jacobian


class TransitionWidthTest(unittest.TestCase):
    def setUp(self):
        features = np.zeros((20, 13))
        features[:, 1] = np.arange(20)
        features[:, 6] = np.arange(20) * 2
        features[:, 12] = 3
        self.inputs = input_matrix(features)
        self.parameters = np.ones(len(ACTIVE))
        self.parameters[np.flatnonzero(ACTIVE == 8)[0]] = 100.0

    def test_r_one_is_exact_baseline(self):
        baseline, baseline_jac = baseline_prediction(self.inputs, self.parameters)
        extended, extended_jac = prediction_and_jacobian(
            self.inputs, np.r_[self.parameters, 0.0])
        np.testing.assert_allclose(extended, baseline, atol=1e-10)
        np.testing.assert_allclose(extended_jac[:, :-1], baseline_jac, atol=1e-10)

    def test_width_and_curvature_jacobian(self):
        parameters = np.r_[self.parameters, np.log(4.0)]
        prediction, jac = prediction_and_jacobian(self.inputs, parameters)
        for column in (0, int(np.flatnonzero(ACTIVE == 8)[0]), len(ACTIVE)):
            shifted = parameters.copy()
            shifted[column] += 1e-5
            difference = (prediction_and_jacobian(self.inputs, shifted)[0] -
                          prediction) / 1e-5
            np.testing.assert_allclose(difference, jac[:, column],
                                       atol=1e-3, rtol=1e-3)

    def test_nested_fit_can_start_at_baseline(self):
        truth = np.r_[self.parameters, np.log(4.0)]
        target = prediction_and_jacobian(self.inputs, truth)[0]
        baseline = fit_model(self.inputs, target, [self.parameters], "raw", 100, False)
        extended = fit_model(self.inputs, target,
                             [np.r_[baseline.x, 0.0], np.r_[baseline.x, np.log(4.0)]],
                             "raw", 100, True)
        self.assertLessEqual(extended.cost, baseline.cost + 1e-5)
        self.assertGreaterEqual(np.exp(extended.x[-1]), 1.0)


if __name__ == "__main__":
    unittest.main()
