#!/usr/bin/env python3
"""Focused checks for the fixed-curvature offline fitting experiment."""

import unittest

import numpy as np

from AnalyzeLatencyPhaseLeastSquares import (
    ACTIVE, PARAMETER_SCALES, input_matrix, prediction_and_jacobian, shape,
)
from CompareLatencyFixedCurvature import (
    CURVATURE_COLUMN, FREE_COLUMNS, fit_curvature, full_parameters,
    model_prediction_and_jacobian, segment_masks,
)


class FixedCurvatureTest(unittest.TestCase):
    def test_parameter_mapping_preserves_other_parameters(self):
        original = np.arange(1, len(ACTIVE) + 1, dtype=float)
        fixed = full_parameters(original[FREE_COLUMNS], 900.0)
        np.testing.assert_array_equal(fixed[FREE_COLUMNS], original[FREE_COLUMNS])
        self.assertEqual(ACTIVE[CURVATURE_COLUMN], 8)
        self.assertEqual(fixed[CURVATURE_COLUMN] * PARAMETER_SCALES[8], 900.0)

    def test_fixed_fit_keeps_curvature_and_refits_other_parameters(self):
        features = np.zeros((40, 13))
        features[:, 1] = np.arange(40) % 10
        features[:, 6] = np.arange(40)
        features[:, 12] = 10
        inputs = input_matrix(features)
        truth = np.ones(len(ACTIVE))
        truth[CURVATURE_COLUMN] = 900.0
        truth[np.flatnonzero(ACTIVE == 7)[0]] = 100.0
        truth /= PARAMETER_SCALES[ACTIVE]
        target = prediction_and_jacobian(inputs, truth)[0]
        start = truth.copy()
        start[np.flatnonzero(ACTIVE == 7)[0]] = 200.0 / PARAMETER_SCALES[7]
        result = fit_curvature(inputs, target, [start], 900.0, "raw", 100)
        fitted = full_parameters(result.x, 900.0)
        self.assertEqual(fitted[CURVATURE_COLUMN] * PARAMETER_SCALES[8], 900.0)
        self.assertLess(np.mean(np.abs(prediction_and_jacobian(inputs, fitted)[0] - target)),
                        np.mean(np.abs(prediction_and_jacobian(inputs, start)[0] - target)))

    def test_high_rising_requires_high_pressure_and_queue_increase(self):
        features = np.zeros((3, 13))
        features[:, 6] = (0, 40, 40)
        inputs = input_matrix(features)
        event_time = np.array([0, 30_000_000, 60_000_000])
        masks = segment_masks(inputs, np.ones(3), event_time)
        np.testing.assert_array_equal(masks["high_rising"], (False, True, False))

    def test_concave_shape_is_increasing_and_jacobian_matches_difference(self):
        features = np.zeros((4, 13))
        features[:, 1] = (0, 1, 2, 3)
        inputs = input_matrix(features)
        parameters = np.ones(len(ACTIVE))
        parameters[CURVATURE_COLUMN] = 1.0
        for shape in ("concave_05", "concave_09"):
            predicted, jac = model_prediction_and_jacobian(inputs, parameters, shape)
            self.assertTrue(np.all(np.diff(predicted) > 0))
            self.assertTrue(np.all(np.diff(predicted, n=2) < 0))
            shifted = parameters.copy()
            shifted[CURVATURE_COLUMN] += 1e-5
            finite_difference = (
                model_prediction_and_jacobian(inputs, shifted, shape)[0] - predicted
            ) / 1e-5
            np.testing.assert_allclose(finite_difference, jac[:, CURVATURE_COLUMN],
                                       atol=1e-4, rtol=1e-4)

    def test_curvature_rescaling_requires_inverse_outer_scale(self):
        u = np.array([-10.0, -1.0, 0.0, 2.0, 100.0])
        zero = np.full(len(u), -0.5)
        original = 1500.0 * (shape(u, 900.0) - shape(zero, 900.0))
        ratio = np.sqrt(100.0 / 900.0)
        transformed = shape(ratio * u, 100.0) - shape(ratio * zero, 100.0)
        self.assertGreater(np.max(np.abs(1500.0 * transformed - original)), 100.0)
        np.testing.assert_allclose(1500.0 / ratio * transformed, original,
                                   atol=1e-10)


if __name__ == "__main__":
    unittest.main()
