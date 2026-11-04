#!/usr/bin/env python3

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from scipy.optimize._numdiff import approx_derivative

from AnalyzeLatencyPhaseLeastSquares import (
    ACTIVE, PARAMETER_SCALES, fit, input_matrix, load_host,
    prediction_and_jacobian, pressure_masks,
    waiting_constraint_matrix,
)


class AnalyzeLatencyPhaseLeastSquaresTest(unittest.TestCase):
    def test_scaled_formula_matches_cpp_fixed_sample(self):
        parameters = np.asarray([
            4.0, 20.0, 4.0, 3.0, 2.0, 1.0, 0.0, 100.0, 10.0,
            1.0, 12.0, 5.0, 3.0, 2.0, 1.0,
            1.0 / PARAMETER_SCALES[15], 2.0,
        ])
        raw = [[2, 3, 1024, 2048, 4194304, 2097152, 4, 512,
                3072, 9437184, 1572864, 5, 250000]]
        inputs = input_matrix(raw)
        prediction, jacobian = prediction_and_jacobian(
            inputs, parameters[ACTIVE], "none")
        self.assertAlmostEqual(66741.79706653491, prediction[0], places=8)
        finite_difference = approx_derivative(
            lambda values: prediction_and_jacobian(inputs, values, "none")[0],
            parameters[ACTIVE], method="3-point")
        np.testing.assert_allclose(jacobian[0], finite_difference,
                                   rtol=1e-5, atol=1e-6)

    def test_pressure_masks_are_disjoint_and_input_only(self):
        raw = np.zeros((5, 13))
        raw[:, 1] = [0, 4, 5, 0, 40]
        raw[:, 6] = [0, 4, 0, 33, 1]
        low, middle, high = pressure_masks(input_matrix(raw))
        np.testing.assert_array_equal(low, [True, True, False, False, False])
        np.testing.assert_array_equal(middle, [False, False, True, False, True])
        np.testing.assert_array_equal(high, [False, False, False, True, False])

    def test_formula_and_gradient(self):
        raw = [[0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
        inputs = input_matrix(raw)
        parameters = np.zeros(17)
        parameters[1] = 320.0 / PARAMETER_SCALES[1]
        parameters[7] = 100.0 / PARAMETER_SCALES[7]
        parameters[8] = 4.0 / PARAMETER_SCALES[8]
        parameters[15] = 1.0 / PARAMETER_SCALES[15]
        active = parameters[ACTIVE]
        predicted, analytic = prediction_and_jacobian(inputs, active)
        self.assertAlmostEqual(100 + 1500 * (np.sqrt(5) - 1), predicted[0])
        numerical = approx_derivative(
            lambda values: prediction_and_jacobian(inputs, values)[0],
            active, method="3-point")
        np.testing.assert_allclose(analytic[0], numerical, rtol=1e-5, atol=1e-6)

    def test_interaction_gradient(self):
        raw = [[0, 3, 0, 2048, 0, 0, 20, 0, 4096, 0, 0, 100, 5000]]
        inputs = input_matrix(raw)
        parameters = np.zeros(17)
        parameters[7] = 100.0 / PARAMETER_SCALES[7]
        parameters[8] = 4.0 / PARAMETER_SCALES[8]
        parameters[15] = 1.0 / PARAMETER_SCALES[15]
        active = np.concatenate((parameters[ACTIVE], [2.0, 3.0, 4.0]))
        for kind in ("queue", "decode", "decode_wait"):
            values = active[:len(ACTIVE) + 1] if kind == "decode_wait" else active
            _, analytic = prediction_and_jacobian(inputs, values, kind)
            numerical = approx_derivative(
                lambda values: prediction_and_jacobian(inputs, values, kind)[0],
                values, method="3-point")
            np.testing.assert_allclose(analytic[0], numerical, rtol=1e-5, atol=1e-6)

    def test_logged_model_prediction_matches(self):
        parameters = np.zeros(17)
        parameters[7] = 100.0 / PARAMETER_SCALES[7]
        parameters[8] = 4.0 / PARAMETER_SCALES[8]
        parameters[15] = 1.0 / PARAMETER_SCALES[15]
        event = {"type": "sample", "event_time_us": 1_000_000,
                 "target_ms": 100.0, "features": [0] * 13,
                 "base_prediction_before_ms": 100.0,
                 "prediction_before_ms": 100.0}
        update = {"type": "update", "parameters_before": parameters.tolist(),
                  "parameters_after": parameters.tolist()}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "mi-test.jsonl"
            with path.open("w", encoding="utf-8") as stream:
                stream.write(json.dumps(update) + "\n")
                for _ in range(25):
                    stream.write(json.dumps(event) + "\n")
            inputs, target, _, logged, logged_final, starts, difference = load_host(path)
        self.assertEqual((25, 14), inputs.shape)
        self.assertEqual(25, len(target))
        self.assertEqual(25, len(logged))
        self.assertEqual(25, len(logged_final))
        self.assertEqual(3, len(starts))
        self.assertLess(difference, 1e-6)

    def test_load_host_uses_requested_number_of_fit_starts(self):
        parameters = np.zeros(17)
        parameters[7] = 100.0 / PARAMETER_SCALES[7]
        parameters[8] = 4.0 / PARAMETER_SCALES[8]
        parameters[15] = 1.0 / PARAMETER_SCALES[15]
        event = {"type": "sample", "event_time_us": 1_000_000,
                 "target_ms": 100.0, "features": [0] * 13,
                 "base_prediction_before_ms": 100.0,
                 "prediction_before_ms": 100.0}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "mi-test.jsonl"
            with path.open("w", encoding="utf-8") as stream:
                for index in range(5):
                    after = parameters.copy()
                    after[0] = index
                    stream.write(json.dumps({
                        "type": "update",
                        "parameters_before": parameters.tolist(),
                        "parameters_after": after.tolist(),
                    }) + "\n")
                    stream.write(json.dumps(event) + "\n")
            *_, starts, _ = load_host(path, fit_start_count=5)
        self.assertEqual(5, len(starts))
        self.assertEqual([1.0, 2.0, 3.0, 4.0],
                         [start[0] for start in starts[1:]])

    def test_constrained_fit_rejects_negative_waiting_cost(self):
        raw = [
            [0, 1, 0, 0, 0, 0, 0, 1, 1024, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 2, 2048, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 4, 4096, 0, 0, 0, 0],
        ]
        inputs = input_matrix(raw)
        initial = np.zeros(17)
        initial[7] = 100.0 / PARAMETER_SCALES[7]
        initial[8] = 4.0 / PARAMETER_SCALES[8]
        initial[15] = 1.0 / PARAMETER_SCALES[15]
        target = np.array([100.0, 80.0, 60.0])
        unconstrained = fit(inputs, target, [initial], "raw", "none")
        unconstrained_waiting = (
            waiting_constraint_matrix(inputs, len(unconstrained.x)) @
            unconstrained.x)
        self.assertLess(float(np.min(unconstrained_waiting)), -1.0)
        result = fit(inputs, target, [initial], "raw", "none",
                     enforce_nonnegative_waiting=True)
        waiting = waiting_constraint_matrix(inputs, len(result.x)) @ result.x
        self.assertTrue(result.success, result.message)
        self.assertLessEqual(result.constraint_violation, 1e-8)
        self.assertGreaterEqual(float(np.min(waiting)), -1e-8)


if __name__ == "__main__":
    unittest.main()
