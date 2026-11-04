#!/usr/bin/env python3

import importlib.util
import math
import pathlib
import tempfile
import unittest


MODULE_PATH = pathlib.Path(__file__).with_name("latency_model_state.py")
SPEC = importlib.util.spec_from_file_location("latency_model_state", MODULE_PATH)
state = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(state)


PARAMETERS = [float(index) / 10.0 for index in range(11)]
CAPACITY_SPLIT_PARAMETERS = [float(index) / 10.0 for index in range(18)]


class LatencyModelStateTest(unittest.TestCase):
    def test_binary_version_matches_current_model(self):
        self.assertEqual(19, state.BINARY_VERSION)

    def test_round_trip_empty_state(self):
        for parameters in (PARAMETERS, CAPACITY_SPLIT_PARAMETERS):
            content = state.encode_latency_model_state(parameters, revision=7)

            decoded = state.decode_latency_model_state(content)

            self.assertEqual(7, decoded["revision"])
            self.assertEqual(parameters, decoded["parameters"])
            self.assertEqual([0.0] * len(parameters), decoded["adam_moment_1"])
            self.assertEqual([0.0] * len(parameters), decoded["adam_moment_2"])
            self.assertEqual(1, decoded["optimizer_step"])

    def test_rejects_invalid_parameters(self):
        for parameters in (PARAMETERS[:-1], PARAMETERS + [1.0, 2.0],
                           PARAMETERS[:-1] + [math.inf]):
            with self.assertRaises(ValueError):
                state.encode_latency_model_state(parameters)

    def test_round_trip_nonzero_optimizer_state(self):
        moment_1 = [index + 0.25 for index in range(len(PARAMETERS))]
        moment_2 = [index + 1.5 for index in range(len(PARAMETERS))]
        content = state.encode_latency_model_state(
            PARAMETERS,
            revision=9,
            adam_moment_1=moment_1,
            adam_moment_2=moment_2,
            optimizer_step=123,
        )

        decoded = state.decode_latency_model_state(content)

        self.assertEqual(9, decoded["revision"])
        self.assertEqual(moment_1, decoded["adam_moment_1"])
        self.assertEqual(moment_2, decoded["adam_moment_2"])
        self.assertEqual(123, decoded["optimizer_step"])

    def test_rejects_invalid_optimizer_state(self):
        invalid_states = (
            {"adam_moment_1": [0.0] * (len(PARAMETERS) - 1)},
            {"adam_moment_1": [0.0] * (len(PARAMETERS) - 1) + [math.nan]},
            {"adam_moment_2": [0.0] * (len(PARAMETERS) - 1) + [-1.0]},
            {"optimizer_step": 0},
        )
        for values in invalid_states:
            with self.assertRaises(ValueError):
                state.encode_latency_model_state(PARAMETERS, **values)

    def test_write_is_atomic_and_stable(self):
        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory) / "nested" / "model.state"
            first = state.write_latency_model_state(path, PARAMETERS)
            second = state.write_latency_model_state(path, PARAMETERS)

            self.assertEqual(first, second)
            self.assertTrue(path.is_file())
            self.assertFalse(path.with_name(path.name + ".tmp").exists())


if __name__ == "__main__":
    unittest.main()
