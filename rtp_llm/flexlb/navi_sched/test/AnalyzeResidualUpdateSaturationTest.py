#!/usr/bin/env python3

import tempfile
import unittest
from pathlib import Path

from AnalyzeResidualUpdateSaturation import (
    RESIDUAL_COUNT,
    audit,
    parse_model_state,
    residual_correction,
)


def model_state(sequence, event_time_us, correction, residual_parameters):
    fields = ["0"] * 25
    fields[0] = "M"
    fields[1] = "mi-test"
    fields[4] = str(sequence)
    fields[5] = "0"
    fields[6] = str(event_time_us)
    fields[7] = "200"
    fields[9] = "1"
    fields[14] = "0"
    fields[15] = "999"
    fields[21] = "32"
    fields[22] = str(32 + correction)
    fields[23] = str(correction)
    fields[24] = "17"
    fields.extend(["0"] * 17)
    fields.append(str(RESIDUAL_COUNT))
    fields.extend(str(value) for value in residual_parameters)
    return "\t".join(fields) + "\n"


class AnalyzeResidualUpdateSaturationTest(unittest.TestCase):
    def test_residual_correction_interpolation_and_idle_decay(self):
        parameters = [0.0] * RESIDUAL_COUNT
        parameters[1] = 10.0
        parameters[2] = 30.0
        self.assertAlmostEqual(
            20.0, residual_correction(2 ** 5.5, parameters, 1_000_000,
                                      1_000_000))
        self.assertAlmostEqual(
            20.0 * 2 / 3,
            residual_correction(2 ** 5.5, parameters, 1_000_000,
                                181_000_000))
        self.assertEqual(
            0.0, residual_correction(2 ** 5.5, parameters, 1_000_000,
                                     301_000_000))

    def test_audit_identifies_clipped_low_pressure_update(self):
        initial = [0.0] * RESIDUAL_COUNT
        updated = initial.copy()
        updated[1] = 3.0
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trace.tsv"
            path.write_text(model_state(1, 1_000_000, 0.0, initial) +
                            model_state(2, 2_000_000, 3.0, updated),
                            encoding="utf-8")
            groups, checked, mismatched, max_difference = audit(path)
        self.assertEqual(2, checked)
        self.assertEqual(0, mismatched)
        self.assertEqual(0.0, max_difference)
        self.assertEqual(1, len(groups[("MI", "low")]))
        clipped, ratio, innovation, limit = groups[("MI", "low")][0]
        self.assertTrue(clipped)
        self.assertEqual(168.0, innovation)
        self.assertEqual(100.0, limit)
        self.assertEqual(1.68, ratio)

    def test_rejects_wrong_trace_shape(self):
        with self.assertRaisesRegex(ValueError, "model-state trace"):
            parse_model_state("M\tbroken\n")

    def test_waiting_count_is_not_waiting_hit_tokens(self):
        fields = model_state(1, 1_000_000, 0.0, [0.0] * RESIDUAL_COUNT)
        state = parse_model_state(fields)
        self.assertEqual(0, state["waiting_count"])


if __name__ == "__main__":
    unittest.main()
