#!/usr/bin/env python3

import unittest

import numpy as np

from EvaluateDecodeProgressFeature import (
    _decode_progress_design,
    evaluate_pairs,
    fit_robust_decode_progress_correction,
    reconstruct_decode_progress,
)


SESSION = "test-session"


def lifecycle(sequence, kind, request_id, monotonic_ns, batch_id=1, **fields):
    return {
        "session": SESSION,
        "sequence": sequence,
        "event": kind,
        "request_id": request_id,
        "monotonic_ns": monotonic_ns,
        "batch_id": batch_id,
        **fields,
    }


def execution_step(completed_ns, decode_requests=1, decode_tokens=1):
    return {
        "session": SESSION,
        "event": "execution_step",
        "success": True,
        "completed_perf_ns": completed_ns,
        "decode_requests": decode_requests,
        "decode_tokens": decode_tokens,
    }


class EvaluateDecodeProgressFeatureTest(unittest.TestCase):
    def test_reconstructs_only_progress_visible_before_admission(self):
        events = [
            lifecycle(1, "admitted", "running", 5, 1),
            lifecycle(2, "slot_acquire", "running", 10, 1),
            execution_step(15, 0, 0),
            lifecycle(3, "first_output", "running", 16, 1, output_tokens=1),
            execution_step(20),
            lifecycle(4, "admitted", "target-a", 22, 2),
            execution_step(25),
            lifecycle(5, "admitted", "target-b", 27, 3),
            lifecycle(6, "slot_release", "running", 30, 1),
            lifecycle(7, "terminal", "running", 31, 1, status="FINISHED_STOPPED"),
            lifecycle(8, "admitted", "target-c", 32, 4),
            lifecycle(9, "terminal", "target-a", 40, 2, status="FINISHED_STOPPED"),
            lifecycle(10, "terminal", "target-b", 41, 3, status="FINISHED_STOPPED"),
            lifecycle(11, "terminal", "target-c", 42, 4, status="FINISHED_ABORTED"),
        ]
        progress, diagnostics = reconstruct_decode_progress(events)
        self.assertEqual(1, progress[2]["active_decode_count"])
        self.assertEqual(2.0, progress[2]["generated_decode_tokens"])
        self.assertEqual(3.0, progress[3]["generated_decode_tokens"])
        self.assertEqual(0, progress[4]["active_decode_count"])
        self.assertTrue(progress[4]["target_aborted"])
        self.assertEqual(2, diagnostics["decode_steps"])
        self.assertEqual(1.0, diagnostics["unit_decode_step_ratio"])

    def test_pair_direction_uses_non_abort_pairs(self):
        pairs = [
            {
                "batches": [2, 3],
                "target_gap_ms": 20.0,
                "sides": [
                    {"batch_id": 2, "target_ms": 30.0, "crosses_abort_wave": False},
                    {"batch_id": 3, "target_ms": 10.0, "crosses_abort_wave": False},
                ],
            },
            {
                "batches": [4, 5],
                "target_gap_ms": 10.0,
                "sides": [
                    {"batch_id": 4, "target_ms": 20.0, "crosses_abort_wave": True},
                    {"batch_id": 5, "target_ms": 10.0, "crosses_abort_wave": False},
                ],
            },
        ]
        progress = {
            2: {"active_decode_count": 2, "generated_decode_tokens": 20.0,
                "mean_generated_decode_tokens": 10.0, "target_aborted": False},
            3: {"active_decode_count": 2, "generated_decode_tokens": 40.0,
                "mean_generated_decode_tokens": 20.0, "target_aborted": False},
        }
        result = evaluate_pairs(pairs, progress)
        self.assertEqual(1, result["pair_count"])
        self.assertEqual(1, result["slower_has_fewer_generated_tokens"])
        self.assertEqual(1, result["generated_difference_over_5pct"])
        self.assertEqual(0, result["active_decode_count_difference_over_1"])

    def test_pressure_design_uses_decode_progress_and_current_pressure(self):
        rows = [
            {
                "batch_id": 1,
                "waiting_batch_size": 3,
                "running_batch_size": 5,
            },
            {
                "batch_id": 2,
                "waiting_batch_size": 7,
                "running_batch_size": 11,
            },
        ]
        progress = {
            1: {"active_decode_count": 2, "generated_decode_tokens": 20},
            2: {"active_decode_count": 4, "generated_decode_tokens": 80},
        }
        names, design = _decode_progress_design(
            rows, progress, "decode_progress_pressure", ["rising", "falling"]
        )
        self.assertEqual(8, len(names))
        np.testing.assert_array_equal(
            [2, 20, 10, 6, 60, 30, 10, 100], design[0]
        )
        np.testing.assert_array_equal(
            [4, 80, 20, 28, 560, 140, 44, 880], design[1]
        )

    def test_robust_correction_improves_independent_time_blocks(self):
        rows = []
        progress = {}
        for index in range(96):
            generated = 10 * (1 + (index * 7) % 17)
            target = 1000.0 + 3.0 * generated
            batch_id = index + 1
            rows.append(
                {
                    "batch_id": batch_id,
                    "event_time_us": index * 30_000_000,
                    "target_ms": target,
                    "prediction_before_ms": 1000.0,
                    "waiting_batch_size": 10 + index % 40,
                    "running_batch_size": 20,
                    "preceding_request_count": 30 + index % 20,
                }
            )
            progress[batch_id] = {
                "active_decode_count": 10,
                "generated_decode_tokens": generated,
                "mean_generated_decode_tokens": generated / 10,
                "target_aborted": False,
            }
        result = fit_robust_decode_progress_correction(
            rows, progress, "generated_tokens"
        )
        self.assertIsNotNone(result)
        self.assertIsNotNone(result["selected_ridge_strength"])
        held = result["evaluations"]["held_all"]
        self.assertLess(
            held["corrected"]["mae_ms"], held["baseline"]["mae_ms"] * 0.1
        )


if __name__ == "__main__":
    unittest.main()
