#!/usr/bin/env python3

import unittest

from AnalyzeVllmDecodeSchedulerState import reconstruct_decode_scheduler_state
from AnalyzeVllmDecodeSchedulerState import reconstruct_sample_scheduler_state


SESSION = "test-session"


def event(
    sequence,
    kind,
    request_id,
    monotonic_ns,
    batch_id=1,
    event_time_us=None,
    **fields,
):
    return {
        "session": SESSION,
        "sequence": sequence,
        "event": kind,
        "request_id": request_id,
        "monotonic_ns": monotonic_ns,
        "event_time_us": (
            monotonic_ns // 1000 if event_time_us is None else event_time_us
        ),
        "batch_id": batch_id,
        **fields,
    }


def step(completed_ns, requests=1, event_time_us=None):
    return {
        "session": SESSION,
        "event": "execution_step",
        "success": True,
        "completed_perf_ns": completed_ns,
        "event_time_us": (
            completed_ns // 1000 if event_time_us is None else event_time_us
        ),
        "decode_requests": requests,
        "decode_tokens": requests,
    }


class AnalyzeVllmDecodeSchedulerStateTest(unittest.TestCase):
    def test_reconstructs_context_remaining_distribution_and_release(self):
        events = [
            event(1, "admitted", "a", 1, prompt_tokens=100, max_tokens=4),
            event(2, "slot_acquire", "a", 2),
            event(3, "first_output", "a", 3, output_tokens=1),
            event(4, "admitted", "b", 4, prompt_tokens=1000, max_tokens=8),
            event(5, "slot_acquire", "b", 5),
            event(6, "first_output", "b", 6, output_tokens=1),
            step(7, 2),
            event(7, "admitted", "target", 8, 2, prompt_tokens=10, max_tokens=2),
            step(9, 2),
            event(8, "slot_release", "a", 5_000_000_008),
            event(9, "terminal", "a", 5_000_000_009,
                  status="FINISHED_LENGTH_CAPPED"),
            event(10, "slot_release", "b", 20_000_000_008),
            event(11, "terminal", "b", 20_000_000_009,
                  status="FINISHED_LENGTH_CAPPED"),
            event(12, "slot_acquire", "target", 20_000_000_010),
            event(13, "terminal", "target", 20_000_000_011,
                  status="FINISHED_ABORTED"),
        ]
        states, diagnostics = reconstruct_decode_scheduler_state(events)
        target = states[2]
        self.assertEqual(2, target["active_decode_count"])
        self.assertEqual(4.0, target["generated_decode_tokens"])
        self.assertEqual(8.0, target["running_decode_remaining_token_budget"])
        self.assertEqual(1100.0, target["running_decode_prompt_tokens"])
        self.assertEqual(1104.0, target["running_decode_context_tokens"])
        self.assertEqual(2.0, target["remaining_token_min"])
        self.assertEqual(2, target["remaining_le_8_count"])
        self.assertEqual(1, target["active_release_5s_count"])
        self.assertEqual(1, target["active_release_10s_count"])
        self.assertEqual(2, target["active_release_30s_count"])
        self.assertAlmostEqual(20_000.000002, target["queue_to_first_slot_ms"])
        self.assertTrue(target["target_aborted"])
        self.assertEqual(1.0, diagnostics["unit_decode_step_ratio"])
        self.assertEqual(2 / 3, diagnostics["length_capped_fraction"])

    def test_reconstructs_forward_snapshot_and_calibrates_async_callback_lag(self):
        events = [
            event(1, "admitted", "a", 1, prompt_tokens=100, max_tokens=4,
                  event_time_us=1000),
            event(2, "slot_acquire", "a", 2, event_time_us=2000),
            step(2_500_000, event_time_us=2500),
            event(3, "first_output", "a", 3, output_tokens=1,
                  event_time_us=3000),
            event(4, "admitted", "b", 4, prompt_tokens=1000, max_tokens=8,
                  event_time_us=3500),
            event(5, "slot_acquire", "b", 5, event_time_us=4000),
            step(4_500_000, requests=2, event_time_us=4500),
            event(6, "first_output", "b", 6, output_tokens=1,
                  event_time_us=5000),
            # The callback is complete, but update_from_output has not committed
            # this step when the forwarding thread captures the snapshot.
            step(6_500_000, requests=2, event_time_us=6500),
            event(7, "admitted", "target", 8_000_000, batch_id=2,
                  prompt_tokens=10, max_tokens=2, event_time_us=8000),
            event(8, "slot_release", "a", 12_000_000,
                  event_time_us=12000),
            event(9, "slot_release", "b", 20_000_000,
                  event_time_us=20000),
            event(10, "slot_acquire", "target", 20_000_001,
                  batch_id=2, event_time_us=20001),
            event(11, "terminal", "target", 20_000_002,
                  batch_id=2, event_time_us=20002,
                  status="FINISHED_LENGTH_CAPPED"),
        ]
        rows = [{
            "batch_id": 2,
            "event_time_us": 8000,
            "target_ms": 1.0,
            "running_decode_request_count": 2,
            "running_decode_generated_tokens": 3,
        }]
        states, diagnostics = reconstruct_sample_scheduler_state(events, rows)
        target = states[2]
        self.assertEqual(2, target["active_decode_count"])
        self.assertEqual(3.0, target["generated_decode_tokens"])
        self.assertEqual(9.0, target["running_decode_remaining_token_budget"])
        self.assertEqual(1100.0, target["running_decode_prompt_tokens"])
        self.assertEqual(1103.0, target["running_decode_context_tokens"])
        self.assertEqual({"1": 1}, diagnostics["selected_callback_lags"])
        self.assertEqual(0, diagnostics["count_mismatch_count"])
        self.assertEqual(0, diagnostics["progress_mismatch_count"])


if __name__ == "__main__":
    unittest.main()
