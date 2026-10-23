"""Cross-layer evidence regression tests for the stricter offline audit."""

import unittest

from prefill_traffic_audit import audit_run
from test_prefill_traffic_audit import evidence


def correlated_evidence(dispatch="NON_BATCH"):
    requests, stages, events = evidence(dispatch)
    epoch_shift = 1_700_000_000_000
    for request in requests:
        request["worker"] = "127.0.0.1:12345"
        request["planned_epoch_ms"] += epoch_shift
    for stage in stages:
        stage["start_epoch_ms"] += epoch_shift
        stage["end_epoch_ms"] += epoch_shift
    for event in events:
        if event["event"] != "prefill_done":
            continue
        event["engine_address"] = "127.0.0.1:12345"
        for field in ("engine_arrival_ms", "prefill_start_ms", "prefill_done_ms"):
            event[field] += epoch_shift
    return requests, stages, events


def immediate_stream_evidence():
    requests, stages, events = correlated_evidence()
    requests[0].update(
        schedule_launched_s=100.01,
        schedule_launch_lag_s=0.01,
        schedule_completed_s=100.29,
        stream_started_s=100.3,
        stream_launch_completed_s=100.50001,
        stream_launch_lag_s=100.50001 - 100.29,
        first_received_s=100.45,
        ttft_s=0.45,
        ended_s=100.5,
    )
    return requests, stages, events


class CrossLayerAuditTest(unittest.TestCase):
    def test_matching_worker_and_clocks_accept_both_delivery_modes(self):
        for dispatch in ("NON_BATCH", "BATCH"):
            report = audit_run(*correlated_evidence(dispatch), dispatch=dispatch)
            self.assertEqual(2, report["successful"])
            self.assertEqual(2, report["stages"][0]["completion_requests"])

    def test_engine_clock_offset_in_either_direction_is_rejected(self):
        for shift_ms in (-86_400_000, 86_400_000):
            with self.subTest(shift_ms=shift_ms):
                requests, stages, events = correlated_evidence()
                for event in events:
                    if event["event"] == "prefill_done":
                        for field in (
                            "engine_arrival_ms",
                            "prefill_start_ms",
                            "prefill_done_ms",
                        ):
                            event[field] += shift_ms
                with self.assertRaisesRegex(ValueError, "client.*interval"):
                    audit_run(requests, stages, events, dispatch="NON_BATCH")

    def test_prefill_cannot_complete_after_client_first_response(self):
        requests, stages, events = correlated_evidence()
        for event in events:
            if event["event"] == "prefill_done":
                # Still before client end, but later than the reported first token.
                event["prefill_done_ms"] += 300
        with self.assertRaisesRegex(ValueError, "client.*interval"):
            audit_run(requests, stages, events, dispatch="NON_BATCH")

    def test_millisecond_clock_rounding_tolerance_is_bounded(self):
        requests, stages, events = correlated_evidence()
        events[0]["engine_arrival_ms"] = requests[0]["planned_epoch_ms"] - 1
        audit_run(requests, stages, events, dispatch="NON_BATCH")
        events[0]["engine_arrival_ms"] = requests[0]["planned_epoch_ms"] - 3
        with self.assertRaisesRegex(ValueError, "client.*interval"):
            audit_run(requests, stages, events, dispatch="NON_BATCH")

    def test_executed_worker_must_match_routing_response(self):
        requests, stages, events = correlated_evidence()
        requests[0]["worker"] = "127.0.0.9:12345"
        with self.assertRaisesRegex(ValueError, "routing worker"):
            audit_run(requests, stages, events, dispatch="NON_BATCH")

    def test_worker_evidence_is_required_and_typed(self):
        for side, field in (("client", "worker"), ("engine", "engine_address")):
            for invalid in (None, "", "   ", 12345):
                with self.subTest(side=side, invalid=invalid):
                    requests, stages, events = correlated_evidence()
                    owner = requests[0] if side == "client" else events[0]
                    if invalid is None:
                        owner.pop(field)
                    else:
                        owner[field] = invalid
                    with self.assertRaisesRegex(ValueError, "worker address"):
                        audit_run(requests, stages, events, dispatch="NON_BATCH")


class ImmediateStreamAuditTest(unittest.TestCase):
    def test_reader_can_finish_before_stream_launcher_returns(self):
        result = audit_run(*immediate_stream_evidence(), dispatch="NON_BATCH")
        self.assertEqual(2, result["successful"])

    def test_reader_cannot_finish_before_stream_started(self):
        requests, stages, events = immediate_stream_evidence()
        requests[0].update(
            stream_started_s=100.51,
            stream_launch_completed_s=100.52,
            stream_launch_lag_s=100.52 - 100.29,
        )
        with self.assertRaisesRegex(ValueError, "stream launch timestamps"):
            audit_run(requests, stages, events, dispatch="NON_BATCH")

    def test_launcher_return_still_has_bounded_delay(self):
        requests, stages, events = immediate_stream_evidence()
        requests[0].update(
            stream_launch_completed_s=100.56,
            stream_launch_lag_s=100.56 - 100.29,
        )
        with self.assertRaisesRegex(ValueError, "stream launch fell behind"):
            audit_run(requests, stages, events, dispatch="NON_BATCH")

    def test_launcher_cannot_return_before_stream_started(self):
        requests, stages, events = immediate_stream_evidence()
        requests[0].update(
            stream_launch_completed_s=100.299,
            stream_launch_lag_s=100.299 - 100.29,
        )
        with self.assertRaisesRegex(ValueError, "stream launch timestamps"):
            audit_run(requests, stages, events, dispatch="NON_BATCH")


if __name__ == "__main__":
    unittest.main()
