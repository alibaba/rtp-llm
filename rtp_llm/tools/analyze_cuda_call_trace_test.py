import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

_SPEC = importlib.util.spec_from_file_location(
    "analyze_cuda_call_trace", Path(__file__).with_name("analyze_cuda_call_trace.py")
)
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)


class CudaTraceAnalysisTest(unittest.TestCase):
    def header(self):
        return {
            "type": "metadata",
            "trace_id": "host-pid-start",
            "file_generation": "0",
            "first_writer_seq": "1",
        }

    def record(self, seq, kind, trace="host-pid-start", **fields):
        return {
            "trace_id": trace,
            "writer_seq": str(seq),
            "type": kind,
            "domain": 1,
            "tid": 10,
            "call_id": "5",
            "correlation_id": 17,
            "activity_kind": 1 if kind == "activity" else 0,
            "params_known": True,
            **fields,
        }

    def analyze(self, records, extra="", **kwargs):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trace.jsonl"
            path.write_text("".join(json.dumps(r) + "\n" for r in records) + extra)
            return _MODULE.analyze([path], **kwargs)

    def test_unreturned_api_survives_and_request_id_is_exact(self):
        rid = "3611341739697827841"
        result = self.analyze(
            [self.header(), self.record(1, "api_enter", request_id=rid)]
        )
        self.assertEqual(result["open_in_retained_records_count"], 1)
        self.assertEqual(result["open_in_retained_records"][0]["request_id"], rid)
        self.assertEqual(result["coverage_issue_count"], 0)

    def test_same_correlation_other_process_cannot_close_or_join(self):
        result = self.analyze(
            [
                self.header(),
                self.record(1, "api_enter"),
                self.record(1, "api_exit", trace="other-instance"),
                self.record(2, "activity", trace="other-instance"),
            ]
        )
        self.assertEqual(result["open_in_retained_records_count"], 1)
        self.assertEqual(result["gpu_correlations"][0]["api_candidates"], [])

    def test_exit_is_not_completion_and_activity_matches_retained_call(self):
        result = self.analyze(
            [
                self.header(),
                self.record(1, "api_enter"),
                self.record(2, "api_exit"),
                self.record(3, "activity", start_ns="100", end_ns="200"),
            ]
        )
        self.assertEqual(result["open_in_retained_records_count"], 0)
        self.assertEqual(len(result["gpu_correlations"][0]["api_candidates"]), 1)
        self.assertNotIn("deadlock", result)

    def test_event_generation_and_context_are_not_conflated(self):
        result = self.analyze(
            [self.header()]
            + [
                self.record(
                    seq,
                    "activity",
                    activity_kind=36,
                    context_uid=context,
                    event_id=7,
                    event_sync_id=generation,
                )
                for seq, context, generation in [
                    (1, 1, "20"),
                    (2, 1, "21"),
                    (3, 2, "20"),
                ]
            ]
        )
        self.assertEqual(len(result["event_generation_groups"]), 3)

    def test_stream_wait_invalid_event_id_is_not_an_event_dependency(self):
        result = self.analyze(
            [self.header()]
            + [
                self.record(
                    seq,
                    "activity",
                    activity_kind=38,
                    event_id=4294967295,
                    event_sync_id="4294967295",
                    stream_id=seq,
                )
                for seq in (1, 2)
            ]
        )
        self.assertEqual(result["event_generation_groups"], [])

    def test_rotation_duplicates_are_deduplicated_and_loss_is_visible(self):
        enter = self.record(1, "api_enter")
        result = self.analyze(
            [self.header(), enter, enter, self.record(2, "drop")],
            extra='{"partial":',
        )
        self.assertEqual(result["open_in_retained_records_count"], 1)
        self.assertEqual(result["coverage_issue_count"], 2)

    def test_ambiguous_call_id_is_reported(self):
        result = self.analyze(
            [
                self.header(),
                self.record(1, "api_enter"),
                self.record(2, "api_enter"),
                self.record(3, "api_exit"),
            ]
        )
        self.assertIn(
            "ambiguous_call_identity",
            [issue["reason"] for issue in result["coverage_issues"]],
        )
        self.assertEqual(result["open_in_retained_records_count"], 2)

    def test_real_segment_header_and_missing_rotation(self):
        header = self.header()
        result = self.analyze(
            [header, self.record(1, "api_enter"), self.record(9, "api_exit")]
        )
        self.assertEqual(result["metadata"], [header])
        self.assertEqual(
            result["coverage_issues"][0]["reason"], "retained_sequence_gap"
        )
        self.assertEqual(result["coverage_issues"][0]["first_missing"], 2)
        self.assertEqual(result["coverage_issues"][0]["last_missing"], 8)

    def test_many_reused_call_ids_remain_ambiguous_after_exit(self):
        records = [self.header()]
        records += [self.record(seq, "api_enter") for seq in range(1, 4001)]
        records += [self.record(4001, "api_exit")]
        # EXIT before a different ENTER cannot close it; an EXIT after a third
        # unique ENTER can. Summary joins must preserve both ordering cases.
        records += [
            self.record(4002, "api_exit", call_id="6"),
            self.record(4003, "api_enter", call_id="6"),
            self.record(4004, "api_enter", call_id="7"),
            self.record(4005, "api_exit", call_id="7"),
        ]
        result = self.analyze(records, limit=2)
        self.assertEqual(result["open_in_retained_records_count"], 4001)
        self.assertEqual(len(result["open_in_retained_records"]), 2)
        ambiguity = next(
            issue
            for issue in result["coverage_issues"]
            if issue["reason"] == "ambiguous_call_identity"
        )
        self.assertEqual(ambiguity["count"], 4000)

    def test_full_disk_sequence_can_still_have_producer_loss(self):
        result = self.analyze(
            [
                self.header(),
                self.record(1, "api_enter", record_seq="1103021334"),
                self.record(2, "flush_begin", record_seq="1103047923"),
            ]
        )
        self.assertEqual(result["producer_records_not_retained_count"], 26588)
        self.assertEqual(result["producer_sequence_gap_count"], 1)
        self.assertEqual(
            result["producer_sequence_gaps"][0]["first_missing"], "1103021335"
        )
        self.assertNotIn(
            "retained_sequence_gap",
            [issue["reason"] for issue in result["coverage_issues"]],
        )

    def test_priority_queue_reordering_does_not_invent_loss(self):
        result = self.analyze(
            [self.header()]
            + [
                self.record(seq, "mark", record_seq=str(producer))
                for seq, producer in enumerate((14, 10, 12, 11, 13), 1)
            ]
        )
        self.assertEqual(result["producer_sequence_gap_count"], 0)
        self.assertEqual(result["producer_records_not_retained_count"], 0)

    def test_producer_sequences_are_process_local_and_output_is_bounded(self):
        result = self.analyze(
            [self.header()]
            + [
                self.record(seq, "mark", record_seq=str(10 + seq * 2))
                for seq in range(1, 5)
            ]
            + [self.record(1, "mark", trace="other", record_seq="9999")],
            limit=1,
        )
        self.assertEqual(result["producer_sequence_gap_count"], 3)
        self.assertEqual(result["producer_records_not_retained_count"], 3)
        self.assertEqual(len(result["producer_sequence_gaps"]), 1)

    def test_snapshot_exposes_loss_even_without_retained_drop_record(self):
        snapshot = {
            "trace_id": "host-pid-start",
            "host_dropped": "26588",
            "open_overflow": "32",
            "open_calls": [],
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "open.json"
            path.write_text(json.dumps(snapshot))
            result = self.analyze([self.header()], snapshots=[path])
        self.assertEqual(
            result["independent_open_snapshots"][0]["loss_counters"],
            {"host_dropped": 26588, "open_overflow": 32},
        )
        self.assertEqual(
            result["coverage_issues"][0]["reason"], "snapshot_reports_loss"
        )

    def test_bounded_report_keeps_large_late_gap_and_all_issue_counts(self):
        snapshot = {
            "trace_id": "host-pid-start",
            "host_dropped": "26588",
            "open_calls": [self.record(0, "api_enter")],
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "open.json"
            path.write_text(json.dumps(snapshot))
            result = self.analyze(
                [self.header()]
                + [
                    self.record(seq, "mark", record_seq=str(producer))
                    for seq, producer in enumerate((1, 3, 5, 26594), 1)
                ],
                snapshots=[path],
                limit=1,
            )
        self.assertEqual(len(result["producer_sequence_gaps"]), 1)
        self.assertEqual(result["producer_sequence_gaps"][0]["missing_count"], 26588)
        self.assertEqual(
            result["coverage_issue_counts"]["producer_records_not_retained"], 3
        )
        self.assertEqual(result["coverage_issue_counts"]["snapshot_reports_loss"], 1)
        self.assertEqual(
            result["coverage_issue_counts"]["queue_derived_active_state_incomplete"], 1
        )

    def snapshot_analysis(self, **fields):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "open.json"
            path.write_text(json.dumps({"trace_id": "host-pid-start", **fields}))
            return self.analyze([self.header()], snapshots=[path])

    def test_bulk_loss_does_not_make_independent_live_calls_queue_derived(self):
        result = self.snapshot_analysis(
            active_state_source="callback_slots",
            bulk_dropped="1000",
            active_state_overflow="0",
            snapshot_missed_slots="0",
            open_calls=[self.record(0, "api_enter")],
        )
        self.assertEqual(result["open_in_retained_records_count"], 0)
        sample = result["independent_open_snapshots"][0]
        self.assertEqual(
            sample["active_state_coverage"], "bounded_non_atomic_callback_sample"
        )
        self.assertEqual(len(sample["snapshot"]["open_calls"]), 1)

    def test_live_slot_exhaustion_is_not_presented_as_complete(self):
        for field in (
            "active_state_overflow",
            "untracked_api_calls",
            "snapshot_missed_slots",
        ):
            with self.subTest(field=field):
                result = self.snapshot_analysis(
                    active_state_source="callback_slots", **{field: "1"}
                )
                self.assertEqual(
                    result["independent_open_snapshots"][0]["active_state_coverage"],
                    "incomplete_callback_sample",
                )
                self.assertIn(
                    "active_state_coverage_limited",
                    [issue["reason"] for issue in result["coverage_issues"]],
                )

    def test_optional_callback_gap_does_not_invalidate_captured_live_slots(self):
        unavailable = [
            {"domain": 1, "cbid": 101, "name": "cudaOptionalApi", "result": 1}
        ]
        for count, callbacks in ((1, unavailable), (1, []), (0, unavailable)):
            with self.subTest(count=count, callbacks=callbacks):
                result = self.snapshot_analysis(
                    active_state_source="callback_slots",
                    optional_callback_unavailable=count,
                    unavailable_callbacks=callbacks,
                    open_calls=[self.record(0, "api_enter")],
                )
                sample = result["independent_open_snapshots"][0]
                self.assertEqual(
                    sample["subscription_coverage"], "optional_callbacks_unavailable"
                )
                self.assertEqual(
                    sample["active_state_coverage"],
                    "bounded_non_atomic_callback_sample",
                )
                self.assertEqual(sample["loss_counters"], {})
                self.assertEqual(len(sample["snapshot"]["open_calls"]), 1)
                self.assertEqual(
                    result["coverage_issue_counts"],
                    {"snapshot_subscription_coverage_limited": 1},
                )

    def test_optional_callback_history_and_snapshot_count_past_output_limit(self):
        callbacks = [
            {"domain": 1, "cbid": 101, "name": "cudaOptionalApi", "result": 1},
            {"domain": 2, "cbid": 202, "name": "cuOptionalApi", "result": 1},
        ]
        history = [
            self.record(seq, "callback_unavailable", **callback)
            for seq, callback in enumerate(callbacks, 1)
        ]
        snapshot = {
            "trace_id": "host-pid-start",
            "active_state_source": "callback_slots",
            "optional_callback_unavailable": 2,
            "unavailable_callbacks": callbacks,
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "open.json"
            path.write_text(json.dumps(snapshot))
            result = self.analyze(
                [self.header(), *history, history[0]], snapshots=[path], limit=1
            )
        self.assertEqual(result["record_counts"]["callback_unavailable"], 2)
        self.assertEqual(len(result["coverage_issues"]), 1)
        self.assertEqual(
            result["coverage_issue_counts"],
            {
                "recorder_callback_unavailable": 2,
                "snapshot_subscription_coverage_limited": 1,
            },
        )
        self.assertEqual(result["coverage_issue_count"], 3)

    def test_optional_callback_gap_preserves_close_mismatch_staleness(self):
        result = self.snapshot_analysis(
            active_state_source="callback_slots",
            optional_callback_unavailable=1,
            close_mismatches="1",
        )
        sample = result["independent_open_snapshots"][0]
        self.assertEqual(sample["active_state_coverage"], "callback_state_may_be_stale")
        self.assertEqual(sample["loss_counters"], {"close_mismatches": 1})
        self.assertEqual(
            result["coverage_issue_counts"]["snapshot_subscription_coverage_limited"],
            1,
        )

    def test_callback_close_mismatch_is_stale_even_when_slots_are_incomplete(self):
        for overflow in ("0", "1"):
            with self.subTest(overflow=overflow):
                result = self.snapshot_analysis(
                    active_state_source="callback_slots",
                    close_mismatches="2",
                    active_state_overflow=overflow,
                    open_calls=[self.record(0, "api_enter")],
                )
                sample = result["independent_open_snapshots"][0]
                self.assertEqual(
                    sample["active_state_coverage"], "callback_state_may_be_stale"
                )
                self.assertEqual(sample["loss_counters"]["close_mismatches"], 2)
                self.assertEqual(result["open_in_retained_records_count"], 0)
                self.assertIn(
                    "callback_active_state_may_be_stale",
                    [issue["reason"] for issue in result["coverage_issues"]],
                )

    def test_old_open_table_with_dropped_exits_is_explicitly_unreliable(self):
        result = self.snapshot_analysis(
            host_dropped="9", open_calls=[self.record(1, "api_enter")]
        )
        self.assertEqual(
            result["independent_open_snapshots"][0]["active_state_coverage"],
            "queue_derived_may_be_stale",
        )
        self.assertIn(
            "queue_derived_active_state_incomplete",
            [issue["reason"] for issue in result["coverage_issues"]],
        )


if __name__ == "__main__":
    unittest.main()
