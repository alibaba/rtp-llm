import json
from pathlib import Path
import tempfile
import unittest

from AnalyzeVllmRequestLifecycles import MARKER, analyze, execution_audit, read_events


def event(sequence, kind, rid, second, batch=1, **extra):
    return {"session": "process-1", "sequence": sequence, "event": kind,
            "request_id": rid, "monotonic_ns": int(second * 1e9),
            "event_time_us": int(second * 1e6), "batch_id": batch,
            "running_count": 1, "waiting_count": 0,
            "audit_errors": 0, "audit_failures": 0, **extra}


def life(start_seq, rid, start, end, batch):
    return [event(start_seq, "admitted", rid, start, batch),
            event(start_seq + 1, "slot_acquire", rid, start + 1, batch),
            event(start_seq + 2, "first_output", rid, start + 2, batch),
            event(start_seq + 3, "terminal", rid, end, batch, output_tokens=256),
            event(start_seq + 4, "kv_free", rid, end + .01, batch),
            event(start_seq + 5, "slot_release", rid, end + .02, batch)]


def order(events):
    rows = sorted(events, key=lambda e: e["monotonic_ns"])
    return [{**row, "sequence": i + 1} for i, row in enumerate(rows)]


class RequestLifecycleAnalysisTest(unittest.TestCase):
    def test_legacy_decode_default_counts_are_unknown_not_zero(self):
        event = {"event": "execution_step", "prefill_tokens": 0, "running_count": 0,
                 "waiting_count": 0, "decode_requests": 512}
        result = execution_audit(event)
        self.assertIsNone(result["running_count"])
        self.assertIsNone(result["waiting_count"])
        self.assertEqual(result["uncollected_raw_counts"]["running_count"], 0)
        self.assertEqual(result["decode_requests"], 512)
        self.assertEqual(event["running_count"], 0)
        sampled = execution_audit({**event, "workload_counts_collected": True,
                                   "running_count": 512, "waiting_count": 80})
        self.assertEqual(sampled["running_count"], 512)
        self.assertEqual(sampled["waiting_count"], 80)
        prefill = execution_audit({**event, "prefill_tokens": 100, "running_count": 5})
        self.assertEqual(prefill["running_count"], 5)

    def test_distant_samples_can_share_same_decode_predecessor(self):
        rows = order(life(1, "decode", 0, 900, 1) + life(1, "a", 100, 150, 2)
                     + life(1, "b", 700, 760, 3))
        result = analyze(rows, [2, 3])
        self.assertTrue(result["sequence_complete"])
        self.assertTrue(result["fully_drained"])
        self.assertEqual(result["comparisons"][0]["admission_separation_s"], 600)
        self.assertEqual(result["comparisons"][0]["shared_unfinished_request_ids"], ["decode"])
        predecessor = result["batches"]["2"]["predecessors"]["decode"]
        self.assertTrue(predecessor["holding_slot_at_arrival"])
        self.assertEqual(predecessor["remaining_lifetime_ms"], 800000)
        self.assertEqual(predecessor["final_output_tokens"], 256)
        self.assertEqual(len(result["requests"]), 3)
        self.assertEqual(result["requests"]["a"]["durations_ms"], {
            "admission_to_first_slot": 1000,
            "first_slot_to_first_output": 1000,
            "first_output_to_terminal": 48000,
            "admission_to_terminal": 50000,
        })

    def test_disjoint_cohorts_and_complete_request_traces(self):
        rows = order(life(1, "a", 0, 90, 1) + life(1, "b", 600, 690, 2))
        result = analyze(rows + rows, [1, 2])
        self.assertEqual(result["request_count"], 2)
        self.assertEqual(len(result["requests"]["a"]["events"]), 6)
        self.assertEqual(result["comparisons"][0]["shared_unfinished_request_ids"], [])
        self.assertTrue(result["sequence_complete"])

    def test_gap_and_missing_acquisition_are_not_complete(self):
        rows = life(1, "a", 0, 90, 1)
        result = analyze(rows[:1] + rows[2:])
        self.assertFalse(result["sequence_complete"])
        self.assertEqual(result["error_count"], 2)
        self.assertTrue(any("sequence gap" in error for error in result["errors"]))
        self.assertTrue(any("without captured acquisition" in error for error in result["errors"]))

    def test_missing_tail_is_not_a_completed_request(self):
        result = analyze(life(1, "a", 0, 90, 1)[:3])
        self.assertTrue(result["sequence_complete"])
        self.assertFalse(result["fully_drained"])
        self.assertIsNone(result["requests"]["a"]["slots"][0]["release"])

    def test_restart_requires_selecting_a_single_session(self):
        rows = life(1, "a", 0, 90, 1)
        rows += [{**e, "session": "process-2"} for e in rows]
        with self.assertRaises(ValueError):
            analyze(rows)
        self.assertEqual(analyze(rows, session="process-2")["request_count"], 1)

    def test_malformed_tail_and_input_limits(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "audit.log"
            path.write_text("[INFO] [" + MARKER + json.dumps(event(1, "admitted", "a", 0))
                            + "] []\n" + MARKER + "{broken tail")
            rows, malformed = read_events([path])
            self.assertEqual(malformed, 1)
            self.assertFalse(analyze(rows, malformed_lines=malformed)["sequence_complete"])
            with self.assertRaisesRegex(ValueError, "byte limit"):
                read_events([path], max_bytes=1)
            with self.assertRaisesRegex(ValueError, "event limit"):
                read_events([path, path], max_events=1)

    def test_dropped_and_conflicting_events_are_visible(self):
        rows = life(1, "a", 0, 90, 1)
        rows[-1]["audit_failures"] = 1
        rows += [{**rows[0], "request_id": "other"}]
        result = analyze(rows)
        self.assertFalse(result["sequence_complete"])
        self.assertTrue(any("dropped events" in error for error in result["errors"]))
        self.assertTrue(any("conflicting duplicate" in error for error in result["errors"]))

    def test_waiting_interval_exposes_later_admission_and_slot_release(self):
        target = life(1, "target", 10, 60, 2)
        target[1]["monotonic_ns"] = 40_000_000_000
        target[2]["monotonic_ns"] = 41_000_000_000
        rows = order(life(1, "before", 0, 20, 1) + target
                     + life(1, "later", 25, 30, 3))
        result = analyze(rows, [2])
        batch = result["batches"]["2"]
        self.assertEqual(batch["later_admitted_acquisitions_before_target"], ["later"])
        self.assertEqual([e["request_id"] for e in batch["slot_releases_while_waiting"]],
                         ["before", "later"])
        self.assertEqual([e["request_id"] for e in batch["slot_acquisitions_while_waiting"]], ["later"])

    def test_successors_cover_whole_lifecycle_not_just_queue(self):
        rows = order(life(1, "target", 10, 100, 1)
                     + life(1, "later", 50, 90, 2)
                     + life(1, "after", 110, 120, 3))
        result = analyze(rows, [1])
        batch = result["batches"]["1"]
        self.assertEqual(set(batch["successors_during_target"]), {"later"})
        later = batch["successors_during_target"]["later"]
        self.assertEqual(later["arrival_after_target_ms"], 40000)
        self.assertEqual(len(later["slot_acquires_during_target"]), 1)
        self.assertEqual(later["final_output_tokens"], 256)
        self.assertTrue(later["complete"])
        self.assertEqual(result["request_count"], 3)

    def test_terminal_with_deferred_kv_release_stays_in_predecessors(self):
        old = life(1, "old", 0, 20, 1)
        old[4]["monotonic_ns"] = 100_000_000_000
        rows = order(old + life(1, "target", 50, 80, 2))
        result = analyze(rows, [2])
        predecessor = result["batches"]["2"]["predecessors"]["old"]
        self.assertFalse(predecessor["holding_slot_at_arrival"])
        self.assertEqual(predecessor["remaining_lifetime_ms"], 0)
        self.assertEqual(predecessor["remaining_resource_lifetime_ms"], 50000)

    def test_missing_tail_does_not_prove_lifecycle_independence(self):
        rows = order(life(1, "a", 0, 90, 1)[:3] + life(1, "b", 600, 690, 2))
        result = analyze(rows, [1, 2])
        comparison = result["comparisons"][0]
        self.assertIsNone(comparison["target_resource_intervals_disjoint"])
        self.assertFalse(comparison["captured_lifecycles_complete"])
        self.assertEqual(comparison["shared_lifecycle_request_ids"], ["a", "b"])

    def test_contiguous_terminal_only_trace_is_still_incomplete(self):
        rows = [event(1, "terminal", "unknown", 10), event(2, "kv_free", "unknown", 11)]
        result = analyze(rows)
        self.assertFalse(result["sequence_complete"])
        self.assertFalse(result["fully_drained"])
        self.assertEqual(result["errors"], ["request lifecycle without captured admission unknown"])

    def test_shared_successor_is_part_of_pair_interference_check(self):
        rows = order(life(1, "a", 0, 100, 1) + life(1, "b", 600, 700, 2)
                     + life(1, "bridge", 50, 900, 3))
        result = analyze(rows, [1, 2])
        comparison = result["comparisons"][0]
        self.assertEqual(comparison["shared_unfinished_request_ids"], [])
        self.assertEqual(comparison["shared_lifecycle_request_ids"], ["bridge"])
        self.assertTrue(comparison["target_resource_intervals_disjoint"])
        self.assertTrue(comparison["captured_lifecycles_complete"])


if __name__ == "__main__":
    unittest.main()
