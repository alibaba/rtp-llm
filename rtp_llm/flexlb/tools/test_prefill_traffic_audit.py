"""Run with: python -m unittest discover -s tools -p 'test_prefill_traffic_audit.py'."""

import copy
import unittest

from prefill_traffic_audit import (
    DEADLINE_ERROR,
    DEADLINE_ERRORS,
    OVERLOAD_ERROR,
    audit_run,
    validate_environment,
    validate_trace,
)


def evidence(dispatch="NON_BATCH"):
    requests, events = [], []
    for index, (length, hit) in enumerate(((1000, 0), (3000, 2000))):
        rid = index + 1
        started = 100.0 + index * 0.1
        requests.append(
            {
                "stage": "steady",
                "request_id": rid,
                "session": "session" if index else "",
                "input_tokens": length,
                "keys": [rid * 10, rid * 10 + 1],
                "planned_s": started,
                "planned_epoch_ms": started * 1000,
                "started_s": started,
                "pacing_lag_s": 0.0,
                "ended_s": 101.0,
                "success": True,
                "code": 200,
                "enqueued_by_master": dispatch == "BATCH",
                "worker": "127.0.0.1:12345",
                "error": "",
                "first_received_s": 100.5,
                "ttft_s": 100.5 - started,
            }
        )
        events.extend(
            [
                {
                    "event": "prefill_done",
                    "rid": rid,
                    "engine_name": "prefill-0",
                    "engine_address": "127.0.0.1:12345",
                    "engine_incarnation": "incarnation-1",
                    "batch_id": 9 if dispatch == "BATCH" else -1,
                    "batch_size": 2,
                    "engine_arrival_ms": 100100,
                    "prefill_start_ms": 100200,
                    "prefill_done_ms": 100400,
                    "exec_ms": 200,
                    "input_len": length,
                    "cache_hit_tokens": hit,
                    "cancelled": False,
                },
                {"event": "decode_done", "rid": rid, "cancelled": False},
            ]
        )
    stages = [
        {
            "stage": "steady",
            "start_s": 100.0,
            "drained_s": 112.0,
            "start_epoch_ms": 100000,
            "end_epoch_ms": 110000,
            "seconds": 10.0,
            "allow_rejections": False,
            "worker_count": 1,
        }
    ]
    return requests, stages, events


def reject(row, error=OVERLOAD_ERROR):
    row.update(success=False, code=8431, error=error)
    row.pop("first_received_s", None)
    row.pop("ttft_s", None)


class TrafficAuditTest(unittest.TestCase):
    def test_weighted_hits_and_distinct_fixed_window_and_drain_denominators(self):
        result = audit_run(*evidence(), dispatch="NON_BATCH")
        stage = result["stages"][0]
        self.assertEqual(stage["success_token_hit_rate"], 0.5)
        self.assertEqual(stage["completion_input_tpm"], 24000)
        self.assertEqual(stage["completion_compute_tpm"], 12000)
        self.assertEqual(stage["input_tpm_including_drain"], 20000)
        self.assertEqual(stage["compute_tpm_including_drain"], 10000)
        self.assertEqual(stage["offered_input_tpm"], 24000)
        self.assertEqual(stage["offered_session_fraction"], 0.5)
        self.assertEqual(stage["ttft_samples"], 2)
        self.assertEqual(result["requests"], result["successful"] + result["rejected"])

    def test_engine_coalescing_is_allowed_in_non_batch_dispatch(self):
        result = audit_run(*evidence(), dispatch="NON_BATCH")
        self.assertEqual(result["multi_request_batches"], 1)

    def test_batch_execution_duration_is_not_multiplied_by_member_count(self):
        result = audit_run(*evidence("BATCH"), dispatch="BATCH")
        self.assertEqual(result["prefill_execution_batches"], 1)
        self.assertEqual(
            result["stages"][0]["completed_batch_service_ms_by_worker"],
            {"prefill-0": 200},
        )

    def test_engine_regrouping_can_combine_distinct_master_batch_ids(self):
        requests, stages, events = evidence("BATCH")
        events[2]["batch_id"] = 10
        result = audit_run(requests, stages, events, dispatch="BATCH")
        self.assertEqual(result["prefill_execution_batches"], 1)

    def test_per_member_start_timestamp_does_not_split_one_execution_batch(self):
        requests, stages, events = evidence("BATCH")
        events[2]["prefill_start_ms"] += 2
        self.assertEqual(
            audit_run(requests, stages, events, dispatch="BATCH")[
                "prefill_execution_batches"
            ],
            1,
        )

    def test_fixed_window_includes_earlier_arrivals_but_excludes_right_boundary(self):
        requests, stages, events = evidence()
        requests[0]["stage"] = "warmup"
        # A completed warmup is included on the completion axis, even though it
        # is absent from the measured phase's arrival cohort.
        result = audit_run(requests, stages, events, dispatch="NON_BATCH")["stages"][0]
        self.assertEqual(result["arrivals"], 1)
        self.assertEqual(result["completion_requests"], 2)
        self.assertEqual(result["success_token_hit_rate"], 2 / 3)
        self.assertEqual(result["completion_token_hit_rate"], 0.5)
        for event in (events[0], events[2]):
            event["prefill_done_ms"] = stages[0]["end_epoch_ms"]
        for row in requests:
            row["first_received_s"] = 110.0
            row["ttft_s"] = row["first_received_s"] - row["started_s"]
            row["ended_s"] = 110.1
        result = audit_run(requests, stages, events, dispatch="NON_BATCH")["stages"][0]
        self.assertEqual(result["completion_requests"], 0)
        self.assertEqual(result["completion_input_tpm"], 0)
        self.assertIsNone(result["completion_token_hit_rate"])

    def test_valid_overload_is_counted_separately_and_cannot_inflate_hit_rate(self):
        requests, stages, events = evidence()
        reject(requests[1])
        stages[0]["allow_rejections"] = True
        events = events[:2]
        events[0]["batch_size"] = 1
        result = audit_run(requests, stages, events, dispatch="NON_BATCH")
        stage = result["stages"][0]
        self.assertEqual(result["rejected"], 1)
        self.assertEqual(stage["success_fraction"], 0.5)
        self.assertEqual(stage["success_token_hit_rate"], 0)
        self.assertEqual(stage["offered_input_tpm"], 24000)
        self.assertEqual(stage["completion_input_tpm"], 6000)

    def test_hard_deadline_needs_elapsed_time_and_absent_engine_evidence(self):
        requests, stages, events = evidence()
        reject(requests[1], DEADLINE_ERROR)
        requests[1]["ended_s"] = requests[1]["started_s"] + 6
        stages[0]["allow_rejections"] = True
        events = events[:2]
        events[0]["batch_size"] = 1
        audit_run(requests, stages, events, dispatch="NON_BATCH")
        requests[1]["ended_s"] -= 0.1
        with self.assertRaisesRegex(ValueError, "unexpected scheduling"):
            audit_run(requests, stages, events, dispatch="NON_BATCH")

    def test_unsent_deadline_is_distinct_from_transport_failure(self):
        requests, stages, events = evidence()
        reject(requests[1], "request expired before send")
        requests[1]["code"] = 8511
        requests[1]["ended_s"] = requests[1]["started_s"] + 6
        stages[0]["allow_rejections"] = True
        events = events[:2]
        events[0]["batch_size"] = 1
        audit_run(requests, stages, events, dispatch="NON_BATCH")
        requests[1]["code"] = 8510
        with self.assertRaisesRegex(ValueError, "unexpected scheduling"):
            audit_run(requests, stages, events, dispatch="NON_BATCH")
        requests[1]["code"] = 8511
        requests[1]["ended_s"] -= 1
        with self.assertRaisesRegex(ValueError, "unexpected scheduling"):
            audit_run(requests, stages, events, dispatch="NON_BATCH")
        requests[1]["ended_s"] += 1
        with self.assertRaisesRegex(
            ValueError, "rejected request 2 has Engine evidence"
        ):
            audit_run(
                requests,
                stages,
                events + [{"rid": 2, "event": "admission"}],
                dispatch="NON_BATCH",
            )

    def test_each_deadline_trigger_needs_elapsed_time_code_and_no_engine_events(self):
        for message in DEADLINE_ERRORS:
            with self.subTest(message=message):
                requests, stages, events = evidence()
                reject(requests[1], message)
                requests[1]["ended_s"] = requests[1]["started_s"] + 6
                stages[0]["allow_rejections"] = True
                events = events[:2]
                events[0]["batch_size"] = 1
                audit_run(requests, stages, events, dispatch="NON_BATCH")
                requests[1]["ended_s"] -= 1
                with self.assertRaisesRegex(ValueError, "unexpected scheduling"):
                    audit_run(requests, stages, events, dispatch="NON_BATCH")
                requests[1]["ended_s"] += 1
                requests[1]["code"] = 8510
                with self.assertRaisesRegex(ValueError, "unexpected scheduling"):
                    audit_run(requests, stages, events, dispatch="NON_BATCH")
                requests[1]["code"] = 8431
                with self.assertRaisesRegex(
                    ValueError, "rejected request 2 has Engine evidence"
                ):
                    audit_run(
                        requests,
                        stages,
                        events + [{"rid": 2, "event": "admission"}],
                        dispatch="NON_BATCH",
                    )

    def test_trace_hash_changes_with_cache_keys_and_preserves_clock_translation(self):
        original = evidence()
        baseline = audit_run(*original, dispatch="NON_BATCH")["input_trace_sha256"]
        translated = copy.deepcopy(original)
        for row in translated[0]:
            for key in ("planned_s", "started_s", "ended_s", "first_received_s"):
                row[key] += 10
            row["planned_epoch_ms"] += 10000
        for key in ("start_s", "drained_s"):
            translated[1][0][key] += 10
        for key in ("start_epoch_ms", "end_epoch_ms"):
            translated[1][0][key] += 10000
        for event in translated[2][::2]:
            for key in ("engine_arrival_ms", "prefill_start_ms", "prefill_done_ms"):
                event[key] += 10000
        self.assertEqual(
            baseline, audit_run(*translated, dispatch="NON_BATCH")["input_trace_sha256"]
        )
        original[0][0]["keys"][0] += 1
        self.assertNotEqual(
            baseline, audit_run(*original, dispatch="NON_BATCH")["input_trace_sha256"]
        )

    def test_mode_label_cannot_substitute_for_real_batch_delivery(self):
        with self.assertRaisesRegex(ValueError, "dispatch mode"):
            audit_run(*evidence(), dispatch="BATCH")
        with self.assertRaisesRegex(ValueError, "dispatch mode"):
            audit_run(*evidence("BATCH"), dispatch="NON_BATCH")

    def test_singleton_only_batch_run_does_not_prove_multi_request_batching(self):
        requests, stages, events = evidence("BATCH")
        for offset, event in enumerate(events[::2]):
            event["batch_size"] = 1
            event["prefill_done_ms"] += offset
        with self.assertRaisesRegex(ValueError, "did not execute a multi-request"):
            audit_run(requests, stages, events, dispatch="BATCH")

    def test_success_requires_both_unique_engine_terminals(self):
        for removed in range(4):
            with self.subTest(removed=removed):
                requests, stages, events = evidence()
                events.pop(removed)
                with self.assertRaisesRegex(ValueError, "exactly one"):
                    audit_run(requests, stages, events, dispatch="NON_BATCH")
        for duplicated in range(4):
            with self.subTest(duplicated=duplicated):
                requests, stages, events = evidence()
                events.append(copy.deepcopy(events[duplicated]))
                with self.assertRaisesRegex(ValueError, "exactly one"):
                    audit_run(requests, stages, events, dispatch="NON_BATCH")

    def test_rejected_request_must_not_have_any_engine_evidence(self):
        requests, stages, events = evidence()
        reject(requests[1])
        stages[0]["allow_rejections"] = True
        for kind in ("prefill_done", "decode_done", "admission"):
            with self.subTest(kind=kind):
                candidate = events[:2] + [{"rid": 2, "event": kind}]
                candidate[0]["batch_size"] = 1
                with self.assertRaisesRegex(
                    ValueError, "rejected request 2 has Engine evidence"
                ):
                    audit_run(requests, stages, candidate, dispatch="NON_BATCH")

    def test_rejection_message_substring_and_transport_failure_are_not_expected(self):
        for code, message in (
            (8431, "unrelated: " + OVERLOAD_ERROR),
            (8431, OVERLOAD_ERROR + "; unexpected failure"),
            (200, OVERLOAD_ERROR),
            (500, OVERLOAD_ERROR),
            (8431, "Prefill queue overload"),
        ):
            with self.subTest(code=code, message=message):
                requests, stages, events = evidence()
                reject(requests[1], message)
                requests[1]["code"] = code
                stages[0]["allow_rejections"] = True
                with self.assertRaisesRegex(ValueError, "unexpected scheduling"):
                    audit_run(requests, stages, events, dispatch="NON_BATCH")

    def test_low_load_contract_does_not_allow_overload_rejections(self):
        requests, stages, events = evidence()
        reject(requests[1])
        events = events[:2]
        events[0]["batch_size"] = 1
        with self.assertRaisesRegex(ValueError, "phase steady rejected traffic"):
            audit_run(requests, stages, events, dispatch="NON_BATCH")

    def test_allowed_rejections_still_must_meet_declared_success_fraction(self):
        requests, stages, events = evidence()
        reject(requests[1])
        stages[0].update(allow_rejections=True, min_success_fraction=0.6)
        events = events[:2]
        events[0]["batch_size"] = 1
        with self.assertRaisesRegex(ValueError, "minimum success fraction"):
            audit_run(requests, stages, events, dispatch="NON_BATCH")
        stages[0]["min_success_fraction"] = 0.5
        audit_run(requests, stages, events, dispatch="NON_BATCH")

    def test_recovery_tail_requires_nonempty_lossless_arrivals(self):
        requests, stages, events = evidence()
        stages[0].update(require_tail_zero_rejections=True, settle_seconds=0.05)
        result = audit_run(requests, stages, events, dispatch="NON_BATCH")
        self.assertEqual(result["stages"][0]["recovery_tail_arrivals"], 1)
        stages[0]["settle_seconds"] = 0.5
        with self.assertRaisesRegex(ValueError, "no recovery tail arrivals"):
            audit_run(requests, stages, events, dispatch="NON_BATCH")
        stages[0].update(settle_seconds=0.05, allow_rejections=True)
        reject(requests[1])
        events = events[:2]
        events[0]["batch_size"] = 1
        with self.assertRaisesRegex(ValueError, "rejects recovery tail"):
            audit_run(requests, stages, events, dispatch="NON_BATCH")

    def test_response_must_confirm_dispatch_path(self):
        requests, stages, events = evidence()
        requests[0]["enqueued_by_master"] = True
        with self.assertRaisesRegex(ValueError, "response contradicts dispatch"):
            audit_run(requests, stages, events, dispatch="NON_BATCH")

    def test_zero_success_cannot_establish_throughput(self):
        requests, stages, _ = evidence()
        for row in requests:
            reject(row)
        stages[0]["allow_rejections"] = True
        with self.assertRaisesRegex(ValueError, "zero successful"):
            audit_run(requests, stages, [], dispatch="NON_BATCH")

    def test_bad_client_evidence_fails(self):
        changes = (
            ("success", None, "terminal outcome"),
            ("code", 500, "success without admission"),
            ("error", "transport closed", "stream error"),
            ("stream_error_code", 9, "stream error"),
            ("input_tokens", -1, "input_tokens"),
            ("input_tokens", 1.5, "noninteger"),
            ("ttft_s", float("nan"), "ttft_s"),
            ("ttft_s", float("inf"), "ttft_s"),
            ("ttft_s", -0.1, "ttft_s"),
            ("ttft_s", 0.1, "first-token"),
            ("first_received_s", 103, "first-token"),
            ("ended_s", 99, "time order"),
            ("pacing_lag_s", 0.3, "inconsistent pacing"),
            ("keys", [], "cache-key"),
            ("session", None, "session"),
        )
        for key, value, message in changes:
            with self.subTest(key=key, value=value):
                requests, stages, events = evidence()
                requests[0][key] = value
                with self.assertRaisesRegex(ValueError, message):
                    audit_run(requests, stages, events, dispatch="NON_BATCH")

    def test_success_cannot_omit_first_token_evidence(self):
        for key in ("ttft_s", "first_received_s"):
            requests, stages, events = evidence()
            del requests[0][key]
            with self.assertRaises(KeyError):
                audit_run(requests, stages, events, dispatch="NON_BATCH")

    def test_actual_late_launch_is_rejected_even_when_lag_field_is_consistent(self):
        requests, stages, events = evidence()
        requests[0].update(started_s=100.3, pacing_lag_s=0.3, ttft_s=0.2)
        with self.assertRaisesRegex(ValueError, "load generator fell behind"):
            audit_run(requests, stages, events, dispatch="NON_BATCH")

    def test_async_rpc_and_stream_launch_lag_are_audited(self):
        requests, stages, events = evidence()
        requests[0].update(
            schedule_launched_s=100.01,
            schedule_launch_lag_s=0.01,
            schedule_completed_s=100.1,
            stream_started_s=100.11,
            stream_launch_completed_s=100.12,
            stream_launch_lag_s=0.02,
        )
        result = audit_run(requests, stages, events, dispatch="NON_BATCH")
        self.assertEqual(result["stages"][0]["max_schedule_launch_lag_s"], 0.01)
        self.assertEqual(result["stages"][0]["max_stream_launch_lag_s"], 0.02)
        for changes, message in (
            (
                {
                    "schedule_launched_s": 100.3,
                    "schedule_launch_lag_s": 0.3,
                    "schedule_completed_s": 100.4,
                },
                "Schedule launch fell behind",
            ),
            ({"schedule_launch_lag_s": 0}, "inconsistent Schedule launch lag"),
            ({"schedule_completed_s": 99}, "inconsistent Schedule timestamps"),
            (
                {"stream_launch_completed_s": 100.4, "stream_launch_lag_s": 0.3},
                "stream launch fell behind",
            ),
            ({"stream_started_s": 100.05}, "inconsistent stream launch timestamps"),
            ({"stream_launch_lag_s": 0}, "inconsistent stream launch lag"),
            ({"stream_launch_lag_s": float("nan")}, "stream_launch_lag_s"),
        ):
            with self.subTest(changes=changes):
                bad = copy.deepcopy(requests)
                bad[0].update(changes)
                with self.assertRaisesRegex(ValueError, message):
                    audit_run(bad, stages, events, dispatch="NON_BATCH")

    def test_partial_async_launch_timestamps_are_invalid(self):
        requests, stages, events = evidence()
        requests[0]["schedule_launch_lag_s"] = 0
        with self.assertRaises(KeyError):
            audit_run(requests, stages, events, dispatch="NON_BATCH")

    def test_bad_engine_evidence_fails(self):
        changes = (
            ("input_len", 1001, "token mismatch"),
            ("cache_hit_tokens", 1001, "impossible cache"),
            ("cache_hit_tokens", -1, "cache_hit_tokens"),
            ("batch_size", 3, "incomplete or contradictory"),
            ("exec_ms", 201, "inconsistent Prefill execution"),
            ("exec_ms", float("inf"), "exec_ms"),
            ("prefill_start_ms", 99900, "Engine time order"),
            ("cancelled", True, "cancelled"),
            ("engine_name", "", "worker identity"),
        )
        for key, value, message in changes:
            with self.subTest(key=key, value=value):
                requests, stages, events = evidence()
                events[0][key] = value
                with self.assertRaisesRegex(ValueError, message):
                    audit_run(requests, stages, events, dispatch="NON_BATCH")

    def test_partial_run_wrong_phase_and_clock_mismatch_fail(self):
        for key, value, message in (
            ("drained_s", 100.9, "not drained"),
            ("end_epoch_ms", 109999, "inconsistent fixed window"),
            ("seconds", 0, "empty measurement"),
            ("worker_count", 0, "worker_count"),
            ("allow_rejections", "true", "rejection expectation"),
        ):
            with self.subTest(key=key):
                requests, stages, events = evidence()
                stages[0][key] = value
                with self.assertRaisesRegex(ValueError, message):
                    audit_run(requests, stages, events, dispatch="NON_BATCH")
        requests, stages, events = evidence()
        requests[0]["stage"] = "typo"
        with self.assertRaisesRegex(ValueError, "unreported phase"):
            audit_run(requests, stages, events, dispatch="NON_BATCH")
        requests[0]["stage"] = "steady"
        requests[0]["planned_epoch_ms"] += 3
        with self.assertRaisesRegex(ValueError, "inconsistent planned clocks"):
            audit_run(requests, stages, events, dispatch="NON_BATCH")

    def test_duplicate_requests_and_unknown_events_fail(self):
        requests, stages, events = evidence()
        with self.assertRaisesRegex(ValueError, "duplicate request"):
            audit_run(requests + [requests[0]], stages, events, dispatch="NON_BATCH")
        with self.assertRaisesRegex(ValueError, "unknown request"):
            audit_run(requests, stages, events + [{"rid": 99}], dispatch="NON_BATCH")


class EvidenceIntegrityTest(unittest.TestCase):
    def trace_evidence(self):
        requests, stages, _ = evidence()
        plan = {
            "warmup": [],
            "affinity": [],
            "calibration": [],
            "traffic": copy.deepcopy(requests),
        }
        for row in plan["traffic"]:
            row["offset_s"] = row["planned_s"] - stages[0]["start_s"]
        return requests, plan, stages

    def environment_evidence(self):
        _, stages, _ = evidence()
        topology = [
            {
                "epoch_ms": 99000,
                "expected": 1,
                "master": {
                    "success": True,
                    "worker_summary": {"PREFILL": {"alive": 1, "discovered": 1}},
                },
            }
        ]
        worker = dict(
            name="prefill-0",
            role="prefill",
            running=0,
            scheduler_running=0,
            waiting=0,
            active_decode_requests=0,
            response_buffers=0,
            prefill_contexts=0,
            decode_waiting_for_kv=0,
            active_kv_tokens=0,
            prefill_waiting_batches=0,
        )
        return stages, topology, {"engines": [worker]}

    def test_trace_inventory_and_reordered_results_are_valid(self):
        requests, plan, stages = self.trace_evidence()
        result = validate_trace(list(reversed(requests)), plan, stages)
        self.assertEqual(result, {"planned_requests": 2, "planned_traffic_requests": 2})

    def test_trace_missing_extra_or_duplicate_request_fails(self):
        requests, plan, stages = self.trace_evidence()
        with self.assertRaisesRegex(ValueError, "inventories differ"):
            validate_trace(requests[:1], plan, stages)
        with self.assertRaisesRegex(ValueError, "duplicate actual"):
            validate_trace(requests + [requests[0]], plan, stages)
        plan["warmup"].append(copy.deepcopy(plan["traffic"][0]))
        with self.assertRaisesRegex(ValueError, "duplicate planned"):
            validate_trace(requests, plan, stages)

    def test_changed_trace_payload_and_arrival_fail(self):
        for field, value in (
            ("stage", "other"),
            ("session", "other"),
            ("input_tokens", 999),
            ("keys", [999]),
            ("planned_s", 101),
        ):
            with self.subTest(field=field):
                requests, plan, stages = self.trace_evidence()
                requests[0][field] = value
                with self.assertRaisesRegex(ValueError, "changed planned"):
                    validate_trace(requests, plan, stages)

    def test_environment_requires_real_master_acknowledgement(self):
        args = self.environment_evidence()
        self.assertEqual(validate_environment(*args)["drained_final_workers"], 1)
        args[1][0]["master"]["worker_summary"]["PREFILL"]["alive"] = 0
        with self.assertRaisesRegex(ValueError, "acknowledgement contradicts"):
            validate_environment(*args)

    def test_topology_target_ack_after_phase_end_is_not_sufficient(self):
        stages, topology, snapshot = self.environment_evidence()
        topology[0]["epoch_ms"] = stages[0]["end_epoch_ms"]
        with self.assertRaisesRegex(ValueError, "did not converge"):
            validate_environment(stages, topology, snapshot)

    def test_changed_fleet_requires_matching_target_in_its_phase(self):
        stages, topology, snapshot = self.environment_evidence()
        stages.append(
            dict(
                stage="out", start_epoch_ms=110000, end_epoch_ms=120000, worker_count=2
            )
        )
        worker = copy.deepcopy(snapshot["engines"][0])
        worker["name"] = "prefill-1"
        snapshot["engines"].append(worker)
        with self.assertRaisesRegex(ValueError, "did not converge"):
            validate_environment(stages, topology, snapshot)
        topology.append(
            {
                "epoch_ms": 111000,
                "expected": 2,
                "master": {
                    "success": True,
                    "worker_summary": {"PREFILL": {"alive": 2, "discovered": 2}},
                },
            }
        )
        result = validate_environment(stages, topology, snapshot)
        self.assertEqual(result["topology"][-1]["convergence_s"], 1)

    def test_removed_worker_must_have_complete_drain_evidence(self):
        stages, topology, snapshot = self.environment_evidence()
        removal = {
            "mode": "graceful",
            "drained": True,
            "remaining_work": {"running_tasks": 0},
            "admission": {"rejected_rpcs_total": 0},
        }
        topology.append({"remove": removal})
        self.assertEqual(
            validate_environment(stages, topology, snapshot)[
                "gracefully_removed_workers"
            ],
            1,
        )
        for field, value, message in (
            ("mode", "abrupt", "gracefully drained"),
            ("drained", False, "gracefully drained"),
            ("remaining_work", {"running_tasks": 1}, "still owns work"),
            ("remaining_work", {}, "still owns work"),
            ("admission", {"rejected_rpcs_total": 1}, "rejected RPCs"),
        ):
            with self.subTest(field=field, value=value):
                bad = copy.deepcopy(topology)
                bad[-1]["remove"][field] = value
                with self.assertRaisesRegex(ValueError, message):
                    validate_environment(stages, bad, snapshot)

    def test_final_owner_leaks_and_missing_counters_fail(self):
        for field in (
            "running",
            "scheduler_running",
            "waiting",
            "active_decode_requests",
            "response_buffers",
            "prefill_contexts",
            "decode_waiting_for_kv",
            "active_kv_tokens",
            "prefill_waiting_batches",
            "leak_detected",
        ):
            with self.subTest(field=field):
                stages, topology, snapshot = self.environment_evidence()
                snapshot["engines"][0][field] = 1
                with self.assertRaises(ValueError):
                    validate_environment(stages, topology, snapshot)
        stages, topology, snapshot = self.environment_evidence()
        del snapshot["engines"][0]["running"]
        with self.assertRaisesRegex(ValueError, "missing/nonzero"):
            validate_environment(stages, topology, snapshot)

    def test_final_fleet_size_must_match_last_phase(self):
        stages, topology, snapshot = self.environment_evidence()
        snapshot["engines"][0]["role"] = "decode"
        with self.assertRaisesRegex(ValueError, "final Prefill worker count"):
            validate_environment(stages, topology, snapshot)


if __name__ == "__main__":
    unittest.main()
