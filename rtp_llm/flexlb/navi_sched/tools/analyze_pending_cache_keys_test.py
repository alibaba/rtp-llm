#!/usr/bin/env python3

import json
import pathlib
import tempfile
import unittest

import analyze_pending_cache_keys as audit


def audit_line(
    request_id,
    local_batch_id,
    arrival_time_us,
    start_time_us,
    finish_time_us,
    agent_hit_tokens,
    rtp_hit_tokens,
    block_hashes,
    outcome="finished",
    total_tokens=768,
    block_size=256,
    master_hit_tokens=None,
    agent_pending_aware_hit_tokens=None,
    schedule_batch_id=None,
    hybrid_tail_required=1,
):
    if master_hit_tokens is None:
        master_hit_tokens = agent_hit_tokens
    if schedule_batch_id is None:
        schedule_batch_id = local_batch_id
    pending_aware_field = ""
    if agent_pending_aware_hit_tokens is not None:
        pending_aware_field = "agent_pending_aware_hit_tokens={} ".format(
            agent_pending_aware_hit_tokens
        )
    return (
        "[navi-cache-key-audit] request_id={} schedule_batch_id={} "
        "local_batch_id={} expected_batch_size=1 arrival_time_us={} "
        "ready_time_us={} start_time_us={} finish_time_us={} cache_revision=1 "
        "cached_blocks=0 cache_checksum=0 full_ready_blocks=0 "
        "full_ready_checksum=0 tail_ready_blocks=0 tail_ready_checksum=0 "
        "hybrid_tail_required={} total_tokens={} block_size={} "
        "master_hit_tokens={} agent_hit_tokens={} {}rtp_hit_tokens={} "
        "hybrid_candidate_blocks=-1 hybrid_flat_matched_blocks=-1 "
        "hybrid_min_full_reuse_blocks=-1 hybrid_selected_reuse_blocks=-1 "
        "outcome={} block_hashes=[{}]"
    ).format(
        request_id,
        schedule_batch_id,
        local_batch_id,
        arrival_time_us,
        arrival_time_us,
        start_time_us,
        finish_time_us,
        hybrid_tail_required,
        total_tokens,
        block_size,
        master_hit_tokens,
        agent_hit_tokens,
        pending_aware_field,
        rtp_hit_tokens,
        outcome,
        ",".join(str(value) for value in block_hashes),
    )


def lifecycle_line(
    event_time_us,
    sequence,
    event_type,
    event_hashes,
    *,
    added=(),
    removed=(),
    cause="request_store",
    cause_request_id=None,
    cause_batch_id=None,
    cause_schedule_batch_id=None,
    group_id=0,
):
    payload = {
        "event_time_us": event_time_us,
        "sequence": sequence,
        "event_type": event_type,
        "group_id": group_id,
        "cause": cause,
        "cause_request_id": cause_request_id,
        "cause_batch_id": cause_batch_id,
        "cause_schedule_batch_id": cause_schedule_batch_id,
        "event_hashes": list(event_hashes),
        "added": list(added),
        "removed": list(removed),
        "added_full_ready": [],
        "removed_full_ready": [],
        "added_tail_ready": [],
        "removed_tail_ready": [],
    }
    return "[navi-cache-lifecycle-audit] " + json.dumps(
        payload, separators=(",", ":"), sort_keys=True
    )


class PendingCacheKeyAuditTest(unittest.TestCase):
    def parse(self, lines):
        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory) / "10_0_0_1.agent.log"
            path.write_text("\n".join(lines) + "\n")
            return audit.parse_log(path)

    def parse_with_lifecycle(self, lines):
        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory) / "10_0_0_1.agent.log"
            path.write_text("\n".join(lines) + "\n")
            return audit.parse_log(path), audit.parse_lifecycle_log(path)

    def test_reports_master_physical_and_pending_aware_accuracy_separately(self):
        rows = self.parse(
            [
                audit_line(
                    1,
                    1,
                    100,
                    200,
                    300,
                    0,
                    16,
                    [11, 22],
                    total_tokens=17,
                    block_size=16,
                    master_hit_tokens=16,
                    agent_pending_aware_hit_tokens=16,
                    hybrid_tail_required=0,
                ),
                # Old logs remain readable but are excluded from pending-aware stats.
                audit_line(
                    2,
                    2,
                    400,
                    500,
                    600,
                    0,
                    0,
                    [33],
                    total_tokens=1,
                    block_size=16,
                    hybrid_tail_required=0,
                ),
            ]
        )

        summary = audit.summarize(audit.analyze_host(rows, linear_step=1))

        self.assertEqual(2, summary["prediction_accuracy"]["master"]["sample_count"])
        self.assertEqual(
            2, summary["prediction_accuracy"]["master"]["exact_request_count"]
        )
        self.assertEqual(
            1,
            summary["prediction_accuracy"]["agent_physical"][
                "underpredicted_request_count"
            ],
        )
        self.assertEqual(
            1,
            summary["prediction_accuracy"]["agent_pending_aware"]["sample_count"],
        )
        self.assertEqual(
            1,
            summary["prediction_accuracy"]["agent_pending_aware"][
                "exact_request_count"
            ],
        )

    def test_lifecycle_replay_attributes_add_remove_readd_clear_and_unknown(self):
        rows, events = self.parse_with_lifecycle(
            [
                audit_line(
                    1,
                    1,
                    100,
                    200,
                    250,
                    16,
                    48,
                    [11, 22, 33],
                    total_tokens=49,
                    block_size=16,
                    hybrid_tail_required=0,
                ),
                audit_line(
                    2,
                    2,
                    300,
                    400,
                    450,
                    48,
                    16,
                    [41, 42, 43],
                    total_tokens=49,
                    block_size=16,
                    hybrid_tail_required=0,
                ),
                lifecycle_line(
                    150,
                    4,
                    "stored",
                    [22],
                    added=[22],
                    cause_request_id="readd",
                    cause_batch_id=14,
                ),
                lifecycle_line(
                    150,
                    1,
                    "stored",
                    [22],
                    added=[22],
                    cause_request_id="producer",
                ),
                lifecycle_line(
                    150,
                    3,
                    "removed",
                    [22],
                    cause="request_allocation",
                    cause_request_id="duplicate-remove",
                ),
                lifecycle_line(
                    150,
                    2,
                    "removed",
                    [22],
                    removed=[22],
                    cause="request_allocation",
                    cause_request_id="evictor",
                ),
                lifecycle_line(
                    160,
                    5,
                    "stored",
                    [33],
                    added=[33],
                    cause="unknown",
                ),
                # Both window boundaries are inclusive.
                lifecycle_line(
                    300,
                    6,
                    "stored",
                    [42],
                    added=[42],
                    cause_request_id="producer-2",
                ),
                lifecycle_line(
                    400,
                    7,
                    "cleared",
                    [],
                    removed=[42, 43],
                    cause="prefix_cache_reset",
                    group_id=None,
                ),
            ]
        )

        details = audit.analyze_host(
            rows, linear_step=1, lifecycle_events=events
        )
        by_request = {row["request_id"]: row for row in details}
        multi = by_request[1]["multi_hit_lifecycle_attribution"]
        less = by_request[2]["less_hit_lifecycle_attribution"]

        self.assertEqual([1, 2, 3, 4], [row["sequence"] for row in multi[0]["events"]])
        self.assertEqual(4, multi[0]["attributed_event"]["sequence"])
        self.assertEqual("present", multi[0]["replayed_state_after_window"])
        self.assertTrue(multi[0]["cause_known"])
        self.assertEqual("unknown", multi[1]["attributed_event"]["cause"])
        self.assertFalse(multi[1]["cause_known"])
        self.assertEqual(7, less[0]["attributed_event"]["sequence"])
        self.assertEqual(7, less[1]["attributed_event"]["sequence"])
        self.assertFalse(less[1]["events"][0]["event_hash_match"])
        self.assertEqual("absent", less[1]["replayed_state_after_window"])

        summary = audit.summarize(details)
        self.assertEqual(2, summary["lifecycle_multi_hit_hash_count"])
        self.assertEqual(2, summary["lifecycle_explained_multi_hit_hash_count"])
        self.assertEqual(2, summary["lifecycle_less_hit_hash_count"])
        self.assertEqual(2, summary["lifecycle_explained_less_hit_hash_count"])
        self.assertEqual(
            1, summary["lifecycle_attributed_unknown_cause_hash_count"]
        )

    def test_pending_aware_underprediction_attributes_same_batch_execution_store(self):
        rows, events = self.parse_with_lifecycle(
            [
                audit_line(
                    101,
                    7,
                    100,
                    200,
                    400,
                    0,
                    32,
                    [11, 22, 33],
                    total_tokens=33,
                    block_size=16,
                    agent_pending_aware_hit_tokens=0,
                    schedule_batch_id=77,
                    hybrid_tail_required=0,
                ),
                lifecycle_line(
                    250,
                    1,
                    "stored",
                    [11, 22],
                    added=[11, 22],
                    cause_request_id="102",
                    cause_batch_id=7,
                    cause_schedule_batch_id=77,
                ),
            ]
        )

        detail = audit.analyze_host(
            rows, linear_step=1, lifecycle_events=events
        )[0]
        self.assertFalse(detail["all_multi_hit_hashes_lifecycle_explained"])
        self.assertTrue(
            detail["all_pending_multi_hit_hashes_lifecycle_explained"]
        )
        self.assertTrue(
            detail["all_pending_multi_hit_hashes_same_batch_store_explained"]
        )
        self.assertEqual(
            [1, 1],
            [
                item["attributed_event"]["sequence"]
                for item in detail["pending_multi_hit_lifecycle_attribution"]
            ],
        )

        summary = audit.summarize([detail])
        self.assertEqual(2, summary["pending_lifecycle_multi_hit_hash_count"])
        self.assertEqual(
            2, summary["pending_lifecycle_explained_multi_hit_hash_count"]
        )
        self.assertEqual(
            2,
            summary[
                "pending_same_batch_store_explained_multi_hit_hash_count"
            ],
        )

    def test_finish_time_index_preserves_candidate_boundaries(self):
        rows = self.parse(
            [
                audit_line(1, 1, 10, 20, 100, 0, 0, [1]),
                audit_line(2, 2, 10, 20, 101, 0, 0, [2]),
                audit_line(3, 3, 10, 20, 200, 0, 0, [3]),
                audit_line(4, 4, 150, 250, 300, 0, 0, [4]),
                audit_line(5, 5, 100, 300, 400, 0, 0, [5]),
            ]
        )
        index = audit.build_candidate_source_index(rows)

        candidates = audit.candidate_sources(rows[4], index)

        self.assertEqual([2, 3], [row["request_id"] for row in candidates])

    def test_pending_source_explains_actual_hit_and_detects_overprediction(self):
        rows = self.parse(
            [
                audit_line(1, 1, 100, 200, 300, 0, 0, [11, 22, 33]),
                audit_line(2, 2, 150, 400, 500, 0, 512, [11, 22, 44]),
                audit_line(3, 3, 160, 410, 510, 0, 256, [11, 22, 55]),
                audit_line(
                    4, 4, 170, 250, 260, 0, -1, [66], outcome="aborted"
                ),
            ]
        )
        details = audit.analyze_host(rows, linear_step=16)
        summary = audit.summarize(details)

        self.assertEqual(768, rows[0]["total_tokens"])
        self.assertEqual(0, rows[0]["master_hit_tokens"])
        self.assertEqual(1, rows[0]["hybrid_tail_required"])
        self.assertEqual(-1, rows[0]["hybrid_selected_reuse_blocks"])
        self.assertEqual(3, summary["request_count"])
        self.assertEqual(2, summary["positive_mismatch_request_count"])
        self.assertEqual(3, summary["positive_extra_block_count"])
        self.assertEqual(3, summary["physically_explained_extra_block_count"])
        self.assertEqual(3, summary["overlay_explained_extra_block_count"])
        self.assertEqual(2, summary["overlay_explained_positive_request_count"])
        self.assertEqual(1, summary["overlay_overpredicted_request_count"])
        self.assertEqual(0, summary["overlay_underpredicted_request_count"])
        by_request = {row["request_id"]: row for row in details}
        self.assertEqual(2, by_request[2]["overlay_hit_blocks"])
        self.assertEqual(2, by_request[3]["overlay_hit_blocks"])
        self.assertEqual(1, by_request[3]["overlay_error_blocks"])
        self.assertEqual(
            1, by_request[2]["extra_block_evidence"][0]["overlay_source_count"]
        )

    def test_future_arrival_and_same_batch_are_not_pending_evidence(self):
        rows = self.parse(
            [
                audit_line(10, 10, 50, 400, 500, 0, 256, [71]),
                audit_line(11, 11, 100, 200, 300, 0, 0, [71]),
                audit_line(12, 12, 600, 700, 800, 0, 0, [81]),
                audit_line(13, 12, 650, 900, 1000, 0, 256, [81]),
            ]
        )
        details = audit.analyze_host(rows, linear_step=16)
        by_request = {row["request_id"]: row for row in details}

        self.assertFalse(by_request[10]["all_extra_blocks_overlay_explained"])
        self.assertEqual(
            0, by_request[10]["extra_block_evidence"][0]["overlay_source_count"]
        )
        self.assertFalse(by_request[13]["all_extra_blocks_overlay_explained"])
        self.assertEqual(
            0, by_request[13]["extra_block_evidence"][0]["overlay_source_count"]
        )

    def test_same_schedule_split_is_physical_but_not_overlay_evidence(self):
        rows = self.parse(
            [
                audit_line(20, 20, 100, 200, 300, 0, 0, [91]),
                audit_line(21, 21, 150, 400, 500, 0, 256, [91]),
            ]
        )
        rows[1]["schedule_batch_id"] = rows[0]["schedule_batch_id"]

        details = audit.analyze_host(rows, linear_step=16)
        by_request = {row["request_id"]: row for row in details}
        evidence = by_request[21]["extra_block_evidence"][0]

        self.assertEqual(1, evidence["physical_source_count"])
        self.assertEqual(0, evidence["overlay_source_count"])
        self.assertTrue(by_request[21]["all_extra_blocks_physically_explained"])
        self.assertFalse(by_request[21]["all_extra_blocks_overlay_explained"])

    def test_pending_tail_policy_and_aligned_match_limit(self):
        rows = self.parse(
            [
                audit_line(
                    30, 30, 100, 200, 300, 0, 0, [1, 2, 3, 4, 5],
                    total_tokens=81, block_size=16,
                ),
                # Index 3 is periodic tail-ready for linear_step=2.
                audit_line(
                    31, 31, 150, 400, 500, 0, 64, [1, 2, 3, 4],
                    total_tokens=65, block_size=16,
                ),
                # Index 2 is not tail-ready, so matching rolls back to index 1.
                audit_line(
                    32, 32, 160, 410, 510, 0, 32, [1, 2, 3],
                    total_tokens=49, block_size=16,
                ),
                # An aligned target cannot reuse its final complete block.
                audit_line(
                    33, 33, 170, 420, 520, 0, 32, [1, 2, 3],
                    total_tokens=48, block_size=16,
                ),
            ]
        )

        details = audit.analyze_host(rows, linear_step=2)
        by_request = {row["request_id"]: row for row in details}

        self.assertEqual(4, by_request[31]["overlay_hit_blocks"])
        self.assertEqual(2, by_request[32]["overlay_hit_blocks"])
        self.assertEqual(2, by_request[33]["matchable_block_count"])
        self.assertEqual(2, by_request[33]["overlay_hit_blocks"])

    def test_aligned_source_marks_previous_and_final_blocks_tail_ready(self):
        rows = self.parse(
            [
                audit_line(
                    40, 40, 100, 200, 300, 0, 0, [11, 12, 13, 14],
                    total_tokens=64, block_size=16,
                ),
                audit_line(
                    41, 41, 150, 400, 500, 0, 48, [11, 12, 13],
                    total_tokens=49, block_size=16,
                ),
            ]
        )

        details = audit.analyze_host(rows, linear_step=16)
        by_request = {row["request_id"]: row for row in details}

        self.assertEqual(3, by_request[41]["overlay_hit_blocks"])
        self.assertEqual(
            1,
            by_request[41]["extra_block_evidence"][-1][
                "overlay_tail_source_count"
            ],
        )

    def test_unaligned_source_marks_last_complete_block_tail_ready(self):
        rows = self.parse(
            [
                audit_line(
                    50, 50, 100, 200, 300, 0, 0, [21, 22, 23],
                    total_tokens=63, block_size=16,
                ),
                audit_line(
                    51, 51, 150, 400, 500, 0, 48, [21, 22, 23],
                    total_tokens=49, block_size=16,
                ),
            ]
        )

        details = audit.analyze_host(rows, linear_step=16)
        by_request = {row["request_id"]: row for row in details}

        self.assertEqual(3, by_request[51]["overlay_hit_blocks"])

    def test_referenced_reservation_matches_production_and_exposes_alternative(self):
        rows = self.parse(
            [
                audit_line(
                    60, 60, 100, 200, 300, 32, 0, [31, 32],
                    total_tokens=47, block_size=16, master_hit_tokens=32,
                ),
                audit_line(
                    61, 61, 150, 400, 500, 0, 32, [31, 32],
                    total_tokens=33, block_size=16, master_hit_tokens=0,
                ),
                audit_line(
                    62, 62, 600, 700, 800, 16, 0, [41],
                    total_tokens=17, block_size=16, master_hit_tokens=16,
                ),
            ]
        )

        produced = audit.analyze_host(
            rows, linear_step=16, reservation_scope="produced"
        )
        produced_by_request = {row["request_id"]: row for row in produced}

        self.assertEqual(
            set(), audit.source_pending_hashes(rows[0], 16, "produced")[0]
        )
        self.assertEqual(
            {31, 32},
            audit.source_pending_hashes(rows[0], 16, "referenced")[0],
        )
        self.assertEqual(0, produced_by_request[61]["overlay_hit_blocks"])
        self.assertEqual(-2, produced_by_request[61]["overlay_error_blocks"])
        self.assertEqual(1, produced_by_request[62]["overlay_hit_blocks"])
        self.assertEqual(1, produced_by_request[62]["overlay_error_blocks"])

        referenced = audit.analyze_host(rows, linear_step=16)
        referenced_by_request = {row["request_id"]: row for row in referenced}
        self.assertEqual(2, referenced_by_request[61]["overlay_hit_blocks"])
        self.assertEqual(0, referenced_by_request[61]["overlay_error_blocks"])
        self.assertEqual(1, referenced_by_request[62]["overlay_hit_blocks"])

    def test_master_lifecycle_correlates_pending_source_terminal_states(self):
        lines = [
            "[root][2026-09-02 11:20:00.000][1] "
            "[navi-master-pending-cache-lifecycle] phase=reserved "
            "agent=agent_0 batch_id=7 schedule_time_us=1788319200000000 "
            "request_count=1 pending_batches=1 pending_blocks=3",
            "[root][2026-09-02 11:20:00.001][1] "
            "[navi-master-pending-cache-lifecycle] phase=all_requests_received "
            "agent=agent_0 batch_id=7",
            "[root][2026-09-02 11:20:00.002][1] "
            "[navi-master-cache-decision-audit] request_id=11 "
            "schedule_time_us=1788319200002000 batch_id=8 agent=agent_0 "
            "pending_first_source_batch_id=7 pending_tail_source_batch_id=7",
            "[root][2026-09-02 11:20:00.003][1] "
            "[navi-master-pending-cache-lifecycle] phase=batch_finished "
            "agent=agent_0 batch_id=7 retired=1 abnormal=0",
            "[root][2026-09-02 11:20:00.004][1] "
            "[navi-master-pending-cache-lifecycle] phase=reserved "
            "agent=agent_0 batch_id=9 schedule_time_us=1788319200004000 "
            "request_count=1 pending_batches=1 pending_blocks=3",
            "[root][2026-09-02 11:21:00.005][1] "
            "[navi-master-pending-cache-lifecycle] phase=expired "
            "agent=agent_0 expired_batches=1",
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory) / "master.log"
            path.write_text("\n".join(lines) + "\n")
            decisions, events = audit.parse_master_logs([path])

        report = audit.analyze_master_lifecycle(decisions, events, 60000.0)

        self.assertEqual(
            {"batch_finished_retired": 1, "expired": 1},
            report["reservation_status_counts"],
        )
        self.assertEqual(
            {"batch_finished_retired": 1},
            report["decision_source_status_counts"],
        )
        self.assertEqual(1, report["decision_source_count"])

    def test_master_log_trailing_bracket_is_not_part_of_scalar_value(self):
        lines = [
            "[root][2026-09-02 11:20:00.002][1] "
            "[navi-master-cache-decision-audit] request_id=11 "
            "schedule_time_us=1788319200002000 batch_id=8 agent=agent_0 "
            "pending_first_source_batch_id=7 pending_tail_source_batch_id=7]",
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory) / "master.log"
            path.write_text("\n".join(lines) + "\n")
            decisions, events = audit.parse_master_logs([path])

        self.assertEqual([], events)
        self.assertEqual(1, len(decisions))
        self.assertEqual(8, decisions[0]["batch_id"])
        self.assertEqual(7, decisions[0]["pending_tail_source_batch_id"])

    def test_master_lifecycle_compares_source_and_target_execution_order(self):
        rows = self.parse(
            [
                audit_line(1, 7, 100, 200, 500, 0, 0, [11]),
                audit_line(2, 8, 150, 400, 600, 0, 0, [22]),
            ]
        )
        decision = {
            "request_id": 2,
            "schedule_time_us": 150,
            "batch_id": 8,
            "agent": "10.0.0.1:19000",
            "pending_first_source_batch_id": 7,
            "pending_tail_source_batch_id": 0,
            "source_path": "master.log",
            "line_number": 1,
        }

        report = audit.analyze_master_lifecycle(
            [decision],
            [],
            60000.0,
            audit.build_agent_schedule_index({"10.0.0.1": rows}),
        )

        source = report["decision_sources"][0]
        self.assertEqual(1, source["source_agent_execution"]["request_count"])
        self.assertEqual(500, source["source_agent_execution"]["last_finish_time_us"])
        self.assertEqual(400, source["target_agent_execution"]["first_start_time_us"])
        self.assertFalse(source["source_completed_before_target_started"])

    def test_master_lifecycle_resolves_local_source_and_schedule_target_ids(self):
        rows = self.parse(
            [
                audit_line(
                    1, 7, 100, 200, 500, 0, 0, [11], schedule_batch_id=700
                ),
                audit_line(
                    2, 8, 150, 400, 600, 0, 0, [22], schedule_batch_id=800
                ),
            ]
        )
        decision = {
            "request_id": 2,
            "schedule_time_us": 150,
            "batch_id": 800,
            "agent": "10.0.0.1:19000",
            "pending_first_source_batch_id": 7,
            "pending_tail_source_batch_id": 0,
            "source_path": "master.log",
            "line_number": 1,
        }

        report = audit.analyze_master_lifecycle(
            [decision],
            [],
            60000.0,
            audit.build_agent_schedule_index({"10.0.0.1": rows}),
        )

        source = report["decision_sources"][0]
        self.assertEqual(1, source["source_agent_execution"]["request_count"])
        self.assertEqual(1, source["target_agent_execution"]["request_count"])
        self.assertEqual(500, source["source_agent_execution"]["last_finish_time_us"])
        self.assertEqual(400, source["target_agent_execution"]["first_start_time_us"])


if __name__ == "__main__":
    unittest.main()
