"""Counterexamples for offline batch evidence and owned master recovery."""

import json
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import Mock, patch

from flexlb_test_framework.engine_ops import EngineOps
from flexlb_test_framework.scenario.actions import status_protocol
from flexlb_test_framework.scenario.actions.execution_evidence import (
    avoidance,
    execution_batches,
    partial_outcome_bounds,
    rotation,
)
from flexlb_test_framework.scenario.actions.master import (
    decode_residual_bound,
    owner_clean,
)


def event(rid, batch, engine, arrival=0, start=1, end=10):
    return dict(
        event="prefill_done",
        rid=rid,
        batch_id=batch,
        engine_name=engine,
        engine_arrival_ms=arrival,
        prefill_start_ms=start,
        prefill_done_ms=end,
    )


class ExecutionEvidenceTests(unittest.TestCase):
    def test_recovery_under_other_owner_load_is_not_a_distribution_oracle(self):
        from flexlb_test_framework.scenario.actions import master_observation as obs

        rows = [
            dict(
                wire_request_id=i,
                business_finished=True,
                business_error_code=None,
                schedule={"status": "OK"},
                stream={"status": "OK"},
                cancel={"requested_s": None},
                prefill_addr="P0",
            )
            for i in range(20)
        ]
        ctx = NS(resource=lambda *a: NS(snapshot_records=lambda: rows))
        with patch.object(
            obs, "_pools", return_value={"prefill": ["P0", "P1"]}
        ), patch.object(obs, "_artifact", return_value="probe.json"):
            result = obs.probe_distribution(
                ctx,
                {"requests": "r", "mode": "recovery", "max_prefill_share": 0.75},
                NS(),
            )
            self.assertEqual("PASS", result.checks[0].status)
            for row in rows[:2]:
                row["business_finished"] = False
            result = obs.probe_distribution(
                ctx,
                {"requests": "r", "mode": "recovery", "max_prefill_share": 0.75},
                NS(),
            )
            self.assertEqual("FAIL", result.checks[0].status)
            rows[2]["prefill_addr"] = "unknown"
            with self.assertRaisesRegex(ValueError, "unknown Prefill"):
                obs.probe_distribution(
                    ctx,
                    {"requests": "r", "mode": "recovery", "max_prefill_share": 0.75},
                    NS(),
                )

    def test_capacity_uses_distinct_batches_and_half_open_intervals(self):
        from flexlb_test_framework.scenario.actions.admission import execution_capacity

        rows = [
            event(1, 1, "P0", 0, 1, 10),
            event(2, 1, "P0", 0, 1, 10),
            event(3, 2, "P0", 0, 10, 20),
        ]
        self.assertEqual({"P0": 1}, execution_capacity(rows, {1, 2, 3})["peaks"])
        rows[-1]["prefill_start_ms"] = 9
        self.assertEqual({"P0": 2}, execution_capacity(rows, {1, 2, 3})["peaks"])
        with self.assertRaisesRegex(ValueError, "missing"):
            execution_capacity(rows, {1, 2, 3, 4})

    def test_single_merged_and_cross_engine_batches(self):
        rows = [event(1, 1, "P0"), event(2, 1, "P0"), event(3, 1, "P1")]
        self.assertEqual(1, len(execution_batches(rows, {1, 2})))
        self.assertEqual(2, len(execution_batches(rows, {1, 2, 3})))
        self.assertEqual([1, 2], execution_batches(rows, {2})[0]["rids"])
        self.assertEqual(1, len(execution_batches([rows[0]], {1})))

    def test_missing_rid_is_invalid_not_zero_batches(self):
        with self.assertRaisesRegex(ValueError, "missing"):
            execution_batches([event(1, 1, "P0")], {1, 2})

    def test_avoidance_only_busy_target_fails_and_own_batch_not_busy(self):
        busy = event(90, 1, "P0", 0, 1, 20)
        for chosen, violations in [("P0", 1), ("P1", 0)]:
            result = avoidance(
                [busy, event(2, 2, chosen, 5, 21, 30)], {2}, ["P0", "P1"]
            )
            self.assertEqual(violations, result["violations"])
        self.assertEqual(
            0, avoidance([event(2, 2, "P0", 1, 1, 10)], {2}, ["P0", "P1"])["violations"]
        )

    def test_avoidance_includes_completion_millisecond(self):
        rows = [event(90, 1, "P0", 0, 1, 20), event(91, 2, "P1", 0, 1, 10)]
        # At the recorded completion millisecond both engines are busy.
        for arrival, expected in [(10, 0), (11, 1)]:
            result = avoidance(
                rows + [event(2, 3, "P0", arrival, 21, 30)], {2}, ["P0", "P1"]
            )
            self.assertEqual(expected, result["violations"])

    def test_rotation_guards_sticky_and_consecutive_runs(self):
        def run(seq, limit):
            return rotation([dict(engine=x) for x in seq], limit)["passed"]

        self.assertTrue(run("ABAB", 1))
        self.assertFalse(run("ABBA", 1))
        self.assertTrue(run("ABBA", 2))
        self.assertFalse(run("ABBBA", 2))
        self.assertFalse(run("AAAA", 2))

    def test_three_batches_allow_three_failures_and_missing_sidecar_is_explicit(self):
        e = [
            event(1, 1, "P0"),
            event(2, 2, "P1"),
            event(3, 2, "P1"),
            event(4, 3, "P0", 2, 3, 4),
        ]
        v = partial_outcome_bounds(e, {1, 2, 3, 4}, 1, 3)
        self.assertEqual(
            (1, 3, False), (v["success_min"], v["failure_max"], v["degraded"])
        )
        self.assertTrue(partial_outcome_bounds(None, {1, 2, 3, 4}, 1, 3)["degraded"])

    def test_partial_zero_failures_still_fails(self):
        records = [
            dict(wire_request_id=i, issued_s=0, consumer_exit_s=1) for i in range(4)
        ]
        ctx = NS(
            resource=lambda *a: NS(snapshot_records=lambda: records),
            env=NS(run_dir=Path("/missing")),
        )
        p = dict(
            requests={},
            failure_min=1,
            error_code=8500,
            per_execution_batch=dict(fallback_success_min=1, fallback_failure_max=3),
        )
        with patch.object(status_protocol, "_terminal_records"), patch.object(
            status_protocol, "_artifact", return_value="fixture"
        ), patch(
            "flexlb_test_framework.scenario.actions.elastic.request_success",
            return_value=True,
        ):
            result = status_protocol.execute_outcomes(ctx, p, NS(check=lambda: None))
        self.assertEqual("FAIL", result.checks[0].status)

    def test_bound_tracks_client_configuration_and_owner_stays_strict(self):
        for n in [1, 4, 16]:
            client = NS(flow=NS(_overrides={"MAX_CONCURRENCY": str(n)}))
            b = decode_residual_bound(client)
            self.assertEqual(n, b)
            self.assertTrue(owner_clean(0, {"prefill": [1, 0], "decode": [b, 0]}, 1, b))
            self.assertFalse(
                owner_clean(0, {"prefill": [1, 0], "decode": [b + 1, 0]}, 1, b)
            )
            self.assertFalse(
                owner_clean(1, {"prefill": [0, 0], "decode": [0, 0]}, 1, b)
            )

    def test_channel_invalidation_is_targeted_idempotent_and_recreates(self):
        ops = EngineOps.__new__(EngineOps)
        old, engine, new = Mock(), Mock(), Mock()
        ops._channels = {"master": old, "engine": engine}
        ops.invalidate_channel("master")
        ops.invalidate_channel("master")
        old.close.assert_called_once()
        engine.close.assert_not_called()
        with patch(
            "flexlb_test_framework.engine_ops.grpc.insecure_channel", return_value=new
        ):
            self.assertIs(new, ops._channel("master"))
        self.assertIs(engine, ops._channel("engine"))
