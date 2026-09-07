"""Admission evidence predicates and bounded submission ownership; no JVM startup."""

import json
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from flexlb_ft.scenario import compile_scenarios, load_scenarios
from flexlb_ft.scenario.actions import admission
from flexlb_ft.scenario.actions.elastic import ClientRecords
from flexlb_ft.scenario.catalog import handlers
from flexlb_ft.scenario.runtime import Deadline, RuntimeContext


class AdmissionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.ctx = RuntimeContext({}, None, self.tmp.name, time.monotonic, time.sleep)
        self.ctx.env_epoch = 1
        self.ctx.ops = None
        self.ctx.instance_deadline_s = time.monotonic() + 60
        self.deadline = Deadline(time.monotonic() + 5, time.monotonic, time.sleep)

    def row(self, code=200, error=None, elapsed=1):
        rows = ClientRecords(1)
        r = rows.issue(1, lambda: 10.0)
        rows.update(
            r,
            schedule=dict(
                status="OK" if code == 200 else "REJECTED",
                started_s=10.0,
                ended_s=10.0 + elapsed,
                error=error,
            ),
            stream=dict(status="OK"),
            consumer_exit_s=10.0 + elapsed,
            transport_terminal_s=10.0 + elapsed,
            business_finished=code == 200,
        )
        r["schedule_response"] = dict(
            code=code, error_message=error, success=code == 200
        )
        return r

    def check(self, rows, **params):
        h = self.ctx.register_resource("admission_rows", rows)
        params = dict(
            dict(
                rows=h,
                metric="success_count",
                expected=1,
                op="eq",
                min_samples=1,
                scope="all",
            ),
            **params
        )
        return admission._check(self.ctx, params, self.deadline).checks[0]

    def test_capacity_uses_exact_response_code_and_rejects_missing_samples(self):
        for code, expected in [(8502, "PASS"), (85020, "FAIL"), (8431, "FAIL")]:
            with self.subTest(code=code):
                result = self.check(
                    [self.row(code, "8502 TooManyRequests QUEUE_FULL")],
                    metric="all_reject_code",
                    expected=8502,
                    scope="rejected",
                )
                self.assertEqual(expected, result.status)
        self.assertEqual(
            "FAIL",
            self.check(
                [], metric="all_reject_code", expected=8502, scope="rejected"
            ).status,
        )

    def test_fast_reject_is_strict_less_than_three_seconds(self):
        self.assertEqual(
            "PASS",
            self.check(
                [self.row(8502, "QUEUE_FULL", 2.99)],
                metric="latency_max",
                expected=3,
                op="lt",
            ).status,
        )
        self.assertEqual(
            "FAIL",
            self.check(
                [self.row(8502, "QUEUE_FULL", 3)],
                metric="latency_max",
                expected=3,
                op="lt",
            ).status,
        )

    def test_text_predicate_cannot_turn_a_success_or_empty_error_into_rejection(self):
        for row in [self.row(), self.row(8502, "")]:
            self.assertEqual(
                "FAIL",
                self.check(
                    [row],
                    metric="all_error_contains",
                    expected=True,
                    text=["queue depth"],
                ).status,
            )

    def test_missing_latency_timestamp_is_error(self):
        row = self.row(8502, "error")
        row["schedule"]["started_s"] = None
        with self.assertRaises(ValueError):
            self.check([row], metric="latency_max", expected=3, op="lt")

    def test_missing_occupancy_counter_is_not_zero_or_occupied(self):
        with patch.object(admission, "_http", return_value={}), patch.object(
            admission, "_engines", return_value={"prefill-0": {"waiting": 1}}
        ):
            with self.assertRaises(ValueError):
                admission._occupy(self.ctx, {"targets": ["prefill-0"]}, self.deadline)
        self.assertTrue(all(x["status"] == "PASS" for x in self.ctx.cleanup(2)))

    def test_submission_exception_is_retained_and_cleanup_attempts_consumer_reap(self):
        batch = Mock()
        batch.entries = []
        batch.snapshot_records.return_value = []
        batch.submit.side_effect = ValueError("invalid protocol")
        with patch.object(admission, "RequestBatch", return_value=batch):
            out = admission._wave(
                self.ctx, admission._traffic_validate({}, None), self.deadline
            )
            with self.assertRaisesRegex(ValueError, "invalid protocol"):
                admission._wait(self.ctx, {"wave": out.output["wave"]}, self.deadline)
            cleanup = self.ctx.cleanup(2)
        self.assertTrue(all(x["status"] == "PASS" for x in cleanup))
        batch.cancel.assert_called()
        batch.cleanup.assert_called_once()
        self.assertTrue(list(Path(self.tmp.name).glob("admission-wave-*.json")))

    def test_queue_family_preserves_current_profile_axes_and_topologies(self):
        h = handlers()
        h.update({x.name: x for x in admission.HANDLERS})
        root = Path(__file__).resolve().parents[1] / "scenarios/admission"
        plans = compile_scenarios(load_scenarios(root), handlers=h)
        self.assertEqual(6, len(plans))
        for plan in plans:
            self.assertIn(plan["profile"], ("batch-window", "single-batch"))
            variant = plan["variant_id"]
            self.assertEqual(
                4 if variant == "queue_depth" else 2, plan["environment"]["n_decode"]
            )
            overrides = plan["environment"].get("config_overrides", {})
            if variant == "slo_deadline":
                self.assertEqual(1500, overrides["queue_timeout_ms"])
            if variant == "master_capacity":
                self.assertEqual(2, overrides["max_outstanding"])
                self.assertEqual(60000, overrides["queue_timeout_ms"])
