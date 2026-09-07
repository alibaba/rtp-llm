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
            **params,
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


class AdmissionProgramsTest(unittest.TestCase):
    """Execute complete shipped programs with explicit external-I/O fixtures."""

    setUp = AdmissionTests.setUp
    row = AdmissionTests.row

    def run_program(
        self,
        variant,
        profile="batch-window",
        bad_code=False,
        bad_latency=False,
        cleanup_error=False,
    ):
        import copy
        import threading
        from dataclasses import replace

        from flexlb_ft.scenario.contracts import CheckResult, StageOutput
        from flexlb_ft.scenario.runtime import execute_instance

        root = Path(__file__).resolve().parents[1] / "scenarios/admission"
        registry = handlers()
        registry.update({h.name: h for h in admission.HANDLERS})
        plans = compile_scenarios(load_scenarios(root), handlers=registry)
        plan = next(
            p for p in plans if p["variant_id"] == variant and p["profile"] == profile
        )
        state = SimpleNamespace(sent=0, active=False, lock=threading.Lock())
        test = self

        class Batch:
            def __init__(self, ctx, params):
                self.ctx = ctx
                self.entries = []
                self.records = []

            def submit(self, deadline):
                with state.lock:
                    state.sent += 1
                    i = state.sent
                code, text, latency = 200, None, 0.1
                if variant == "master_capacity" and 2 < i <= 4:
                    code, text = (
                        85020 if bad_code else 8502
                    ), "8502 TooManyRequests QUEUE_FULL"
                if variant == "slo_deadline" and state.active:
                    code, text, latency = (
                        8431,
                        "queue deadline expired",
                        (0.5 if bad_latency else 1.5),
                    )
                if variant == "queue_depth" and state.active and i >= 3:
                    code, text = 8510, "queue depth limit exceeded"
                row = test.row(code, text, latency)
                row["wire_request_id"] = i
                self.records = [row]
                self.entries = [
                    dict(
                        record=row,
                        response=SimpleNamespace(
                            code=code, success=code == 200, error_message=text or ""
                        ),
                    )
                ]

            def wait(self, deadline):
                pass

            def cancel(self, reason):
                pass

            def cleanup(self, deadline):
                if cleanup_error:
                    raise RuntimeError("fixture consumer cleanup failed")

            def snapshot_records(self):
                return copy.deepcopy(self.records)

        def external(handler):
            def execute(ctx, params, deadline):
                if handler.name == "engine_inject":
                    state.active = True
                if handler.name == "engine_clear":
                    state.active = False
                output = {}
                for key, kind in handler.outputs.items():
                    output[key] = (
                        True
                        if kind == "boolean"
                        else (
                            1.0 if kind == "number" else ctx.register_resource(kind, {})
                        )
                    )
                return StageOutput(
                    output,
                    [
                        CheckResult(name, "PASS", detail="external I/O fixture")
                        for name in handler.checks
                    ],
                )

            return replace(handler, execute=execute)

        for name in [
            "engine_control",
            "engine_inject",
            "engine_clear",
            "master_mark",
            "master_ready",
            "master_direct_clean",
        ]:
            registry[name] = external(registry[name])
        backend = SimpleNamespace(
            setup=lambda *args: (SimpleNamespace(), None), teardown=lambda *args: None
        )
        with patch.object(admission, "RequestBatch", Batch), patch.object(
            admission, "_http", return_value={}
        ), patch.object(
            admission,
            "_engines",
            side_effect=lambda *args: {
                f"prefill-{i}": {"waiting": int(state.sent >= 2), "running": 0}
                for i in range(2)
            },
        ):
            return execute_instance(
                plan, backend, registry, Path(self.tmp.name) / f"{variant}-{profile}"
            )

    def test_all_six_compiled_programs_execute_every_declared_check(self):
        for variant in ("queue_depth", "slo_deadline", "master_capacity"):
            for profile in ("batch-window", "single-batch"):
                with self.subTest(variant=variant, profile=profile):
                    result = self.run_program(variant, profile)
                    self.assertEqual("PASS", result["status"], result)
                    self.assertTrue(
                        all(s["status"] == "PASS" for s in result["stages"])
                    )
                    self.assertTrue(
                        all(c["status"] == "PASS" for c in result["cleanup"])
                    )
                    self.assertTrue(
                        any(
                            s["action"] == "admission_check" and s["checks"]
                            for s in result["stages"]
                        )
                    )

    def test_wrong_numeric_capacity_code_fails_its_actual_program_check(self):
        result = self.run_program("master_capacity", bad_code=True)
        self.assertEqual("FAIL", result["status"], result)
        stage = next(s for s in result["stages"] if s["id"] == "typed_code")
        self.assertEqual("FAIL", stage["checks"][0]["status"])

    def test_too_early_slo_failure_is_not_a_valid_queue_deadline(self):
        result = self.run_program("slo_deadline", bad_latency=True)
        self.assertEqual("FAIL", result["status"], result)
        self.assertEqual(
            "FAIL",
            next(s for s in result["stages"] if s["id"] == "waited")["checks"][0][
                "status"
            ],
        )

    def test_green_business_checks_cannot_hide_consumer_cleanup_error(self):
        result = self.run_program("master_capacity", cleanup_error=True)
        self.assertEqual("ERROR", result["status"], result)
        self.assertTrue(any(c["status"] == "ERROR" for c in result["cleanup"]))
