"""Protocol checks use independent owner evidence and real bounded fake-RPC drivers."""

import ast
import copy
import json
import tempfile
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch

from flexlb_test_framework.scenario.actions import status_protocol as status
from flexlb_test_framework.scenario.contracts import PlanContext
from flexlb_test_framework.scenario.runtime import Deadline, RuntimeContext
from test_scenario_backend import Ops, Stream


def owner_frame(decode_load=0):
    return {
        "inflight": {
            "scheduler_inflight": 0,
            "prefill_endpoints": [
                {"ip_port": "p", "inflight_batches": 0, "inflight_requests": 0}
            ],
            "decode_endpoints": [
                {
                    "ip_port": "d",
                    "reserved_total": decode_load,
                    "master_queued": 0,
                    "confirmed_accepted": 0,
                    "confirmed_running": 0,
                    "total_load": decode_load,
                    "engine_load": 0,
                    "active_dispatch_permits": 0,
                    "engine_capacity_used": 0,
                }
            ],
        }
    }


class StatusProtocolTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.ctx = RuntimeContext({}, None, self.tmp.name, time.monotonic, time.sleep)
        self.ctx.env_epoch = 1
        self.ctx.env = NS(
            mock_http_port=10000, master_http_port=10001, master_management_port=10002
        )
        self.ctx.ops = Ops()
        self.ctx.instance_deadline_s = time.monotonic() + 5
        self.plan = PlanContext("test", {}, profiles=("batch-window",))
        self.addCleanup(lambda: self.ctx.cleanup(2))

    def deadline(self):
        return Deadline(time.monotonic() + 2, time.monotonic, time.sleep)

    def test_decode_total_load_and_fingerprint_never_use_missing_expected_field(self):
        frame = owner_frame(3)
        self.assertEqual(3, status.metric(frame, "decode_total_load"))
        self.assertNotEqual(
            status.metric(frame, "fingerprint"),
            status.metric(owner_frame(), "fingerprint"),
        )
        del frame["inflight"]["decode_endpoints"][0]["total_load"]
        with self.assertRaises(KeyError):
            status.metric(frame, "decode_total_load")

    def test_sparse_successful_master_summary_proves_absent_role_only(self):
        # Exact relevant fields from the preserved full385 batch-02 HTTP
        # response for instance-6863b441...; the original result stays ERROR.
        info = {
            "success": True,
            "code": 200,
            "ready": False,
            "worker_summary": {
                "DECODE": {"discovered": 1, "alive": 1, "maxQueueTokens": 0}
            },
        }
        original = copy.deepcopy(info)
        self.assertEqual(0, status.metric({"info": info}, "alive_prefill"))
        self.assertEqual(1, status.metric({"info": info}, "alive_decode"))
        self.assertEqual(original, info)
        restored = copy.deepcopy(info)
        restored["worker_summary"]["PREFILL"] = {"discovered": 1, "alive": 1}
        self.assertEqual(1, status.metric({"info": restored}, "alive_prefill"))
        # The producer explicitly emits null for an entirely empty directory.
        self.assertEqual(
            0,
            status.metric(
                {"info": {"success": True, "code": 200, "worker_summary": None}},
                "alive_prefill",
            ),
        )

    def test_missing_or_malformed_master_evidence_is_not_absent_role_zero(self):
        bad = [
            None,
            {},
            {"success": True, "code": 200},
            {"worker_summary": {}},
            {"success": False, "code": 500, "worker_summary": {}},
            {"success": True, "code": 200, "worker_summary": []},
            {"success": True, "code": 200, "worker_summary": {"DECODE": {}}},
            {
                "success": True,
                "code": 200,
                "worker_summary": {"PREFILL": {"discovered": 0}},
            },
            {
                "success": True,
                "code": 200,
                "worker_summary": {"DECODE": {"discovered": 1, "alive": True}},
            },
            {
                "success": True,
                "code": 200,
                "worker_summary": {"DECODE": {"discovered": 0, "alive": 1}},
            },
        ]
        for info in bad:
            with self.subTest(info=info), self.assertRaises(
                (KeyError, TypeError, RuntimeError, ValueError)
            ):
                status.metric({"info": info}, "alive_prefill")

    def test_unknown_fault_fields_and_channels_rejected_before_io(self):
        for params in [
            {"fault": "status_fake_task", "config": {"batchId": 1}},
            {"fault": "missing"},
            {"fault": "enqueue_ack_drop", "expected_http": [200, 400]},
            {"fault": "status_fake_task", "selection": "landing"},
        ]:
            with self.subTest(params=params), self.assertRaises(ValueError):
                status.validate_control(params, self.plan)

    def test_injection_cleanup_registered_before_partial_failure_and_epoch_safe(self):
        p = status.validate_control({"fault": "status_suppress_finished"}, self.plan)
        calls = []

        def http(ctx, server, path, deadline, body=None, allowed=(200,), text=False):
            calls.append(body)
            if body["engine"] == "p1" and body["enabled"]:
                raise OSError("injection reply missing")
            return 200, {}

        with patch.object(status, "_targets", return_value=["p0", "p1"]), patch.object(
            status, "_http", side_effect=http
        ):
            with self.assertRaises(OSError):
                status.execute_control(self.ctx, p, self.deadline())
            self.assertEqual(1, len(self.ctx._cleanup))
            results = self.ctx.cleanup(1)
        self.assertEqual(["PASS"], [r["status"] for r in results])
        self.assertEqual([True, True, False, False], [x["enabled"] for x in calls])
        with patch.object(status, "_targets", return_value=["p0"]), patch.object(
            status, "_http", return_value=(200, {})
        ):
            status.execute_control(self.ctx, p, self.deadline())
        self.ctx.env_epoch = 2
        with patch.object(status, "_http") as http:
            results = self.ctx.cleanup(1)
        http.assert_not_called()
        self.assertEqual("ERROR", results[0]["status"])

    def test_prepared_cohort_is_bound_before_dispatch_and_deferred_wait_fetches(self):
        p = status.validate_prepare(
            {"count": 3, "concurrency": 2, "consume": "deferred"}, self.plan
        )
        result = status.execute_prepare(self.ctx, p, self.deadline())
        requests = self.ctx.resource(result.output["requests"], "requests")
        before = requests.snapshot_records()
        ids = [r["wire_request_id"] for r in before]
        self.assertEqual(3, len(set(ids)))
        self.assertTrue(all(r["issued_s"] is None for r in before))
        self.assertEqual(0, self.ctx.ops.fetch_count)
        requests.dispatch(self.deadline())
        self.assertEqual(
            ids, [r["wire_request_id"] for r in requests.snapshot_records()]
        )
        self.assertEqual(0, self.ctx.ops.fetch_count)
        self.assertEqual(
            {"completed": True, "error_count": 0}, requests.wait(self.deadline())
        )
        self.assertEqual(3, self.ctx.ops.fetch_count)
        self.assertTrue(all(event.is_set() for event in requests.done))
        self.assertEqual(2, len(requests.exit_records))

    def test_undispatched_records_cannot_pass_outcome_checks(self):
        p = status.validate_prepare({"count": 1}, self.plan)
        result = status.execute_prepare(self.ctx, p, self.deadline())
        with self.assertRaises(RuntimeError):
            status.execute_outcomes(
                self.ctx,
                {"requests": result.output["requests"], "success_min": 0},
                self.deadline(),
            )

    def test_ack_and_execution_errors_are_separate_normal_checks(self):
        def record(phase):
            return dict(
                issued_s=1,
                consumer_exit_s=2,
                transport_terminal_s=2,
                consumer_done=True,
                consumer_completion_verified=True,
                business_finished=False,
                business_error_code=8500 if phase == "execution" else None,
                business_error_message="injected",
                cancel={"requested_s": None},
                schedule={
                    "status": "OK" if phase == "execution" else "REJECTED",
                    "error": "8500",
                    "ended_s": 2,
                },
                stream={
                    "status": "OK" if phase == "execution" else None,
                    "started_s": 1 if phase == "execution" else None,
                    "ended_s": 2 if phase == "execution" else None,
                },
            )

        for phase in ("schedule", "execution"):
            provider = NS(snapshot_records=lambda: [record(phase)])
            handle = self.ctx.register_resource("requests", provider)
            for expected in ("schedule", "execution"):
                result = status.execute_outcomes(
                    self.ctx,
                    {"requests": handle, "error_code": 8500, "failure_phase": expected},
                    self.deadline(),
                )
                self.assertEqual(
                    "PASS" if phase == expected else "FAIL", result.checks[0].status
                )
        provider = NS(
            snapshot_records=lambda: [
                dict(
                    record("execution"),
                    stream={
                        "status": "DEADLINE_EXCEEDED",
                        "started_s": 1,
                        "ended_s": 2,
                    },
                )
            ]
        )
        handle = self.ctx.register_resource("requests", provider)
        with self.assertRaises(TimeoutError):
            status.execute_outcomes(self.ctx, {"requests": handle}, self.deadline())

    def test_expected_rpc_failure_requires_typed_status_and_all_exit_evidence(self):
        class RpcFailure(Exception):
            def code(self):
                return NS(name="INTERNAL")

        self.ctx.ops.pb2_grpc.RpcServiceStub = lambda channel: NS(
            FetchResponse=lambda *args, **kwargs: Stream(error=RpcFailure("injected"))
        )
        p = status.validate_prepare(
            {"count": 1, "expected_rpc_statuses": ["INTERNAL"]}, self.plan
        )
        result = status.execute_prepare(self.ctx, p, self.deadline())
        cohort = self.ctx.resource(result.output["requests"], "requests")
        cohort.dispatch(self.deadline())
        self.assertEqual(
            {"completed": False, "error_count": 1}, cohort.wait(self.deadline())
        )
        record = cohort.snapshot_records()[0]
        self.assertTrue(cohort.children[0].status_done.is_set())
        self.assertTrue(record["consumer_done"])
        self.assertTrue(record["consumer_completion_verified"])
        status._terminal_records([record], ["INTERNAL"])
        for key in ("consumer_exit_s", "transport_terminal_s"):
            broken = dict(record, **{key: None})
            with self.assertRaises(RuntimeError):
                status._terminal_records([broken], ["INTERNAL"])
        broken = copy.deepcopy(record)
        broken["stream"]["status"] = "ERROR"
        with self.assertRaises(RuntimeError):
            status._terminal_records([broken], ["INTERNAL"])
        with self.assertRaises(TimeoutError):
            cohort.wait(Deadline(time.monotonic() - 1, time.monotonic, time.sleep))

    def test_ack_drop_observes_completed_future_without_rewriting_raw_timeout(self):
        import grpc

        calls = []

        class Call:
            def result(self, timeout):
                raise grpc.FutureTimeoutError()

            def done(self):
                calls.append("done")
                return True

            def cancelled(self):
                calls.append("cancelled")
                return False

            def code(self):
                calls.append("code")
                return grpc.StatusCode.DEADLINE_EXCEEDED

            def cancel(self):
                calls.append("cleanup_cancel")
                return False

        self.ctx.ops.future = lambda *a, **kw: Call()
        p = status.validate_prepare(
            {
                "count": 4,
                "concurrency": 4,
                "expected_rpc_statuses": ["DEADLINE_EXCEEDED"],
                "observe_schedule_future_terminal": True,
            },
            self.plan,
        )
        handle = status.execute_prepare(self.ctx, p, self.deadline()).output["requests"]
        cohort = self.ctx.resource(handle, "requests")
        cohort.dispatch(self.deadline())
        self.assertEqual(
            {"completed": False, "error_count": 4}, cohort.wait(self.deadline())
        )
        records = cohort.snapshot_records()
        self.assertEqual(4, len(records))
        self.assertEqual(
            (0, 0), (self.ctx.ops.fetch_count, self.ctx.ops.generate_count)
        )
        for record in records:
            self.assertEqual("ERROR", record["schedule"]["status"])
            self.assertIn("FutureTimeoutError", record["schedule"]["error"])
            self.assertEqual(
                "DEADLINE_EXCEEDED", record["schedule"]["future_terminal"]["code"]
            )
            self.assertIsNone(record["stream"]["started_s"])
            self.assertFalse(record["business_finished"])
            self.assertFalse(record.get("consumer_completion_verified", False))
        self.assertEqual(4, calls.count("code"))
        self.assertLess(calls.index("code"), calls.index("cleanup_cancel"))
        with self.assertRaises(RuntimeError):
            status._terminal_records(records, ["DEADLINE_EXCEEDED"])

    def test_schedule_future_timeout_without_terminal_proof_remains_error(self):
        import grpc

        for enabled, done, cancelled, code in (
            (False, True, False, grpc.StatusCode.DEADLINE_EXCEEDED),
            (True, False, False, grpc.StatusCode.DEADLINE_EXCEEDED),
            (True, True, True, grpc.StatusCode.DEADLINE_EXCEEDED),
            (True, True, False, grpc.StatusCode.CANCELLED),
            (True, True, False, grpc.StatusCode.INTERNAL),
            (True, True, False, grpc.StatusCode.OK),
        ):
            with self.subTest(
                enabled=enabled, done=done, cancelled=cancelled, code=code
            ):
                observations = []

                def actual_code():
                    observations.append("code")
                    return code

                def result(timeout):
                    raise grpc.FutureTimeoutError()

                self.ctx.ops.future = lambda *a, **kw: NS(
                    result=result,
                    done=lambda: done,
                    cancelled=lambda: cancelled,
                    code=actual_code,
                    cancel=lambda: False,
                )
                p = status.validate_prepare(
                    {
                        "count": 1,
                        "expected_rpc_statuses": ["DEADLINE_EXCEEDED"],
                        "observe_schedule_future_terminal": enabled,
                    },
                    self.plan,
                )
                handle = status.execute_prepare(self.ctx, p, self.deadline()).output[
                    "requests"
                ]
                cohort = self.ctx.resource(handle, "requests")
                with self.assertRaises(grpc.FutureTimeoutError):
                    cohort.dispatch(self.deadline())
                record = cohort.snapshot_records()[0]
                self.assertEqual("ERROR", record["schedule"]["status"])
                self.assertIsNone(record["stream"]["started_s"])
                if not enabled or not done or cancelled:
                    self.assertEqual([], observations)

    def test_schedule_terminal_opt_in_is_strict_and_only_ack_drop_enables_it(self):
        from flexlb_test_framework.scenario import compile_scenarios, load_scenarios
        from flexlb_test_framework.scenario.catalog import handlers

        for params in (
            {"observe_schedule_future_terminal": True},
            {"observe_schedule_future_terminal": "true"},
        ):
            with self.subTest(params=params), self.assertRaises(ValueError):
                status.validate_prepare(params, self.plan)
        root = Path(__file__).resolve().parents[1]
        plans = compile_scenarios(
            load_scenarios(root / "scenarios"), handlers=handlers()
        )
        enabled = [
            (p["variant_id"], s["id"])
            for p in plans
            for s in p["stages"]
            if s["params"].get("observe_schedule_future_terminal")
        ]
        self.assertEqual([("ack_drop", "uncertain_cohort")], enabled)

    def test_prefill_acceptance_cannot_be_inflated_by_decode(self):
        frame = {
            "mock": {
                "p": {"role": "PREFILL", "accepted": 2},
                "d": {"role": "DECODE", "accepted": 20},
            }
        }
        self.assertEqual(2, status.metric(frame, "prefill_accepted"))
        self.assertEqual(22, status.metric(frame, "accepted"))

    def test_ttl_environment_matches_full_expected_spec_and_rejects_base_fallbacks(
        self,
    ):
        from dataclasses import asdict

        from flexlb_cfg import OMIT, ConfigOverride, render_env
        from flexlb_test_framework.harness import EnvSpec, fault_env_perf
        from flexlb_test_framework.scenario.backend import make_env_spec
        from flexlb_test_framework.scenario.catalog import handlers
        from flexlb_test_framework.scenario.compiler import compile_scenarios
        from flexlb_test_framework.scenario.loader import load_scenarios

        root = Path(__file__).resolve().parents[1]
        old = EnvSpec(
            label="fault_ttl_batch_window",
            n_prefill=2,
            n_decode=2,
            perf=fault_env_perf(),
            master_profile="batch-window",
            discovery="discovery_file",
            config_overrides=ConfigOverride(
                ordering="priority", queue_timeout_ms=OMIT, request_timeout_ms=30000
            ),
        )
        registry = handlers()
        registry.update({h.name: h for h in status.HANDLERS})
        plans = compile_scenarios(
            load_scenarios(root / "scenarios/status/status_protocol.yaml"),
            handlers=registry,
        )
        plan = next(p for p in plans if p["variant_id"] == "inflight_ttl_cleanup")
        self.assertEqual("batch-window", plan["profile"])

        def assert_equivalent(environment):
            actual = make_env_spec(environment, plan["profile"], {})
            expected_fields, actual_fields = asdict(old), asdict(actual)
            for fields in (expected_fields, actual_fields):
                fields.pop("label")  # fresh scenario label is intentional
                fields.pop("config_overrides")  # compare the rendered config below
            self.assertEqual(expected_fields, actual_fields)
            self.assertEqual(
                render_env(old.master_profile, old.config_overrides),
                render_env(actual.master_profile, actual.config_overrides),
            )

        assert_equivalent(plan["environment"])
        self.assertNotIn(
            "queueTimeoutMs", plan["environment"]["resolved_config"]["scheduler"]
        )
        # Each inherited base setting changes a real test channel: expiry,
        # Prefill timing, or discovery/retirement. None may silently return.
        for channel, value in (
            ("queue_timeout_ms", 10000),
            ("perf_preset", "default"),
            ("discovery", "file"),
        ):
            broken = copy.deepcopy(plan["environment"])
            if channel == "queue_timeout_ms":
                broken["config_overrides"][channel] = value
            else:
                broken[channel] = value
            with self.subTest(channel=channel), self.assertRaises(AssertionError):
                assert_equivalent(broken)

    def test_ttl_suppressed_wave_keeps_serial_schedule_shape_and_no_fetch(self):
        from flexlb_test_framework.scenario import compile_scenarios, load_scenarios
        from flexlb_test_framework.scenario.catalog import handlers

        root = Path(__file__).resolve().parents[1]
        plan = next(
            p
            for p in compile_scenarios(
                load_scenarios(root / "scenarios/status/status_protocol.yaml"),
                handlers=handlers(),
            )
            if p["variant_id"] == "inflight_ttl_cleanup"
        )
        stages = {s["id"]: s for s in plan["stages"]}
        params = stages["silent"]["params"]
        # Drive the real prepared-cohort submission with its first Schedule
        # blocked: no second call may start until the first call returns.
        requests = status.StatusRequests(self.ctx, params)
        self.ctx.register_resource("requests", requests, requests.cleanup)
        entered, release = threading.Event(), threading.Event()
        calls, errors = [], []
        original = self.ctx.ops.future

        def future(req, timeout, metadata=None):
            calls.append((req, timeout))
            call = original(req, timeout, metadata)
            if len(calls) == 1:
                result = call.result

                def first_result(timeout):
                    entered.set()
                    if not release.wait(1):
                        raise RuntimeError("test did not release first Schedule")
                    return result(timeout)

                call.result = first_result
            return call

        self.ctx.ops.future = future

        def dispatch():
            try:
                requests.dispatch(Deadline(time.monotonic() + 190))
            except Exception as exc:
                errors.append(exc)

        worker = threading.Thread(target=dispatch)
        worker.start()
        try:
            self.assertTrue(entered.wait(1))
            self.assertEqual(1, len(requests.threads))
            self.assertEqual(1, len(calls))
        finally:
            release.set()
            worker.join(2)
        self.assertFalse(worker.is_alive())
        self.assertEqual([], errors)
        self.assertEqual(6, len(calls))
        self.assertTrue(
            all(
                shape == {"input_len": 2048, "output_len": 10} and timeout == 30
                for (_, shape), timeout in calls
            )
        )
        self.assertEqual(
            (0, 0), (self.ctx.ops.fetch_count, self.ctx.ops.generate_count)
        )
        self.assertTrue(
            all(
                r["stream"]["started_s"] is None and not r["business_finished"]
                for r in requests.snapshot_records()
            )
        )
        self.assertGreaterEqual(stages["silent_dispatch"]["timeout_s"], 6 * 30)
        self.assertEqual(0.5, stages["accepted_window"]["params"]["interval_s"])
        self.assertEqual(2, stages["ttl_drain"]["params"]["interval_s"])

    def test_all_expected_cases_have_explicit_programs_and_profile_mapping(self):
        from flexlb_test_framework.scenario.catalog import handlers
        from flexlb_test_framework.scenario.compiler import (
            compile_scenarios,
            plan_counts,
        )
        from flexlb_test_framework.scenario.loader import load_scenarios

        root = Path(__file__).resolve().parents[1]
        docs = load_scenarios(root / "scenarios/status")
        registry = handlers()
        registry.update({h.name: h for h in status.HANDLERS})
        plans = compile_scenarios(docs, handlers=registry)
        self.assertEqual(3, plan_counts(plans)["logical_scenarios"])
        self.assertEqual(28, plan_counts(plans)["variants"])
        expected = json.loads((root / "tests/fixtures/instance_ids.json").read_text())
        actual_ids = sorted(p["id"] for p in plans)
        scenario_ids = {p["scenario_id"] for p in plans}
        self.assertEqual(
            actual_ids, [i for i in expected if i.split("::")[0] in scenario_ids]
        )
        for _, doc in docs:
            for variant in doc["variants"]:
                self.assertTrue(variant["stages"])
                self.assertNotIn("stage_overrides", variant)
                self.assertTrue(
                    all(s["action"] != "legacy_case" for s in variant["stages"])
                )
        zombie = next(p for p in plans if p["variant_id"] == "zombie_fake_running")
        self.assertEqual([], zombie["findings"])
        stages = {s["id"]: s for s in zombie["stages"]}
        self.assertEqual(60, stages["active_ghost_window"]["params"]["duration_s"])
        self.assertEqual(95, stages["clear_retirement_window"]["params"]["duration_s"])
        unbatched = next(
            p for p in plans if p["variant_id"] == "unbatched_single_request"
        )
        self.assertEqual(
            4, sum(s["id"].endswith("_ignored") for s in unbatched["stages"])
        )
        multi = next(p for p in plans if p["variant_id"] == "ack_multi_error")
        checks = [
            s["params"]["metric"]
            for s in multi["stages"]
            if s["action"] == "status_check"
        ]
        self.assertEqual(["scheduler", "scheduler", "master_http"], checks)
        execution = next(p for p in plans if p["variant_id"] == "execution_partial")
        stages = {s["id"]: s for s in execution["stages"]}
        self.assertEqual("last", stages["no_resurrection"]["params"]["aggregate"])
        ids = [s["id"] for s in execution["stages"]]
        self.assertLess(ids.index("serial_fail_off"), ids.index("serial_perf_restore"))
        self.assertLess(ids.index("serial_perf_restore"), ids.index("serial_drained"))
        duplicate = next(p for p in plans if p["variant_id"] == "duplicate_finished")
        self.assertFalse(
            any(s["id"] == "scheduler_retires" for s in duplicate["stages"])
        )
        decode_first = next(
            p for p in plans if p["variant_id"] == "decode_before_prefill"
        )
        ids = [s["id"] for s in decode_first["stages"]]
        self.assertLess(
            ids.index("p_terminal_restore"), ids.index("prefill_after_clear")
        )
        self.assertNotIn("prefill_eventually_retires", ids)
        special = next(p for p in plans if p["variant_id"] == "special_ids")
        stages = {s["id"]: s for s in special["stages"]}
        for name in ("zero_batch_on", "negative_batch_on"):
            self.assertEqual(8500, stages[name]["params"]["config"]["error_code"])
        unknown_batch = next(p for p in plans if p["variant_id"] == "unknown_batchid")
        ids = [s["id"] for s in unknown_batch["stages"]]
        self.assertLess(ids.index("slow_prefill"), ids.index("baseline_request"))
        nofetch = next(p for p in plans if p["variant_id"] == "normal_no_fetch")
        self.assertFalse(
            any(s["action"] == "status_control" for s in nofetch["stages"])
        )
        self.assertFalse(
            any(
                s["action"] == "wait"
                and s["params"]["requests"].get("$ref")
                == "stages.unfetched.output.requests"
                for s in nofetch["stages"]
            )
        )

    def test_expected_cleanup_does_not_add_a_prefill_member_condition(self):
        frame = owner_frame()
        frame["inflight"]["prefill_endpoints"][0]["inflight_requests"] = 3
        self.assertEqual(0, status.metric(frame, "cleanup_inflight"))
        self.assertEqual(3, status.metric(frame, "all_inflight"))
        self.assertEqual(3, status.metric(frame, "prefill_requests"))

    def test_debug_source_cannot_omit_a_required_owner(self):
        pages = {"scheduler": {}, "queues": {}}
        for role, endpoint, generation in [
            ("prefill", "p", "1"),
            ("decode", "d", "2"),
            ("engine", "p", "1"),
            ("engine", "d", "2"),
        ]:
            pages[f"{role}/{generation}"] = {
                "metadata": {"endpoint": endpoint, "endpoint_generation": generation}
            }
        capture = NS(
            payload={"endpointDirectoryTruncated": False, "components": pages},
            component=lambda name: pages[name],
        )
        status._debug_directory(capture, 1, 1)
        del pages["engine/2"]
        with self.assertRaisesRegex(RuntimeError, "incomplete"):
            status._debug_directory(capture, 1, 1)

    def test_fingerprint_baseline_stability_is_rejected(self):
        plan = PlanContext(
            "test",
            {"before": {"snapshot": "snapshot"}, "after": {"snapshot": "snapshot"}},
        )
        with self.assertRaises(ValueError):
            status.validate_check(
                {
                    "snapshot": {"$ref": "stages.after.output.snapshot"},
                    "baseline": {"$ref": "stages.before.output.snapshot"},
                    "metric": "fingerprint",
                    "aggregate": "stable",
                    "op": "eq",
                    "expected": True,
                },
                plan,
            )

    def test_metrics_readiness_records_fallback_and_pins_epoch(self):
        p = status.validate_metrics_ready({"duration_s": 1}, self.plan)
        with patch.object(
            status, "_http", side_effect=[(404, "not ready"), (200, "metric 1\n")]
        ):
            result = status.execute_metrics_ready(self.ctx, p, self.deadline())
        data = self.ctx.resource(result.output["snapshot"], "snapshot").to_dict()
        self.assertEqual([404, 200], [a["http_status"] for a in data["attempts"]])
        self.assertEqual((1, "prometheus"), self.ctx.status_metrics_source)
        self.ctx.env_epoch = 2
        with self.assertRaises(RuntimeError), patch.object(status, "_http") as http:
            status._frame(self.ctx, {"include": ["ttl"]}, self.deadline())
        http.assert_not_called()

    def test_stability_false_is_not_ignored_and_missing_owner_source_is_error(self):
        frames = [owner_frame(), owner_frame()]
        result = status._frozen(self.ctx, "test", {"frames": frames})
        args = {
            "snapshot": result.output["snapshot"],
            "metric": "scheduler",
            "aggregate": "stable",
            "op": "eq",
            "expected": False,
        }
        self.assertEqual(
            "FAIL",
            status.execute_check(self.ctx, args, self.deadline()).checks[0].status,
        )
        frames = [{}]
        result = status._frozen(self.ctx, "missing", {"frames": frames})
        args["snapshot"] = result.output["snapshot"]
        with self.assertRaises(KeyError):
            status.execute_check(self.ctx, args, self.deadline())

    def test_sampling_source_failure_is_not_an_empty_success(self):
        p = status.validate_sample({}, self.plan)
        with patch.object(status, "_http", side_effect=OSError("source missing")):
            with self.assertRaises(OSError):
                status.execute_sample(self.ctx, p, self.deadline())


if __name__ == "__main__":
    unittest.main()
