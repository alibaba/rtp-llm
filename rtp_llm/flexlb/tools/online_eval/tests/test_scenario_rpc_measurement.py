import copy
import sys
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from flexlb_test_framework.scenario import compile_scenarios, load_scenarios
from flexlb_test_framework.scenario.actions import engine_fault, master
from flexlb_test_framework.scenario.actions.elastic import ClientRecords
from flexlb_test_framework.scenario.catalog import handlers
from flexlb_test_framework.scenario.observed import ObservedRequestBatch
from flexlb_test_framework.scenario.runtime import execute_instance
from test_scenario_backend import Ops, Stream


class ScaledClock:
    def __call__(self):
        return time.monotonic() * 100

    def sleep(self, value):
        time.sleep(value / 100)


class DelayOps(Ops):
    def __init__(self, backend, ctx, batch):
        super().__init__(batch)
        self.backend, self.ctx = backend, ctx

    def future(self, req, timeout, metadata=None):
        future = super().future(req, timeout, metadata)
        original = future.result

        def result(timeout):
            if self.backend.active and self.backend.fault_type == "enqueue_delay":
                self.ctx.sleeper(self.backend.injected_delta)
            response = original(timeout)
            if self.backend.active and self.backend.reject_during_fault:
                response.code, response.success = 500, False
                response.error_message = "injected rejected Schedule"
            return response

        future.result = result
        return future

    def _stream(self):
        backend, ctx = self.backend, self.ctx

        class DelayedStream(Stream):
            def __iter__(self):
                if backend.active and backend.fault_type == "generate_delay":
                    ctx.sleeper(backend.injected_delta)
                for output in super().__iter__():
                    output.flatten_output.finished = [backend.finished]
                    yield output

        stream = DelayedStream()
        self.streams.append(stream)
        return stream

    def fetch(self, req, timeout):
        self.fetch_count += 1
        return self._stream()

    def generate(self, req, timeout):
        self.generate_count += 1
        return self._stream()


class Backend:
    def __init__(
        self, injected_delta=1.5, corrupt=None, finished=True, reject_during_fault=False
    ):
        self.injected_delta, self.corrupt = injected_delta, corrupt
        self.active = set()
        self.fault_type = None
        self.finished, self.reject_during_fault = finished, reject_during_fault

    def setup(self, ctx, environment, deadline):
        self.ops = DelayOps(
            self, ctx, ctx.instance["effective_axes"]["dispatcher"] == "BATCH"
        )
        return object(), self.ops

    def teardown(self, ctx, deadline):
        if self.active:
            raise RuntimeError("fault cleanup did not run before environment cleanup")

    def http(self, ops, endpoint, deadline, body=None):
        if endpoint == "snapshot":
            return {
                "engines": [
                    dict(
                        name=name, role="prefill", grpc_addr="host:1234", stopped=False
                    )
                    for name in ("prefill-0", "prefill-1")
                ]
            }
        if body["enabled"]:
            self.active.add(body["engine"])
            self.fault_type = body["type"]
        else:
            self.active.discard(body["engine"])
        return dict(status="ok", engine=body["engine"], port=1234, type=body["type"])


class RpcMeasurementTests(unittest.TestCase):
    def plans(self):
        return [
            plan
            for plan in compile_scenarios(
                load_scenarios(ROOT / "scenarios/engine_fault"), handlers=handlers()
            )
            if "delay" in plan["variant_id"]
        ]

    def run_plan(self, plan, backend):
        clock = ScaledClock()
        original = ObservedRequestBatch.submit

        def submit(batch, deadline):
            original(batch, deadline)
            if backend.corrupt:
                backend.corrupt(batch.entries[0]["record"])

        with patch.object(
            ObservedRequestBatch, "submit", submit
        ), tempfile.TemporaryDirectory() as root, patch.object(
            engine_fault, "_http", side_effect=backend.http
        ), patch.object(
            master,
            "_master_json",
            return_value={
                "scheduler_inflight": 0,
                "prefill_endpoints": [{"inflight_batches": 0}],
                "decode_endpoints": [{"total_load": 0}],
            },
        ):
            return execute_instance(
                plan,
                backend,
                artifact_dir=root,
                handlers=handlers(),
                clock=clock,
                sleeper=clock.sleep,
            )

    def test_all_shipped_variants_execute_real_handlers_and_exact_checks(self):
        plans = self.plans()
        self.assertEqual(len(plans), 6)
        for plan in plans:
            with self.subTest(instance=plan["id"]):
                result = self.run_plan(plan, Backend())
                self.assertEqual(result["status"], "PASS", result["error"])
                checks = [
                    check for stage in result["stages"] for check in stage["checks"]
                ]
                self.assertEqual(
                    len(checks),
                    7 if plan["profile"] in ("batch-window", "single-batch") else 6,
                )

    def test_insufficient_delay_is_business_fail_not_execution_error(self):
        result = self.run_plan(self.plans()[0], Backend(injected_delta=0.5))
        self.assertEqual(result["status"], "FAIL", result)
        latency = next(row for row in result["stages"] if row["id"] == "latency")
        failed = [row["id"] for row in latency["checks"] if row["status"] == "FAIL"]
        self.assertEqual(failed, ["latency_increased"])
        self.assertTrue(all(row["status"] == "PASS" for row in result["cleanup"]))

    def test_missing_exit_or_timestamp_cannot_pass_latency_check(self):
        for corrupt in (
            lambda record: record.update(consumer_completion_verified=False),
            lambda record: record["stream"].update(first_output_s=None),
            lambda record: record.update(transport_terminal_s=float("nan")),
        ):
            plan = next(p for p in self.plans() if "generate_delay" in p["id"])
            result = self.run_plan(plan, Backend(corrupt=corrupt))
            # All terminal fields must be valid even when the selected metric
            # uses only a subset; corrupt terminal evidence is not a finding.
            self.assertEqual(result["status"], "ERROR", result)
            self.assertEqual(
                next(row for row in result["stages"] if row["id"] == "latency")[
                    "status"
                ],
                "ERROR",
            )

    def test_ttft_eof_without_finished_preserves_old_success_and_actual_verdict(self):
        plan = next(p for p in self.plans() if "generate_delay" in p["id"])
        result = self.run_plan(plan, Backend(finished=False))
        self.assertEqual(result["status"], "PASS", result["error"])
        latency = next(s for s in result["stages"] if s["id"] == "latency")
        for check in latency["checks"][:3]:
            self.assertIs(check["actual"], True)
        plan = next(p for p in self.plans() if "enqueue_delay" in p["id"])
        result = self.run_plan(plan, Backend(finished=False))
        self.assertEqual(result["status"], "FAIL")
        self.assertEqual(
            next(s for s in result["stages"] if s["id"] == "inject")["status"],
            "BLOCKED",
        )

    def test_delayed_schedule_rejection_remains_failure_and_still_recovers(self):
        for mode in ("generate_delay", "enqueue_delay"):
            plan = next(p for p in self.plans() if mode in p["id"])
            result = self.run_plan(plan, Backend(reject_during_fault=True))
            self.assertEqual(result["status"], "FAIL", result["error"])
            self.assertEqual(
                next(s for s in result["stages"] if s["id"] == "recovery")["status"],
                "PASS",
            )
            latency = next(s for s in result["stages"] if s["id"] == "latency")
            self.assertIs(
                next(c for c in latency["checks"] if c["id"] == "delayed_success")[
                    "actual"
                ],
                False,
            )

    def test_reused_cohort_is_rejected(self):
        plan = copy.deepcopy(self.plans()[0])
        stage = next(s for s in plan["stages"] if s["id"] == "latency")
        stage["params"]["recovery"] = stage["params"]["baseline"]
        self.assertEqual(self.run_plan(plan, Backend())["status"], "ERROR")


if __name__ == "__main__":
    unittest.main()
