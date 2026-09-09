import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from flexlb_test_framework.scenario import compile_scenarios, load_scenarios
from flexlb_test_framework.scenario.actions import engine_fault, master
from flexlb_test_framework.scenario.backend import JavaMockBackend
from flexlb_test_framework.scenario.catalog import handlers
from flexlb_test_framework.scenario.runtime import execute_instance
from test_scenario_backend import Ops, Stream


class Clock:
    now = 0

    def __call__(self):
        return self.now

    def sleep(self, value):
        self.now += value


class RpcError(Exception):
    def __init__(self, status):
        self.status = status

    def code(self):
        return NS(name=self.status)


class FaultOps(Ops):
    def __init__(self, backend, batch):
        super().__init__(batch)
        self.backend = backend
        self.schedule_pb2 = NS(
            FlexlbCancelRequestPB=lambda **kw: NS(**kw),
            CANCEL_REASON_CLIENT_CANCELLED=1,
        )
        self.pb2.CancelRequestPB = lambda **kw: NS(**kw)
        self.schedule_pb2_grpc = NS(
            FlexlbServiceStub=lambda channel: NS(
                Schedule=NS(future=self.future),
                Cancel=lambda req, timeout: "cancel transport ok",
            )
        )
        self.pb2_grpc = NS(
            RpcServiceStub=lambda channel: NS(
                FetchResponse=self.fetch,
                GenerateStreamCall=self.generate,
                Cancel=lambda req, timeout: "cancel transport ok",
            )
        )

    def future(self, req, timeout, metadata=None):
        result = super().future(req, timeout, metadata)
        self.responses[-1].HasField = lambda field: False
        return result

    def fault_stream(self):
        fault = next(iter(self.backend.active.values()), None)
        if fault and not self.backend.ignore_fault:
            if self.backend.expire_stage:
                self.backend.clock.now += 1000
            return Stream(
                error=RpcError(
                    self.backend.status_override
                    or ("DEADLINE_EXCEEDED" if fault == "no_respond" else "UNKNOWN")
                )
            )
        return Stream()

    def fetch(self, req, timeout):
        self.fetch_count += 1
        return self.fault_stream()

    def generate(self, req, timeout):
        self.generate_count += 1
        return self.fault_stream()


class Backend(JavaMockBackend):
    def __init__(
        self, clock, ignore_fault=False, status_override=None, expire_stage=False
    ):
        super().__init__({})
        self.active, self.clock = {}, clock
        self.ignore_fault, self.status_override, self.expire_stage = (
            ignore_fault,
            status_override,
            expire_stage,
        )

    def setup(self, ctx, environment, deadline):
        self.ops = FaultOps(
            self, ctx.instance["effective_axes"]["dispatcher"] == "BATCH"
        )
        return object(), self.ops

    def teardown(self, ctx, deadline):
        if self.active:
            raise RuntimeError("owned fault was not cleared")

    def http(self, ops, endpoint, deadline, body=None):
        if endpoint == "snapshot":
            return {
                "engines": [
                    dict(
                        name=f"prefill-{i}",
                        role="prefill",
                        grpc_addr=f"host:{100+i}",
                        stopped=False,
                        inject_config=dict(no_respond=False, enqueue_error=False),
                    )
                    for i in range(2)
                ]
            }
        name = body["engine"]
        if body["enabled"]:
            self.active[name] = body["type"]
        else:
            self.active.pop(name, None)
        return dict(
            status="ok", engine=name, port=100 + int(name[-1]), type=body["type"]
        )


class FaultProbeTests(unittest.TestCase):
    def plans(self):
        return [
            p
            for p in compile_scenarios(
                load_scenarios(ROOT / "scenarios/engine_fault"), handlers=handlers()
            )
            if p["variant_id"] == "enqueue_error"
        ]

    def run_plan(self, plan, **kwargs):
        clock = Clock()
        backend = Backend(clock, **kwargs)
        with tempfile.TemporaryDirectory() as root, patch.object(
            engine_fault, "_http", side_effect=backend.http
        ), patch.object(
            master,
            "_master_json",
            return_value={
                "scheduler_inflight": 0,
                "prefill_endpoints": [{"inflight_batches": 0}],
                "decode_endpoints": [{"total_load": 0}],
            },
        ) as owner:
            result = execute_instance(
                plan,
                backend,
                handlers=handlers(),
                artifact_dir=root,
                clock=clock,
                sleeper=clock.sleep,
            )
        return result, backend, owner.call_count

    def test_all_error_programs_execute_actual_request_consumers_and_owner_condition(
        self,
    ):
        plans = self.plans()
        self.assertEqual(len(plans), 4)
        for plan in plans:
            result, backend, owner_reads = self.run_plan(plan)
            self.assertEqual(result["status"], "PASS", result["error"])
            self.assertEqual(
                owner_reads, int(plan["effective_axes"]["dispatcher"] == "BATCH")
            )
            probe = next(row for row in result["stages"] if row["id"] == "probe")
            record = probe["checks"][0]["evidence"]["record"]
            self.assertTrue(record["consumer_done"])
            self.assertTrue(record["consumer_completion_verified"])
            self.assertFalse(record["business_finished"])
            self.assertEqual(backend.ops.counter, 2)
            self.assertTrue(all(row["status"] == "PASS" for row in result["cleanup"]))

    def test_ack_without_fault_effect_fails_and_cleanup_still_clears(self):
        result, backend, _ = self.run_plan(self.plans()[0], ignore_fault=True)
        self.assertEqual(result["status"], "FAIL", result["error"])
        self.assertEqual(backend.ops.counter, 1)
        self.assertEqual(backend.active, {})

    def test_unlisted_transport_status_is_error(self):
        result, _, _ = self.run_plan(self.plans()[0], status_override="DATA_LOSS")
        self.assertEqual(result["status"], "ERROR")
        self.assertIn("unexpected or unavailable", result["error"])

    def test_stage_deadline_never_becomes_expected_request_timeout(self):
        result, _, _ = self.run_plan(self.plans()[0], expire_stage=True)
        self.assertEqual(result["status"], "TIMEOUT", result["error"])
        self.assertEqual(
            next(row for row in result["stages"] if row["id"] == "probe")["status"],
            "TIMEOUT",
        )


if __name__ == "__main__":
    unittest.main()
