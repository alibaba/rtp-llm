"""Run a complete cancellation YAML through real handlers and owned cleanup.

The fake RPC service drives ledger state independently of checks. This exercises
four unknown-ID profiles, not all 66 programs or real Java cancellation behavior.
"""

import copy
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch

from flexlb_test_framework.scenario.actions import cancel
from flexlb_test_framework.scenario.catalog import handlers
from flexlb_test_framework.scenario.compiler import compile_scenarios
from flexlb_test_framework.scenario.loader import load_scenarios
from flexlb_test_framework.scenario.runtime import execute_instance
from test_scenario_backend import Ops


class UnknownBackend:
    def __init__(self, mode):
        self.mode = mode
        self.calls = []
        self.scheduler = 0
        self.ops = Ops()
        self.ops.schedule_pb2 = NS(
            FlexlbCancelRequestPB=lambda **kw: NS(**kw),
            CANCEL_REASON_CLIENT_CANCELLED=1,
        )
        self.ops.schedule_pb2_grpc.FlexlbServiceStub = lambda channel: NS(
            Cancel=self.cancel
        )

    def setup(self, ctx, environment, deadline):
        self.calls.append("setup")
        return NS(), self.ops

    def cancel(self, request, timeout):
        self.calls.append(("master_cancel", request.request_id))
        if self.mode == "mutates_ledger":
            self.scheduler += 1
        return NS(found=self.mode == "wrong_found")

    def teardown(self, ctx, deadline):
        self.calls.append("teardown")

    def http(self, ctx, server, path, deadline, *args, **kwargs):
        if server == "mock":
            return 200, {"engines": [{"name": "prefill-0", "role": "prefill"}]}
        if self.mode == "missing_source":
            return 200, {}
        prefill = [
            {"ip_port": f"p{i}", "inflight_batches": 0, "inflight_requests": 0}
            for i in range(2)
        ]
        decode = []
        for i in range(4):
            row = {"ip_port": f"d{i}"}
            row.update(
                dict.fromkeys(
                    [
                        "reserved_total",
                        "master_queued",
                        "confirmed_accepted",
                        "confirmed_running",
                        "total_load",
                        "engine_load",
                        "active_dispatch_permits",
                        "engine_capacity_used",
                    ],
                    0,
                )
            )
            decode.append(row)
        return 200, copy.deepcopy(
            {
                "scheduler_inflight": self.scheduler,
                "prefill_endpoints": prefill,
                "decode_endpoints": decode,
            }
        )


class CancelProgramsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.registry = handlers()
        cls.registry.update({h.name: h for h in cancel.HANDLERS})
        root = Path(__file__).resolve().parents[1] / "scenarios/cancel"
        cls.plans = [
            p
            for p in compile_scenarios(load_scenarios(root), handlers=cls.registry)
            if p["variant_id"] == "unknown_rid"
        ]
        assert len(cls.plans) == 4

    def run_programs(self, mode, expected, failed_stage=None):
        for plan in self.plans:
            with self.subTest(profile=plan["profile"], mode=mode):
                backend = UnknownBackend(mode)
                with tempfile.TemporaryDirectory() as root, patch.object(
                    cancel.status, "_http", side_effect=backend.http
                ):
                    result = execute_instance(
                        plan,
                        backend,
                        handlers=self.registry,
                        artifact_dir=root,
                    )
                    self.assertTrue((Path(root) / "result.json").is_file())
                self.assertEqual(expected, result["status"], result)
                self.assertTrue(all(r["status"] == "PASS" for r in result["cleanup"]))
                self.assertEqual("setup", backend.calls[0])
                self.assertEqual("teardown", backend.calls[-1])
                self.assertEqual(0, backend.ops.fetch_count)
                self.assertEqual(0, backend.ops.generate_count)
                if mode != "missing_source":
                    self.assertEqual(
                        1, sum(isinstance(c, tuple) for c in backend.calls)
                    )
                if failed_stage:
                    failed = next(
                        s for s in result["stages"] if s["id"] == failed_stage
                    )
                    self.assertEqual(expected, failed["status"])

    def test_complete_unknown_id_programs_pass_without_schedule_stream_or_ledger_mutation(
        self,
    ):
        self.run_programs("correct", "PASS")

    def test_master_wrong_found_is_an_ordinary_contract_failure(self):
        self.run_programs("wrong_found", "FAIL", "unknown_is_typed_not_found")

    def test_unknown_cancel_mutating_ledger_is_not_hidden_by_successful_rpc(self):
        self.run_programs("mutates_ledger", "FAIL", "unknown_does_not_mutate_ledger")

    def test_missing_owner_source_is_error_and_teardown_still_runs(self):
        self.run_programs("missing_source", "ERROR", "clean_baseline")


if __name__ == "__main__":
    unittest.main()
