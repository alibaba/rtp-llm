"""Formal program with real Schedule and consumer workers; external IO is fake."""

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch

from flexlb_ft.scenario import compile_scenarios, load_scenarios
from flexlb_ft.scenario.actions import priority
from flexlb_ft.scenario.actions import priority_preemption as preempt
from flexlb_ft.scenario.catalog import handlers
from flexlb_ft.scenario.runtime import execute_instance
from test_scenario_backend import Ops
from test_scenario_priority import Clean

ROOT = Path(__file__).resolve().parents[1]


class Backend:
    def __init__(self, reverse=False, missing=False, rejected=None):
        self.ops = Ops(batch=False)
        self.ops.master_http_port = 1
        self.reverse, self.missing = reverse, missing
        original_future = self.ops.future

        def future(req, timeout, metadata=None):
            call = original_future(req, timeout, metadata)
            if req[0] == rejected:
                response = self.ops.responses[-1]
                response.code = 8400
                response.success = False
                response.error_message = "fixture yielded"
            return call

        self.ops.future = future
        self.shapes = []
        original = self.ops.build_schedule_request

        def build(rid, **shape):
            self.shapes.append(shape)
            return original(rid, **shape)

        self.ops.build_schedule_request = build

    def setup(self, ctx, environment, deadline):
        return NS(), self.ops

    def teardown(self, ctx, deadline):
        pass

    def http(self, ops, endpoint, deadline, body=None):
        if endpoint == "set_perf":
            return dict(status="ok", engine=body["engine"], port=1234)
        if endpoint != "snapshot":
            raise AssertionError(endpoint)
        lifecycle = {
            str(i): dict(running_ms=(12 - i if self.reverse and i > 1 else i) * 4000)
            for i in range(1, 11)
        }
        if self.missing:
            lifecycle.pop("6")
        return {
            "engines": [
                dict(
                    name="p0",
                    role="prefill",
                    grpc_addr="prefill",
                    port=1234,
                    stopped=False,
                    request_lifecycle=lifecycle,
                    running=1,
                    waiting=0,
                )
            ]
        }


class PreemptionPrograms(unittest.TestCase):
    def plan(self):
        registry = handlers()
        registry.update({h.name: h for h in preempt.HANDLERS})
        plans = compile_scenarios(
            load_scenarios(ROOT / "scenarios/priority/priority_preemption.yaml"),
            handlers=registry,
        )
        self.assertEqual(1, len(plans))
        return plans[0], registry

    def run_program(self, **kwargs):
        plan, registry = self.plan()
        backend = Backend(**kwargs)
        with tempfile.TemporaryDirectory() as tmp, patch.object(
            priority, "_http", backend.http
        ), patch.object(preempt, "_http", backend.http), patch(
            "flexlb_ft.scenario.actions.engine_control._http", backend.http
        ), patch(
            "flexlb_ft.scenario.actions.balance.urllib.request.urlopen",
            return_value=Clean(),
        ):
            result = execute_instance(
                plan, backend, handlers=registry, artifact_dir=tmp
            )
            cohorts = [
                json.loads(p.read_text())
                for p in Path(tmp).glob("priority-wave-*.json")
            ]
        return result, cohorts, backend

    def test_complete_same_priority_program_uses_ten_real_consumers(self):
        result, cohorts, backend = self.run_program()
        self.assertEqual("PASS", result["status"], result)
        self.assertEqual(10, backend.ops.generate_count)
        self.assertEqual(0, backend.ops.fetch_count)
        self.assertEqual([1, 9], sorted(map(len, cohorts)))
        self.assertTrue(
            all(
                r["consumer_done"] and r["consumer_completion_verified"]
                for rows in cohorts
                for r in rows
            )
        )
        self.assertTrue(all(c["status"] == "PASS" for c in result["cleanup"]))
        self.assertEqual(10, len(backend.shapes))
        self.assertTrue(
            all(
                s["priority"] == 50 and s["input_len"] == 2048 and s["output_len"] == 2
                for s in backend.shapes
            )
        )

    def test_inverted_peer_dispatch_fails_pr4_and_p6(self):
        result, _, _ = self.run_program(reverse=True)
        checks = {
            c["id"]: c["status"]
            for s in result["stages"]
            if s["id"] == "same_priority"
            for c in s["checks"]
        }
        self.assertEqual("FAIL", result["status"])
        self.assertEqual({"PR4": "FAIL", "AT3": "PASS", "P6_terminal": "FAIL"}, checks)

    def test_missing_peer_lifecycle_is_error(self):
        result, _, _ = self.run_program(missing=True)
        self.assertEqual("ERROR", result["status"])
        self.assertTrue(all(c["status"] == "PASS" for c in result["cleanup"]))

    def test_compiled_owner_config_and_drain_order(self):
        plan, _ = self.plan()
        stages = {s["id"]: s for s in plan["stages"]}
        order = list(stages)
        self.assertEqual("NON_BATCH", plan["effective_axes"]["dispatcher"])
        self.assertEqual("SINGLE", plan["effective_axes"]["decision"])
        self.assertEqual("PRIORITY", plan["effective_axes"]["ordering"])
        cfg = plan["environment"]["resolved_config"]
        self.assertEqual(60000, cfg["scheduler"]["queueTimeoutMs"])
        self.assertEqual(
            ["PREFILL_QUEUED"],
            cfg["scheduler"]["ordering"]["preemption"]["allowedVictimStages"],
        )
        self.assertEqual(0, stages["placeholder"]["params"]["gap_s"])
        self.assertEqual(0.15, stages["wave"]["params"]["gap_s"])
        self.assertEqual(90, stages["wave_settled"]["timeout_s"])
        self.assertEqual(35, stages["placeholder_drain"]["timeout_s"])
        self.assertEqual(315, stages["wave_drain"]["timeout_s"])
        self.assertLess(order.index("wave_settled"), order.index("placeholder_drain"))
        self.assertLess(order.index("placeholder_drain"), order.index("wave_drain"))
        self.assertEqual(30, stages["master_clean"]["timeout_s"])

    def test_incoming_eviction_is_not_zero_eviction(self):
        result, _, backend = self.run_program(rejected=10)
        checks = {
            c["id"]: c
            for s in result["stages"]
            if s["id"] == "same_priority"
            for c in s["checks"]
        }
        self.assertEqual("FAIL", result["status"])
        self.assertFalse(checks["PR4"]["actual"]["zero_eviction"])
        self.assertEqual(8400, checks["AT3"]["actual"])
        self.assertEqual(9, backend.ops.generate_count)

    def test_rejected_placeholder_blocks_wave(self):
        result, _, backend = self.run_program(rejected=1)
        self.assertEqual("FAIL", result["status"])
        self.assertEqual(1, len(backend.shapes))
        self.assertEqual(
            "FAIL",
            next(s for s in result["stages"] if s["id"] == "placeholder_admitted")[
                "status"
            ],
        )
