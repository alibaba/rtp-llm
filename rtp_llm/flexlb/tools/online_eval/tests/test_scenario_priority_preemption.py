"""Formal program with real Schedule and consumer workers; external IO is fake."""

import json
import tempfile
import unittest
from functools import partial
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch

from environment_expectations import environment as expected_environment
from flexlb_test_framework.scenario import compile_scenarios, load_scenarios
from flexlb_test_framework.scenario.actions import priority
from flexlb_test_framework.scenario.actions import priority_preemption as preempt
from flexlb_test_framework.scenario.catalog import handlers
from flexlb_test_framework.scenario.runtime import execute_instance
from test_scenario_backend import Ops
from test_scenario_priority import Clean

ROOT = Path(__file__).resolve().parents[1]


class Backend:
    def __init__(
        self,
        reverse=False,
        missing=False,
        rejected=None,
        queued=False,
        expiry=False,
        reason=0,
        terminals=None,
        comparator=False,
        invert_fifo=False,
        pure_priority=True,
    ):
        self.ops = Ops(batch=False)
        self.ops.master_http_port = 1
        self.reverse, self.missing = reverse, missing
        self.queued = queued
        self.comparator, self.invert_fifo = comparator, invert_fifo
        self.pure_priority = pure_priority
        self.environments = []
        original_future = self.ops.future

        def future(req, timeout, metadata=None):
            call = original_future(req, timeout, metadata)
            if req[0] == rejected:
                response = self.ops.responses[-1]
                response.code = 8400
                response.success = False
                response.error_message = "fixture yielded"
            if expiry and req[0] not in (1, 13):
                response = self.ops.responses[-1]
                response.code = 8511
                response.success = False
                response.admission_reject_reason = reason
            if terminals and req[0] in terminals:
                response = self.ops.responses[-1]
                response.code = terminals[req[0]]
                response.success = response.code == 200
            return call

        self.ops.future = future
        self.shapes = []
        original = self.ops.build_schedule_request

        def build(rid, **shape):
            self.shapes.append(shape)
            return original(rid, **shape)

        self.ops.build_schedule_request = build

    def setup(self, ctx, environment, deadline):
        self.environments.append(environment)
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
        if self.queued:
            order = (
                [1, 10, 4, 5, 2, 3, 6, 7, 8, 9, 11, 20, 12, 13, 14, 15, 16, 17, 18, 19]
                if self.pure_priority
                else [
                    1,
                    2,
                    10,
                    4,
                    5,
                    3,
                    6,
                    7,
                    8,
                    9,
                    11,
                    12,
                    20,
                    13,
                    14,
                    15,
                    16,
                    17,
                    18,
                    19,
                ]
            )
            if self.reverse:
                order[2], order[3] = order[3], order[2]
            lifecycle = {
                str(rid): dict(running_ms=i * 4000) for i, rid in enumerate(order)
            }
        if self.comparator:
            order = [1, 2, 4, 5, 6, 3, 7, 8, 9, 10, 11, 12]
            if self.pure_priority:
                order[:6] = [1, 4, 5, 6, 2, 3]
            if self.invert_fifo:
                order[8], order[9] = order[9], order[8]
            lifecycle = {
                str(rid): dict(running_ms=i * 3000) for i, rid in enumerate(order)
            }
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
    def plan(self, variant="same_priority_zero_eviction"):
        registry = handlers()
        registry.update({h.name: h for h in preempt.HANDLERS})
        plans = compile_scenarios(
            load_scenarios(ROOT / "scenarios/priority/priority_preemption.yaml"),
            handlers=registry,
        )
        self.assertEqual(14, len(plans))
        return next(p for p in plans if p["variant_id"] == variant), registry

    def run_program(self, variant="same_priority_zero_eviction", **kwargs):
        plan, registry = self.plan(variant)
        backend = Backend(**kwargs)
        with tempfile.TemporaryDirectory() as tmp, patch.object(
            priority, "_http", backend.http
        ), patch.object(preempt, "_http", backend.http), patch(
            "flexlb_test_framework.scenario.actions.engine_control._http", backend.http
        ), patch(
            "flexlb_test_framework.scenario.actions.balance.urllib.request.urlopen",
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

    def test_queued_two_rounds_preserve_twenty_consumers(self):
        result, cohorts, backend = self.run_program(
            variant="prefill_queued", queued=True
        )
        self.assertEqual("PASS", result["status"], result)
        self.assertEqual(20, backend.ops.generate_count)
        self.assertEqual([1, 1, 9, 9], sorted(map(len, cohorts)))
        self.assertTrue(
            all(r["consumer_completion_verified"] for wave in cohorts for r in wave)
        )
        self.assertTrue(all(c["status"] == "PASS" for c in result["cleanup"]))
        plan, _ = self.plan("prefill_queued")
        stages = {s["id"]: s for s in plan["stages"]}
        order = list(stages)
        self.assertLess(order.index("r1_master_clean"), order.index("r2_placeholder"))
        for n in (1, 2):
            self.assertEqual(30, stages[f"r{n}_master_clean"]["timeout_s"])
            self.assertEqual(90, stages[f"r{n}_wave_settled"]["timeout_s"])
            self.assertEqual(35, stages[f"r{n}_placeholder_drain"]["timeout_s"])
            self.assertEqual(315, stages[f"r{n}_wave_drain"]["timeout_s"])
            self.assertLess(
                order.index(f"r{n}_wave_settled"),
                order.index(f"r{n}_placeholder_drain"),
            )

    def test_queued_priority_inversion_blocks_second_round(self):
        result, _, backend = self.run_program(
            variant="prefill_queued", queued=True, reverse=True
        )
        self.assertEqual("FAIL", result["status"])
        self.assertEqual(10, backend.ops.generate_count)
        checks = next(s for s in result["stages"] if s["id"] == "r1_same_priority")[
            "checks"
        ]
        self.assertTrue(all(c["status"] == "FAIL" for c in checks))

    def test_legacy_first_peer_exemption_fails_strict_yaml_contract(self):
        result, _, _ = self.run_program(
            variant="prefill_queued", queued=True, pure_priority=False
        )
        self.assertEqual("FAIL", result["status"])
        first = next(s for s in result["stages"] if s["id"] == "r1_same_priority")
        self.assertEqual("FAIL", first["status"])

    def test_expectation_flag_preserves_legacy_and_rejects_non_booleans(self):
        self.assertTrue(preempt._first_peer_params({})["first_peer_exempt"])
        priorities = [30, 30, 70, 70, 70]
        self.assertEqual([0, 2, 3, 4, 1], preempt._priority_indices(priorities, True))
        self.assertEqual([2, 3, 4, 0, 1], preempt._priority_indices(priorities, False))
        for value in (0, 1, "false", None):
            with self.subTest(value=value), self.assertRaises(ValueError):
                preempt._first_peer_params({"first_peer_exempt": value})

    def test_queued_second_placeholder_rejection_blocks_second_wave(self):
        result, _, backend = self.run_program(
            variant="prefill_queued", queued=True, rejected=11
        )
        self.assertEqual("FAIL", result["status"])
        self.assertEqual(11, len(backend.shapes))
        self.assertEqual(10, backend.ops.generate_count)

    def test_expiry_preserves_two_successful_placeholders_and_twenty_rejections(self):
        result, cohorts, backend = self.run_program(
            variant="timeout_attribution", expiry=True
        )
        self.assertEqual("PASS", result["status"], result)
        self.assertEqual(22, len(backend.shapes))
        self.assertEqual(2, backend.ops.generate_count)
        self.assertEqual([1, 1, 9, 11], sorted(map(len, cohorts)))
        plan, _ = self.plan("timeout_attribution")
        self.assertEqual(
            7000, plan["environment"]["resolved_config"]["scheduler"]["queueTimeoutMs"]
        )
        stages = {s["id"]: s for s in plan["stages"]}
        self.assertEqual(12000, stages["slow"]["params"]["perf"]["prefill_fixed_ms"])
        self.assertEqual(
            10000, stages["round2_slow"]["params"]["perf"]["prefill_fixed_ms"]
        )
        self.assertEqual(385, stages["r1_wave_drain"]["timeout_s"])
        self.assertTrue(all(c["status"] == "PASS" for c in result["cleanup"]))

    def test_expiry_wrong_reason_fails_pr7_even_with_correct_code(self):
        result, _, backend = self.run_program(
            variant="timeout_attribution", expiry=True, reason=1
        )
        self.assertEqual("FAIL", result["status"])
        self.assertEqual(12, len(backend.shapes))
        checks = next(s for s in result["stages"] if s["id"] == "r1_same_priority")[
            "checks"
        ]
        self.assertEqual(
            {"PR7": "FAIL", "P6_terminal": "PASS"},
            {c["id"]: c["status"] for c in checks},
        )

    def test_expiry_missing_reason_is_error(self):
        result, _, backend = self.run_program(
            variant="timeout_attribution", expiry=True, reason=None
        )
        self.assertEqual("ERROR", result["status"])
        self.assertEqual(12, len(backend.shapes))
        self.assertTrue(all(c["status"] == "PASS" for c in result["cleanup"]))

    def test_comparator_rebuilds_q2_then_f1_and_drains_twelve_consumers(self):

        result, cohorts, backend = self.run_program(
            variant="comparator_frozen_weak", comparator=True
        )
        self.assertEqual("PASS", result["status"], result)
        self.assertEqual(12, backend.ops.generate_count)
        self.assertEqual([1, 1, 5, 5], sorted(map(len, cohorts)))
        self.assertEqual(2, len(backend.environments))
        for environment, factory in zip(
            backend.environments,
            (
                partial(expected_environment, "priority_comparator"),
                partial(expected_environment, "fifo_comparator"),
            ),
        ):
            spec = factory(NS(profile="single-nonbatch"))
            self.assertEqual(
                spec.resolved_config,
                environment["resolved_config"],
            )
        self.assertTrue(
            all(r["consumer_completion_verified"] for wave in cohorts for r in wave)
        )
        self.assertTrue(all(c["status"] == "PASS" for c in result["cleanup"]))
        plan, _ = self.plan("comparator_frozen_weak")
        stages = {s["id"]: s for s in plan["stages"]}
        order = list(stages)
        self.assertLess(order.index("r1_master_clean"), order.index("fifo_environment"))
        self.assertLess(order.index("fifo_environment"), order.index("fifo_slow"))
        self.assertEqual(175, stages["r1_wave_drain"]["timeout_s"])
        self.assertEqual(175, stages["r2_wave_drain"]["timeout_s"])

    def test_comparator_fifo_inversion_fails_second_half(self):
        result, _, backend = self.run_program(
            variant="comparator_frozen_weak", comparator=True, invert_fifo=True
        )
        self.assertEqual("FAIL", result["status"])
        self.assertEqual(12, backend.ops.generate_count)
        stages = {s["id"]: s for s in result["stages"]}
        self.assertEqual("PASS", stages["r1_same_priority"]["status"])
        self.assertEqual("PASS", stages["r2_same_priority"]["status"])
        self.assertEqual("FAIL", stages["comparator_verdict"]["status"])

    def test_comparator_rejected_first_placeholder_never_rebuilds_environment(self):
        result, _, backend = self.run_program(
            variant="comparator_frozen_weak", comparator=True, rejected=1
        )
        self.assertEqual("FAIL", result["status"])
        self.assertEqual(1, len(backend.environments))
        self.assertEqual(1, len(backend.shapes))

    def test_comparator_first_half_failure_still_executes_fifo_and_fails_final(self):
        result, cohorts, backend = self.run_program(
            variant="comparator_frozen_weak", comparator=True, pure_priority=False
        )
        self.assertEqual("FAIL", result["status"], result)
        self.assertEqual(12, backend.ops.generate_count)
        self.assertEqual(2, len(backend.environments))
        self.assertEqual([1, 1, 5, 5], sorted(map(len, cohorts)))
        stages = {s["id"]: s for s in result["stages"]}
        self.assertEqual("PASS", stages["fifo_environment"]["status"])
        self.assertEqual("PASS", stages["r2_master_clean"]["status"])
        checks = {c["id"]: c for c in stages["comparator_verdict"]["checks"]}
        self.assertEqual(
            {"priority_half": False, "fifo_half": True}, checks["PR9"]["actual"]
        )
        self.assertEqual("FAIL", checks["PR9"]["status"])
        self.assertEqual("FAIL", checks["P6"]["status"])
