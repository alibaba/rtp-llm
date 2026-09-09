"""Whole RPC fault programs must retain every old Master-owned ledger check."""

import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from flexlb_test_framework.scenario import compile_scenarios, load_scenarios
from flexlb_test_framework.scenario.actions import engine_fault, master
from flexlb_test_framework.scenario.actions.rpc_measurement import _owner_counts
from flexlb_test_framework.scenario.catalog import handlers
from flexlb_test_framework.scenario.runtime import execute_instance
from test_scenario_rpc_fault_probe import Backend as ProbeBackend
from test_scenario_rpc_fault_probe import Clock
from test_scenario_rpc_measurement import Backend as DelayBackend
from test_scenario_rpc_measurement import ScaledClock

ROOT = Path(__file__).resolve().parents[1]
CLEAN = dict(
    scheduler_inflight=0,
    prefill_endpoints=[dict(ip_port="P0", inflight_batches=0)],
    decode_endpoints=[dict(ip_port="D0", inflight_requests=0, total_load=0)],
)


class RpcOwnerCleanupTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.plans = [
            p
            for p in compile_scenarios(
                load_scenarios(ROOT / "scenarios/engine_fault"), handlers=handlers()
            )
            if p["scenario_id"] == "engine_rpc_fault" and p["profile"] == "batch-window"
        ]

    def run_program(self, plan, observation):
        clock = ScaledClock() if "delay" in plan["variant_id"] else Clock()
        backend = (
            DelayBackend() if "delay" in plan["variant_id"] else ProbeBackend(clock)
        )
        with tempfile.TemporaryDirectory() as root, patch.object(
            engine_fault, "_http", side_effect=backend.http
        ), patch.object(master, "_master_json", return_value=observation):
            result = execute_instance(
                plan,
                backend,
                artifact_dir=root,
                handlers=handlers(),
                clock=clock,
                sleeper=clock.sleep,
            )
            evidence = [
                json.loads(p.read_text())
                for p in Path(root).glob("rpc-fault-owner-*.json")
            ]
        return result, evidence

    def test_rpc_fault_ids_reject_endpoint_ownership_with_scheduler_zero(self):
        self.assertEqual(len(self.plans), 3)
        for plan in self.plans:
            stage = next(s for s in plan["stages"] if s["id"] == "owner_clean")
            self.assertEqual(
                stage["timeout_s"], 10 if "delay" in plan["variant_id"] else 95
            )
            for role in ("prefill", "decode"):
                raw = copy.deepcopy(CLEAN)
                if role == "prefill":
                    raw["prefill_endpoints"][0]["inflight_batches"] = 3
                else:
                    raw["decode_endpoints"][0]["total_load"] = 7
                with self.subTest(instance=plan["id"], role=role):
                    result, evidence = self.run_program(plan, raw)
                    self.assertEqual(result["status"], "TIMEOUT", result)
                    self.assertEqual(
                        next(s for s in result["stages"] if s["id"] == "owner_clean")[
                            "status"
                        ],
                        "TIMEOUT",
                    )
                    self.assertTrue(
                        all(c["status"] == "PASS" for c in result["cleanup"])
                    )
                    self.assertEqual(evidence[0]["samples"][-1]["raw"], raw)
                    self.assertFalse(evidence[0]["complete"])

    def test_actual_program_requires_endpoint_evidence_not_missing_zero(self):
        for plan in self.plans:
            with self.subTest(instance=plan["id"]):
                raw = copy.deepcopy(CLEAN)
                raw["decode_endpoints"][0] = {"ip_port": "D0"}
                result, evidence = self.run_program(plan, raw)
                self.assertEqual(result["status"], "ERROR", result)
                self.assertEqual(evidence[0]["samples"][-1]["raw"], raw)

    def test_clean_endpoint_schema_passes_each_old_program(self):
        for plan in self.plans:
            with self.subTest(instance=plan["id"]):
                result, evidence = self.run_program(plan, CLEAN)
                self.assertEqual(result["status"], "PASS", result)
                owner = next(s for s in result["stages"] if s["id"] == "owner_clean")
                self.assertEqual(owner["checks"][0]["id"], "all_owners_zero")
                self.assertEqual(
                    owner["checks"][0]["actual"],
                    dict(scheduler=0, prefill=[0], decode=[0]),
                )
                self.assertTrue(evidence[0]["complete"])

    def test_decode_or_and_strict_invalid_evidence(self):
        for row, expected in (
            (dict(inflight_requests=0, total_load=7), 7),
            (dict(total_load=2), 2),
            (dict(inflight_requests=3), 3),
        ):
            raw = copy.deepcopy(CLEAN)
            raw["decode_endpoints"] = [row]
            self.assertEqual(_owner_counts(raw)["decode"], [expected])
        for row in (
            {},
            dict(inflight_requests=0, total_load=None),
            dict(inflight_requests=False),
            dict(total_load=-1),
            dict(total_load=float("nan")),
            dict(total_load="0"),
        ):
            raw = copy.deepcopy(CLEAN)
            raw["decode_endpoints"] = [row]
            with self.subTest(row=row), self.assertRaises(ValueError):
                _owner_counts(raw)
        for key, value in (
            ("prefill_endpoints", []),
            ("decode_endpoints", None),
            ("prefill_endpoints", [{}]),
            ("scheduler_inflight", True),
        ):
            raw = dict(CLEAN, **{key: value})
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                _owner_counts(raw)


if __name__ == "__main__":
    unittest.main()
