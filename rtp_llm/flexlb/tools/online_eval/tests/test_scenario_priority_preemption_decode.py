"""Two EV2 waves with real consumer workers and external control IO fixtures."""

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch

import test_scenario_priority_preemption as programs
from environment_expectations import environment as expected_environment
from flexlb_test_framework.scenario.actions import priority_preemption as preempt
from flexlb_test_framework.scenario.backend import make_env_spec
from flexlb_test_framework.scenario.runtime import execute_instance
from test_scenario_backend import lease_manifest
from test_scenario_priority import Clean


class DecodeBackend(programs.Backend):

    def __init__(self, incoming_success=False, guard=None, incoming_code=8403):
        super().__init__(
            terminals={5: 200 if incoming_success else incoming_code, 10: incoming_code}
        )
        self.pressure = {f"d{i}": 0 for i in range(4)}
        self.controls = []
        self.guard = guard

    def http(self, ops, endpoint, deadline, body=None):
        if endpoint == "set_kv_pressure":
            self.controls.append((body["engine"], body["active_kv_tokens"]))
            self.pressure[body["engine"]] = body["active_kv_tokens"]
            return dict(status="ok", engine=body["engine"])
        if endpoint == "set_perf":
            return dict(status="ok", engine=body["engine"], port=1234)
        if endpoint != "snapshot":
            raise AssertionError(endpoint)
        rows = []
        for role, count in (("prefill", 2), ("decode", 4)):
            for i in range(count):
                name = f"{role[0]}{i}"
                row = dict(
                    name=name,
                    role=role,
                    stopped=False,
                    grpc_addr=f"127.0.0.1:{1234 + len(rows)}",
                    request_lifecycle={},
                )
                if role == "decode":
                    row["available_kv_tokens"] = 6291456 - self.pressure[name]
                    row["request_lifecycle"] = {
                        str(i + 6): dict(end_state="running", running_ms=1000)
                    }
                    if self.pressure[name] and name == "d0":
                        if self.guard == "missing":
                            del row["available_kv_tokens"]
                        elif self.guard == "available":
                            row["available_kv_tokens"] = 1
                rows.append(row)
        return dict(engines=rows)


class DecodePrograms(unittest.TestCase):
    def run_program(self, **kwargs):
        plan, registry = programs.PreemptionPrograms().plan("decode_engine_owned")
        backend = DecodeBackend(**kwargs)
        with tempfile.TemporaryDirectory() as tmp, patch.object(
            preempt, "_http", backend.http
        ), patch(
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
            running = [
                json.loads(p.read_text())
                for p in Path(tmp).glob("preemption-decode-running-*.json")
            ]
        return result, cohorts, running, backend

    def test_both_ev2_waves_drain_eight_consumers_and_two_rejections(self):
        result, cohorts, running, backend = self.run_program()
        self.assertEqual("PASS", result["status"], result)
        self.assertEqual(10, len(backend.shapes))
        self.assertEqual(8, backend.ops.generate_count)
        self.assertEqual([1, 1, 4, 4], sorted(map(len, cohorts)))
        admitted = [
            r for wave in cohorts for r in wave if r["schedule"]["status"] == "OK"
        ]
        self.assertEqual(8, len(admitted))
        self.assertTrue(all(r["consumer_completion_verified"] for r in admitted))
        self.assertEqual(["6", "7", "8", "9"], [r["request_id"] for r in running[0]])
        self.assertEqual({f"d{i}": 0 for i in range(4)}, backend.pressure)
        self.assertEqual(8, sum(tokens > 0 for name, tokens in backend.controls))
        self.assertTrue(all(c["status"] == "PASS" for c in result["cleanup"]))
        spec = expected_environment("decode_reservation", NS(profile="single-nonbatch"))
        plan, _ = programs.PreemptionPrograms().plan("decode_engine_owned")
        self.assertEqual(
            spec.resolved_config,
            plan["environment"]["resolved_config"],
        )
        self.assertEqual(
            (2, 4), (plan["environment"]["n_prefill"], plan["environment"]["n_decode"])
        )
        rebuilt = make_env_spec(
            plan["environment"], "single-nonbatch", lease_manifest()
        )
        self.assertEqual(spec.master_env, rebuilt.master_env)
        order = [s["id"] for s in plan["stages"]]
        self.assertLess(order.index("r1_settled"), order.index("r1_pressure"))
        self.assertLess(order.index("r2_running"), order.index("r2_pressure"))
        self.assertLess(order.index("r1_master_clean"), order.index("release_pressure"))
        self.assertLess(order.index("r2_guard"), order.index("r2_incoming"))
        stages = {s["id"]: s for s in plan["stages"]}
        self.assertEqual(80, stages["r2_running"]["timeout_s"])
        self.assertEqual(140, stages["r1_occupants_drain"]["timeout_s"])
        self.assertEqual(35, stages["r2_incoming_drain"]["timeout_s"])

    def test_deadline_rejection_preserves_ev2_contract(self):
        result, _, _, _ = self.run_program(incoming_code=8511)
        self.assertEqual("PASS", result["status"], result)

    def test_successful_incoming_fails_reserved_ev2_contract(self):
        result, _, _, backend = self.run_program(incoming_success=True)
        self.assertEqual("FAIL", result["status"], result)
        self.assertEqual(5, len(backend.shapes))
        self.assertEqual(5, backend.ops.generate_count)
        self.assertTrue(all(v == 0 for v in backend.pressure.values()))

    def test_one_available_decode_fails_before_incoming(self):
        result, _, _, backend = self.run_program(guard="available")
        self.assertEqual("FAIL", result["status"], result)
        self.assertEqual(4, len(backend.shapes))
        self.assertTrue(all(v == 0 for v in backend.pressure.values()))

    def test_missing_decode_capacity_is_error_not_expected_negative_sentinel(self):
        result, _, _, backend = self.run_program(guard="missing")
        self.assertEqual("ERROR", result["status"], result)
        self.assertEqual(4, len(backend.shapes))
        self.assertTrue(all(v == 0 for v in backend.pressure.values()))

    def test_decode_running_uses_fresh_twenty_seconds_per_request(self):
        from flexlb_test_framework.scenario.runtime import Deadline

        clock = [0.0]

        def sleep(seconds):
            clock[0] += seconds

        wave = preempt.PriorityWave.__new__(preempt.PriorityWave)
        wave.entries = [None] * 4
        wave.records = lambda: [dict(wire_request_id=i) for i in range(1, 5)]

        def snapshot(ops, endpoint, deadline):
            return dict(
                engines=[
                    dict(
                        role="decode",
                        request_lifecycle={
                            str(i): dict(end_state="running")
                            for i in range(1, 5)
                            if clock[0] >= i * 12
                        },
                    )
                ]
            )

        with tempfile.TemporaryDirectory() as tmp, patch.object(
            preempt, "_http", snapshot
        ):
            ctx = NS(
                resource=lambda handle, kind: wave,
                artifact_dir=Path(tmp),
                clock=lambda: clock[0],
                sleeper=sleep,
                ops=NS(),
            )
            result = preempt._decode_running(
                ctx, {"requests": {}}, Deadline(80, ctx.clock, sleep)
            )
        self.assertEqual("PASS", result.checks[0].status)
        self.assertGreaterEqual(clock[0], 48)
        self.assertLess(clock[0], 48.2)

    def test_decode_running_never_outlives_parent_deadline(self):
        from flexlb_test_framework.scenario.runtime import Deadline, StageTimeout

        clock = [0.0]

        def sleep(seconds):
            clock[0] += seconds

        wave = preempt.PriorityWave.__new__(preempt.PriorityWave)
        wave.entries = [None] * 4
        wave.records = lambda: [dict(wire_request_id=i) for i in range(1, 5)]

        def snapshot(ops, endpoint, deadline):
            return dict(
                engines=[
                    dict(
                        role="decode",
                        request_lifecycle={
                            str(i): dict(end_state="running")
                            for i in range(1, 5)
                            if clock[0] >= i * 12
                        },
                    )
                ]
            )

        with tempfile.TemporaryDirectory() as tmp, patch.object(
            preempt, "_http", snapshot
        ):
            ctx = NS(
                resource=lambda handle, kind: wave,
                artifact_dir=Path(tmp),
                clock=lambda: clock[0],
                sleeper=sleep,
                ops=NS(),
            )
            with self.assertRaises(StageTimeout):
                preempt._decode_running(
                    ctx, {"requests": {}}, Deadline(25, ctx.clock, sleep)
                )
        self.assertAlmostEqual(25, clock[0])

    def test_decode_running_rejects_malformed_timestamp_evidence(self):
        from flexlb_test_framework.scenario.runtime import Deadline

        wave = preempt.PriorityWave.__new__(preempt.PriorityWave)
        wave.entries = [None] * 4
        wave.records = lambda: [dict(wire_request_id=i) for i in range(1, 5)]
        snapshot = dict(
            engines=[
                dict(
                    role="decode",
                    request_lifecycle={"1": dict(running_ms="not-a-timestamp")},
                )
            ]
        )
        with tempfile.TemporaryDirectory() as tmp, patch.object(
            preempt, "_http", return_value=snapshot
        ):
            ctx = NS(
                resource=lambda handle, kind: wave,
                artifact_dir=Path(tmp),
                clock=lambda: 0.0,
                sleeper=lambda seconds: None,
                ops=NS(),
            )
            with self.assertRaises(ValueError):
                preempt._decode_running(
                    ctx, {"requests": {}}, Deadline(80, ctx.clock, ctx.sleeper)
                )
