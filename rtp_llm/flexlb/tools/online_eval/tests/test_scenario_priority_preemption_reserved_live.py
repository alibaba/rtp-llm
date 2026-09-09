"""Actual deferred Fetch workers for the two live Decode reservation profiles."""

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch

import test_scenario_priority_preemption as programs
from environment_expectations import environment as expected_environment
from flexlb_test_framework.scenario.actions import priority
from flexlb_test_framework.scenario.actions import priority_preemption as preempt
from flexlb_test_framework.scenario.backend import make_env_spec
from flexlb_test_framework.scenario.runtime import execute_instance
from test_scenario_priority import Clean
from test_scenario_priority_preemption_live import LiveBackend


class ReservedBackend(LiveBackend):
    def __init__(
        self, victim_code=8400, kv=428, saw_victim=False, physical_pressure=False
    ):
        super().__init__(victim_code=200)
        self.kv, self.saw_victim = kv, saw_victim
        self.pressure = 0
        self.physical_pressure = physical_pressure
        original = self.ops.future

        def future(req, timeout, metadata=None):
            call = original(req, timeout, metadata)
            if req[0] == 2:
                response = self.ops.responses[-1]
                response.code, response.success = victim_code, victim_code == 200
            return call

        self.ops.future = future

    def http(self, ops, endpoint, deadline, body=None):
        if endpoint == "set_kv_pressure":
            self.pressure = body["active_kv_tokens"]
            return dict(status="ok", engine=body["engine"])
        raw = super().http(ops, endpoint, deadline, body)
        if endpoint == "snapshot":
            raw["engines"] = raw["engines"][:2]
            raw["engines"][1].update(
                available_blocks=5,
                cache_blocks=5,
                total_kv_tokens=5120,
                held_blocks=0,
                referenced_blocks=0,
                available_kv_tokens=5120 - self.pressure,
            )
            if self.physical_pressure and self.pressure:
                raw["engines"][1]["available_blocks"] -= 1
            lifecycle = {str(i): dict(running_ms=i * 4000) for i in (1, 3)}
            if self.saw_victim:
                lifecycle["2"] = dict(running_ms=2000)
            raw["engines"][0]["request_lifecycle"] = lifecycle
        return raw

    def metrics(self, ctx, server, endpoint, deadline, **kwargs):
        labels = 'stage="decode_reserved",victim_priority="30"'
        kv_name = "flexlb_auto_tpm_victim_kv_tokens_seconds"
        return (
            200,
            f"flexlb_auto_tpm_victim_count_total{{{labels}}} 1\n"
            f"{kv_name}_sum{{{labels}}} {self.kv / 1000}\n"
            f"{kv_name}_count{{{labels}}} 1\n"
            f"{kv_name}_max{{{labels}}} {self.kv / 1000}\n"
            f'{kv_name}_bucket{{{labels},le="+Inf"}} 9999\n'
            f'{kv_name}{{{labels},quantile="0.5"}} 9999\n',
        )

    def setup(self, ctx, environment, deadline):
        env, ops = super().setup(ctx, environment, deadline)
        env.master_http_port = 12345
        return env, ops

    def debug(self, **kwargs):
        from test_debug_client import snapshot
        from flexlb_test_framework.debug_client import Capture

        raw = snapshot()
        page = raw["components"].pop("scheduler")
        page["rows"] = [
            dict(
                request_id=str(kwargs["request_id"]),
                queued=True,
                owns_request=True,
                ownership="reserved",
                has_dispatch_permit=False,
                has_protocol_owner=False,
            )
        ]
        raw["components"]["decode/1"] = page
        return Capture(raw, 1, 2)


class ReservedPrograms(unittest.TestCase):
    def run_program(self, variant="decode_reserved_live_single", **kwargs):
        plan, registry = programs.PreemptionPrograms().plan(variant)
        backend = ReservedBackend(**kwargs)
        with tempfile.TemporaryDirectory() as tmp, patch.object(
            preempt, "_http", backend.http
        ), patch.object(priority, "_http", backend.http), patch(
            "flexlb_test_framework.scenario.actions.engine_control._http", backend.http
        ), patch(
            "flexlb_test_framework.scenario.actions.balance._http", backend.http
        ), patch(
            "flexlb_test_framework.scenario.actions.balance.urllib.request.urlopen",
            return_value=Clean(),
        ), patch(
            "flexlb_test_framework.scenario.actions.status_protocol._http",
            backend.metrics,
        ):
            with patch(
                "flexlb_test_framework.debug_client.DebugClient.snapshot",
                side_effect=backend.debug,
            ):
                result = execute_instance(
                    plan, backend, handlers=registry, artifact_dir=tmp
                )
            observations = [
                json.loads(p.read_text())
                for p in Path(tmp).glob("preemption-live-reserved-*.json")
            ]
        checks = {c["id"]: c["status"] for s in result["stages"] for c in s["checks"]}
        return result, observations, checks, backend, plan

    def test_both_profiles_use_five_block_pool_and_master_local_layout(self):
        for variant in ("decode_reserved_live_single", "decode_reserved_live_window"):
            with self.subTest(variant=variant):
                result, obs, checks, backend, plan = self.run_program(variant)
                self.assertEqual("PASS", result["status"], result)
                old = expected_environment(
                    "reserved_preemption", NS(profile=plan["profile"])
                )
                env = backend.environments[0]
                actual = make_env_spec(env, plan["profile"], {"master_base": 28000})
                expected = old.resolved_config
                expected["scheduler"]["decision"] = dict(
                    type="FIXED_WINDOW",
                    maxRequests=32,
                    maxCollectionWaitMs=3000,
                    maxPredictedExecutionMs=550,
                )
                self.assertEqual(expected, env["resolved_config"])
                self.assertEqual(
                    (
                        old.n_prefill,
                        old.n_decode,
                        old.perf,
                        dict(old.master_env, FLEXLB_DEBUG_ENABLED="true"),
                        old.decode_cache_blocks,
                    ),
                    (
                        actual.n_prefill,
                        actual.n_decode,
                        actual.perf,
                        actual.master_env,
                        actual.decode_cache_blocks,
                    ),
                )
                self.assertEqual(5, actual.decode_cache_blocks)
                self.assertEqual(
                    [(90, 512, 2), (30, 1536, 2), (70, 3500, 2)],
                    [
                        (r["priority"], r["input_len"], r["output_len"])
                        for r in backend.shapes[:3]
                    ],
                )
                self.assertEqual(4, len(backend.shapes))
                self.assertEqual([1, 3, 4], [r["rid"] for r in backend.fetches])
                self.assertEqual(
                    [1, 3], [r["schedule_count"] for r in backend.fetches[:2]]
                )
                self.assertEqual(0, backend.ops.generate_count)
                self.assertTrue(
                    all(59 < r["timeout"] <= 60 for r in backend.fetches[:2])
                )
                rows = [obs[0]["placeholder"][0], obs[0]["wave"][1]]
                self.assertTrue(
                    all(
                        r["fetch_invocations"] == 1
                        and r["consumer_done"]
                        and r["consumer_completion_verified"]
                        and r["transport_terminal_s"] is not None
                        for r in rows
                    )
                )
                self.assertLessEqual(
                    rows[0]["consumer_exit_s"], rows[1]["stream"]["started_s"]
                )
                self.assertEqual(0, obs[0]["wave"][0].get("fetch_invocations", 0))
                self.assertEqual(428, obs[0]["victim_kv_total"])
                self.assertEqual(
                    ("PASS", "PASS", "PASS", "PASS"),
                    tuple(checks[k] for k in ("PR10", "PR5", "PR6", "P6")),
                )
                self.assertTrue(all(c["status"] == "PASS" for c in result["cleanup"]))

    def test_427_tokens_do_not_cover_original_428_deficit(self):
        result, _, checks, _, _ = self.run_program(kv=427)
        self.assertEqual("FAIL", result["status"], result)
        self.assertEqual(
            ("PASS", "PASS", "FAIL", "PASS"),
            tuple(checks[k] for k in ("PR10", "PR5", "PR6", "P6")),
        )

    def test_8429_cannot_stand_for_shadow_8400(self):
        result, _, checks, _, _ = self.run_program(victim_code=8429)
        self.assertEqual("FAIL", result["status"], result)
        self.assertEqual(
            ("FAIL", "FAIL", "FAIL", "PASS"),
            tuple(checks[k] for k in ("PR10", "PR5", "PR6", "P6")),
        )

    def test_engine_seen_victim_fails_before_issuing_incoming(self):
        result, observations, checks, backend, _ = self.run_program(saw_victim=True)
        self.assertEqual("ERROR", result["status"], result)
        wave = next(s for s in result["stages"] if s["id"] == "wave")
        self.assertIn("victim reached engine", wave["error"])
        self.assertEqual(2, len(backend.shapes))
        self.assertEqual([], observations)
        self.assertNotIn("PR10", checks)

    def test_pressure_that_changes_physical_pool_is_rejected_and_cleared(self):
        result, observations, _, backend, _ = self.run_program(physical_pressure=True)
        self.assertEqual("ERROR", result["status"])
        stage = next(s for s in result["stages"] if s["id"] == "pressure")
        self.assertIn("physical Decode capacity", stage["error"])
        self.assertEqual(1, len(backend.shapes))
        self.assertEqual(0, backend.pressure)
        self.assertEqual([], observations)
