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
    def __init__(self, victim_code=8400, kv=428, saw_victim=False):
        super().__init__(victim_code=200)
        self.kv, self.saw_victim = kv, saw_victim
        original = self.ops.future

        def future(req, timeout, metadata=None):
            call = original(req, timeout, metadata)
            if req[0] == 2:
                response = self.ops.responses[-1]
                response.code, response.success = victim_code, victim_code == 200
            return call

        self.ops.future = future

    def http(self, ops, endpoint, deadline, body=None):
        raw = super().http(ops, endpoint, deadline, body)
        if endpoint == "snapshot":
            raw["engines"] = raw["engines"][:2]
            lifecycle = {str(i): dict(running_ms=i * 4000) for i in (1, 3)}
            if self.saw_victim:
                lifecycle["2"] = dict(running_ms=2000)
            raw["engines"][0]["request_lifecycle"] = lifecycle
        return raw

    def metrics(self, ctx, server, endpoint, deadline, **kwargs):
        return (
            200,
            f'flexlb_auto_tpm_victim_count{{stage="decode_reserved"}} 1\nflexlb_auto_tpm_victim_kv_tokens{{stage="decode_reserved"}} {self.kv}\n',
        )


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
            result = execute_instance(
                plan, backend, handlers=registry, artifact_dir=tmp
            )
            observations = [
                json.loads(p.read_text())
                for p in Path(tmp).glob("preemption-live-reserved-*.json")
            ]
        checks = {c["id"]: c["status"] for s in result["stages"] for c in s["checks"]}
        return result, observations, checks, backend, plan

    def test_both_profiles_use_five_block_pool_preserve_shapes_and_real_fetch(self):
        for variant in ("decode_reserved_live_single", "decode_reserved_live_window"):
            with self.subTest(variant=variant):
                result, obs, checks, backend, plan = self.run_program(variant)
                self.assertEqual("PASS", result["status"], result)
                old = expected_environment(
                    "reserved_preemption", NS(profile=plan["profile"])
                )
                env = backend.environments[0]
                actual = make_env_spec(env, plan["profile"], {"master_base": 28000})
                self.assertEqual(
                    old.resolved_config,
                    env["resolved_config"],
                )
                self.assertEqual(
                    (
                        old.n_prefill,
                        old.n_decode,
                        old.perf,
                        old.master_env,
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
                    [(90, 512, 2), (30, 512, 2), (70, 3500, 2)],
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

    def test_engine_seen_victim_fails_master_local_proof(self):
        result, _, checks, _, _ = self.run_program(saw_victim=True)
        self.assertEqual("FAIL", result["status"], result)
        self.assertEqual(
            ("PASS", "FAIL", "PASS", "PASS"),
            tuple(checks[k] for k in ("PR10", "PR5", "PR6", "P6")),
        )
