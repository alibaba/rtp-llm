"""NOT_FOUND choreography, real consumers, and frozen-status cleanup."""

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


class NotFoundBackend(programs.Backend):
    def __init__(
        self,
        incoming_code=8431,
        cancel_count=1,
        cancelled=False,
        never_finishes=False,
        missing_census=False,
    ):
        super().__init__(terminals={2: incoming_code})
        self.cancel_count, self.cancelled = cancel_count, cancelled
        self.never_finishes, self.missing_census = never_finishes, missing_census
        self.injected = False
        self.ever_injected = False
        self.injections = []

    def http(self, ops, endpoint, deadline, body=None):
        if endpoint != "snapshot":
            raise AssertionError(endpoint)
        end = (
            "finished" if self.ever_injected and not self.never_finishes else "running"
        )
        arrived = len(self.ops.responses) >= 2
        engines = [
            dict(
                name=f"{role[0]}0",
                role=role,
                stopped=False,
                grpc_addr=f"{role}0",
                inflight=0,
                leak_detected=False,
                request_lifecycle={"1": dict(running_ms=100, end_state=end)},
                cancelled_rids=(
                    [1] if arrived and self.cancelled and role == "decode" else []
                ),
                rpc_counts=(
                    {"cancel": self.cancel_count}
                    if arrived and role == "prefill"
                    else {}
                ),
            )
            for role in ("prefill", "decode")
        ]
        if self.missing_census:
            engines[0].pop("rpc_counts")
        return dict(engines=engines)

    def control(
        self, ctx, server, path, deadline, body=None, allowed=(200,), text=False
    ):
        if server != "mock":
            raise AssertionError(server)
        if path == "snapshot":
            return 200, self.http(ctx.ops, path, deadline)
        if path != "inject":
            raise AssertionError(path)
        self.injections.append(dict(body))
        if body["enabled"]:
            self.ever_injected = True
        self.injected = body["enabled"]
        return 200, dict(status="ok")


class NotFoundPrograms(unittest.TestCase):
    def run_program(self, **kwargs):
        plan, registry = programs.PreemptionPrograms().plan("cancel_not_found")
        backend = NotFoundBackend(**kwargs)
        with tempfile.TemporaryDirectory() as tmp, patch.object(
            preempt, "_http", backend.http
        ), patch.object(priority, "_http", backend.http), patch(
            "flexlb_test_framework.scenario.actions.balance.urllib.request.urlopen",
            return_value=Clean(),
        ), patch(
            "flexlb_test_framework.scenario.actions.status_protocol._http",
            backend.control,
        ):
            result = execute_instance(
                plan, backend, handlers=registry, artifact_dir=tmp
            )
            observations = [
                json.loads(p.read_text())
                for p in Path(tmp).glob("preemption-nf-verdict-*.json")
            ]
        checks = {c["id"]: c["status"] for s in result["stages"] for c in s["checks"]}
        return result, observations, checks, backend, plan

    def test_exact_nf_config_shapes_and_independent_outcome_planes(self):
        result, obs, checks, backend, plan = self.run_program()
        self.assertEqual("PASS", result["status"], result)
        old = expected_environment("cancel_not_found", NS(profile="single-nonbatch"))
        env = backend.environments[0]
        actual = make_env_spec(env, plan["profile"], {"master_base": 28000})
        self.assertEqual(
            old.resolved_config,
            env["resolved_config"],
        )
        self.assertEqual(
            (old.n_prefill, old.n_decode, old.perf, old.master_env),
            (actual.n_prefill, actual.n_decode, actual.perf, actual.master_env),
        )
        self.assertEqual(
            [(30, 512, 200), (70, 512, 2)],
            [
                (r["priority"], r["input_len"], r["output_len"])
                for r in backend.shapes[:2]
            ],
        )
        self.assertEqual(3, len(backend.shapes))
        self.assertNotIn("priority", backend.shapes[2])
        self.assertEqual([301], backend.shapes[2]["block_keys"])
        self.assertEqual(2, backend.ops.generate_count)
        row = obs[0]["victim"][0]
        self.assertTrue(
            row["consumer_done"]
            and row["consumer_completion_verified"]
            and row["transport_terminal_s"] is not None
            and row["consumer_exit_s"] is not None
        )
        self.assertEqual(8431, obs[0]["incoming_code"])
        self.assertEqual(1, obs[0]["cancel_delta"])
        self.assertEqual(["d0", "p0"], sorted(obs[0]["before"]["missing_cancel_keys"]))
        self.assertEqual([], obs[0]["cancelled_by"])
        self.assertEqual(
            ("PASS", "PASS", "PASS"), tuple(checks[k] for k in ("PR6", "AT5", "P6"))
        )
        self.assertTrue(backend.injections[0]["enabled"])
        self.assertTrue(
            all(
                r["engine"] == "d0" and r["type"] == "status_no_respond"
                for r in backend.injections
            )
        )
        self.assertFalse(backend.injected)
        self.assertTrue(all(c["status"] == "PASS" for c in result["cleanup"]))

    def test_8403_construction_miss_is_not_relaxed_to_8431(self):
        result, _, checks, _, _ = self.run_program(incoming_code=8403)
        self.assertEqual("FAIL", result["status"], result)
        self.assertEqual(
            ("FAIL", "PASS", "PASS"), tuple(checks[k] for k in ("PR6", "AT5", "P6"))
        )

    def test_zero_cancel_delta_fails_only_at5(self):
        result, _, checks, _, _ = self.run_program(cancel_count=0)
        self.assertEqual("FAIL", result["status"], result)
        self.assertEqual(
            ("PASS", "FAIL", "PASS"), tuple(checks[k] for k in ("PR6", "AT5", "P6"))
        )

    def test_engine_cancel_evidence_invalidates_normal_victim_claim(self):
        result, obs, checks, _, _ = self.run_program(cancelled=True)
        self.assertEqual("FAIL", result["status"], result)
        self.assertTrue(obs[0]["victim_completed"])
        self.assertEqual(["d0"], obs[0]["cancelled_by"])
        self.assertEqual(
            ("FAIL", "PASS", "PASS"), tuple(checks[k] for k in ("PR6", "AT5", "P6"))
        )

    def test_three_second_finish_window_stops_incoming_and_clears_fault(self):
        result, obs, _, backend, _ = self.run_program(never_finishes=True)
        self.assertEqual("TIMEOUT", result["status"], result)
        self.assertEqual([], obs)
        self.assertEqual(1, len(backend.shapes))
        self.assertFalse(backend.injected)
        stages = {s["id"]: s["status"] for s in result["stages"]}
        self.assertEqual("TIMEOUT", stages["victim_engine_finished"])
        self.assertEqual("BLOCKED", stages["incoming"])
        self.assertTrue(all(c["status"] == "PASS" for c in result["cleanup"]))

    def test_missing_census_map_is_error_before_freeze(self):
        result, _, _, backend, _ = self.run_program(missing_census=True)
        self.assertEqual("ERROR", result["status"], result)
        self.assertEqual([], backend.injections)
        self.assertEqual(1, len(backend.shapes))
