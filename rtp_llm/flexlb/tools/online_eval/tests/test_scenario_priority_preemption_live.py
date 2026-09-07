"""Live BATCH eviction fixtures preserve deferred and serial survivor consumption."""

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch

import test_scenario_priority_preemption as programs
from flexlb_ft.harness import render_env
from flexlb_ft.scenario.actions import priority
from flexlb_ft.scenario.actions import priority_preemption as preempt
from flexlb_ft.scenario.backend import make_env_spec
from flexlb_ft.scenario.runtime import execute_instance
from flexlb_ft.support.priority import _pq_live_spec
from test_scenario_priority import Clean


class LiveBackend(programs.Backend):
    def __init__(
        self,
        victim_code=8400,
        saw_victim=False,
        missing_metric=False,
        wrong_tags=False,
        missing_owner=False,
    ):
        super().__init__(terminals={3: victim_code})
        self.ops.batch = True
        self.missing_owner = missing_owner
        self.saw_victim, self.missing_metric, self.wrong_tags = (
            saw_victim,
            missing_metric,
            wrong_tags,
        )
        self.fetches = []
        fetch = self.ops.fetch

        def wrapped(req, timeout):
            self.fetches.append(
                dict(
                    rid=req["request_id"],
                    schedule_count=len(self.ops.responses),
                    timeout=timeout,
                )
            )
            return fetch(req, timeout)

        self.ops.fetch = wrapped

    def http(self, ops, endpoint, deadline, body=None):
        if endpoint == "set_perf":
            return dict(status="ok", engine=body["engine"], port=1234)
        if endpoint != "snapshot":
            raise AssertionError(endpoint)
        lifecycle = {str(i): dict(running_ms=i * 4000) for i in (1, 2, 4)}
        if self.saw_victim:
            lifecycle["3"] = dict(running_ms=3000)
        result = dict(
            engines=[
                dict(
                    name=f"{role[0]}{i}",
                    role=role,
                    stopped=False,
                    grpc_addr=f"{role}{i}",
                    running=1 if role == "prefill" else 0,
                    waiting=0,
                    inflight=0,
                    leak_detected=False,
                    request_lifecycle=lifecycle if role == "prefill" else {},
                )
                for role, count in [("prefill", 1), ("decode", 4)]
                for i in range(count)
            ]
        )

        if self.missing_owner:
            result["engines"].pop()
        return result

    def metrics(self, ctx, server, endpoint, deadline, **kwargs):
        if self.missing_metric:
            return 200, "jvm_threads_live_threads 1\n"
        incoming = "50" if self.wrong_tags else "70"
        return (
            200,
            f'flexlb_auto_tpm_victim_count{{stage="prefill_queued",victim_priority="30",incoming_priority="{incoming}"}} 1\nflexlb_auto_tpm_priority_preempt_count{{stage="prefill_queued"}} 1\n',
        )


class LivePrograms(unittest.TestCase):
    def run_program(self, variant="prefill_queued_live_single", **kwargs):
        plan, registry = programs.PreemptionPrograms().plan(variant)
        backend = LiveBackend(**kwargs)
        with tempfile.TemporaryDirectory() as tmp, patch.object(
            preempt, "_http", backend.http
        ), patch.object(priority, "_http", backend.http), patch(
            "flexlb_ft.scenario.actions.engine_control._http", backend.http
        ), patch(
            "flexlb_ft.scenario.actions.balance._http", backend.http
        ), patch(
            "flexlb_ft.scenario.actions.balance.urllib.request.urlopen",
            return_value=Clean(),
        ), patch(
            "flexlb_ft.scenario.actions.status_protocol._http", backend.metrics
        ):
            result = execute_instance(
                plan, backend, handlers=registry, artifact_dir=tmp
            )
            observations = [
                json.loads(p.read_text())
                for p in Path(tmp).glob("preemption-live-prefill-*.json")
            ]
        checks = {c["id"]: c["status"] for s in result["stages"] for c in s["checks"]}
        return result, observations, checks, backend, plan

    def test_both_profiles_match_exact_old_env_and_deferred_survivors(self):
        for variant in ("prefill_queued_live_single", "prefill_queued_live_window"):
            with self.subTest(variant=variant):
                result, obs, checks, backend, plan = self.run_program(variant)
                self.assertEqual("PASS", result["status"], result)
                old = _pq_live_spec(NS(profile=plan["profile"]))
                env = backend.environments[0]
                actual = make_env_spec(env, plan["profile"], {"master_base": 28000})
                self.assertEqual(
                    json.loads(render_env(old.master_profile, old.config_overrides)),
                    env["resolved_config"],
                )
                self.assertEqual(
                    (old.n_prefill, old.n_decode, old.perf, old.master_env),
                    (actual.n_prefill, actual.n_decode, actual.perf, actual.master_env),
                )
                self.assertEqual(5, len(backend.shapes))
                self.assertEqual(
                    [50, 30, 30, 70], [r["priority"] for r in backend.shapes[:4]]
                )
                self.assertEqual([1, 2, 4, 5], [r["rid"] for r in backend.fetches])
                self.assertEqual(
                    [4, 4, 4], [r["schedule_count"] for r in backend.fetches[:3]]
                )
                self.assertTrue(
                    all(59 < r["timeout"] <= 60 for r in backend.fetches[:3])
                )
                self.assertEqual(0, backend.ops.generate_count)
                rows = obs[0]["placeholder"] + [obs[0]["wave"][i] for i in (0, 2)]
                self.assertTrue(
                    all(
                        r["fetch_invocations"] == 1
                        and r["consumer_completion_verified"]
                        and r["consumer_done"]
                        for r in rows
                    )
                )
                self.assertLessEqual(
                    rows[0]["consumer_exit_s"], rows[1]["stream"]["started_s"]
                )
                self.assertLessEqual(
                    rows[1]["consumer_exit_s"], rows[2]["stream"]["started_s"]
                )
                self.assertEqual(0, obs[0]["wave"][1].get("fetch_invocations", 0))
                self.assertEqual(
                    {"PR10": "PASS", "PR5": "PASS", "PR6": "PASS", "P6": "PASS"},
                    {k: checks[k] for k in ("PR10", "PR5", "PR6", "P6")},
                )
                self.assertTrue(all(c["status"] == "PASS" for c in result["cleanup"]))

    def test_8429_cannot_replace_exact_master_local_8400(self):
        result, _, checks, _, _ = self.run_program(victim_code=8429)
        self.assertEqual("FAIL", result["status"], result)
        self.assertEqual(
            ("FAIL", "FAIL", "FAIL", "PASS"),
            tuple(checks[k] for k in ("PR10", "PR5", "PR6", "P6")),
        )

    def test_engine_seen_victim_fails_never_delivered_only(self):
        result, _, checks, _, _ = self.run_program(saw_victim=True)
        self.assertEqual("FAIL", result["status"], result)
        self.assertEqual(
            ("PASS", "FAIL", "PASS", "PASS"),
            tuple(checks[k] for k in ("PR10", "PR5", "PR6", "P6")),
        )

    def test_missing_count_cannot_be_one_eviction(self):
        result, _, checks, _, _ = self.run_program(missing_metric=True)
        self.assertEqual("FAIL", result["status"], result)
        self.assertEqual(
            ("PASS", "PASS", "FAIL", "PASS"),
            tuple(checks[k] for k in ("PR10", "PR5", "PR6", "P6")),
        )

    def test_priority_tagged_metric_remains_diagnostic(self):
        result, obs, _, _, _ = self.run_program(wrong_tags=True)
        self.assertEqual("PASS", result["status"], result)
        self.assertIsNone(obs[0]["tagged_victim_diagnostic"])

    def test_incomplete_engine_inventory_cannot_prove_never_delivered(self):
        result, obs, checks, backend, _ = self.run_program(missing_owner=True)
        self.assertEqual("ERROR", result["status"], result)
        self.assertEqual([], obs)
        self.assertEqual(4, len(backend.shapes))
        self.assertNotIn("PR5", checks)
