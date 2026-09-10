"""O1 complete program with real Schedule/consumer workers and private log fixtures."""

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


class ObservabilityBackend(programs.Backend):
    def __init__(
        self,
        duplicate=8406,
        missing_bucket=False,
        foreign_pv=False,
        reverse_pair=False,
        missing_log=False,
        missing_lifecycle=False,
        expired_first=False,
    ):
        terminals = {i: 8511 for i in range(2, 10) if i != 6}
        if expired_first:
            terminals[6] = 8511
            terminals.pop(2)
        super().__init__(terminals=terminals)
        self.missing_bucket, self.foreign_pv = missing_bucket, foreign_pv
        self.reverse_pair, self.missing_log = reverse_pair, missing_log
        self.missing_lifecycle = missing_lifecycle or expired_first
        self.calls = []
        original = self.ops.future

        def future(req, timeout, metadata=None):
            self.calls.append(req)
            call = original(req, timeout, metadata)
            if req[1].get("priority") == 40:
                response = self.ops.responses[-1]
                response.code, response.success = duplicate, duplicate == 200
            return call

        self.ops.future = future

    def setup(self, ctx, environment, deadline):
        self.environments.append(environment)
        private = ctx.artifact_dir / "master-logs"
        private.mkdir()
        ctx.master_log_dir = private
        run_dir = ctx.artifact_dir / "environment"
        run_dir.mkdir()
        (run_dir / "flexlb_master.log").write_text("startup\n")
        (private / "flexlb.log").write_text(
            "ordinary\n" if self.missing_log else "[request-scheduler] fixture\n"
        )
        (private / "pv.log").write_text(
            "2026-09-08 [publisher] INFO pvLogger - "
            + json.dumps(
                dict(
                    requestId=100 if self.foreign_pv else 1, admissionRejectReason=None
                )
            )
            + "\n"
        )
        return NS(run_dir=run_dir, master_log_dir=private), self.ops

    def http(self, ops, endpoint, deadline, body=None):
        raw = super().http(ops, endpoint, deadline, body)
        if endpoint == "snapshot":
            lifecycle = raw["engines"][0]["request_lifecycle"]
            lifecycle["10"]["running_ms"] = 8000
            lifecycle["6"]["running_ms"] = 12000
            if self.reverse_pair:
                lifecycle["10"]["running_ms"] = 100000
        if endpoint == "snapshot" and self.missing_lifecycle:
            raw["engines"][0]["request_lifecycle"].pop("6", None)
        return raw

    def metrics(self, ctx, server, endpoint, deadline, **kwargs):
        values = [(30, 4), (50, 3), (70, 2), (90, 1)]
        if self.missing_bucket:
            values.pop()
        # A present zero latency sample is legal; a missing victim series remains weak-zero.
        return (
            200,
            "\n".join(
                f'flexlb_auto_tpm_request_count{{priority="{p}"}} {v}'
                for p, v in values
            )
            + '\nflexlb_auto_tpm_schedule_latency_ms_count{result="success"} 0\n',
        )


class ObservabilityPrograms(unittest.TestCase):
    def run_program(self, **kwargs):
        plan, registry = programs.PreemptionPrograms().plan("observability_integrity")
        backend = ObservabilityBackend(**kwargs)
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
                for p in Path(tmp).glob("preemption-observability-*.json")
                if "-duplicate-" not in p.name
            ]
            backend.input_evidence = [
                json.loads(p.read_text())
                for p in Path(tmp).glob("preemption-o1-input-*.json")
            ]
        checks = {c["id"]: c["status"] for s in result["stages"] for c in s["checks"]}
        return result, observations, checks, backend, plan

    def test_exact_o1_and_all_planes_with_owned_consumers(self):
        result, obs, checks, backend, plan = self.run_program()
        self.assertEqual("PASS", result["status"], result)
        self.assertEqual(11, len(backend.calls))
        self.assertEqual([1, 1] + list(range(2, 11)), [r[0] for r in backend.calls])
        self.assertEqual(
            [50, 40, 30, 30, 50, 50, 70, 70, 30, 30, 90],
            [r[1]["priority"] for r in backend.calls],
        )
        self.assertTrue(
            all(
                (r[1]["input_len"], r[1]["output_len"]) == (2048, 2)
                for r in backend.calls
            )
        )
        self.assertEqual(3, backend.ops.generate_count)
        completed = [
            r
            for r in obs[0]["placeholder"] + obs[0]["wave"]
            if r["schedule"]["status"] == "OK"
        ]
        self.assertEqual(3, len(completed))
        self.assertTrue(
            all(
                r["consumer_done"]
                and r["consumer_completion_verified"]
                and r["consumer_exit_s"] is not None
                and r["transport_terminal_s"] is not None
                for r in completed
            )
        )
        spec = expected_environment("observability", NS(profile="single-nonbatch"))
        env = backend.environments[0]
        self.assertEqual(
            spec.resolved_config,
            env["resolved_config"],
        )
        actual_spec = make_env_spec(env, "single-nonbatch", {"master_base": 28000})
        self.assertEqual(spec.perf, actual_spec.perf)
        self.assertEqual(
            (spec.n_prefill, spec.n_decode),
            (actual_spec.n_prefill, actual_spec.n_decode),
        )
        self.assertEqual(spec.master_env, actual_spec.master_env)
        self.assertTrue(actual_spec.master_debug_log)
        self.assertTrue(env["master_debug_log"])
        self.assertEqual("flexlb_auto_tpm", env["metric_whitelist"])
        self.assertEqual(["ph", "70a", "90"], obs[0]["completed"])
        self.assertEqual(7, len(obs[0]["expired"]))
        self.assertIsNone(obs[0]["victim_total"])
        self.assertEqual(0, obs[0]["latency_success"])
        self.assertEqual(
            {"AT8": "PASS", "P6": "PASS", "AT6": "PASS"},
            {k: checks[k] for k in ("AT8", "P6", "AT6")},
        )
        self.assertTrue(all(c["status"] == "PASS" for c in result["cleanup"]))

    def test_missing_lifecycle_keeps_source_evidence_and_error(self):
        result, _, _, backend, _ = self.run_program(missing_lifecycle=True)
        self.assertEqual(result["status"], "ERROR")
        self.assertIn("request_id=6, matches=0", result["error"])
        self.assertIn("preemption-o1-input-", result["error"])
        self.assertEqual(len(backend.input_evidence), 1)
        evidence = backend.input_evidence[0]
        self.assertEqual(len(evidence["wave"]), 9)
        self.assertNotIn("6", evidence["snapshot"]["engines"][0]["request_lifecycle"])
        self.assertTrue(all(c["status"] == "PASS" for c in result["cleanup"]))

    def test_witnessed_expiry_of_expected_completed_request_is_contract_failure(self):
        result, obs, checks, _, _ = self.run_program(expired_first=True)
        self.assertEqual(result["status"], "FAIL", result)
        self.assertIsNone(result["error"])
        self.assertIsNone(obs[0]["dispatch"][0])
        self.assertEqual(obs[0]["completed"], ["ph", "30a", "90"])
        self.assertEqual(checks["P6"], "FAIL")
        self.assertTrue(all(c["status"] == "PASS" for c in result["cleanup"]))

    def test_missing_required_metric_fails_at8_preserves_p6_at6(self):
        result, _, checks, _, _ = self.run_program(missing_bucket=True)
        self.assertEqual("FAIL", result["status"], result)
        self.assertEqual(
            ("FAIL", "PASS", "PASS"), tuple(checks[k] for k in ("AT8", "P6", "AT6"))
        )

    def test_duplicate_nonrejection_fails_only_at6_and_never_opens_extra_consumer(self):
        result, _, checks, backend, _ = self.run_program(duplicate=200)
        self.assertEqual("FAIL", result["status"], result)
        self.assertEqual(
            ("PASS", "PASS", "FAIL"), tuple(checks[k] for k in ("AT8", "P6", "AT6"))
        )
        self.assertEqual(3, backend.ops.generate_count)

    def test_foreign_pv_rid_prefix_does_not_supply_channel_evidence(self):
        result, obs, checks, _, _ = self.run_program(foreign_pv=True)
        self.assertEqual("FAIL", result["status"], result)
        self.assertEqual([], obs[0]["pv_selected"])
        self.assertEqual(
            ("FAIL", "PASS", "PASS"), tuple(checks[k] for k in ("AT8", "P6", "AT6"))
        )

    def test_reverse_actual_dispatch_fails_all_three_invariants(self):
        result, _, checks, _, _ = self.run_program(reverse_pair=True)
        self.assertEqual("FAIL", result["status"], result)
        self.assertEqual(
            ("FAIL", "FAIL", "FAIL"), tuple(checks[k] for k in ("AT8", "P6", "AT6"))
        )

    def test_missing_private_debug_marker_fails_only_at8(self):
        result, _, checks, _, _ = self.run_program(missing_log=True)
        self.assertEqual("FAIL", result["status"], result)
        self.assertEqual(
            ("FAIL", "PASS", "PASS"), tuple(checks[k] for k in ("AT8", "P6", "AT6"))
        )
