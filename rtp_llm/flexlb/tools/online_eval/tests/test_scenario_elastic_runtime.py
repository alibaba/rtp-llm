"""Compile the shipped elastic YAML and execute its full stage/check contract.

Only external resources are faked. Windows, the high-hit guard, final verdict,
reference validation, output validation and fail-stop behavior use real code.
"""

import copy
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from flexlb_test_framework.scenario import compile_scenarios
from flexlb_test_framework.scenario.actions import elastic as e
from flexlb_test_framework.scenario.contracts import CheckResult, StageOutput
from flexlb_test_framework.scenario.loader import load_scenarios
from flexlb_test_framework.scenario.runtime import execute_instance


class Clock:
    def __init__(self):
        self.now = 0

    def __call__(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds


class Backend:
    def setup(self, ctx, environment, deadline):
        return object(), NS(master_http_port=1)

    def teardown(self, ctx, deadline):
        pass


class ElasticRuntimeTests(unittest.TestCase):
    def test_actual_mutation_handlers_follow_typed_reference_and_budget(self):
        source = self.source()
        source.pop("variants")
        source["stages"] = [
            dict(id="setup", action="setup"),
            dict(id="add", action="elastic_add", params=dict(role="prefill")),
            dict(
                id="remove",
                action="elastic_remove",
                params=dict(
                    engine={"$ref": "stages.add.output.engine"}, drain_timeout_ms=5000
                ),
            ),
            dict(id="teardown", action="teardown"),
        ]
        handlers = {h.name: h for h in e.HANDLERS}
        plan = compile_scenarios([("mutation.yaml", source)], handlers=handlers)[0]
        self.assertEqual(plan["resource_budget"]["max_dynamic_additions"], 1)
        engine = dict(role="prefill", grpc_addr="127.0.0.1:12345")
        responses = [
            dict(status="ok", action="added", engine="p2", port=12345, http_port=12344),
            dict(
                status="ok",
                action="removed",
                engine="p2",
                port=12345,
                mode="graceful",
                drained=False,
            ),
        ]
        clock = Clock()
        with tempfile.TemporaryDirectory() as root, patch.object(
            e, "_snapshot", side_effect=[{}, {"p2": engine}, {"p2": engine}, {}]
        ), patch.object(e, "_http", side_effect=responses) as http:
            result = execute_instance(
                plan,
                Backend(),
                handlers=handlers,
                artifact_dir=root,
                clock=clock,
                sleeper=clock.sleep,
            )
        self.assertEqual(result["status"], "PASS", result)
        self.assertEqual(http.call_args.args[3]["engine"], "p2")
        # This program asserts membership only; it deliberately makes no
        # business-drain claim from drained=false or from a successful ack.
        self.assertEqual(result["stages"][2]["checks"][0]["id"], "membership")

    def source(self):
        source = load_scenarios(ROOT / "scenarios/elastic/lifecycle.yaml")[0][1]
        source["variants"] = [
            v for v in source["variants"] if v["id"].startswith("kv_skew_")
        ]
        return source

    def run_pilot(self, rate=1.0, flow_failed=False):
        clock = Clock()
        mutated = []

        def external(action, ctx, params, deadline):
            register = ctx.register_resource
            records = e.ClientRecords(ctx.env_epoch)
            for rid in range(20):
                record = records.issue(rid, clock)
                record["rpc_observation"] = dict(complete=True, legacy_success=True)
                records.update(
                    record,
                    business_finished=True,
                    schedule=dict(status="OK"),
                    stream=dict(status="OK"),
                    transport_terminal_s=clock(),
                    consumer_exit_s=clock(),
                    consumer_done=True,
                    consumer_completion_verified=True,
                )
            if action == "elastic_seed":
                return StageOutput(
                    output=dict(
                        requests=register("requests", records),
                        families=register("snapshot", dict(hot="p0", cold="p1")),
                    ),
                    checks=[
                        CheckResult("seed_success", "PASS"),
                        CheckResult("skew", "PASS"),
                    ],
                )
            if action == "elastic_metrics_start":
                samples = [
                    dict(
                        time_s=t,
                        engines={
                            name: dict(
                                role="prefill",
                                mock_engine_cache_key_hits_total=t * 10 * rate,
                                mock_engine_cache_keys_requested_total=t * 10,
                                mock_engine_waiting=1,
                                mock_engine_cache_blocks=100,
                                mock_engine_available_blocks=20,
                            )
                            for name in ("p0", "p1")
                        },
                    )
                    for t in range(121)
                ]
                metrics = NS(
                    skew_started_s=clock(),
                    snapshot=lambda: dict(
                        samples=samples, errors=[], env_epoch=ctx.env_epoch
                    ),
                )
                return StageOutput(
                    output=dict(observation=register("observation", metrics))
                )
            if action == "elastic_flow_start":
                return StageOutput(output=dict(flow=register("flow", records)))
            if action == "elastic_scale":
                mutated.append(clock())
                return StageOutput(
                    output=dict(
                        scale=register(
                            "snapshot",
                            dict(
                                started_s=clock(),
                                ended_s=clock(),
                                response=dict(drained=False),
                            ),
                        ),
                        drained=True,
                    ),
                    checks=[CheckResult("pre_scale_skew", "PASS")],
                )
            if action == "elastic_flow_stop":
                result = e.completeness(records.snapshot_records())
                if flow_failed:
                    result.update(zero_errors=False, failed_request_ids=[42])
                return StageOutput(
                    output=dict(
                        complete=True, issued=20, result=register("snapshot", result)
                    )
                )
            if action == "elastic_recovery":
                return StageOutput(
                    output=dict(
                        requests=register("requests", records), success_rate=1.0
                    )
                )
            raise AssertionError(action)

        handlers = {}
        for handler in e.HANDLERS:
            if handler.name in {
                "elastic_baseline",
                "elastic_window",
                "elastic_verdict",
            }:
                handlers[handler.name] = handler
            else:
                handlers[handler.name] = replace(
                    handler,
                    execute=lambda ctx, p, d, name=handler.name: external(
                        name, ctx, p, d
                    ),
                )
        plans = compile_scenarios([("pilot.yaml", self.source())], handlers=handlers)
        results = []
        with patch(
            "flexlb_test_framework.harness.http_post_json",
            return_value=(
                200,
                dict(worker_summary={"PREFILL": dict(discovered=1, alive=1)}),
            ),
        ):
            for plan in plans:
                clock.now = 0
                with tempfile.TemporaryDirectory() as root:
                    results.append(
                        execute_instance(
                            plan,
                            Backend(),
                            handlers=handlers,
                            artifact_dir=root,
                            clock=clock,
                            sleeper=clock.sleep,
                        )
                    )
        return results, mutated

    def test_high_hit_runs_all_stages_for_both_victims(self):
        results, mutated = self.run_pilot()
        self.assertEqual(len(mutated), 2)
        for result in results:
            self.assertEqual(result["status"], "PASS", result)
            rows = {row["id"]: row for row in result["stages"]}
            self.assertTrue(all(row["status"] == "PASS" for row in rows.values()))
            self.assertEqual([c["id"] for c in rows["baseline"]["checks"]], ["traffic"])
            self.assertEqual(rows["transient"]["checks"], [])
            self.assertEqual(rows["steady"]["checks"], [])
            self.assertEqual(
                {c["id"] for c in rows["verdict"]["checks"]},
                {"PC", "PQ", "PK", "P6", "P2"},
            )

    def test_low_nonempty_hit_is_not_a_new_gate(self):
        results, mutated = self.run_pilot(rate=0.5)
        self.assertEqual(len(mutated), 2)
        for result in results:
            self.assertEqual(result["status"], "PASS", result)

    def test_p6_failure_still_fails_after_high_hit(self):
        results, _ = self.run_pilot(flow_failed=True)
        for result in results:
            self.assertEqual(result["status"], "FAIL")
            verdict = next(row for row in result["stages"] if row["id"] == "verdict")
            checks = {check["id"]: check for check in verdict["checks"]}
            self.assertEqual(checks["P6"]["status"], "FAIL")
            self.assertEqual(checks["P6"]["actual"]["failed_request_ids"], [42])

    def test_baseline_cannot_bypass_guard_with_generic_window(self):
        source = self.source()
        for variant in source["variants"]:
            for stage in variant["stages"]:
                if stage["id"] == "baseline":
                    stage["action"] = "elastic_window"
        with self.assertRaisesRegex(ValueError, "requires elastic_baseline"):
            compile_scenarios(
                [("pilot.yaml", source)], handlers={h.name: h for h in e.HANDLERS}
            )

    def test_baseline_action_rejects_post_scale_phase(self):
        source = self.source()
        for variant in source["variants"]:
            for stage in variant["stages"]:
                if stage["id"] == "transient":
                    stage["action"] = "elastic_baseline"
        with self.assertRaisesRegex(ValueError, "requires phase baseline"):
            compile_scenarios(
                [("pilot.yaml", source)], handlers={h.name: h for h in e.HANDLERS}
            )
