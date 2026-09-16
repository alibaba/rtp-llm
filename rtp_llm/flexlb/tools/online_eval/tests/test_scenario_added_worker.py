"""Execute the shipped added-worker YAML with fake external services."""

import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import test_scenario_elastic_concurrent_profiles as rpc_fixture
from environment_expectations import environment as expected_environment
from flexlb_cfg import render_env
from flexlb_test_framework.scenario import compile_scenarios
from flexlb_test_framework.scenario.actions import elastic as e
from flexlb_test_framework.scenario.actions import elastic_added_worker as aw
from flexlb_test_framework.scenario.actions import engine_control as ec
from flexlb_test_framework.scenario.backend import make_env_spec
from flexlb_test_framework.scenario.loader import load_scenarios
from flexlb_test_framework.scenario.runtime import execute_instance


class Clock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds


class AddedWorkerTests(unittest.TestCase):
    def run_program(
        self,
        resumed=True,
        survivor_ok=True,
        profile="batch-window",
        reset=False,
        opposite=False,
        late=False,
        initial=0,
    ):
        clock = Clock()
        state = dict(
            stopped=False,
            added=False,
            accepted=initial,
            flow_starts=0,
            restarted=False,
            engines={"prefill-0": {"grpc_addr": "127.0.0.1:10005"}},
        )
        handlers = {h.name: h for h in [*e.HANDLERS, *ec.HANDLERS, *aw.HANDLERS]}
        plans = compile_scenarios(
            load_scenarios(ROOT / "scenarios/elastic/added_worker_fault.yaml"),
            handlers=handlers,
        )
        plan = next(p for p in plans if p["profile"] == profile)
        self.assertEqual(plan["resource_budget"]["max_dynamic_additions"], 1)
        old = expected_environment("master_kill", profile)
        spec = make_env_spec(plan["environment"], profile, {"master_base": 28000})
        self.assertEqual(
            plan["environment"]["resolved_config"],
            old.resolved_config,
        )
        for key in (
            "perf",
            "n_prefill",
            "n_decode",
            "discovery",
            "prefill_cache_blocks",
            "decode_cache_blocks",
        ):
            self.assertEqual(getattr(spec, key), getattr(old, key))
        self.assertNotIn("queueTimeoutMs", str(plan["environment"]["resolved_config"]))

        with tempfile.TemporaryDirectory() as temp:
            file = Path(temp) / "discovery.json"

            class Backend:
                def setup(self, ctx, environment, deadline):
                    file.write_text(
                        json.dumps(
                            {
                                "mock.prefill.hosts.address": [
                                    "127.0.0.1:10000",
                                    "127.0.0.1:10002",
                                ]
                            }
                        )
                    )
                    ops = NS(master_http_port=1)
                    rpc_fixture.ProfileTests().driver(opposite=opposite)(
                        ops, environment, state, clock
                    )
                    counter = iter(range(20001, 30001))
                    ops.next_request_id = lambda: next(counter)
                    stub_factory = ops.pb2_grpc.RpcServiceStub

                    def stub(ch):
                        original = stub_factory(ch)

                        def consume(method):
                            def call(req, timeout):
                                result = getattr(original, method)(req, timeout)
                                if state["stopped"] and not survivor_ok:
                                    return iter(
                                        [
                                            NS(
                                                HasField=lambda name: True,
                                                error_info=NS(
                                                    error_code=8431,
                                                    error_message="survivor failed",
                                                ),
                                                flatten_output=NS(finished=[False]),
                                            )
                                        ]
                                    )
                                if not state["stopped"]:
                                    if not state["restarted"] or resumed:
                                        state["accepted"] += 1
                                return result

                            return call

                        return NS(
                            FetchResponse=consume("FetchResponse"),
                            GenerateStreamCall=consume("GenerateStreamCall"),
                        )

                    ops.pb2_grpc.RpcServiceStub = stub
                    # Survivor uses its own legacy fixed key rather than pump's cold key.
                    original_schedule = ops.schedule_pb2_grpc.FlexlbServiceStub

                    def schedule_stub(ch):
                        wrapped = original_schedule(ch)

                        def future(req, timeout):
                            if state["stopped"]:
                                state["survivor_actual_shape"] = copy.deepcopy(
                                    vars(req)
                                )
                                response = NS(
                                    code=200,
                                    success=True,
                                    error_message="",
                                    target="127.0.0.1:10001",
                                    enqueued_by_master=("nonbatch" not in profile)
                                    != opposite,
                                )
                                return NS(result=lambda: response, cancel=lambda: True)
                            future = wrapped.Schedule.future(req, timeout)
                            original_result = future.result

                            def result():
                                if late:
                                    clock.now += 21  # within the 30s Schedule timeout
                                return original_result()

                            future.result = result
                            return future

                        return NS(Schedule=NS(future=future))

                    ops.schedule_pb2_grpc.FlexlbServiceStub = schedule_stub
                    state["ops"] = ops
                    return NS(discovery_file=file), ops

                def start_requests(self, ctx, params, deadline):
                    state["survivor_shape"] = params
                    records = e.RecordedRequests(ctx.ops, ctx.env_epoch, clock)
                    r = records.issue(ctx.ops.next_request_id(), clock)
                    records.run(
                        r,
                        {
                            k: params[k]
                            for k in ("input_len", "output_len", "block_keys")
                        },
                        schedule_timeout_s=30,
                        stream_timeout_s=10,
                    )
                    state["survivor_records"] = records.snapshot_records()
                    return ctx.register_resource(
                        "requests", records, cleanup=lambda d: records.cancel_active()
                    )

                def wait_requests(self, ctx, requests, deadline):
                    rows = requests.snapshot_records()
                    result = e.completeness(rows)
                    return dict(
                        completed=result["complete"],
                        error_count=len(result["failed_request_ids"]),
                    )

                def teardown(self, ctx, deadline):
                    state["cleaned"] = True

            def http(ops, endpoint, deadline, body=None):
                if endpoint == "snapshot":
                    engines = []
                    if state["added"]:
                        engines.append(
                            dict(
                                name="prefill-2",
                                role="prefill",
                                grpc_addr="127.0.0.1:10005",
                                stopped=state["stopped"],
                                accepted=state["accepted"],
                            )
                        )
                    return dict(engines=copy.deepcopy(engines))
                if endpoint == "add_engine":
                    state["added"] = True
                    file.write_text(
                        json.dumps(
                            {
                                "mock.prefill.hosts.address": [
                                    "127.0.0.1:10000",
                                    "127.0.0.1:10002",
                                    "127.0.0.1:10004",
                                ]
                            }
                        )
                    )
                    return dict(
                        status="ok",
                        engine="prefill-2",
                        port=10005,
                        http_port=10004,
                        action="added",
                    )
                if endpoint in {"stop_engine", "start_engine"}:
                    state["stopped"] = endpoint == "stop_engine"
                    if endpoint == "start_engine":
                        state["restarted"] = True
                        if reset:
                            state["accepted"] = 0
                    return dict(status="ok", engine="prefill-2", port=10005)
                raise AssertionError(endpoint)

            def master(*args, **kwargs):
                return 200, dict(
                    worker_summary={
                        "PREFILL": dict(
                            discovered=3, alive=2 if state["stopped"] else 3
                        )
                    }
                )

            with patch.object(e, "_http", side_effect=http), patch.object(
                ec, "_http", side_effect=http
            ), patch(
                "flexlb_test_framework.harness.http_post_json", side_effect=master
            ):
                result = execute_instance(
                    plan,
                    Backend(),
                    handlers=handlers,
                    artifact_dir=Path(temp) / "artifacts",
                    clock=clock,
                    sleeper=clock.sleep,
                )
            self.assertTrue(state["cleaned"])
            state["artifacts"] = [
                json.loads(p.read_text())
                for p in (Path(temp) / "artifacts").glob("elastic-added-probe-*.json")
            ]
            state["plan"] = plan
        return result, state

    def test_full_program_preserves_stop_survivor_restart_and_fresh_traffic(self):
        result, state = self.run_program()
        self.assertEqual(result["status"], "PASS", result)
        self.assertEqual(len(state["artifacts"]), 2)
        self.assertEqual(state["survivor_shape"]["stream_timeout_s"], 10)
        rows = {row["id"]: row for row in result["stages"]}
        self.assertEqual(rows["stopped_topology"]["checks"][1]["actual"]["alive"], 2)
        self.assertEqual(rows["restored_topology"]["checks"][1]["actual"]["alive"], 3)
        self.assertEqual(rows["resumed_traffic"]["checks"][0]["actual"], 1)

    def test_pre_stop_traffic_cannot_prove_post_restart_traffic(self):
        result, _ = self.run_program(resumed=False)
        self.assertEqual(result["status"], "FAIL")
        row = next(r for r in result["stages"] if r["id"] == "resumed_traffic")
        self.assertEqual(row["checks"][0]["actual"], 0)

    def test_survivor_failure_blocks_restart_without_hiding_cleanup(self):
        result, state = self.run_program(survivor_ok=False)
        self.assertEqual(result["status"], "FAIL")
        self.assertTrue(state["stopped"])
        self.assertEqual(
            next(r for r in result["stages"] if r["id"] == "restart")["status"],
            "BLOCKED",
        )
        self.assertTrue(all(r["status"] == "PASS" for r in result["cleanup"]))

    def test_all_profiles_actual_driver_configs_and_terminal_records(self):
        for profile in [
            "batch-window",
            "single-batch",
            "single-nonbatch",
            "window-nonbatch",
        ]:
            with self.subTest(profile=profile):
                result, state = self.run_program(profile=profile)
                self.assertEqual(result["status"], "PASS", result)
                expected = (
                    "GenerateStreamCall" if "nonbatch" in profile else "FetchResponse"
                )
                self.assertEqual({m for m, _ in state["protocol_calls"]}, {expected})
                self.assertEqual(
                    state["survivor_records"][0]["stream"]["method"], expected
                )
                self.assertTrue(state["survivor_records"][0]["business_finished"])
                self.assertEqual(state["survivor_actual_shape"]["block_keys"], [7])
                for artifact in state["artifacts"]:
                    self.assertTrue(
                        all(
                            r["business_finished"]
                            and r["consumer_exit_s"] is not None
                            and r["transport_terminal_s"] is not None
                            for r in artifact["requests"]
                        )
                    )

    def test_successful_opposite_protocol_fails(self):
        for profile in ["single-batch", "single-nonbatch", "window-nonbatch"]:
            result, _ = self.run_program(profile=profile, opposite=True)
            stage = next(s for s in result["stages"] if s["id"] == "first_traffic")
            self.assertEqual(
                next(c for c in stage["checks"] if c["id"] == "protocol")["status"],
                "FAIL",
                result,
            )

    def test_last_admitted_request_can_complete_after_issuance_window(self):
        result, state = self.run_program(late=True)
        self.assertEqual(result["status"], "PASS", result)
        self.assertTrue(
            all(
                a["samples"][-1]["time_s"] > a["issuance_end_s"]
                for a in state["artifacts"]
            )
        )

    def test_restart_counter_reset_cannot_pass_with_one_fresh_request(self):
        result, _ = self.run_program(reset=True, initial=10)
        rows = {s["id"]: s for s in result["stages"]}
        self.assertEqual(rows["resumed_traffic"]["status"], "PASS", result)
        self.assertEqual(rows["cross_restart_growth"]["status"], "FAIL", result)

    def test_existing_counter_cannot_prove_recovery(self):
        result, _ = self.run_program(resumed=False, initial=10)
        rows = {s["id"]: s for s in result["stages"]}
        self.assertEqual(rows["resumed_traffic"]["status"], "FAIL", result)
