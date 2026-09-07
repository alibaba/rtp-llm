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
from flexlb_ft.scenario import compile_scenarios
from flexlb_ft.scenario.actions import elastic as e
from flexlb_ft.scenario.actions import engine_control as ec
from flexlb_ft.scenario.loader import load_scenarios
from flexlb_ft.scenario.runtime import execute_instance


class Clock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds


class AddedWorkerTests(unittest.TestCase):
    def run_program(self, resumed=True, survivor_ok=True):
        clock = Clock()
        state = dict(stopped=False, added=False, accepted=0, flow_starts=0)
        flows = []
        handlers = {h.name: h for h in [*e.HANDLERS, *ec.HANDLERS]}
        plan = compile_scenarios(
            load_scenarios(ROOT / "scenarios/elastic/added_worker_fault.yaml"),
            handlers=handlers,
        )[0]
        self.assertEqual(plan["resource_budget"]["max_dynamic_additions"], 1)
        self.assertEqual(plan["legacy_case_ids"], ["elastic_stop_after_add"])
        config = plan["environment"]["resolved_config"]
        self.assertEqual(config["scheduler"]["ordering"]["type"], "PRIORITY")
        self.assertEqual(config["scheduler"]["decision"]["type"], "FIXED_WINDOW")
        self.assertEqual(config["dispatcher"]["type"], "BATCH")
        self.assertEqual(config["dispatcher"]["maxInflightBatchesPerPrefillWorker"], 4)
        self.assertNotIn("queueTimeoutMs", str(config))

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
                    return NS(discovery_file=file), NS(master_http_port=1)

                def start_requests(self, ctx, params, deadline):
                    state["survivor_shape"] = params
                    return ctx.register_resource("requests", object())

                def wait_requests(self, ctx, requests, deadline):
                    return dict(completed=survivor_ok, error_count=int(not survivor_ok))

                def teardown(self, ctx, deadline):
                    state["cleaned"] = True

            class Flow(e.ClientRecords):
                pump_error = None

                def __init__(self, ops, epoch, clock):
                    super().__init__(epoch)
                    flows.append(self)

                def start(self):
                    state["flow_starts"] += 1
                    if state["flow_starts"] == 1 or resumed:
                        state["accepted"] += 1
                    r = self.issue(state["flow_starts"], clock)
                    self.update(
                        r,
                        business_finished=True,
                        schedule=dict(status="OK"),
                        stream=dict(status="OK"),
                        consumer_exit_s=clock(),
                        transport_terminal_s=clock(),
                    )

                def stop(self, deadline, cancel=False):
                    self.stopped = True
                    return e.completeness(self.snapshot_records())

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
            ), patch.object(e, "ColdFlow", Flow), patch(
                "flexlb_ft.harness.http_post_json", side_effect=master
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
            self.assertTrue(all(flow.stopped for flow in flows))
        return result, state

    def test_full_program_preserves_stop_survivor_restart_and_fresh_traffic(self):
        result, state = self.run_program()
        self.assertEqual(result["status"], "PASS", result)
        self.assertEqual(state["flow_starts"], 2)
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
