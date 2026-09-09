"""Strict balance-window math and formal steady-recovery YAML execution."""

import copy
import json
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from flexlb_test_framework.scenario import compile_scenarios
from flexlb_test_framework.scenario.actions import elastic as e
from flexlb_test_framework.scenario.actions import elastic_balance as balance
from flexlb_test_framework.scenario.actions import elastic_concurrent as concurrent
from flexlb_test_framework.scenario.actions import engine_control as control
from flexlb_test_framework.scenario.catalog import handlers as builtin_handlers
from flexlb_test_framework.scenario.loader import load_scenarios
from flexlb_test_framework.scenario.runtime import execute_instance


class Clock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds
        time.sleep(0.0001)


class BalanceTests(unittest.TestCase):
    def run_program(
        self, drift=False, missing=False, high_queue=False, recovery_errors=0
    ):
        handlers = builtin_handlers()
        plans = compile_scenarios(
            load_scenarios(ROOT / "scenarios/elastic/lifecycle.yaml"), handlers=handlers
        )
        plan = next(p for p in plans if p["variant_id"] == "steady_recovery")
        self.assertEqual(len(plan["stages"]), 14)
        self.assertEqual(
            next(s for s in plan["stages"] if s["id"] == "flow")["params"][
                "interval_s"
            ],
            0.2,
        )
        clock = Clock()
        state = dict(removed=None, cleaned=False, recovery=0)
        engines = {
            f"decode-{i}": dict(
                name=f"decode-{i}", role="decode", grpc_addr=f"127.0.0.1:{10001+2*i}"
            )
            for i in range(4)
        }
        rid = iter(range(1, 10000))
        with tempfile.TemporaryDirectory() as temp:
            discovery = Path(temp) / "discovery.json"

            def sync():
                discovery.write_text(
                    json.dumps(
                        {
                            "mock.decode.hosts.address": [
                                "127.0.0.1:"
                                + str(int(v["grpc_addr"].split(":")[-1]) - 1)
                                for v in engines.values()
                            ]
                        }
                    )
                )

            class Backend:
                def setup(self, ctx, environment, deadline):
                    self.environment = environment
                    sync()
                    return NS(discovery_file=discovery), NS(
                        next_request_id=lambda: next(rid), master_http_port=2
                    )

                def teardown(self, ctx, deadline):
                    state["cleaned"] = True

            class Metrics:
                def __init__(self, ctx, max_duration_s):
                    self.thread = NS(start=lambda: None, ident=None)

                def stop(self, deadline):
                    state["metrics_stopped"] = True

                def snapshot(self):
                    samples = []
                    for t in range(int(clock()) + 1):
                        rows = {}
                        for i in range(4):
                            if (
                                state["removed"] is not None
                                and t >= state["removed"]
                                and i == 0
                            ):
                                continue
                            count = t * 30
                            # Two adjacent 3s early windows drift; tail returns to
                            # equal shares so the oscillation check is independent.
                            if drift and t >= 40 and i > 0:
                                count += min(max(t - 40, 0), 6) * (
                                    30 if i == 1 else -15
                                )
                            row = dict(
                                role="decode",
                                mock_engine_completed_total=count,
                                mock_engine_cache_blocks=100,
                                mock_engine_available_blocks=80,
                                mock_engine_waiting=3 if high_queue and t >= 80 else 1,
                                mock_engine_decode_ms_avg=10,
                                rtp_llm_generate_tps=10,
                                mock_engine_cache_key_hits_total=t * 10,
                                mock_engine_cache_keys_requested_total=t * 20,
                            )
                            if missing and i == 2 and t >= 80:
                                row.pop("mock_engine_available_blocks")
                            rows[f"decode-{i}"] = row
                        samples.append(dict(time_s=float(t), engines=rows))
                    return dict(samples=samples, errors=[], env_epoch=1)

            def http(ops, endpoint, deadline, body=None):
                if endpoint == "snapshot":
                    return dict(engines=copy.deepcopy(list(engines.values())))
                self.assertEqual(endpoint, "remove_engine")
                self.assertEqual(
                    body,
                    dict(engine="decode-0", mode="graceful", drain_timeout_ms=60000),
                )
                engines.pop("decode-0")
                sync()
                state["removed"] = clock()
                return dict(
                    status="ok",
                    action="removed",
                    engine="decode-0",
                    port=10001,
                    mode="graceful",
                    drained=True,
                )

            def run(records, record, shape, **kwargs):
                is_batch = len(shape["block_keys"]) == 3
                if is_batch:
                    state["recovery"] += 1
                    good = state["recovery"] > recovery_errors
                else:
                    self.assertEqual(
                        shape,
                        dict(
                            input_len=2048,
                            output_len=2,
                            block_keys=[record["wire_request_id"] * 100 + 1],
                        ),
                    )
                    self.assertEqual(kwargs["stream_timeout_s"], 30)
                    good = True
                records.update(
                    record,
                    business_finished=good,
                    schedule=dict(status="OK"),
                    stream=dict(status="OK"),
                    transport_terminal_s=clock(),
                    consumer_exit_s=clock(),
                )

            def master(*args, **kwargs):
                return 200, dict(
                    worker_summary=dict(
                        DECODE=dict(discovered=len(engines), alive=len(engines))
                    )
                )

            with patch.object(e, "ElasticMetrics", Metrics), patch.object(
                e, "_http", side_effect=http
            ), patch.object(
                concurrent, "mutation_http", side_effect=http
            ), patch.object(
                e.RecordedRequests, "run", run
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
            self.assertTrue(state["metrics_stopped"])
            self.assertFalse(
                any(
                    t.name == "elastic-recorded-flow" and t.is_alive()
                    for t in threading.enumerate()
                )
            )
            artifacts = {
                p.name: json.loads(p.read_text())
                for p in (Path(temp) / "artifacts").glob(
                    "elastic-steady-verdict-*.json"
                )
            }
            return result, state, artifacts

    def test_complete_steady_program_preserves_tail_windows_and_observations(self):
        result, state, artifacts = self.run_program()
        self.assertEqual(result["status"], "PASS", result)
        self.assertEqual(state["recovery"], 20)
        self.assertEqual(sum(len(s["checks"]) for s in result["stages"]), 15)
        detail = next(iter(artifacts.values()))
        self.assertEqual(
            detail["tail_share"],
            dict.fromkeys(["decode-1", "decode-2", "decode-3"], 1 / 3),
        )
        self.assertEqual(detail["observations"]["steady"]["hit_rate"]["value"], 0.5)
        self.assertEqual(
            detail["observations"]["steady"]["generate_tps_ratio"]["value"], 1
        )

    def test_early_persistent_drift_fails_independently_of_balanced_tail(self):
        result, _, _ = self.run_program(drift=True)
        checks = {
            c["id"]: c["status"]
            for s in result["stages"]
            if s["id"] == "verdict"
            for c in s["checks"]
        }
        self.assertEqual(checks["share_max"], "PASS")
        self.assertEqual(checks["oscillation"], "FAIL")

    def test_missing_occupancy_is_error_and_queue_three_is_failure(self):
        result, _, _ = self.run_program(missing=True)
        self.assertEqual(
            next(s for s in result["stages"] if s["id"] == "verdict")["status"], "ERROR"
        )
        result, _, _ = self.run_program(high_queue=True)
        checks = {
            c["id"]: c["status"]
            for s in result["stages"]
            if s["id"] == "verdict"
            for c in s["checks"]
        }
        self.assertEqual(checks["waiting_peak"], "FAIL")

    def test_empty_subwindow_cannot_join_nonadjacent_drift(self):
        self.assertEqual(balance.oscillations({"d": [(0, 0.8), (2, 0.8)]}, 0.5), [])
        self.assertEqual(balance.oscillations({"d": [(0, 0.6), (1, 0.6)]}, 0.5), [])
        self.assertEqual(balance.oscillations({"d": [(0, 0.8), (1, 0.2)]}, 0.5), [])
        self.assertEqual(len(balance.oscillations({"d": [(0, 0.8), (1, 0.8)]}, 0.5)), 1)

    def test_counter_reset_and_missing_samples_never_become_zero(self):
        window = dict(
            start_s=0,
            end_s=3,
            data=dict(
                errors=[],
                samples=[
                    dict(time_s=t, engines={"d": {"counter": v}})
                    for t, v in [(0, 10), (1, 11), (2, 0), (3, 1)]
                ],
            ),
        )
        with self.assertRaisesRegex(ValueError, "counter reset"):
            balance.deltas(balance.points(window, ["d"], "counter"))
        window["data"]["samples"] = window["data"]["samples"][:1]
        with self.assertRaisesRegex(ValueError, "insufficient"):
            balance.points(window, ["d"], "counter")


if __name__ == "__main__":
    unittest.main()
