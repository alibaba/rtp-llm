"""Rebalance profile contracts through actual Python RPC consumers."""

import json
import sys
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace as NS

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import test_scenario_elastic_lifecycle as original
from environment_expectations import environment as expected_environment
from flexlb_cfg import render_env
from flexlb_test_framework.scenario.backend import make_env_spec


class ProfileTests(unittest.TestCase):
    def driver(self, new_count=20, opposite=False, error=False):
        owner = self

        def factory(ops, environment, state, clock):
            lock = threading.Lock()
            barrier = threading.Barrier(10)
            state["methods"] = []
            state["shapes"] = {}
            batch = environment["effective_axes"]["dispatcher"] == "BATCH"

            class Stream:
                def cancel(self):
                    return True

                def __iter__(self):
                    yield NS(
                        HasField=lambda name: error,
                        error_info=NS(error_code=8431, error_message="fixture"),
                        flatten_output=NS(finished=[not error]),
                    )

            class Future:
                def __init__(self, response, index):
                    self.response, self.index = response, index

                def cancel(self):
                    return True

                def result(self):
                    if self.index % 50 < 10:
                        barrier.wait(timeout=3)
                    return self.response

            def build(rid, input_len=2048, output_len=2, block_keys=None):
                shape = dict(
                    input_len=input_len, output_len=output_len, block_keys=block_keys
                )
                state["shapes"][rid] = shape
                return NS(rid=rid, **shape)

            def schedule(req, timeout):
                owner.assertEqual(timeout, 30)
                owner.assertEqual(req.input_len, 2048)
                owner.assertEqual(req.output_len, 2)
                owner.assertEqual(req.block_keys, [req.rid * 100 + j for j in range(3)])
                with lock:
                    i = state["batches"]
                    state["batches"] += 1
                    name = (
                        "prefill-2"
                        if i >= 50 and i - 50 < new_count
                        else f"prefill-{i%2}"
                    )
                    state["engines"][name]["accepted"] += 1
                return Future(
                    NS(
                        code=200,
                        success=True,
                        error_message="",
                        target=state["engines"][name]["grpc_addr"],
                        enqueued_by_master=batch != opposite,
                    ),
                    i,
                )

            def fetch(req, timeout):
                owner.assertEqual(timeout, 15)
                state["methods"].append("FetchResponse")
                return Stream()

            def genbuild(rid, **shape):
                owner.assertEqual(shape, state["shapes"][rid])
                return NS(rid=rid, roles=False, **shape)

            def roles(req, response):
                req.roles = True

            def generate(req, timeout):
                owner.assertEqual(timeout, 15)
                owner.assertTrue(req.roles)
                state["methods"].append("GenerateStreamCall")
                return Stream()

            ops.build_schedule_request = build
            ops.build_generate_input = genbuild
            ops._copy_role_addrs = roles
            ops.master_target = lambda: "master"
            ops._channel = lambda x: x
            ops.prefill_addr = lambda r: r.target
            ops.schedule_pb2_grpc = NS(
                FlexlbServiceStub=lambda ch: NS(Schedule=NS(future=schedule))
            )
            ops.pb2_grpc = NS(
                RpcServiceStub=lambda ch: NS(
                    FetchResponse=fetch, GenerateStreamCall=generate
                )
            )
            ops.pb2 = NS(FetchRequestPB=lambda **kw: NS(**kw))

        return factory

    def run_case(self, profile="batch-window", **kwargs):
        variant = (
            "rebalance"
            if profile == "batch-window"
            else "rebalance_" + profile.replace("-", "_")
        )
        delayed = kwargs.pop("delayed_old", 0)
        return original.LifecycleTests().run_program(
            variant=variant,
            profile=profile,
            driver_factory=self.driver(**kwargs),
            delayed_old=delayed,
        )

    def test_four_profiles_complete_old_config_and_actual_ten_concurrent_consumers(
        self,
    ):
        for profile in [
            "batch-window",
            "single-batch",
            "single-nonbatch",
            "window-nonbatch",
        ]:
            with self.subTest(profile=profile):
                result, state = self.run_case(profile)
                self.assertEqual(result["status"], "PASS", result)
                old = expected_environment("master_kill", profile)
                plan = state["plan"]
                self.assertEqual(
                    plan["environment"]["resolved_config"],
                    old.resolved_config,
                )
                spec = make_env_spec(
                    plan["environment"], profile, {"master_base": 28000}
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
                expected = (
                    "GenerateStreamCall" if "nonbatch" in profile else "FetchResponse"
                )
                self.assertEqual(state["methods"], [expected] * 100)
                self.assertEqual([len(a) for a in state["artifacts"]], [50, 50])
                self.assertTrue(
                    all(
                        r["business_finished"]
                        and r["transport_terminal_s"] is not None
                        and r["consumer_exit_s"] is not None
                        for a in state["artifacts"]
                        for r in a
                    )
                )
                self.assertEqual(state["flow_count"], 0)
                self.assertEqual(plan["resource_budget"]["max_dynamic_additions"], 1)

    def test_zero_and_exact_sixty_percent_fail_but_below_passes(self):
        for n, status in [(0, "FAIL"), (29, "PASS"), (30, "FAIL")]:
            result, _ = self.run_case(new_count=n)
            row = next(s for s in result["stages"] if s["id"] == "rebalance_share")
            self.assertEqual(row["status"], status, result)

    def test_old_counter_anchor_includes_delayed_accepts_during_add(self):
        result, _ = self.run_case(new_count=30, delayed_old=10)
        self.assertEqual(result["status"], "PASS", result)
        row = next(s for s in result["stages"] if s["id"] == "rebalance_share")
        check = next(c for c in row["checks"] if c["id"] == "new_share")
        self.assertEqual(check["actual"], 0.5)
        self.assertEqual(check["evidence"]["denominator"], 60)

    def test_wrong_response_protocol_cannot_pass_successful_streams(self):
        for p in ["single-batch", "single-nonbatch", "window-nonbatch"]:
            result, _ = self.run_case(p, opposite=True)
            row = next(s for s in result["stages"] if s["id"] == "rebalance_baseline")
            checks = {c["id"]: c["status"] for c in row["checks"]}
            self.assertEqual(
                checks, dict(complete="PASS", no_errors="PASS", protocol="FAIL"), result
            )

    def test_business_error_is_not_protocol_success(self):
        result, _ = self.run_case("window-nonbatch", error=True)
        row = next(s for s in result["stages"] if s["id"] == "rebalance_baseline")
        self.assertEqual(
            {c["id"]: c["status"] for c in row["checks"]},
            dict(complete="PASS", no_errors="FAIL", protocol="PASS"),
            result,
        )
