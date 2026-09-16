"""Explicit lifecycle cohorts and literal nonbatch owner observations."""

import json
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import test_scenario_elastic_lifecycle as original
from environment_expectations import environment as expected_environment
from flexlb_cfg import render_env
from flexlb_test_framework.scenario.actions import elastic_combined as combined
from flexlb_test_framework.scenario.actions import elastic_lifecycle as life
from flexlb_test_framework.scenario.backend import make_env_spec


class ProfileTests(unittest.TestCase):
    def driver(
        self,
        opposite=False,
        probe_error=False,
        flow_error=False,
        late_probe=False,
        recovery_mode=None,
    ):
        owner = self

        def factory(ops, environment, state, clock):
            batch = environment["effective_axes"]["dispatcher"] == "BATCH"
            owner.fixture_state = state
            owner.fixture_clock = clock
            state["recovery_release"] = threading.Event()
            state["recovery_blocked"] = threading.Event()
            state["protocol_calls"] = []
            state["shapes"] = {}
            state["rpc_sources"] = {}

            class Stream:
                def __init__(self, rid):
                    self.rid = rid

                def cancel(self):
                    return True

                def __iter__(self):
                    source = state["rpc_sources"][self.rid]
                    # Only admission probes fail; the terminal recovery remains healthy.
                    bad = (probe_error and source == "probe" and state["adds"] < 4) or (
                        flow_error and source == "flow" and state["flow_count"] == 2
                    )
                    yield NS(
                        HasField=lambda name: bad,
                        error_info=NS(error_code=8431, error_message="fixture"),
                        flatten_output=NS(finished=[not bad]),
                    )

            class RecoveryStream:
                def cancel(self):
                    state["recovery_release"].set()
                    return True

                def __iter__(self):
                    finished = NS(
                        HasField=lambda name: False, flatten_output=NS(finished=[True])
                    )
                    if recovery_mode == "typed_finished":
                        yield NS(
                            HasField=lambda name: True,
                            error_info=NS(error_code=8431, error_message="typed"),
                            flatten_output=NS(finished=[True]),
                        )
                        return
                    if recovery_mode == "finished_open":
                        yield finished
                    state["recovery_blocked"].set()
                    state["recovery_release"].wait(2)
                    if recovery_mode == "open_two_then_29":
                        clock.now += 29
                    if recovery_mode != "finished_open":
                        yield finished

            def stream_for(rid, timeout):
                if timeout == 60 and recovery_mode:
                    if recovery_mode == "open_two_then_29":
                        clock.now += 2
                    state["recovery_open_return_s"] = clock()
                    return RecoveryStream()
                return Stream(rid)

            def build(rid, input_len=2048, output_len=2, block_keys=None):
                owner.assertEqual(input_len, 2048)
                owner.assertEqual(output_len, 2)
                owner.assertEqual(block_keys, [rid * 100 + 1])
                shape = dict(
                    input_len=input_len, output_len=output_len, block_keys=block_keys
                )
                state["shapes"][rid] = shape
                return NS(rid=rid, **shape)

            def schedule(req, timeout):
                owner.assertEqual(timeout, 30)
                state["rpc_sources"][req.rid] = state.get("request_source") or "probe"
                names = sorted(state["engines"])
                name = names[req.rid % len(names)]
                if (
                    late_probe
                    and state["rpc_sources"][req.rid] == "probe"
                    and len(names) == 3
                ):
                    name = next(n for n in names if n not in ["prefill-0", "prefill-1"])
                state["engines"][name]["accepted"] += 1
                state["batches"] += 1
                response = NS(
                    code=200,
                    success=True,
                    error_message="",
                    target=state["engines"][name]["grpc_addr"],
                    enqueued_by_master=batch != opposite,
                )

                def finish():
                    if late_probe and state["rpc_sources"][req.rid] == "probe":
                        clock.now += 16
                    return response

                return NS(result=finish, cancel=lambda: True)

            def fetch(req, timeout):
                owner.assertIn(timeout, [10, 60])
                state["protocol_calls"].append(("FetchResponse", req.request_id))
                return stream_for(req.request_id, timeout)

            def genbuild(rid, **kwargs):
                shape = dict(input_len=2048, **kwargs)
                owner.assertEqual(shape, state["shapes"][rid])
                return NS(rid=rid, roles=False, **shape)

            def generate(req, timeout):
                owner.assertIn(timeout, [10, 60])
                owner.assertTrue(req.roles)
                state["protocol_calls"].append(("GenerateStreamCall", req.rid))
                return stream_for(req.rid, timeout)

            ops.build_schedule_request = build
            ops.build_generate_input = genbuild
            ops._copy_role_addrs = lambda req, response: setattr(req, "roles", True)
            ops._channel = lambda x: x
            ops.master_target = lambda: "master"
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

    def run_case(self, profile="batch-window", grade="normal", **kwargs):
        variant = (
            grade
            if profile == "batch-window"
            else grade + "_" + profile.replace("-", "_")
        )
        return original.LifecycleTests().run_program(
            variant=variant, profile=profile, driver_factory=self.driver(**kwargs)
        )

    def test_four_profiles_both_grades_actual_consumers_and_separate_flow_cohorts(self):
        for profile in [
            "batch-window",
            "single-batch",
            "single-nonbatch",
            "window-nonbatch",
        ]:
            for grade in ["normal", "strict"]:
                with self.subTest(profile=profile, grade=grade):
                    result, state = self.run_case(profile, grade)
                    self.assertEqual(result["status"], "PASS", result)
                    old = expected_environment("master_kill", profile)
                    plan = state["plan"]
                    spec = make_env_spec(
                        plan["environment"], profile, {"master_base": 28000}
                    )
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
                    expected = (
                        "GenerateStreamCall"
                        if "nonbatch" in profile
                        else "FetchResponse"
                    )
                    self.assertEqual(
                        {m for m, _ in state["protocol_calls"]}, {expected}
                    )
                    self.assertEqual(len(state["flow_records"]), 5)
                    self.assertTrue(
                        all(
                            rows
                            and all(
                                r["business_finished"]
                                and r["consumer_exit_s"] is not None
                                and r["transport_terminal_s"] is not None
                                for r in rows
                            )
                            for rows in state["flow_records"]
                        )
                    )
                    ids = [s["id"] for s in plan["stages"]]
                    self.assertLess(
                        ids.index("remove_flow"), ids.index("remove_traffic")
                    )
                    for n in range(1, 4):
                        self.assertLess(
                            ids.index(f"cycle{n}_traffic"), ids.index(f"cycle{n}_flow")
                        )
                        self.assertLess(
                            ids.index(f"cycle{n}_flow"), ids.index(f"cycle{n}_remove")
                        )
                    self.assertEqual(
                        plan["resource_budget"]["max_dynamic_additions"], 4
                    )

    def test_probe_business_error_does_not_pollute_background_zero_error_cohort(self):
        result, state = self.run_case("single-nonbatch", probe_error=True)
        self.assertEqual(result["status"], "PASS", result)
        self.assertTrue(
            all(r["business_finished"] for rows in state["flow_records"] for r in rows)
        )
        self.assertTrue(
            any(
                r["business_error_code"] == 8431
                for a in state["probe_artifacts"]
                for r in a["requests"]
            )
        )

    def test_background_failure_still_fails_removal(self):
        result, _ = self.run_case("window-nonbatch", flow_error=True)
        rows = {s["id"]: s for s in result["stages"]}
        self.assertEqual(rows["remove_zero_errors"]["status"], "FAIL", result)
        self.assertEqual(rows["cycle1_add"]["status"], "BLOCKED", result)

    def test_wrong_protocol_success_cannot_pass(self):
        result, _ = self.run_case("single-nonbatch", opposite=True)
        rows = {s["id"]: s for s in result["stages"]}
        self.assertEqual(rows["preference_protocol"]["status"], "FAIL", result)

    def test_literal_nonbatch_batch_zero_does_not_claim_route_owner_zero(self):
        clock = original.Clock()
        with tempfile.TemporaryDirectory() as root:
            ctx = NS(clock=clock, artifact_dir=Path(root))
            deadline = NS(
                remaining=lambda: 100 - clock(), sleep=clock.sleep, check=lambda: None
            )
            data = dict(
                scheduler_inflight=0,
                prefill_endpoints=[dict(inflight_route_requests=7)],
                decode_endpoints=[dict(total_load=0)],
            )
            with patch.object(life, "_master_get", return_value=data):
                result = combined.literal_accounting(ctx, {}, deadline)
            self.assertTrue(all(c.status == "PASS" for c in result.checks))
            self.assertEqual(result.checks[1].id, "prefill_batches")
            self.assertEqual(
                result.checks[1].evidence["samples"][0]["data"]["prefill_endpoints"][0][
                    "inflight_route_requests"
                ],
                7,
            )

    def test_serial_probe_issuance_cutoff_allows_last_schedule_to_finish_late(self):
        result, state = self.run_case("single-batch", late_probe=True)
        self.assertEqual(result["status"], "PASS", result)
        self.assertEqual(len(state["probe_artifacts"]), 4)
        self.assertTrue(
            all(
                a["samples"][-1]["time_s"] > a["issuance_end_s"]
                for a in state["probe_artifacts"]
            )
        )

    def test_recovery_finished_open_success_is_frozen_before_cleanup(self):
        def observed_wait(done, timeout):
            self.assertEqual(timeout, 30)
            self.assertTrue(self.fixture_state["recovery_blocked"].wait(1))
            self.fixture_clock.now += 30
            return False

        with patch.object(combined, "wait_consumer", side_effect=observed_wait):
            result, state = self.run_case(
                "single-nonbatch", recovery_mode="finished_open"
            )
        self.assertEqual(result["status"], "PASS", result)
        artifact = state["recovery_artifacts"][0]
        self.assertTrue(artifact["frozen"]["legacy_success"])
        self.assertIsNone(artifact["frozen"]["record"]["consumer_exit_s"])
        self.assertIsNotNone(artifact["final_records"][0]["consumer_exit_s"])

    def test_recovery_finished_only_after_cancel_cannot_change_frozen_failure(self):
        def observed_wait(done, timeout):
            self.assertTrue(self.fixture_state["recovery_blocked"].wait(1))
            self.fixture_clock.now += 30
            return False

        with patch.object(combined, "wait_consumer", side_effect=observed_wait):
            result, state = self.run_case("single-batch", recovery_mode="late_finished")
        self.assertEqual(result["status"], "FAIL", result)
        artifact = state["recovery_artifacts"][0]
        self.assertFalse(artifact["frozen"]["legacy_success"])
        self.assertTrue(artifact["final_records"][0]["business_finished"])

    def test_recovery_wait_starts_after_rpc_open_returns(self):
        def observed_wait(done, timeout):
            self.assertEqual(timeout, 30)
            self.assertTrue(self.fixture_state["recovery_blocked"].wait(1))
            self.assertEqual(
                self.fixture_clock(), self.fixture_state["recovery_open_return_s"]
            )
            self.fixture_state["recovery_release"].set()
            return done.wait(1)

        with patch.object(combined, "wait_consumer", side_effect=observed_wait):
            result, state = self.run_case(
                "single-batch", recovery_mode="open_two_then_29"
            )
        self.assertEqual(result["status"], "PASS", result)
        artifact = state["recovery_artifacts"][0]
        self.assertEqual(
            artifact["caller_observation_started_s"], artifact["open_return_s"]
        )
        self.assertEqual(
            artifact["frozen"]["time_s"] - artifact["caller_observation_started_s"], 29
        )

    def test_literal_recovery_retains_typed_error_without_reclassifying_expected_success(
        self,
    ):
        result, state = self.run_case("single-nonbatch", recovery_mode="typed_finished")
        self.assertEqual(result["status"], "PASS", result)
        frozen = state["recovery_artifacts"][0]["frozen"]
        self.assertTrue(frozen["legacy_success"])
        self.assertEqual(frozen["record"]["business_error_code"], 8431)
