"""Crossfire profile configs and actual Python consumers with simulated RPCs."""

import json
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace as NS

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import test_scenario_elastic_concurrent as original
from environment_expectations import environment as expected_environment
from flexlb_cfg import render_env
from flexlb_test_framework.scenario import compile_scenarios
from flexlb_test_framework.scenario.actions import elastic as e
from flexlb_test_framework.scenario.backend import make_env_spec
from flexlb_test_framework.scenario.loader import load_scenarios


class ProfileTests(unittest.TestCase):
    def test_complete_old_configs_and_explicit_profile_variants(self):
        plans = compile_scenarios(
            load_scenarios(ROOT / "scenarios/elastic/concurrent_mutation.yaml"),
            handlers={h.name: h for h in e.HANDLERS},
        )
        expected = {
            "batch-window": "default",
            "single-batch": "single_batch",
            "single-nonbatch": "single_nonbatch",
            "window-nonbatch": "window_nonbatch",
        }
        plans = [p for p in plans if not p["variant_id"].endswith("_convergence")]
        self.assertEqual({p["profile"]: p["variant_id"] for p in plans}, expected)
        for plan in plans:
            profile = plan["profile"]
            old = expected_environment("master_kill", profile)
            new = make_env_spec(plan["environment"], profile, {"master_base": 28000})
            self.assertEqual(
                plan["environment"]["resolved_config"],
                old.resolved_config,
            )
            self.assertEqual(new.perf, old.perf)
            for field in [
                "discovery",
                "prefill_cache_blocks",
                "decode_cache_blocks",
            ]:
                self.assertEqual(getattr(new, field), getattr(old, field))
            self.assertEqual((new.n_prefill, new.n_decode), (4, 8))
            self.assertEqual(plan["stages"][1]["params"]["mutation_window_s"], 30)
            self.assertEqual(plan["resource_budget"]["initial_workers"], 12)
            self.assertEqual(plan["resource_budget"]["max_dynamic_additions"], 65)
            self.assertEqual(len(plan["stages"]), 5 if profile == "batch-window" else 6)
            self.assertEqual(
                sum(len(s["check_ids"]) for s in plan["stages"]),
                7 if profile == "batch-window" else 8,
            )

    def driver(self, opposite=False, fail=False):
        owner = self

        def factory(ops, environment, state, clock):
            batch = environment["effective_axes"]["dispatcher"] == "BATCH"
            state["protocol_calls"] = []
            state["schedule_shapes"] = {}
            state["generate_shapes"] = {}

            class Future:
                def __init__(self, response):
                    self.response = response

                def result(self):
                    return self.response

                def cancel(self):
                    return True

            class Stream:
                def cancel(self):
                    return True

                def __iter__(self):
                    yield NS(
                        HasField=lambda name: fail,
                        error_info=NS(
                            error_code=8431, error_message="simulated request failure"
                        ),
                        flatten_output=NS(finished=[not fail]),
                    )

            def build(rid, input_len=2048, output_len=2, block_keys=None, **kwargs):
                shape = dict(
                    input_len=input_len, output_len=output_len, block_keys=block_keys
                )
                state["schedule_shapes"][rid] = shape
                return NS(rid=rid, **shape)

            def schedule(request, timeout):
                owner.assertEqual(timeout, 30)
                owner.assertEqual(request.input_len, 2048)
                owner.assertEqual(request.output_len, 2)
                owner.assertEqual(request.block_keys, [request.rid * 100 + 1])
                return Future(
                    NS(
                        code=200,
                        success=True,
                        error_message="",
                        target=state["engines"]["prefill-0"]["grpc_addr"],
                        enqueued_by_master=batch != opposite,
                    )
                )

            def fetch(request, timeout):
                owner.assertEqual(timeout, 10)
                state["protocol_calls"].append(("FetchResponse", request.request_id))
                return Stream()

            def build_generate(
                rid, input_len=2048, output_len=2, block_keys=None, **kwargs
            ):
                shape = dict(
                    input_len=input_len, output_len=output_len, block_keys=block_keys
                )
                state["generate_shapes"][rid] = shape
                return NS(rid=rid, roles_copied=False, **shape)

            def copy_roles(request, response):
                request.roles_copied = True

            def generate(request, timeout):
                owner.assertEqual(timeout, 10)
                owner.assertTrue(request.roles_copied)
                owner.assertEqual(
                    state["generate_shapes"][request.rid],
                    state["schedule_shapes"][request.rid],
                )
                state["protocol_calls"].append(("GenerateStreamCall", request.rid))
                return Stream()

            ops.schedule_pb2_grpc = NS(
                FlexlbServiceStub=lambda ch: NS(Schedule=NS(future=schedule))
            )
            ops.pb2_grpc = NS(
                RpcServiceStub=lambda ch: NS(
                    FetchResponse=fetch, GenerateStreamCall=generate
                )
            )
            ops.pb2 = NS(FetchRequestPB=lambda **kwargs: NS(**kwargs))
            ops._channel = lambda x: x
            ops.master_target = lambda: "master"
            ops.build_schedule_request = build
            ops.build_generate_input = build_generate
            ops._copy_role_addrs = copy_roles
            ops.prefill_addr = lambda response: response.target
            return ops

        return factory

    def test_full_new_profiles_use_expected_actual_consumer_branch_and_shape(self):
        for profile in ["single-batch", "single-nonbatch", "window-nonbatch"]:
            with self.subTest(profile=profile):
                result, state = original.ConcurrentTests().run_program(
                    profile=profile, driver_factory=self.driver()
                )
                self.assertEqual(result["status"], "PASS", result)
                expected = (
                    "FetchResponse"
                    if profile == "single-batch"
                    else "GenerateStreamCall"
                )
                self.assertEqual({m for m, _ in state["protocol_calls"]}, {expected})
                self.assertGreaterEqual(state["max_active"], 2)
                self.assertGreater(state["removed"], 0)
                rows = next(iter(state["request_artifacts"].values()))["requests"]
                self.assertTrue(rows)
                self.assertTrue(
                    all(
                        r["stream"]["method"] == expected
                        and r["business_finished"]
                        and r["transport_terminal_s"] is not None
                        and r["consumer_exit_s"] is not None
                        for r in rows
                    )
                )
                self.assertGreaterEqual(state["snapshot_s"], state["last_remove_s"])

    def test_opposite_protocol_cannot_pass_by_returning_successful_streams(self):
        for profile in ["single-batch", "single-nonbatch", "window-nonbatch"]:
            result, state = original.ConcurrentTests().run_program(
                profile=profile, driver_factory=self.driver(opposite=True)
            )
            row = next(s for s in result["stages"] if s["id"] == "protocol")
            self.assertEqual(row["status"], "FAIL", result)
            self.assertEqual(
                next(s for s in result["stages"] if s["id"] == "health")["status"],
                "BLOCKED",
            )
            records = next(iter(state["request_artifacts"].values()))["requests"]
            self.assertTrue(all(r["business_finished"] for r in records))

    def test_nonbatch_failure_still_fails_the_unchanged_health_floor(self):
        result, state = original.ConcurrentTests().run_program(
            profile="window-nonbatch", driver_factory=self.driver(fail=True)
        )
        rows = {s["id"]: s for s in result["stages"]}
        self.assertEqual(rows["protocol"]["status"], "PASS", result)
        self.assertEqual(rows["health"]["status"], "FAIL", result)
        self.assertEqual(
            next(c for c in rows["health"]["checks"] if c["id"] == "success_rate")[
                "status"
            ],
            "FAIL",
        )


if __name__ == "__main__":
    unittest.main()
