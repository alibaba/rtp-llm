"""Performance framework behavior and failure regressions."""

import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from rtp_llm.test.perf_test.config_resolver import load_resolved_config
from rtp_llm.test.perf_test.config_schema import parse_config
from rtp_llm.test.perf_test.decode.model import DecodeCaseSpec
from rtp_llm.test.perf_test.decode.planner import plan_decode_cases
from rtp_llm.test.perf_test.errors import PlanningError


class DecodePlannerTest(unittest.TestCase):

    def test_fixed_batches_reject_uneven_active_dp_workers(self):
        grid, _ = decode_planner_resolved_grid(inputs=[128], outputs=[16], sizes=[1])
        distribution, resolved = decode_planner_resolved_distribution(
            distribution=[64, 96, 128], sizes=[3]
        )
        for config, workload in (
            (grid, None),
            (distribution, resolved.resolved_config["workload"]),
        ):
            with self.subTest(kind=config.workload.kind):
                with self.assertRaisesRegex(PlanningError, "uniform"):
                    plan_decode_cases(
                        config.workload,
                        config.engine,
                        dp_size=2,
                        resolved_workload=workload,
                    )

    def test_rank_batch_grid_is_canonical_global_batch(self):
        payload = decode_planner_payload(
            {
                "kind": "grid",
                "input_lengths": [128],
                "output_lengths": [16],
                "batch": {"strategy": "fixed", "sizes": [2, 4], "unit": "per_dp_rank"},
            }
        )
        config = parse_config(payload)
        cases = plan_decode_cases(config.workload, config.engine, dp_size=4)
        self.assertEqual([8, 16], [case["batch_policy"]["size"] for case in cases])
        self.assertEqual([8, 16], [len(case["input_lengths"]) for case in cases])
        self.assertEqual(["per_dp_rank"] * 2, [case["batch_unit"] for case in cases])

    def test_exact_grid_is_input_by_output_by_batch(self):
        (config, resolved) = decode_planner_resolved_grid(
            inputs=[128, 512], outputs=[16, 64], sizes=[2, 4]
        )
        cases = plan_decode_cases(
            config.workload,
            config.engine,
            dp_size=2,
            resolved_workload=resolved.resolved_config["workload"],
        )
        specs = [DecodeCaseSpec.from_identity(case) for case in cases]
        self.assertEqual(8, len(specs))
        self.assertEqual(
            [
                (128, 16, 2),
                (128, 16, 4),
                (128, 64, 2),
                (128, 64, 4),
                (512, 16, 2),
                (512, 16, 4),
                (512, 64, 2),
                (512, 64, 4),
            ],
            [
                (item.input_lengths[0], item.output_length, item.batch_policy.size)
                for item in specs
            ],
        )
        self.assertEqual((128, 128, 128, 128), specs[1].input_lengths)

    def test_capacity_is_checked_per_dp_worker(self):
        (config, _) = decode_planner_resolved_grid(
            inputs=[128], outputs=[16], sizes=[4]
        )
        engine = replace(config.engine, concurrency_limit=2, max_batch_tokens=256)
        self.assertEqual(1, len(plan_decode_cases(config.workload, engine, dp_size=2)))
        for overrides, field in (
            ({"concurrency_limit": 1}, "concurrency_limit"),
            ({"max_batch_tokens": 255}, "max_batch_tokens"),
        ):
            with self.subTest(field=field):
                with self.assertRaisesRegex(PlanningError, field):
                    plan_decode_cases(
                        config.workload, replace(engine, **overrides), dp_size=2
                    )

        distribution, resolved = decode_planner_resolved_distribution(
            distribution=[128], sizes=[1]
        )
        for config, resolved_workload in (
            (config, None),
            (distribution, resolved.resolved_config["workload"]),
        ):
            for output, runs, steps in (
                (1, 1, 3),
                (2, 1, 3),
                (64, 1, 100),
                (2, 2, 5),
                (32, 2, 50),
                (1600, 1, 1500),
            ):
                workload = config.workload
                if workload.kind == "grid":
                    workload = replace(
                        workload, output_lengths=(max(64, output), output)
                    )
                else:
                    workload = replace(workload, output_length=output)
                profiling = replace(
                    workload.profiling, enabled=True, runs=runs, steps=steps
                )
                with self.subTest(
                    kind=workload.kind, output=output, runs=runs, steps=steps
                ):
                    with self.assertRaisesRegex(
                        PlanningError, "profiling.steps.*" + str(min(output, 1000))
                    ):
                        plan_decode_cases(
                            replace(workload, profiling=profiling),
                            config.engine,
                            dp_size=1,
                            resolved_workload=resolved_workload,
                        )
                    for enabled, valid_steps in (
                        (False, steps),
                        (True, min(output, 1000)),
                    ):
                        cases = plan_decode_cases(
                            replace(
                                workload,
                                profiling=replace(
                                    profiling, enabled=enabled, steps=valid_steps
                                ),
                            ),
                            config.engine,
                            dp_size=1,
                            resolved_workload=resolved_workload,
                        )
                        self.assertTrue(cases)
                        self.assertTrue(
                            all(
                                case["profiling"]["steps"] == valid_steps
                                for case in cases
                            )
                        )

            for args, env in (
                ({"sp_type": "mtp", "gen_num_per_cycle": 5}, {}),
                ({"sp_type": "eagle"}, {}),
                ({"sp_type": "dspark"}, {}),
                ({}, {"SP_TYPE": "mtp"}),
                ({"sp_ty": "mtp"}, {}),
            ):
                engine = replace(config.engine, args=args, env=env)
                profiling = replace(config.workload.profiling, enabled=True, steps=2)
                with self.subTest(kind=config.workload.kind, args=args, env=env):
                    with self.assertRaisesRegex(PlanningError, "SP/MTP.*steps=1"):
                        plan_decode_cases(
                            replace(config.workload, profiling=profiling),
                            engine,
                            dp_size=1,
                            resolved_workload=resolved_workload,
                        )
                    for enabled, steps in ((True, 1), (False, 32)):
                        self.assertTrue(
                            plan_decode_cases(
                                replace(
                                    config.workload,
                                    profiling=replace(
                                        profiling, enabled=enabled, steps=steps
                                    ),
                                ),
                                engine,
                                dp_size=1,
                                resolved_workload=resolved_workload,
                            )
                        )
            # An explicit CLI value takes precedence over its paired env value.
            for args, env in (
                ({"sp_type": "none"}, {"SP_TYPE": "mtp"}),
                ({"sp_type": ""}, {"SP_TYPE": "mtp"}),
                ({}, {"SP_TYPE": "none"}),
            ):
                self.assertTrue(
                    plan_decode_cases(
                        replace(
                            config.workload,
                            profiling=replace(
                                config.workload.profiling, enabled=True, steps=3
                            ),
                        ),
                        replace(config.engine, args=args, env=env),
                        dp_size=1,
                        resolved_workload=resolved_workload,
                    )
                )

    def test_distribution_fixed_prefixes_and_maps_follow_batch_order(self):
        for distribution, source_kind in (
            ([64, 96, 128], "sequence_lengths"),
            ({"sequence_lengths": [64, 96, 128]}, "sequence_lengths"),
            (
                {
                    "description": "generated",
                    "source_csv": "requests.csv",
                    "sampling_params": {},
                    "batch_seq_len_map": {"1": [64], "3": [64, 96, 128], "4": [64] * 4},
                },
                "batch_seq_len_map",
            ),
        ):
            with self.subTest(source_kind=source_kind):
                (config, resolved) = decode_planner_resolved_distribution(
                    distribution=distribution, sizes=[3, 1]
                )
                cases = plan_decode_cases(
                    config.workload,
                    config.engine,
                    dp_size=1,
                    resolved_workload=resolved.resolved_config["workload"],
                )
                self.assertEqual(
                    [[64, 96, 128], [64]], [case["input_lengths"] for case in cases]
                )
                self.assertEqual(
                    {"strategy": "fixed", "size": 3}, cases[0]["batch_policy"]
                )
                self.assertEqual({"source_kind": source_kind}, cases[0]["distribution"])
                self.assertTrue(all((case["output_length"] == 16 for case in cases)))


class DecodeDistributionTest(unittest.TestCase):

    def test_materialized_sequence_lengths_generate_deterministic_actual_cases(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "requests.json").write_text(
                json.dumps({"sequence_lengths": [64, 128, 256, 512]}), encoding="utf-8"
            )
            config_path = root / "perf.json"
            config_path.write_text(
                json.dumps(decode_distribution_source()), encoding="utf-8"
            )
            resolved = load_resolved_config(config_path)
            parsed = parse_config(decode_distribution_source())
            first = plan_decode_cases(
                parsed.workload,
                parsed.engine,
                dp_size=1,
                resolved_workload=resolved.resolved_config["workload"],
            )
            second = plan_decode_cases(
                parsed.workload,
                parsed.engine,
                dp_size=1,
                resolved_workload=resolved.resolved_config["workload"],
            )
            self.assertEqual(first, second)
            self.assertEqual(
                [[64, 128], [64, 128, 256]], [case["input_lengths"] for case in first]
            )
            self.assertEqual([2, 3], [case["batch_policy"]["size"] for case in first])
            self.assertEqual({32}, {case["output_length"] for case in first})


import json
import tempfile
from pathlib import Path

from rtp_llm.test.perf_test.config_resolver import resolve_config
from rtp_llm.test.perf_test.config_schema import parse_config
from rtp_llm.test.perf_test.tests.support.fixtures import fixture_config, fixture_model


def decode_planner_payload(workload):
    return fixture_config(
        model=fixture_model(label="test-model"),
        workload=workload,
        execution={"backend": "local"},
    )


def decode_planner_resolved_grid(*, inputs, outputs, sizes):
    payload = decode_planner_payload(
        {
            "kind": "grid",
            "input_lengths": inputs,
            "output_lengths": outputs,
            "batch": {"strategy": "fixed", "sizes": sizes},
        }
    )
    return (parse_config(payload), resolve_config(payload))


def decode_planner_resolved_distribution(*, distribution, sizes):
    batch = {"strategy": "fixed", "sizes": sizes}
    payload = decode_planner_payload(
        {
            "kind": "distribution",
            "distribution_file": "requests.json",
            "output_length": 16,
            "batch": batch,
        }
    )
    with tempfile.TemporaryDirectory() as directory:
        Path(directory, "requests.json").write_text(json.dumps(distribution))
        return (parse_config(payload), resolve_config(payload, source_dir=directory))


def decode_distribution_source(distribution_file="requests.json"):
    return fixture_config(
        workload={
            "kind": "distribution",
            "distribution_file": distribution_file,
            "output_length": 32,
            "batch": {"strategy": "fixed", "sizes": [2, 3]},
            "measure_runs": 2,
            "warmup_runs": 1,
            "generate_config": {"top_k": 1},
        }
    )


if __name__ == "__main__":
    unittest.main()
