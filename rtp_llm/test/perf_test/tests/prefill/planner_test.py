"""Performance framework behavior and failure regressions."""

import json
import tempfile
import unittest
from pathlib import Path

from rtp_llm.test.perf_test.config_resolver import resolve_config
from rtp_llm.test.perf_test.config_schema import parse_config
from rtp_llm.test.perf_test.errors import PlanningError
from rtp_llm.test.perf_test.prefill.planner import plan_prefill_cases
from rtp_llm.test.perf_test.tests.support.fixtures import (
    fixture_config,
    fixture_engine,
    fixture_workload,
)


class PrefillPlannerTest(unittest.TestCase):

    def test_all_prefill_sources_allow_non_block_aligned_commit_tails(self):
        from rtp_llm.test.perf_test.case_planning import plan_cases as plan_workload

        with tempfile.TemporaryDirectory() as directory:
            Path(directory, "distribution.json").write_text(
                json.dumps(
                    dict(
                        schema_version=1,
                        kind="conditional_request_lengths",
                        input_bucket_width=128,
                        reuse_ratio_bins=2,
                        source_rows=10,
                        incomplete_rows=0,
                        cells=[
                            dict(
                                input_bucket=6,
                                reuse_bin=1,
                                count=10,
                                input_min=768,
                                input_max=895,
                            )
                        ],
                    )
                )
            )
            sources = (
                dict(
                    max_input_len=2048,
                    compute_steps=[256],
                    batch_sizes=[1],
                    cache_lengths=[256],
                ),
                dict(
                    request_batches=[
                        dict(
                            request_groups=[
                                dict(count=1, input_len=1024, cache_len=256)
                            ]
                        )
                    ]
                ),
                dict(
                    random_request_batches=dict(
                        num_cases=2,
                        batch_sizes=[1],
                        min_input_tokens=768,
                        max_input_tokens=1536,
                        input_alignment=256,
                        cold_probability=0,
                    )
                ),
                dict(
                    traffic_request_batches=dict(
                        distribution_file="distribution.json",
                        num_cases=2,
                        batch_sizes=[1],
                        min_input_tokens=768,
                        max_input_tokens=895,
                    )
                ),
            )
            for spec in sources:
                with self.subTest(source=next(iter(spec))):
                    source = prefill_planner_payload()
                    source["engine"]["args"] = {"seq_size_per_block": 256}
                    source["workload"] = dict(
                        kind="cache_grid",
                        cache_alignment=128,
                        commit_tail_tokens=128,
                        workspace_tokens=8192,
                        **spec
                    )
                    resolved = resolve_config(source, source_dir=directory)
                    cases = plan_workload("prefill", source, resolved)
                    self.assertTrue(cases)
                    for case in cases:
                        self.assertEqual(0, case["cache_len"] % 256)
                        for group in case.get("request_groups", ()):
                            self.assertEqual(0, group["cache_len"] % 256)
                    source["model"]["topology"] = dict(tp=1, dp=2, pp=1, ep=2)
                    with self.assertRaisesRegex(
                        PlanningError, "model.topology.dp=1.*DP replica"
                    ):
                        plan_workload(
                            "prefill",
                            source,
                            resolve_config(source, source_dir=directory),
                        )

    def test_generated_cache_lengths_must_align_to_physical_blocks(self):
        payload = prefill_planner_payload()
        payload["engine"]["args"] = {"seq_size_per_block": 256}
        payload["workload"].update(commit_tail_tokens=256, cache_lengths=[64])
        (workload, engine) = prefill_planner_parsed(payload)
        with self.assertRaisesRegex(PlanningError, "cache length 64.*aligned"):
            plan_prefill_cases(workload, engine)

    def test_generated_commit_tails_can_straddle_a_physical_block(self):
        for tail in (128, 384):
            with self.subTest(tail=tail):
                payload = prefill_planner_payload()
                payload["engine"]["args"] = {"seq_size_per_block": 256}
                payload["workload"]["commit_tail_tokens"] = tail
                workload, engine = prefill_planner_parsed(payload)
                cases = plan_prefill_cases(workload, engine)
                self.assertEqual({0, 512}, {case["cache_len"] for case in cases})
                self.assertTrue(all(case["cache_len"] % 256 == 0 for case in cases))

    def test_generated_random_cache_lengths_use_physical_block_stride(self):
        payload = prefill_planner_payload()
        payload["engine"]["args"] = {"seq_size_per_block": 256}
        payload["workload"].update(
            commit_tail_tokens=256, cache_lengths=[], random_cache_count=6
        )
        (workload, engine) = prefill_planner_parsed(payload)
        cases = plan_prefill_cases(workload, engine)
        self.assertTrue(cases)
        self.assertTrue(all((case["cache_len"] % 256 == 0 for case in cases)))

    def test_random_batches_are_deterministic_unique_and_admitted(self):
        payload = prefill_planner_payload()
        payload["engine"].update(
            max_seq_len=1025, max_batch_tokens=900, concurrency_limit=3
        )
        payload["workload"] = {
            "kind": "cache_grid",
            "cache_alignment": 64,
            "commit_tail_tokens": 64,
            "workspace_tokens": 1024,
            "seed": 42,
            "random_request_batches": {
                "num_cases": 12,
                "batch_sizes": [2, 3],
                "min_input_tokens": 128,
                "max_input_tokens": 512,
                "input_alignment": 64,
                "cold_probability": 0.5,
                "batch_input_limits": {"3": 256},
                "kv_budget_tokens": 1024,
            },
        }
        (workload, engine) = prefill_planner_parsed(payload)
        cases = plan_prefill_cases(workload, engine)
        self.assertEqual(cases, plan_prefill_cases(workload, engine))
        self.assertEqual({2, 3}, {case["batch_size"] for case in cases})
        shapes = set()
        for case in cases:
            self.assertEqual(
                "concurrent_grouped_batch_v1", case["measurement_contract"]
            )
            groups = case["request_groups"]
            shape = tuple(sorted(((g["input_len"], g["cache_len"]) for g in groups)))
            self.assertGreater(len(set(shape)), 1)
            shapes.add(shape)
            self.assertLessEqual(sum((g["input_len"] + 1 for g in groups)), 900)
            self.assertLessEqual(
                sum(((g["input_len"] + 64) // 64 * 64 for g in groups)), 1024
            )
            self.assertLessEqual(
                sum(
                    (
                        (g["input_len"] + 64) // 64 * 64 + (64 if g["cache_len"] else 0)
                        for g in groups
                    )
                ),
                1024,
            )
            for g in groups:
                self.assertEqual(1, g["count"])
                self.assertEqual(0, g["input_len"] % 64)
                self.assertEqual(0, g["cache_len"] % 64)
                self.assertLessEqual(g["cache_len"], g["input_len"] - 64)
                if case["batch_size"] == 3:
                    self.assertLessEqual(g["input_len"], 256)
        self.assertEqual(len(cases), len(shapes))

    def test_generated_probe_must_fit_each_budget(self):
        for limit in ("max_seq_len", "max_batch_tokens", "workspace_tokens"):
            with self.subTest(limit=limit):
                payload = prefill_planner_payload()
                payload["workload"].update(
                    max_input_len=32,
                    min_compute_len=8,
                    compute_steps=[8],
                    cache_lengths=[0],
                    batch_sizes=[1],
                    commit_tail_tokens=64,
                    cache_alignment=8,
                )
                payload["engine"]["args"] = {"seq_size_per_block": 64}
                target = "workload" if limit == "workspace_tokens" else "engine"
                payload[target][limit] = 64
                workload, engine = prefill_planner_parsed(payload)
                with self.assertRaisesRegex(PlanningError, "probe.*" + limit):
                    plan_prefill_cases(workload, engine)

        payload = prefill_planner_payload()
        payload["workload"]["workspace_tokens"] = 256
        workload, engine = prefill_planner_parsed(payload)
        with self.assertRaisesRegex(PlanningError, "seed.*workspace_tokens"):
            plan_prefill_cases(workload, engine)

    def test_fixed_cache_compute_sweep_is_deterministic_and_deduplicated(self):
        (workload, engine) = prefill_planner_parsed()
        first = plan_prefill_cases(workload, engine)
        second = plan_prefill_cases(workload, engine)
        self.assertEqual(first, second)
        geometries = [
            (case["batch_size"], case["input_len"], case["cache_len"]) for case in first
        ]
        self.assertEqual(len(geometries), len(set(geometries)))
        self.assertEqual({1, 2}, {case["batch_size"] for case in first})
        self.assertEqual({0, 512}, {case["cache_len"] for case in first})
        for case in first:
            self.assertEqual(case["input_len"] - case["cache_len"], case["compute_len"])
            self.assertGreaterEqual(case["compute_len"], 256)
            self.assertLessEqual(case["input_len"], 2048)
            self.assertEqual(3, case["measure_runs"])
            self.assertEqual(17, case["seed"])

    def test_named_sampled_grids_preserve_top_level_case_geometry(self):
        from rtp_llm.test.perf_test.case_planning import plan_cases

        examples = Path(__file__).resolve().parents[2] / "examples"
        source = json.loads((examples / "prefill_multi_grid_local.json").read_text())
        for grid in source["workload"]["grids"]:
            keys = set(grid) & {"random_request_batches", "traffic_request_batches"}
            if not keys:
                continue
            key = keys.pop()
            with self.subTest(source=key):
                common = {k: v for (k, v) in source["workload"].items() if k != "grids"}
                top = dict(source, workload=dict(common, **{key: grid[key]}))
                named = dict(source, workload=dict(common, grids=[grid]))
                left = resolve_config(top, source_dir=examples)
                right = resolve_config(named, source_dir=examples)
                left_cases = plan_cases("prefill", top, left)
                right_cases = plan_cases("prefill", named, right)
                self.assertEqual(len(left_cases), len(right_cases))
                for a, b in zip(left_cases, right_cases):
                    self.assertEqual(a["request_groups"], b["request_groups"])


def prefill_planner_payload():
    return fixture_config(
        engine=fixture_engine(max_seq_len=4097, concurrency_limit=2),
        workload=fixture_workload(
            commit_tail_tokens=128,
            compute_steps=[512, 256],
            min_compute_len=256,
            cache_lengths=[0, 512],
            batch_sizes=[1, 2],
            measure_runs=3,
            seed=17,
            max_cases=100,
        ),
        execution={"backend": "local"},
    )


def prefill_planner_parsed(payload=None):
    config = parse_config(payload or prefill_planner_payload())
    return (config.workload, config.engine)


if __name__ == "__main__":
    unittest.main()
