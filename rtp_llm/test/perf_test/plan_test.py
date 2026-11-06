"""Core performance behavior contracts."""

import dataclasses
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from rtp_llm.test.perf_test.case_planning import plan_cases
from rtp_llm.test.perf_test.config_resolver import resolve_config
from rtp_llm.test.perf_test.config_schema import _parse_execution, parse_config
from rtp_llm.test.perf_test.decode.model import DecodeCaseSpec
from rtp_llm.test.perf_test.decode.planner import plan_decode_cases
from rtp_llm.test.perf_test.errors import PlanningError, ResourceError
from rtp_llm.test.perf_test.multi_node import decode_runtime, prefill_runtime
from rtp_llm.test.perf_test.plan import (
    AOneResourcePlan,
    ExecutionPlan,
    LocalResourcePlan,
    build_execution_plan,
    case_id,
)
from rtp_llm.test.perf_test.prefill.planner import plan_prefill_cases
from rtp_llm.test.perf_test.test_support import (
    fixture_engine,
    fixture_model,
    fixture_workload,
)


def decode_planner_payload(workload):
    return {
        "schema_version": 1,
        "model": fixture_model(label="test-model"),
        "engine": fixture_engine(),
        "workload": workload,
        "execution": {"backend": "local"},
        "output": {},
        "regression": {},
        "publish": {},
    }


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


def decode_planner_resolved_distribution(*, distribution, max_size=None, sizes=None):
    batch = (
        {"strategy": "fixed", "sizes": sizes}
        if sizes is not None
        else {
            "strategy": "target_tpot",
            "max_size": max_size,
            "candidate_step": 2,
            "target_tpot_ms": 10.0,
        }
    )
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


def prefill_planner_payload():
    return {
        "schema_version": 1,
        "model": fixture_model(),
        "engine": fixture_engine(max_seq_len=4097, concurrency_limit=2),
        "workload": fixture_workload(
            commit_tail_tokens=128,
            compute_steps=[512, 256],
            min_compute_len=256,
            cache_lengths=[0, 512],
            batch_sizes=[1, 2],
            measure_runs=3,
            seed=17,
            max_cases=100,
        ),
        "execution": {"backend": "local"},
        "output": {},
        "regression": {},
        "publish": {},
    }


def prefill_planner_parsed(payload=None):
    config = parse_config(payload or prefill_planner_payload())
    return (config.workload, config.engine)


def plan_config(backend="local"):
    execution = {"backend": "local", "devices": [2, 3]}
    if backend == "aone":
        execution = {
            "backend": "aone",
            "resource": "daily",
            "gpu_type": "H20",
            "max_parallel": 2,
            "shard_count": 2,
        }
    return {
        "schema_version": 1,
        "model": fixture_model(
            label="test-model",
            type="registered_model",
            checkpoint="/models/test",
            topology={"tp": 2, "dp": 1, "pp": 1, "ep": 2},
        ),
        "engine": fixture_engine(),
        "workload": fixture_workload(),
        "execution": execution,
        "output": {},
        "regression": {},
        "publish": {},
    }


def plan_cases():
    return [
        {"input_len": 1024, "cache_len": 512, "batch_size": 1},
        {"batch_size": 1, "cache_len": 0, "input_len": 512},
    ]


class ExecutionPlanTest(unittest.TestCase):

    def test_case_id_uses_only_mode_and_canonical_identity(self):
        left = {"input_len": 1024, "cache_len": 512, "batch_size": 1}
        right = {"batch_size": 1, "cache_len": 512, "input_len": 1024}
        self.assertEqual(case_id("prefill", left), case_id("prefill", right))
        self.assertNotEqual(case_id("prefill", left), case_id("decode", left))
        self.assertNotEqual(
            case_id("prefill", left), case_id("prefill", {**left, "cache_len": 0})
        )

    def test_plan_is_input_order_independent_and_deeply_immutable(self):
        resolved = resolve_config(plan_config())
        forward = build_execution_plan(
            mode="prefill",
            action="run",
            resolved=resolved,
            cases=plan_cases(),
            result_directory="runs/example",
        )
        reverse = build_execution_plan(
            mode="prefill",
            action="run",
            resolved=resolved,
            cases=list(reversed(plan_cases())),
            result_directory="runs/example",
        )
        self.assertEqual(forward, reverse)
        self.assertEqual(
            sorted((case.case_id for case in forward.cases)),
            [case.case_id for case in forward.cases],
        )
        with self.assertRaises(dataclasses.FrozenInstanceError):
            forward.mode = "decode"
        with self.assertRaises(TypeError):
            forward.resolved_config["model"]["label"] = "changed"
        with self.assertRaises(AttributeError):
            forward.cases.append("changed")

    def test_plan_serialization_round_trips(self):
        plan = build_execution_plan(
            mode="prefill",
            action="run",
            resolved=resolve_config(plan_config()),
            cases=plan_cases(),
            result_directory="runs/example",
        )
        restored = ExecutionPlan.from_dict(plan.to_dict())
        self.assertEqual(plan, restored)
        self.assertEqual(plan.to_json(), restored.to_json())
        self.assertIsInstance(restored.resource_plan, LocalResourcePlan)
        self.assertEqual((2, 3), restored.resource_plan.devices)
        self.assertEqual(2, restored.resource_plan.world_size)


class ExtensionSeamsTest(unittest.TestCase):
    """Explicit backend adapters must preserve shared identity and validation."""

    def test_execution_parser_flows_through_resolution_and_file_loading(self):
        import json
        import tempfile
        from pathlib import Path

        from rtp_llm.test.perf_test.config_resolver import load_resolved_config
        from rtp_llm.test.perf_test.config_schema import _parse_execution, parse_config

        source = plan_config()
        expected = resolve_config(source)
        parsed = parse_config(source, execution_parser=_parse_execution)
        self.assertEqual("local", parsed.execution.backend)
        self.assertEqual(
            expected, resolve_config(source, execution_parser=_parse_execution)
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            path.write_text(json.dumps(source))
            self.assertEqual(
                expected, load_resolved_config(path, execution_parser=_parse_execution)
            )

    def test_injected_planner_preserves_identity_and_historical_round_trip(self):
        from rtp_llm.test.perf_test.capabilities import Capability

        source = plan_config("aone")
        source["execution"]["resource"] = "private-pool"
        source["execution"]["gpu_type"] = "private-gpu"

        @dataclasses.dataclass(frozen=True)
        class HistoricalExecution:
            backend: str
            resource: str
            gpu_type: str
            max_parallel: int
            shard_count: int

        resolved = resolve_config(
            source, execution_parser=lambda raw: HistoricalExecution(**raw)
        )

        def planner(*, mode, resolved_config, world_size, case_count):
            self.assertEqual(("prefill", 2, 2), (mode, world_size, case_count))
            execution = resolved_config["execution"]
            return AOneResourcePlan(
                "aone",
                execution["resource"],
                execution["gpu_type"],
                world_size,
                2,
                2,
                world_size,
                1,
                world_size,
            )

        plan = build_execution_plan(
            mode="prefill",
            action="run",
            resolved=resolved,
            cases=plan_cases(),
            result_directory="runs/example",
            resource_planner=planner,
            capability_validator=lambda mode, action, backend: Capability(
                mode, backend, action
            ),
        )
        self.assertEqual(resolved.config_fingerprint, plan.config_fingerprint)
        self.assertEqual(resolved.workload_fingerprint, plan.workload_fingerprint)
        self.assertEqual(
            {case_id("prefill", case) for case in plan_cases()},
            {case.case_id for case in plan.cases},
        )
        self.assertEqual(plan, ExecutionPlan.from_dict(plan.to_dict()))
        damaged = plan.to_dict()
        damaged["resource_plan"]["machine_count"] = 2
        with self.assertRaises(ResourceError):
            ExecutionPlan.from_dict(damaged)


class ExternalPlanPolicyTest(unittest.TestCase):

    def test_explicit_parser_preserves_default_plan_and_source(self):
        root = Path(__file__).parent
        for mode, runtime in (("prefill", prefill_runtime), ("decode", decode_runtime)):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as directory:
                builder = getattr(runtime, f"build_external_{mode}_plan")
                path = (
                    root
                    / "examples"
                    / (
                        "prefill_local.json"
                        if mode == "prefill"
                        else "decode_grid_local.json"
                    )
                )
                import json

                topology = json.loads(path.read_text())["model"]["topology"]
                (expected, source) = builder(path, directory, topology)
                with patch(
                    "rtp_llm.test.perf_test.config_schema._parse_execution",
                    side_effect=AssertionError("default policy forbidden"),
                ):
                    (actual, actual_source) = builder(
                        path, directory, topology, execution_parser=_parse_execution
                    )
                self.assertEqual(expected.to_dict(), actual.to_dict())
                self.assertEqual(source, actual_source)


class PrefillPlannerTest(unittest.TestCase):

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

    def test_generated_probe_must_fit_all_budgets_without_override(self):
        for named in (False, True):
            for inferred in (False, True):
                for limit in ("max_seq_len", "max_batch_tokens", "workspace_tokens"):
                    payload = prefill_planner_payload()
                    payload["workload"].update(
                        max_input_len=32,
                        min_compute_len=8,
                        compute_steps=[8],
                        cache_lengths=[0],
                        batch_sizes=[1],
                        commit_tail_tokens=64,
                        cache_alignment=8 if inferred else 64,
                    )
                    if inferred:
                        payload["engine"]["args"] = {"seq_size_per_block": 64}
                    target = "workload" if limit == "workspace_tokens" else "engine"
                    payload[target][limit] = 64
                    if named:
                        keys = (
                            "max_input_len",
                            "min_compute_len",
                            "compute_steps",
                            "cache_lengths",
                            "batch_sizes",
                            "seed",
                        )
                        grid = {key: payload["workload"].pop(key) for key in keys}
                        payload["workload"]["grids"] = [{"name": "small", **grid}]
                    (workload, engine) = prefill_planner_parsed(payload)
                    with self.subTest(named=named, inferred=inferred, limit=limit):
                        with self.assertRaisesRegex(PlanningError, "probe.*" + limit):
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


class DecodePlannerTest(unittest.TestCase):

    def test_exact_grid_is_input_by_output_by_batch(self):
        (config, resolved) = decode_planner_resolved_grid(
            inputs=[128, 512], outputs=[16, 64], sizes=[1, 4]
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
                (128, 16, 1),
                (128, 16, 4),
                (128, 64, 1),
                (128, 64, 4),
                (512, 16, 1),
                (512, 16, 4),
                (512, 64, 1),
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
            inputs=[128], outputs=[16], sizes=[3]
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

    def test_distribution_fixed_prefixes_and_maps_follow_batch_order(self):
        for distribution, source_kind in (
            ([64, 96, 128], "sequence_lengths"),
            ({"sequence_lengths": [64, 96, 128]}, "sequence_lengths"),
            (
                {"batch_seq_len_map": {"1": [64], "3": [64, 96, 128]}},
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
                    dp_size=2,
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


if __name__ == "__main__":
    unittest.main()
