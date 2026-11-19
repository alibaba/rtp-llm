"""Performance framework behavior and failure regressions."""

import dataclasses
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from rtp_llm.test.perf_test.case_planning import plan_cases
from rtp_llm.test.perf_test.cli import parse_cli
from rtp_llm.test.perf_test.config_resolver import resolve_config
from rtp_llm.test.perf_test.config_schema import _parse_execution, parse_config
from rtp_llm.test.perf_test.entry import _new_plan
from rtp_llm.test.perf_test.errors import ResourceError
from rtp_llm.test.perf_test.plan import (
    AOneResourcePlan,
    ExecutionPlan,
    LocalResourcePlan,
    bind_member_plan,
    build_execution_plan,
    case_id,
)
from rtp_llm.test.perf_test.tests.support.fixtures import PERF_TEST_ROOT
from rtp_llm.test.perf_test.tests.support.plan import plan_cases, plan_config


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


class LatestPlanContractTest(unittest.TestCase):

    def test_only_latest_plan_schema_is_accepted(self):
        from rtp_llm.test.perf_test.errors import PlanningError

        plan = build_execution_plan(
            mode="prefill",
            action="run",
            resolved=resolve_config(plan_config()),
            cases=plan_cases(),
            result_directory="runs/latest",
        )
        self.assertEqual(2, plan.schema_version)
        raw = plan.to_dict()
        raw["schema_version"] = 1
        with self.assertRaisesRegex(PlanningError, "schema_version"):
            ExecutionPlan.from_dict(raw)


class FrozenPlanIntegrityTest(unittest.TestCase):

    def test_changed_resolved_fields_cannot_keep_original_fingerprint(self):
        from rtp_llm.test.perf_test.errors import PlanningError

        plan = build_execution_plan(
            mode="prefill",
            action="run",
            resolved=resolve_config(plan_config()),
            cases=plan_cases(),
            result_directory="runs/latest",
        )
        for section, field, value in (
            ("engine", "max_seq_len", 9999),
            ("workload", "seed", 999),
        ):
            with self.subTest(section=section):
                raw = plan.to_dict()
                raw["resolved_config"][section][field] = value
                with self.assertRaisesRegex(PlanningError, "fingerprint"):
                    ExecutionPlan.from_dict(raw)


class FrozenBindingTest(unittest.TestCase):

    def test_only_result_path_is_bound(self):
        request = parse_cli(
            [
                "--mode=decode",
                "--action=run",
                "--config=" + str(PERF_TEST_ROOT / "examples/decode_grid_local.json"),
            ]
        )
        (plan, _) = _new_plan(request)
        with patch(
            "rtp_llm.test.perf_test.config_resolver.load_resolved_config",
            side_effect=AssertionError("must not reload source"),
        ):
            bound = bind_member_plan(plan, result_directory="/remote/run")
        expected = dataclasses.replace(plan, result_directory="/remote/run")
        self.assertEqual(expected, bound)
        self.assertEqual(plan.config_fingerprint, bound.config_fingerprint)
        self.assertEqual(plan.workload_fingerprint, bound.workload_fingerprint)
        self.assertEqual(plan.cases, bound.cases)

    def test_relative_binding_is_rejected(self):
        with self.assertRaises(ValueError):
            bind_member_plan(None, result_directory="relative")


if __name__ == "__main__":
    unittest.main()
