import io
import json
import tempfile
import unittest
from contextlib import ExitStack
from functools import partial
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from rtp_llm.test.perf_test import entry
from rtp_llm.test.perf_test.aone.config_schema import parse_execution
from rtp_llm.test.perf_test.aone.plan import (
    GangTopology,
    derive_gang_topology,
    plan_resources,
)
from rtp_llm.test.perf_test.capabilities import validate_aone_capability
from rtp_llm.test.perf_test.config_resolver import resolve_config
from rtp_llm.test.perf_test.errors import PlanningError
from rtp_llm.test.perf_test.multi_node import decode_runtime, prefill_runtime
from rtp_llm.test.perf_test.plan import (
    AOneResourcePlan,
    ExecutionPlan,
    build_execution_plan,
)


class AOneEntryTest(unittest.TestCase):

    def test_cli_dry_run_and_invalid_resource_never_create_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = _gang_config(
                _execution(gpus_per_machine=8),
                topology={"tp": 8, "dp": 2, "pp": 1, "ep": 16},
            )
            source["output"]["result_root"] = str(Path(tmp) / "results")
            config = Path(tmp) / "config.json"
            config.write_text(json.dumps(source))
            argv = [
                "--mode=prefill",
                "--action=run",
                "--dry-run",
                "--config",
                str(config),
            ]
            output, errors = io.StringIO(), io.StringIO()
            self.assertEqual(
                0,
                entry.main(
                    argv,
                    stdout=output,
                    stderr=errors,
                    executor_factory=lambda: self.fail("executor"),
                ),
            )
            self.assertEqual(2, json.loads(output.getvalue())["gang"]["machine_count"])
            self.assertFalse((Path(tmp) / "results").exists())
            source["execution"]["resource"] = "unknown"
            config.write_text(json.dumps(source))
            self.assertNotEqual(
                0, entry.main(argv, stdout=io.StringIO(), stderr=errors)
            )
            self.assertIn("unsupported AOne resource", errors.getvalue())
            self.assertFalse((Path(tmp) / "results").exists())


resolve_config = partial(resolve_config, execution_parser=parse_execution)


build_execution_plan = partial(
    build_execution_plan,
    resource_planner=plan_resources,
    capability_validator=validate_aone_capability,
)


def _execution(**fields):
    execution = {
        "backend": "aone",
        "resource": "daily",
        "gpu_type": "H20",
        "max_parallel": 2,
    }
    execution.update(fields)
    return execution


def _gang_config(execution, topology=None, mode="prefill"):
    if mode == "decode":
        workload = {
            "kind": "grid",
            "input_lengths": [128],
            "output_lengths": [16],
            "batch": {"strategy": "fixed", "sizes": [1]},
        }
    else:
        workload = {
            "kind": "cache_grid",
            "max_input_len": 2048,
            "cache_alignment": 64,
            "commit_tail_tokens": 64,
            "compute_steps": [128],
            "batch_sizes": [1],
        }
    return {
        "schema_version": 1,
        "model": {
            "label": "test-model",
            "type": "registered_model",
            "checkpoint": "/models/test",
            "topology": topology or {"tp": 2, "dp": 1, "pp": 1, "ep": 1},
        },
        "engine": {
            "max_seq_len": 4096,
            "max_batch_tokens": 8192,
            "concurrency_limit": 8,
        },
        "workload": workload,
        "execution": execution,
        "output": {},
        "regression": {},
        "publish": {},
    }


def _cases():
    return [
        {"input_len": 1024, "cache_len": 512, "batch_size": 1},
        {"batch_size": 1, "cache_len": 0, "input_len": 512},
    ]


def _build(execution, topology=None, mode="prefill"):
    return build_execution_plan(
        mode=mode,
        action="run",
        resolved=resolve_config(_gang_config(execution, topology, mode)),
        cases=_cases(),
        result_directory="runs/example",
    )


class GangConfigTest(unittest.TestCase):
    def test_default_derives_single_machine(self):
        plan = _build(_execution(shard_count=2))

        resource = plan.resource_plan
        self.assertIsInstance(resource, AOneResourcePlan)
        self.assertEqual(1, resource.machine_count)
        self.assertEqual(2, resource.gpus_per_machine)
        self.assertEqual(2, resource.local_world_size)
        self.assertEqual(2, resource.shard_count)

    def test_gang_derivation(self):
        self.assertEqual(
            GangTopology(
                world_size=16,
                gpus_per_machine=8,
                machine_count=2,
                local_world_size=8,
            ),
            derive_gang_topology(16, 8),
        )

        plan = _build(
            _execution(gpus_per_machine=8),
            topology={"tp": 8, "dp": 2, "pp": 1, "ep": 16},
        )

        resource = plan.resource_plan
        self.assertEqual(2, resource.machine_count)
        self.assertEqual(8, resource.gpus_per_machine)
        self.assertEqual(8, resource.local_world_size)
        self.assertIsNone(resource.shard_count)

    def test_non_divisible_world_size_rejected(self):
        topology = {"tp": 5, "dp": 2, "pp": 1, "ep": 10}

        with self.assertRaisesRegex(PlanningError, "world_size 10.*machine_count 3"):
            _build(_execution(gpus_per_machine=4), topology)

    def test_gang_resource_plan_round_trips(self):
        plan = _build(
            _execution(gpus_per_machine=8),
            topology={"tp": 8, "dp": 2, "pp": 1, "ep": 16},
        )

        restored = ExecutionPlan.from_dict(plan.to_dict())

        self.assertEqual(plan, restored)
        self.assertIsNone(restored.resource_plan.shard_count)
        self.assertEqual(2, restored.resource_plan.machine_count)
        self.assertEqual(8, restored.resource_plan.local_world_size)


class GangPolicyTest(unittest.TestCase):

    def test_rank_plan_preserves_identity_without_local_policy(self):
        from rtp_llm.test.perf_test.aone.gang_worker import _build_rank_group_plan

        for mode, runtime in (("prefill", prefill_runtime), ("decode", decode_runtime)):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "config.json"
                topology = {"tp": 8, "dp": 2, "pp": 1, "ep": 16}
                path.write_text(
                    json.dumps(
                        _gang_config(_execution(gpus_per_machine=8), topology, mode)
                    )
                )
                builder = getattr(runtime, f"build_external_{mode}_plan")
                expected, source = builder(
                    path,
                    str(Path(directory) / "run"),
                    topology,
                    execution_parser=parse_execution,
                    resource_planner=plan_resources,
                    capability_validator=validate_aone_capability,
                )
                context = SimpleNamespace(gang_plan={"topology": topology})
                with ExitStack() as stack:
                    for target in (
                        "config_schema._parse_execution",
                        "plan._aone_resource_plan",
                    ):
                        stack.enter_context(
                            patch(
                                "rtp_llm.test.perf_test." + target,
                                side_effect=AssertionError("public policy forbidden"),
                                create=True,
                            )
                        )
                    actual, actual_source = _build_rank_group_plan(
                        str(path), context, Path(directory)
                    )
                self.assertEqual(expected.to_dict(), actual.to_dict())
                self.assertEqual(source, actual_source)


if __name__ == "__main__":
    unittest.main()
