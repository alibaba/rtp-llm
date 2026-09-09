"""Compiler/parent/child agree on maximum fresh topology across clean epochs."""

import copy
import json
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from flexlb_test_framework.resource_plan import (
    JavaMockBudget,
    ResourcePlanError,
    plan_lane_leases,
)
from flexlb_test_framework.scenario import compile_scenarios
from flexlb_test_framework.scenario.backend import (
    BoundedOps,
    JavaMockBackend,
    make_env_spec,
)
from flexlb_test_framework.scenario.catalog import handlers
from flexlb_test_framework.scenario.compiler import environment
from flexlb_test_framework.scenario.contracts import StageHandler
from flexlb_test_framework.scenario.lease import validate_lease
from flexlb_test_framework.scenario.runtime import Deadline, RuntimeContext
from test_scenario_runtime import source


def program():
    doc = source()
    doc["profiles"] = ["single-nonbatch"]
    doc["environment"] = {
        "n_prefill": 2,
        "n_decode": 2,
        "config_overrides": {"ordering": "priority"},
    }
    doc["stages"].insert(
        1,
        {
            "id": "larger",
            "action": "environment_reconfigure",
            "params": {
                "config_overrides": {"ordering": "priority"},
                "n_prefill": 1,
                "n_decode": 4,
                "metric_whitelist": "flexlb_auto_tpm",
            },
        },
    )
    return doc


def compile_one(doc):
    return compile_scenarios([("test", doc)], handlers=handlers())[0]


class EnvironmentBudgetTest(unittest.TestCase):
    def test_parent_and_child_reserve_peak_without_relabeling_initial_population(self):
        instance = compile_one(program())
        raw = instance["resource_budget"]
        self.assertEqual(raw["initial_workers"], 4)
        self.assertEqual(raw["max_environment_workers"], 5)
        budget = JavaMockBudget.from_metadata(raw)
        self.assertEqual(budget.worker_capacity, 5)
        self.assertEqual(budget.to_manifest()["initial_workers"], 4)
        lease = plan_lane_leases([[budget]], master_base=28000, mock_base=55000)[
            0
        ].to_manifest()
        lease["schema_version"] = 1
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "lease.json"
            path.write_text(json.dumps(lease))
            self.assertEqual(
                validate_lease(path, raw, lease["child_env"])["worker_capacity"], 5
            )
            lease["worker_capacity"] = 4
            path.write_text(json.dumps(lease))
            with self.assertRaisesRegex(ValueError, "capacity"):
                validate_lease(path, raw, lease["child_env"])

    def test_later_stage_inherits_current_topology_and_metric_environment(self):
        doc = program()
        doc["stages"].insert(
            2,
            {
                "id": "again",
                "action": "environment_reconfigure",
                "params": {"config_overrides": {"ordering": "fifo"}},
            },
        )
        instance = compile_one(doc)
        later = instance["stages"][2]["params"]["environments"]["single-nonbatch"]
        self.assertEqual((later["n_prefill"], later["n_decode"]), (1, 4))
        self.assertEqual(later["metric_whitelist"], "flexlb_auto_tpm")
        self.assertEqual(later["effective_axes"]["ordering"], "FIFO")
        self.assertEqual(instance["resource_budget"]["max_environment_workers"], 5)

    def test_action_capabilities_are_checked_in_their_actual_epoch(self):
        doc = program()
        doc["stages"][1]["params"]["config_overrides"] = {"ordering": "fifo"}
        doc["stages"].insert(2, {"id": "priority_only", "action": "needs_priority"})
        registry = handlers()
        registry["needs_priority"] = StageHandler(
            "needs_priority",
            lambda params, plan: params,
            lambda *args: None,
            {},
            requires=frozenset({"priority"}),
        )
        with self.assertRaisesRegex(ValueError, "priority_only.*lacks capabilities"):
            compile_scenarios([("test", doc)], handlers=registry)

    def test_reserved_victim_ports_and_invalid_peak_metadata_fail_before_launch(self):
        doc = program()
        doc["stages"][1]["params"]["n_decode"] = 149
        with self.assertRaisesRegex(ValueError, "reserved victim"):
            compile_one(doc)
        raw = compile_one(program())["resource_budget"]
        for invalid in (None, True, 3, 150, "5"):
            with self.subTest(invalid=invalid), self.assertRaises(ResourcePlanError):
                JavaMockBudget.from_metadata(dict(raw, max_environment_workers=invalid))
        with self.assertRaises(ResourcePlanError):
            JavaMockBudget(4, 145, 5)

    def test_backend_rejects_unplanned_or_unleased_topology_before_process_setup(self):
        instance = compile_one(program())
        target = instance["stages"][1]["params"]["environments"]["single-nonbatch"]
        lease = plan_lane_leases(
            [[JavaMockBudget.from_metadata(instance["resource_budget"])]],
            master_base=28000,
            mock_base=55000,
        )[0].to_manifest()
        with tempfile.TemporaryDirectory() as tmp, patch(
            "flexlb_test_framework.harness.EnvManager.ensure"
        ) as ensure:
            ctx = RuntimeContext(instance, None, tmp, time.monotonic, time.sleep)
            ctx.env_epoch = 1
            for compiled_bound, capacity in ((4, 5), (5, 4)):
                ctx.instance["resource_budget"][
                    "max_environment_workers"
                ] = compiled_bound
                lease["worker_capacity"] = capacity
                with self.assertRaisesRegex(ValueError, "capacity"):
                    JavaMockBackend(lease).setup(
                        ctx, target, Deadline(time.monotonic() + 1)
                    )
            ensure.assert_not_called()

    def test_dynamic_additions_remain_cumulative_beyond_largest_fresh_population(self):
        raw = compile_one(program())["resource_budget"]
        raw["max_dynamic_additions"] = 1
        ops = Mock()
        ops.add_engine.return_value = (200, {"port": 55005})
        bounded = BoundedOps(ops, raw, {"mock_base": 55000})
        self.assertEqual(bounded.add_engine("decode")[0], 200)
        with self.assertRaisesRegex(ValueError, "budget exhausted"):
            bounded.add_engine("decode")


if __name__ == "__main__":
    unittest.main()
