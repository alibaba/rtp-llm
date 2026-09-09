import copy
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from flexlb_test_framework.instance_plan import (
    InstancePlanError,
    parse_catalog,
    plan_instances,
    select_instances,
)
from flexlb_test_framework.resource_plan import ResourcePlanError


def catalog():
    return {
        "schema_version": 1,
        "instances": [
            {
                "id": f"scenario{i}::default::batch-window",
                "scenario_id": f"scenario{i}",
                "variant_id": "default",
                "profile": "batch-window",
                "category": "cancel" if i % 2 else "kv",
                "source_path": f"scenarios/scenario{i}.yaml",
                "source": "yaml",
                "tags": [],
                "requires": [],
                "estimated_duration_s": i + 1,
                "execution": {"timeout_s": 60, "cleanup_timeout_s": 10},
                "resource_budget": {
                    "backend": "java_mock",
                    "initial_workers": 6,
                    "max_dynamic_additions": 0,
                    "bounded": True,
                    "mock_control_offset": -1,
                    "victim_control_offset": 149,
                    "victim_grpc_offset": 150,
                    "reserved_tail_offset": 151,
                },
            }
            for i in range(7)
        ],
    }


def parse(payload=None):
    return parse_catalog(payload or catalog(), source="yaml", profile="batch-window")


class InstancePlanTest(unittest.TestCase):
    def test_serial_and_parallel_execute_identical_instance_set_once(self):
        instances = parse()
        for n in [1, 2, 4, 20]:
            lanes = plan_instances(instances, n)
            self.assertCountEqual(
                [i.id for i in instances], [i.id for lane in lanes for i in lane]
            )
            self.assertTrue(all(lanes))

    def test_lpt_uses_full_id_and_restores_catalog_order(self):
        instances = parse()
        timings = {instances[0].id: 100}
        lanes = plan_instances(instances, 2, timings)
        self.assertEqual([instances[0]], lanes[0])
        self.assertEqual(instances[1:], lanes[1])
        self.assertEqual(lanes, plan_instances(instances, 2, timings))

    def test_unknown_empty_and_profile_excluded_selection_fail(self):
        for args in [
            {"exact_ids": "unknown"},
            {"exact_ids": ","},
            {"exact_ids": ""},
            {"categories": ""},
            {"categories": " , "},
            {"categories": "unknown"},
        ]:
            with self.subTest(args=args), self.assertRaises(InstancePlanError):
                select_instances(parse(), **args)
        with self.assertRaises(InstancePlanError):
            select_instances([])

    def test_selection_is_bounded_and_deduplicates_requested_ids(self):
        instances = parse()
        chosen = select_instances(
            instances,
            exact_ids=instances[0].id + "," + instances[0].id,
            categories="kv",
        )
        self.assertEqual([instances[0]], chosen)
        with self.assertRaises(InstancePlanError):
            select_instances(instances, exact_ids=instances[0].id, categories="cancel")

    def test_duplicate_metadata_and_combined_ids_fail(self):
        data = catalog()
        data["instances"].append(copy.deepcopy(data["instances"][0]))
        with self.assertRaises(InstancePlanError):
            parse(data)
        instances = parse()
        with self.assertRaises(InstancePlanError):
            select_instances(instances + instances)

    def test_unsupported_or_unbounded_resource_budget_rejected(self):
        for key, value in [
            ("backend", "real_gpu"),
            ("bounded", False),
            ("max_dynamic_additions", 144),
            ("victim_grpc_offset", 170),
            ("initial_workers", True),
            ("explicit_port", 61001),
        ]:
            data = catalog()
            data["instances"][0]["resource_budget"][key] = value
            with self.subTest(key=key), self.assertRaises(ResourcePlanError):
                parse(data)

    def test_wrong_schema_profile_and_invalid_time_rejected(self):
        data = catalog()
        data["schema_version"] = True
        with self.assertRaises(InstancePlanError):
            parse(data)
        for key, value in [
            ("profile", "single-batch"),
            ("source", "legacy"),
            ("estimated_duration_s", float("nan")),
            ("estimated_duration_s", 0),
            ("id", "a,b"),
        ]:
            data = catalog()
            data["instances"][0][key] = value
            with self.subTest(key=key), self.assertRaises(InstancePlanError):
                parse(data)


if __name__ == "__main__":
    unittest.main()
