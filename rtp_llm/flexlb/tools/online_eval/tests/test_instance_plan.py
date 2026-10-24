import copy
import unittest

from runtime.resource_plan import ResourcePlanError
from scenario.instance_plan import InstancePlanError, parse_catalog, plan_instances, select_instances


def parse(payload=None):
    return parse_catalog(payload or catalog(), source="yaml", profile="batch-window")


class InstancePlanTest(unittest.TestCase):
    def test_serial_and_parallel_execute_identical_instance_set_once(self):
        instances = parse()
        for n in [1, 2, 4, 20]:
            lanes = plan_instances(instances, n)
            self.assertCountEqual([i.id for i in instances], [i.id for lane in lanes for i in lane])
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
        chosen = select_instances(instances, exact_ids=instances[0].id + "," + instances[0].id, categories="kv")
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
        data["instance_catalog_schema_version"] = True
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

from runtime_fixtures import catalog
