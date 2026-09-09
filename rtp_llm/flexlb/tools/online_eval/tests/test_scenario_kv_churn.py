"""Full churn/LRU programs with an ordered finite cache and controlled retry paths."""

import copy
import unittest
from collections import OrderedDict
from unittest.mock import patch

import test_scenario_kv_affinity as affinity_test
import test_scenario_kv_global as global_test
from flexlb_test_framework.scenario import compile_scenarios, load_scenarios
from flexlb_test_framework.scenario.catalog import handlers
from test_scenario_kv import ROOT


class LruModel(affinity_test.AffinityModel):
    def __init__(
        self, retry=False, stay_wrong=False, forget=False, hide_counters=False
    ):
        super().__init__()
        self.retry, self.stay_wrong, self.forget, self.hide_counters = (
            retry,
            stay_wrong,
            forget,
            hide_counters,
        )
        self.lru = {}
        self.evictions = {}

    def setup(self, ctx, environment, deadline):
        result = super().setup(ctx, environment, deadline)
        self.capacity = environment["prefill_cache_blocks"]
        self.lru = {name: OrderedDict() for name in self.keys}
        self.evictions = {name: 0 for name in self.keys}
        return result

    def select(self, wanted):
        if self.retry and self.rid == 4:
            return "prefill-0"
        if self.retry and self.rid in (2, 3):
            return "prefill-1" if self.rid == 2 or self.stay_wrong else "prefill-0"
        return super().select(wanted)

    def start_requests(self, ctx, params, deadline):
        if len(params["block_keys"]) > self.capacity - 1:
            raise ValueError("request cannot fit while preserving the reserve block")
        handle = super().start_requests(ctx, params, deadline)
        rows = ctx.resource(handle, "requests").snapshot_records()
        for row in rows:
            name = "prefill-" + str(int(row["prefill_addr"].split(":")[-1]) - 100)
            for key in params["block_keys"]:
                self.lru[name].pop(key, None)
                self.lru[name][key] = True
                if len(self.lru[name]) > self.capacity:
                    self.lru[name].popitem(last=False)
                    self.evictions[name] += 1
            self.keys[name] = set(self.lru[name])
        return handle

    def http(self, ops, endpoint, deadline, body=None):
        if endpoint == "snapshot" and self.forget:
            for name in self.keys:
                self.keys[name].clear()
                self.lru[name].clear()
        response = super().http(ops, endpoint, deadline, body)
        if endpoint == "snapshot":
            for row in response["engines"]:
                row.update(
                    referenced_blocks=0,
                    held_blocks=0,
                    cache_keys=len(self.keys[row["name"]]),
                    cache_evictions=self.evictions[row["name"]],
                )
                if self.hide_counters:
                    row.pop("cache_evictions")
        return response


class ChurnTests(unittest.TestCase):
    def run_plan(self, plan, backend):
        with patch(
            "flexlb_test_framework.scenario.actions.kv_capacity._http",
            side_effect=backend.http,
        ):
            return global_test.GlobalKvTests.run_plan(self, plan, backend)

    def plans(self):
        return [
            p
            for p in compile_scenarios(
                load_scenarios(ROOT / "scenarios/kv/cache_churn.yaml"),
                handlers=handlers(),
            )
            if p["variant_id"] != "referenced_occupancy"
        ]

    def test_all_eight_programs_execute_with_real_measurement_handlers(self):
        plans = self.plans()
        self.assertEqual(len(plans), 8)
        for plan in plans:
            with self.subTest(variant=plan["variant_id"], profile=plan["profile"]):
                model = LruModel()
                result = self.run_plan(plan, model)
                self.assertEqual(result["status"], "PASS", result["error"])
                if plan["variant_id"] == "hot_churn":
                    self.assertEqual(model.rid, 50)
                    rate = next(s for s in result["stages"] if s["id"] == "hit_rate")
                    self.assertEqual(rate["checks"][1]["evidence"]["sample_count"], 50)
                else:
                    self.assertEqual(model.rid, 3)
                    self.assertFalse(
                        next(s for s in result["stages"] if s["id"] == "retry")[
                            "output"
                        ]["retried"]
                    )

    def test_retry_is_conditional_and_bounded_to_one_request(self):
        plan = next(p for p in self.plans() if p["variant_id"] == "lru_affinity")
        for wrong in (False, True):
            model = LruModel(retry=True, stay_wrong=wrong)
            result = self.run_plan(plan, model)
            self.assertEqual(
                result["status"], "FAIL" if wrong else "PASS", result["error"]
            )
            self.assertEqual(model.rid, 3 if wrong else 4)
            self.assertTrue(
                next(s for s in result["stages"] if s["id"] == "retry")["output"][
                    "retried"
                ]
            )

    def test_collapse_and_unavailable_cache_counters_do_not_pass(self):
        for variant, model, status, stage in [
            ("hot_churn", LruModel(forget=True), "FAIL", "hit_rate"),
            ("lru_affinity", LruModel(hide_counters=True), "ERROR", "prime_counters"),
        ]:
            plan = next(p for p in self.plans() if p["variant_id"] == variant)
            result = self.run_plan(plan, model)
            self.assertEqual(result["status"], status, result["error"])
            self.assertEqual(
                next(s for s in result["stages"] if s["id"] == stage)["status"], status
            )

    def test_post_request_cache_snapshot_is_rejected_as_hit_evidence(self):
        plan = copy.deepcopy(
            next(p for p in self.plans() if p["variant_id"] == "hot_churn")
        )
        before = plan["stages"].pop(1)
        plan["stages"].insert(3, before)
        result = self.run_plan(plan, LruModel())
        self.assertEqual(result["status"], "ERROR")
        self.assertIn("precede request issue", result["error"])


if __name__ == "__main__":
    unittest.main()
