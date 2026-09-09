"""Affinity program checks retain cohort membership and the seed-inclusive M2 denominator."""

import unittest

import test_scenario_kv_global as global_test
from flexlb_test_framework.scenario import compile_scenarios, load_scenarios
from flexlb_test_framework.scenario.catalog import handlers
from test_scenario_kv import ROOT


class AffinityModel(global_test.CacheModel):
    def __init__(self, ignore_hits=False, swallow_free=False):
        super().__init__()
        self.tie = 0
        self.ignore_hits, self.swallow_free = ignore_hits, swallow_free
        self.shapes = []

    def select(self, wanted):
        scores = {
            name: (
                int(self.clock() - self.last.get(name, -100) < 2),
                0 if self.ignore_hits else -len(wanted & cache),
            )
            for name, cache in self.keys.items()
        }
        best = min(scores.values())
        candidates = sorted(name for name, score in scores.items() if score == best)
        if self.swallow_free and len(candidates) > 1 and self.rid > 1:
            return "prefill-0"
        name = candidates[self.tie % len(candidates)]
        if len(candidates) > 1:
            self.tie += 1
        return name

    def start_requests(self, ctx, params, deadline):
        self.shapes.append(dict(params))
        return super().start_requests(ctx, params, deadline)


class AffinityTests(unittest.TestCase):
    run_plan = global_test.GlobalKvTests.run_plan

    def plans(self):
        return [
            p
            for p in compile_scenarios(
                load_scenarios(ROOT / "scenarios/kv/cache_affinity.yaml"),
                handlers=handlers(),
            )
            if not p["variant_id"].startswith("leader_spill_")
        ]

    def test_all_twelve_programs_execute_with_explicit_cohorts(self):
        plans = self.plans()
        self.assertEqual(len(plans), 12)
        for plan in plans:
            with self.subTest(variant=plan["variant_id"], profile=plan["profile"]):
                backend = AffinityModel()
                result = self.run_plan(plan, backend)
                self.assertEqual(result["status"], "PASS", result["error"])
                if plan["variant_id"] == "hot_tension":
                    self.assertEqual(len(backend.shapes), 41)
                    self.assertTrue(
                        all(s["input_len"] == 16384 for s in backend.shapes)
                    )
                    row = next(s for s in result["stages"] if s["id"] == "holder_total")
                    self.assertEqual(row["checks"][1]["evidence"]["sample_count"], 41)
                elif plan["variant_id"] == "mixed_tiers":
                    self.assertEqual(len(backend.shapes), 32)
                    self.assertEqual(backend.shapes[11]["input_len"], 4096)
                    half = backend.shapes[12:22]
                    self.assertTrue(
                        all(
                            s["block_keys"][:4] == list(range(5001, 5005)) for s in half
                        )
                    )
                    self.assertEqual(
                        len(set(k for s in half for k in s["block_keys"][4:])), 40
                    )
                else:
                    self.assertEqual(len(backend.shapes), 32)
                    self.assertEqual(
                        [
                            s["block_keys"] == list(range(1001, 1009))
                            for s in backend.shapes[2:]
                        ],
                        [i % 5 < 3 for i in range(30)],
                    )

    def test_lost_affinity_is_a_contract_failure(self):
        plan = next(p for p in self.plans() if p["variant_id"] == "mixed_tiers")
        result = self.run_plan(plan, AffinityModel(ignore_hits=True))
        self.assertEqual(result["status"], "FAIL")
        self.assertEqual(
            next(s for s in result["stages"] if s["id"] == "full_concentration")[
                "status"
            ],
            "FAIL",
        )

    def test_hot_holder_overconcentration_fails_seed_inclusive_cap(self):
        plan = next(p for p in self.plans() if p["variant_id"] == "hot_tension")
        result = self.run_plan(plan, AffinityModel(swallow_free=True))
        self.assertEqual(result["status"], "FAIL")
        row = next(s for s in result["stages"] if s["id"] == "holder_total")
        self.assertEqual(row["output"]["share"], 1)
        self.assertEqual(row["checks"][1]["id"], "M2")
        self.assertEqual(row["checks"][1]["status"], "FAIL")


if __name__ == "__main__":
    unittest.main()
