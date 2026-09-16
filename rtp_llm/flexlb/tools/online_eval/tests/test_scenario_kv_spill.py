"""Existing saturation finding stays scoped to its healthy hit-rate contract."""

import copy
import unittest

import test_scenario_kv_churn as churn_test
import test_scenario_kv_global as global_test
from flexlb_test_framework.scenario import compile_scenarios, load_scenarios
from flexlb_test_framework.scenario.catalog import handlers
from test_scenario_kv import ROOT


class SpillModel(churn_test.LruModel):
    def __init__(
        self,
        collapse=False,
        break_recovery=False,
        lose_terminal=False,
        fail_early_recovery=False,
    ):
        super().__init__()
        self.saturation_collapse, self.break_recovery, self.lose_terminal = (
            collapse,
            break_recovery,
            lose_terminal,
        )
        self.fail_early_recovery = fail_early_recovery
        self.tie = 0

    def select(self, wanted):
        if self.rid in (1, 2):
            return f"prefill-{self.rid-1}"
        maximum = max(len(wanted & keys) for keys in self.keys.values())
        names = sorted(
            name for name, keys in self.keys.items() if len(wanted & keys) == maximum
        )
        name = names[self.tie % len(names)]
        if len(names) > 1:
            self.tie += 1
        return name

    def http(self, ops, endpoint, deadline, body=None):
        if endpoint == "snapshot" and (
            (self.saturation_collapse and 18 <= self.rid < 42)
            or (self.break_recovery and self.rid >= 42)
        ):
            for name in self.keys:
                self.keys[name].clear()
                self.lru[name].clear()
        return super().http(ops, endpoint, deadline, body)

    def wait_requests(self, ctx, records, deadline):
        result = super().wait_requests(ctx, records, deadline)
        if self.lose_terminal:
            for row in records.rows:
                if row["wire_request_id"] == 21:
                    records.update(row, consumer_completion_verified=False)
        if self.fail_early_recovery:
            for row in records.rows:
                if row["wire_request_id"] == 43:
                    records.update(row, business_finished=False)
        return result


class SpillTests(unittest.TestCase):
    run_plan = global_test.GlobalKvTests.run_plan

    @classmethod
    def setUpClass(cls):
        cls.plans = [
            p
            for p in compile_scenarios(
                load_scenarios(ROOT / "scenarios/kv/cache_affinity.yaml"),
                handlers=handlers(),
            )
            if p["variant_id"] in {"leader_spill_batch", "leader_spill_nonbatch"}
        ]

    def test_all_four_profiles_run_resolved_and_confirmed_programs(self):
        self.assertEqual(len(self.plans), 4)
        for plan in self.plans:
            for collapse in (False, True):
                with self.subTest(
                    variant=plan["variant_id"],
                    profile=plan["profile"],
                    collapse=collapse,
                ):
                    model = SpillModel(collapse=collapse)
                    result = self.run_plan(plan, model)
                    self.assertEqual(
                        result["status"],
                        "FINDING-CONFIRMED" if collapse else "FINDING-RESOLVED",
                        result["error"],
                    )
                    self.assertEqual(model.rid, 66)
                    self.assertEqual(plan["findings"], ["saturation_hit.M3"])
                    for name, count in [
                        ("baseline_hit", 12),
                        ("saturation_hit", 16),
                        ("recovery_steady_hit", 8),
                    ]:
                        stage = next(s for s in result["stages"] if s["id"] == name)
                        self.assertEqual(
                            stage["checks"][1]["evidence"]["sample_count"], count
                        )
                    stages = plan["stages"]
                    self.assertEqual(
                        len([s for s in stages if s["id"].endswith("_spacing")]), 12
                    )
                    for w in range(4):
                        order = [
                            s["params"]["block_keys"][0]
                            for s in stages
                            if s["action"] == "request"
                            and s["id"].startswith(f"saturation_w{w}_r")
                        ]
                        self.assertEqual(order, [810000, 810000, 811000, 811000])

    def test_recovery_failure_and_terminal_evidence_are_not_expected_findings(self):
        for model, status in [
            (SpillModel(collapse=True, break_recovery=True), "FAIL"),
            (SpillModel(collapse=True, lose_terminal=True), "ERROR"),
            (SpillModel(collapse=True, fail_early_recovery=True), "FAIL"),
        ]:
            result = self.run_plan(self.plans[0], model)
            self.assertEqual(result["status"], status, result["error"])

    def test_replication_uses_last_window_before_final_sync(self):
        class LateReplicationModel(SpillModel):
            final_snapshots = 0

            def http(self, ops, endpoint, deadline, body=None):
                if endpoint == "snapshot" and self.rid == 66:
                    self.final_snapshots += 1
                    if self.final_snapshots >= 2:
                        # A delayed cache update after the final 2s sync changes
                        # holder count, but must not change the old P5 verdict.
                        for keys in self.keys.values():
                            keys.update((810000, 811000))
                return super().http(ops, endpoint, deadline, body)

        for plan in self.plans:
            replication = next(s for s in plan["stages"] if s["id"] == "replication")
            self.assertEqual(
                replication["params"]["snapshot"],
                {"$ref": "stages.recovery_w5_end.output.snapshot"},
            )
            result = self.run_plan(plan, LateReplicationModel())
            self.assertEqual(result["status"], "FINDING-RESOLVED", result["error"])
            late_plan = copy.deepcopy(plan)
            late = next(s for s in late_plan["stages"] if s["id"] == "replication")
            late["params"]["snapshot"] = {
                "$ref": "stages.recovery_digest.output.snapshot"
            }
            result = self.run_plan(late_plan, LateReplicationModel())
            self.assertEqual(result["status"], "FAIL", result["error"])

    def test_finding_cannot_hide_duplicate_saturation_samples(self):
        plan = copy.deepcopy(self.plans[0])
        stage = next(s for s in plan["stages"] if s["id"] == "saturation_hit")
        stage["params"]["samples"] = [stage["params"]["samples"][0]] * 16
        result = self.run_plan(plan, SpillModel(collapse=True))
        self.assertEqual(result["status"], "ERROR")
        self.assertIn("duplicate requests", result["error"])


if __name__ == "__main__":
    unittest.main()
