"""Protection must retain transient failures and reject starvation/stalled recovery."""

import copy
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from flexlb_test_framework.case_config import configure_program
from flexlb_test_framework.scenario import compile_scenarios, load_scenarios
from flexlb_test_framework.scenario.actions.decode_scale_out import assess, sample
from flexlb_test_framework.scenario.catalog import handlers
from flexlb_test_framework.scenario.loader import load_document


class DecodeScaleOutTest(unittest.TestCase):
    def samples(self):
        rows = []
        for i in range(111):
            engine = dict(
                running=6,
                waiting=0,
                accepted=i + 6,
                completed=i,
                kv_admission_fails=0,
                available_blocks=58,
                cache_blocks=64,
                referenced_blocks=6,
                stopped=False,
                http_addr="127.0.0.1:42",
            )
            rows.append(
                dict(
                    time_s=i / 5,
                    engines={"decode-2": engine} if i >= 5 else {},
                    capacity={"127.0.0.1:42": 8} if i >= 6 else {},
                    limit=8,
                )
            )
        return rows

    def verdict(self, rows):
        return assess(rows, "decode-2", 8, 2, 20)

    def test_healthy_new_decode_completes_in_every_window(self):
        v = self.verdict(self.samples())
        self.assertEqual(v["errors"], [])
        self.assertTrue(all(n > 0 for n in v["progress"]))

    def test_routed_decode_can_complete_without_public_rpc_accept_counter(self):
        rows = self.samples()
        for item in rows:
            for engine in item["engines"].values():
                engine["accepted"] = 0
        self.assertEqual(self.verdict(rows)["errors"], [])

    def test_transient_capacity_peak_survives_later_drain(self):
        for field in ("running", "waiting", "master"):
            with self.subTest(field=field):
                rows = self.samples()
                if field == "master":
                    rows[15]["capacity"]["127.0.0.1:42"] = 9
                else:
                    rows[15]["engines"]["decode-2"][field] = 9
                self.assertIn(
                    "new Decode exceeded engine capacity", self.verdict(rows)["errors"]
                )

    def test_never_receiving_and_late_stall_cannot_pass(self):
        for starved in (True, False):
            rows = self.samples()
            for row in rows:
                for engine in row["engines"].values():
                    if starved:
                        engine.update(accepted=0, completed=0)
                    else:
                        engine["completed"] = min(engine["completed"], 60)
            self.assertIn(
                "new Decode did not complete requests in every observation window",
                self.verdict(rows)["errors"],
            )

    def test_disappearance_stop_and_counter_reset_are_retained(self):
        for fault in ("missing", "master_missing", "stopped", "reset"):
            with self.subTest(fault=fault):
                rows = self.samples()
                if fault == "missing":
                    rows[30]["engines"].clear()
                elif fault == "master_missing":
                    rows[30]["capacity"].clear()
                elif fault == "stopped":
                    rows[30]["engines"]["decode-2"]["stopped"] = True
                else:
                    rows[30]["engines"]["decode-2"]["completed"] = 0
                self.assertTrue(self.verdict(rows)["errors"])

    def test_missing_master_capacity_or_changed_limit_cannot_pass(self):
        for fault in ("missing", "changed"):
            rows = self.samples()
            for row in rows:
                if fault == "missing":
                    row["capacity"].clear()
                else:
                    row["limit"] = 132
            self.assertTrue(self.verdict(rows)["errors"])

    def test_kv_rejection_and_invalid_capacity_cannot_pass(self):
        for key, value in (
            ("kv_admission_fails", 1),
            ("available_blocks", 65),
            ("referenced_blocks", 65),
        ):
            rows = self.samples()
            rows[20]["engines"]["decode-2"][key] = value
            self.assertIn(
                "new Decode KV admission/capacity failure", self.verdict(rows)["errors"]
            )

    def test_observation_gap_is_not_silently_ignored(self):
        rows = self.samples()
        del rows[20:30]
        self.assertIn("observation gap exceeds 1s", self.verdict(rows)["errors"])

    def test_missing_numeric_counter_is_an_evidence_error(self):
        row = copy.deepcopy(self.samples()[20]["engines"]["decode-2"])
        row.update(role="decode")
        del row["running"]
        ctx = SimpleNamespace(clock=lambda: 1)
        with patch(
            "flexlb_test_framework.scenario.actions.decode_scale_out._snapshot",
            return_value={"decode-2": row},
        ), patch(
            "flexlb_test_framework.scenario.actions.decode_scale_out._master_json",
            return_value={},
        ):
            with self.assertRaisesRegex(ValueError, "invalid Decode counters"):
                sample(ctx, None)

    def test_python_plan_has_four_profiles_and_one_dynamic_addition(self):
        plans = compile_scenarios(
            load_scenarios(ROOT / "scenarios/elastic/lifecycle.yaml"),
            handlers=handlers(),
        )
        plans = [p for p in plans if p["variant_id"] == "decode_scale_out_protection"]
        self.assertEqual(len(plans), 4)
        for plan in plans:
            actions = [s["action"] for s in plan["stages"]]
            self.assertLess(
                actions.index("decode_scale_flow"), actions.index("elastic_add")
            )
            self.assertLess(
                actions.index("decode_scale_loaded"), actions.index("elastic_add")
            )
            self.assertEqual(actions.count("elastic_add"), 1)
            self.assertEqual(plan["resource_budget"]["initial_workers"], 4)
            self.assertEqual(plan["resource_budget"]["max_dynamic_additions"], 1)

    def test_configuration_rejects_traffic_below_old_pool_capacity(self):
        cfg = load_document(ROOT / "scenarios/elastic/lifecycle.yaml")
        cfg["variants"] = [cfg["variants"][0]]
        cfg["variants"][0]["parameters"]["concurrency"] = 16
        with self.assertRaisesRegex(ValueError, "exceed.*capacity"):
            configure_program(cfg, "test")


if __name__ == "__main__":
    unittest.main()
