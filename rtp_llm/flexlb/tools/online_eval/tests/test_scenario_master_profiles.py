"""Profile-specific old configuration and actual core consumer branch checks."""

import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace as NS

from environment_expectations import environment as expected_environment
from flexlb_test_framework.scenario import compile_scenarios, load_scenarios
from flexlb_test_framework.scenario.backend import RequestBatch, make_env_spec
from flexlb_test_framework.scenario.catalog import handlers
from flexlb_test_framework.scenario.runtime import Deadline, RuntimeContext
from test_scenario_backend import Ops

ROOT = Path(__file__).resolve().parents[1]


def plans(name):
    return compile_scenarios(
        load_scenarios(ROOT / f"scenarios/master/{name}.yaml"), handlers=handlers()
    )


class MasterProfileTests(unittest.TestCase):
    def test_complete_environment_matches_expected_profiles(self):
        kill = [
            p for p in plans("master_lifecycle") if p["variant_id"] == "kill_single"
        ]
        quota = plans("master_dispatch_quota")
        self.assertEqual(
            {p["profile"] for p in kill},
            {"batch-window", "single-batch", "single-nonbatch", "window-nonbatch"},
        )
        self.assertEqual(
            {p["profile"] for p in quota}, {"batch-window", "single-batch"}
        )
        for plan in kill + quota:
            with self.subTest(scenario=plan["scenario_id"], profile=plan["profile"]):
                ctx = NS(profile=plan["profile"])
                if plan in kill:
                    ctx.env_manager = NS(ensure=lambda spec: NS(spec=spec))
                    ctx.engine_ops = lambda env: None
                    expected = expected_environment("master_kill", ctx)
                else:
                    expected = expected_environment("master_quota", ctx)
                actual = make_env_spec(
                    plan["environment"], plan["profile"], {"master_base": 28000}
                )
                self.assertEqual(
                    expected.resolved_config,
                    plan["environment"]["resolved_config"],
                )
                for key in (
                    "n_prefill",
                    "n_decode",
                    "discovery",
                    "perf",
                    "master_profile",
                ):
                    self.assertEqual(getattr(actual, key), getattr(expected, key), key)

    def run_core_request(self, plan, stage_id, wait):
        stage = next(s for s in plan["stages"] if s["id"] == stage_id)
        with tempfile.TemporaryDirectory() as tmp:
            ctx = RuntimeContext(plan, None, tmp, time.monotonic, time.sleep)
            batch_mode = (
                plan["environment"]["resolved_config"]["dispatcher"]["type"] == "BATCH"
            )
            ctx.ops = Ops(batch=batch_mode)
            ctx.instance_deadline_s = time.monotonic() + 5
            batch = RequestBatch(ctx, stage["params"])
            ctx.register_resource("requests", batch, batch.cleanup)
            batch.submit(Deadline(time.monotonic() + 3))
            if wait:
                self.assertEqual(
                    batch.wait(Deadline(time.monotonic() + 3)),
                    {"completed": True, "error_count": 0},
                )
                for row in batch.snapshot_records():
                    self.assertTrue(row["consumer_completion_verified"])
                    self.assertTrue(row["consumer_done"])
                    self.assertIsNotNone(row["consumer_exit_s"])
                    self.assertTrue(row["business_finished"])
                expected = (
                    (stage["params"]["count"], 0)
                    if batch_mode
                    else (0, stage["params"]["count"])
                )
                self.assertEqual(
                    (ctx.ops.fetch_count, ctx.ops.generate_count), expected
                )
            else:
                self.assertEqual((ctx.ops.fetch_count, ctx.ops.generate_count), (0, 0))
                rows = batch.snapshot_records()
                self.assertEqual(len(rows), 4)
                self.assertTrue(all(r["schedule"]["status"] == "OK" for r in rows))
                self.assertTrue(
                    all(
                        r["consumer_exit_s"] is None and not r["business_finished"]
                        for r in rows
                    )
                )
            self.assertTrue(all(c["status"] == "PASS" for c in ctx.cleanup(3)))

    def test_kill_baseline_and_recovery_use_actual_profile_consumer(self):
        for plan in plans("master_lifecycle"):
            if plan["variant_id"] == "kill_single":
                for stage in ("baseline", "recovery"):
                    with self.subTest(profile=plan["profile"], stage=stage):
                        self.run_core_request(plan, stage, True)

    def test_both_quota_profiles_preserve_schedule_only_fill(self):
        for plan in plans("master_dispatch_quota"):
            with self.subTest(profile=plan["profile"]):
                self.run_core_request(plan, "fill", False)
                stages = {s["id"]: s for s in plan["stages"]}
                self.assertEqual(stages["blocked"]["params"]["concurrency"], 10)
                self.assertFalse(stages["blocked"]["params"]["sample_topology"])
                self.assertEqual(stages["block_verdict"]["params"]["expected"], 0.5)
                self.assertEqual(stages["ttl_empty"]["timeout_s"], 95)
                self.assertEqual(stages["recovery"]["params"]["concurrency"], 1)
                self.assertEqual(stages["recovery"]["params"]["count"], 20)
                self.assertEqual(stages["recovered"]["params"]["expected"], 0.9)
