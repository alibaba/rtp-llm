"""Quota rejects requests with its only P absent; coldstart still needs topology."""

import json
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import test_scenario_master as master_test
from flexlb_test_framework.scenario import compile_scenarios, load_scenarios
from flexlb_test_framework.scenario.actions import master
from flexlb_test_framework.scenario.actions.elastic import RecordedRequests
from flexlb_test_framework.scenario.catalog import handlers

ROOT = Path(__file__).resolve().parents[1]


class MasterQuotaTest(unittest.TestCase):
    setUp = master_test.MasterActionsTest.setUp

    @classmethod
    def setUpClass(cls):
        cls.plan = compile_scenarios(
            load_scenarios(ROOT / "scenarios/master/master_dispatch_quota.yaml"),
            handlers=handlers(),
        )[0]

    def run_batch(self, params, successes, missing_topology=True):
        self.ctx.env.spec.master_stable_window_s = 0
        self.ctx.ops = SimpleNamespace(
            next_request_id=Mock(side_effect=range(1, params["count"] + 1))
        )
        calls = []

        def run(records, row, shape, timeout_s):
            calls.append((row["wire_request_id"], shape, timeout_s))
            ok = row["wire_request_id"] <= successes
            records.update(
                row,
                schedule={"status": "OK" if ok else "REJECTED"},
                stream={"status": "OK" if ok else "NOT_STARTED"},
                business_finished=ok,
                consumer_exit_s=time.monotonic(),
                transport_terminal_s=time.monotonic(),
            )

        info = {"worker_summary": {"DECODE": {"discovered": 1, "alive": 1}}}
        if not missing_topology:
            info["worker_summary"]["PREFILL"] = {"discovered": 1, "alive": 1}
        with patch.object(RecordedRequests, "run", run), patch.object(
            master, "_master_json", return_value=info
        ) as observe:
            try:
                out = master._batch(self.ctx, params, self.deadline)
            finally:
                cleanup = self.ctx.cleanup(5)
                self.assertTrue(all(c["status"] == "PASS" for c in cleanup), cleanup)
        return out, observe.call_count, calls

    def test_compiled_blocked_phase_runs_all_ten_without_prefill_role(self):
        stages = {s["id"]: s for s in self.plan["stages"]}
        params = stages["blocked"]["params"]
        self.assertEqual((params["count"], params["concurrency"]), (10, 10))
        self.assertEqual(params["request_timeout_s"], 12)
        self.assertFalse(params["sample_topology"])
        threshold = stages["block_verdict"]["params"]["expected"]
        self.assertEqual(threshold, 0.5)
        for successes in (5, 6):
            with self.subTest(successes=successes):
                out, observations, calls = self.run_batch(params, successes)
                self.assertEqual(observations, 0)
                self.assertEqual(len(calls), 10)
                self.assertEqual(
                    out.output["success_rate"] <= threshold, successes == 5
                )
                for rid, shape, timeout in calls:
                    self.assertEqual(
                        shape,
                        dict(input_len=2048, output_len=2, block_keys=[rid * 100 + 1]),
                    )
                    self.assertGreater(timeout, 0)
                    self.assertLessEqual(timeout, 12)
                artifact = json.loads(Path(out.artifacts[0]).read_text())
                self.assertEqual(len(artifact["records"]), 10)
                self.assertEqual(artifact["topology"], [])
        self.assertEqual(stages["ttl_empty"]["timeout_s"], 95)
        self.assertEqual(stages["recovery"]["params"]["count"], 20)
        self.assertEqual(stages["recovery"]["params"]["concurrency"], 1)
        self.assertTrue(stages["recovery"]["params"]["sample_topology"])

    def test_coldstart_cannot_disable_required_topology_sampling(self):
        plan = SimpleNamespace(path="batch", environment={"master_stable_window_s": 0})
        with self.assertRaisesRegex(ValueError, "coldstart requires topology"):
            master._batch_validate({"coldstart": True, "sample_topology": False}, plan)
        for invalid in (0, "false", None):
            with self.assertRaisesRegex(ValueError, "must be boolean"):
                master._batch_validate({"sample_topology": invalid}, plan)
        with self.assertRaisesRegex(ValueError, "sample_after_s"):
            master._batch_validate(
                {"sample_topology": False, "sample_after_s": 2}, plan
            )
        params = master._batch_validate({"coldstart": True}, plan)
        self.assertTrue(params["sample_topology"])
        with self.assertRaises(KeyError):
            self.run_batch(params, 20)

    def test_default_batch_still_rejects_missing_topology_instead_of_zero(self):
        params = master._batch_validate(
            {}, SimpleNamespace(path="batch", environment={})
        )
        self.assertTrue(params["sample_topology"])
        with self.assertRaises(KeyError):
            self.run_batch(params, 20)


if __name__ == "__main__":
    unittest.main()
