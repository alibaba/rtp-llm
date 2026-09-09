"""Quota recovery waits for Prefill liveness, independently of endpoint ledgers."""

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch

from flexlb_test_framework.scenario import compile_scenarios, load_scenarios
from flexlb_test_framework.scenario.actions import master
from flexlb_test_framework.scenario.catalog import handlers
from flexlb_test_framework.scenario.runtime import Deadline, RuntimeContext


class QuotaAliveTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.now = 0.0
        self.ctx = RuntimeContext({}, NS(), self.tmp.name, lambda: self.now, self.sleep)
        self.ctx.env = NS(spec=NS(n_prefill=1, n_decode=1))

    def sleep(self, seconds):
        self.now += seconds

    def test_missing_then_live_prefill_does_not_require_ledger_or_ready(self):
        raw = [
            dict(worker_summary={}),
            dict(ready=False, worker_summary={"PREFILL": {"alive": 1}}),
        ]
        with patch.object(master, "_master_json", side_effect=raw) as http:
            out = master._prefill_alive(
                self.ctx,
                {"target": "single"},
                Deadline(30, lambda: self.now, self.sleep),
            )
        self.assertEqual("PASS", out.checks[0].status)
        self.assertEqual(0.5, self.now)
        self.assertTrue(
            all(c.args[2] == "/rtp_llm/master/info" for c in http.call_args_list)
        )
        rows = json.loads(Path(out.artifacts[0]).read_text())
        self.assertIsNone(rows[0]["alive"])
        self.assertEqual(1, rows[1]["alive"])

    def test_zero_alive_never_passes(self):
        with patch.object(
            master,
            "_master_json",
            return_value=dict(worker_summary={"PREFILL": {"alive": 0}}),
        ):
            with self.assertRaises(TimeoutError):
                master._prefill_alive(
                    self.ctx,
                    {"target": "single"},
                    Deadline(30, lambda: self.now, self.sleep),
                )
        self.assertEqual(30, self.now)

    def test_invalid_alive_is_error(self):
        for value in (True, -1, "1", None):
            with self.subTest(value=value), patch.object(
                master,
                "_master_json",
                return_value=dict(worker_summary={"PREFILL": {"alive": value}}),
            ):
                with self.assertRaises(ValueError):
                    master._prefill_alive(
                        self.ctx,
                        {"target": "single"},
                        Deadline(30, lambda: self.now, self.sleep),
                    )

    def test_formal_quota_preserves_independent_ttl_and_recovery_boundaries(self):
        path = (
            Path(__file__).resolve().parents[1]
            / "scenarios/master/master_dispatch_quota.yaml"
        )
        plan = compile_scenarios(load_scenarios(path), handlers=handlers())[0]
        stages = {s["id"]: s for s in plan["stages"]}
        self.assertEqual("master_prefill_alive", stages["ready"]["action"])
        self.assertEqual(30, stages["ready"]["timeout_s"])
        self.assertEqual(95, stages["ttl_empty"]["timeout_s"])
        self.assertEqual(2, stages["settle"]["params"]["wait_s"])
        self.assertEqual(20, stages["recovery"]["params"]["count"])
        self.assertEqual(1, stages["recovery"]["params"]["concurrency"])
