"""Formal program shape and independent topology/ledger failure boundaries."""

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch

from flexlb_ft.scenario import compile_scenarios, load_scenarios
from flexlb_ft.scenario.actions import master
from flexlb_ft.scenario.catalog import handlers
from flexlb_ft.scenario.runtime import Deadline, RuntimeContext, StageTimeout


class Clock:
    value = 0.0

    def __call__(self):
        return self.value

    def sleep(self, seconds):
        self.value += seconds


class WraparoundTests(unittest.TestCase):
    def test_formal_loader_compiler_retains_independent_clocks_all_profiles(self):
        root = Path(__file__).resolve().parents[1]
        plans = compile_scenarios(
            load_scenarios(root / "scenarios/master"), handlers=handlers()
        )
        selected = [p for p in plans if p["variant_id"] == "wraparound"]
        self.assertEqual(4, len(selected))
        for p in selected:
            stages = {s["id"]: s for s in p["stages"]}
            order = list(stages)
            self.assertEqual("master_topology_ready", stages["ready_a"]["action"])
            self.assertEqual(60, stages["ready_a"]["timeout_s"])
            self.assertEqual({"target": "A"}, stages["ready_a"]["params"])
            self.assertEqual("master_inflight_clean", stages["clean_a"]["action"])
            self.assertEqual(10, stages["clean_a"]["timeout_s"])
            self.assertEqual(10, stages["switched_a"]["params"]["wait_s"])
            self.assertEqual(5, stages["post_clean"]["params"]["wait_s"])
            self.assertEqual(order.index("restart_a") + 1, order.index("ready_a"))
            self.assertEqual(order.index("kill_b") + 1, order.index("switched_a"))
            self.assertEqual(order.index("switched_a") + 1, order.index("clean_a"))
            self.assertEqual(order.index("clean_a") + 1, order.index("post_clean"))

    def run_gate(self, fn, response, budget):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        clock = Clock()
        ctx = RuntimeContext({}, NS(), tmp.name, clock, clock.sleep)
        ctx.env = NS(spec=NS(n_prefill=2, n_decode=4))

        def http(ctx, target, endpoint, deadline, *args):
            self.assertEqual("A", target)
            expected = (
                "/rtp_llm/master/info"
                if fn is master._wrap_topology
                else "/rtp_llm/inflight_status"
            )
            self.assertEqual(expected, endpoint)
            return response

        return (
            ctx,
            clock,
            patch.object(master, "_master_json", side_effect=http),
            Deadline(budget, clock, clock.sleep),
        )

    def test_topology_does_not_require_ready_discovered_or_ledger(self):
        raw = {
            "worker_summary": {"PREFILL": {"alive": 3}, "DECODE": {"alive": 4}},
            "ready": False,
        }
        ctx, clock, mocked, deadline = self.run_gate(master._wrap_topology, raw, 60)
        with mocked:
            out = master._wrap_topology(ctx, {"target": "A"}, deadline)
        self.assertEqual(["topology"], [c.id for c in out.checks])
        self.assertEqual(0, clock())

    def test_missing_alive_evidence_is_error_not_zero(self):
        ctx, _, mocked, deadline = self.run_gate(
            master._wrap_topology, {"worker_summary": {}}, 60
        )
        with mocked, self.assertRaises(KeyError):
            master._wrap_topology(ctx, {"target": "A"}, deadline)

    def test_low_alive_counts_exhaust_only_topology_budget(self):
        raw = {"worker_summary": {"PREFILL": {"alive": 1}, "DECODE": {"alive": 4}}}
        ctx, clock, mocked, deadline = self.run_gate(master._wrap_topology, raw, 60)
        with mocked, self.assertRaises(StageTimeout):
            master._wrap_topology(ctx, {"target": "A"}, deadline)
        self.assertEqual(60, clock())
        self.assertTrue(
            json.loads(next(ctx.artifact_dir.glob("master-topology*")).read_text())
        )

    def test_clean_ledger_needs_no_topology_or_health_response(self):
        raw = {
            "scheduler_inflight": 0,
            "prefill_endpoints": [{"inflight_batches": 0}],
            "decode_endpoints": [{"total_load": 0}],
        }
        ctx, clock, mocked, deadline = self.run_gate(master._wrap_inflight, raw, 10)
        with mocked:
            out = master._wrap_inflight(ctx, {"target": "A"}, deadline)
        self.assertEqual(["inflight"], [c.id for c in out.checks])
        self.assertEqual(0, clock())

    def test_each_nonzero_owner_exhausts_ten_seconds(self):
        cases = [
            (1, 0, {"total_load": 0}),
            (0, 1, {"total_load": 0}),
            (0, 0, {"total_load": 1}),
            (0, 0, {"inflight_requests": 1, "total_load": 0}),
            (0, 0, {"inflight_requests": 0, "total_load": 1}),
        ]
        for sched, prefill, decode in cases:
            with self.subTest(sched=sched, prefill=prefill, decode=decode):
                raw = {
                    "scheduler_inflight": sched,
                    "prefill_endpoints": [{"inflight_batches": prefill}],
                    "decode_endpoints": [decode],
                }
                ctx, clock, mocked, deadline = self.run_gate(
                    master._wrap_inflight, raw, 10
                )
                with mocked, self.assertRaises(StageTimeout):
                    master._wrap_inflight(ctx, {"target": "A"}, deadline)
                self.assertEqual(10, clock())
                rows = json.loads(
                    next(ctx.artifact_dir.glob("master-inflight*")).read_text()
                )
                self.assertEqual(20, len(rows))
