"""Freeze preserves process/topology while inactivity TTL may clear the ledger."""

import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch

from flexlb_test_framework.scenario import compile_scenarios, load_scenarios
from flexlb_test_framework.scenario.actions import master
from flexlb_test_framework.scenario.catalog import handlers
from flexlb_test_framework.scenario.runtime import Deadline, RuntimeContext


class FreezeTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.ctx = RuntimeContext({}, NS(), self.tmp.name, time.monotonic, time.sleep)
        self.ctx.env = NS(spec=NS(n_prefill=2, n_decode=4))
        self.deadline = Deadline(time.monotonic() + 10, time.monotonic, time.sleep)

    def compare(self, immediate=1, settled_count=0, ready=True, topo=None, pid=42):
        common = dict(target="B", pid=42)
        before = dict(
            common,
            scheduler_inflight=11,
            ready=True,
            topology={"PREFILL": 2, "DECODE": 4},
        )
        # The immediate observation deliberately has no readiness/topology.
        after = dict(common, scheduler_inflight=immediate)
        settled = dict(
            target="B",
            pid=pid,
            scheduler_inflight=settled_count,
            ready=ready,
            topology=topo or {"PREFILL": 2, "DECODE": 4},
        )
        params = {
            k: self.ctx.register_resource("master_state", v, historical=True)
            for k, v in [("before", before), ("after", after), ("settled", settled)]
        }
        return master._continuity(self.ctx, params, self.deadline)

    def test_healthy_late_readiness_does_not_rejudge_early_state(self):
        out = self.compare()
        self.assertTrue(all(c.status == "PASS" for c in out.checks))

    def test_expired_ledger_does_not_imply_process_or_topology_loss(self):
        out = self.compare(immediate=0, settled_count=100)
        checks = {c.id: c.status for c in out.checks}
        self.assertEqual("PASS", checks["discovered_continuity"])
        self.assertEqual("PASS", checks["same_process"])
        self.assertTrue(out.output["retained"])

    def test_late_not_ready_and_topology_loss_remain_failures(self):
        for kwargs in ({"ready": False}, {"topo": {"PREFILL": 1, "DECODE": 4}}):
            with self.subTest(kwargs=kwargs):
                self.assertEqual(
                    "FAIL",
                    next(
                        c
                        for c in self.compare(**kwargs).checks
                        if c.id == "discovered_continuity"
                    ).status,
                )

    def test_process_replacement_during_post_window_fails(self):
        self.assertEqual(
            "FAIL",
            next(
                c for c in self.compare(pid=43).checks if c.id == "same_process"
            ).status,
        )

    def test_immediate_sample_only_calls_inflight_endpoint(self):
        with patch.object(master, "_process", return_value=NS(pid=42)), patch.object(
            master, "_master_json", return_value={"scheduler_inflight": 11}
        ) as http:
            out = master._scheduler_state(self.ctx, {"target": "B"}, self.deadline)
        self.assertEqual("/rtp_llm/inflight_status", http.call_args.args[2])
        self.assertEqual(1, http.call_count)
        state = self.ctx.resource(out.output["state"], "master_state")
        self.assertNotIn("ready", state)
        self.assertEqual(11, state["scheduler_inflight"])

    def test_deadline_probe_requires_exact_transport_deadline_and_no_stream(self):
        for status, stream, count, expected in [
            ("DEADLINE_EXCEEDED", None, 1, "PASS"),
            ("OK", None, 1, "FAIL"),
            ("UNAVAILABLE", None, 1, "FAIL"),
            ("DEADLINE_EXCEEDED", 1, 1, "FAIL"),
            ("DEADLINE_EXCEEDED", None, 0, "FAIL"),
        ]:
            with self.subTest(status=status, stream=stream, count=count):
                rows = [
                    dict(
                        schedule=dict(status=status),
                        stream=dict(started_s=stream),
                        consumer_exit_s=3,
                    )
                ] * count
                handle = self.ctx.register_resource(
                    "snapshot", dict(records=rows), historical=True
                )
                result = master._deadline_probe_check(
                    self.ctx, {"snapshot": handle}, self.deadline
                )
                self.assertEqual(expected, result.checks[0].status)

    def test_formal_program_places_three_observations_on_old_boundaries(self):
        path = Path(__file__).resolve().parents[1] / "scenarios/master"
        plans = compile_scenarios(load_scenarios(path), handlers=handlers())
        selected = [p for p in plans if p["variant_id"] == "freeze_short_long"]
        self.assertEqual(4, len(selected))
        for p in selected:
            stages = {s["id"]: s for s in p["stages"]}
            order = list(stages)
            self.assertLess(order.index("long_freeze"), order.index("deadline_probe"))
            self.assertLess(order.index("deadline_probe"), order.index("long_restore"))
            self.assertFalse(stages["deadline_probe"]["params"]["sample_topology"])
            self.assertEqual("B", stages["deadline_probe"]["params"]["target"])
            self.assertEqual(
                {"$ref": "stages.deadline_straddle.output.rows"},
                stages["retry_seen"]["params"]["rows"],
            )
            self.assertEqual(-0.5, stages["short_post"]["params"]["until_offset_s"])
            self.assertEqual(-0.5, stages["short_burst"]["params"]["until_offset_s"])
            self.assertEqual("master_scheduler_state", stages["after"]["action"])
            self.assertEqual(order.index("long_restore") + 1, order.index("after"))
            self.assertEqual(order.index("after") + 1, order.index("post_long_end"))
            self.assertEqual(10, stages["post_long_end"]["params"]["wait_s"])
            self.assertEqual(order.index("post_long_end") + 1, order.index("ready_b"))
            self.assertEqual("master_topology_state", stages["ready_b"]["action"])
            self.assertEqual(
                {"$ref": "stages.ready_b.output.state"},
                stages["continuity"]["params"]["settled"],
            )

    def test_settled_sample_only_calls_info_once(self):
        raw = {
            "ready": True,
            "worker_summary": {
                "PREFILL": {"discovered": 2},
                "DECODE": {"discovered": 4},
            },
        }
        with patch.object(master, "_process", return_value=NS(pid=42)), patch.object(
            master, "_master_json", return_value=raw
        ) as http:
            out = master._topology_state(self.ctx, {"target": "B"}, self.deadline)
        self.assertEqual("/rtp_llm/master/info", http.call_args.args[2])
        self.assertEqual(1, http.call_count)
        self.assertNotIn(
            "scheduler_inflight", self.ctx.resource(out.output["state"], "master_state")
        )
