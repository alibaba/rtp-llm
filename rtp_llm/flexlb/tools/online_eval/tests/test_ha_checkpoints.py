"""Live evidence cannot hide incomplete requests, owner leaks or absent engines."""

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch

from flexlb_test_framework.ha import LiveClientEvents
from flexlb_test_framework.harness import ClientOps
from flexlb_test_framework.scenario.actions import master_observation as obs
from flexlb_test_framework.scenario.runtime import Deadline, RuntimeContext


class Clock:
    value = 10.0

    def __call__(self):
        return self.value

    def sleep(self, seconds):
        self.value += seconds


def row(rid, event, sequence, start=10000, end=10020, target="A"):
    return dict(
        rid=rid,
        event=event,
        sequence=sequence,
        send_start_epoch_ms=start,
        recorded_epoch_ms=end,
        status="ok",
        route_path="master",
        master_target=target,
        failover=target == "B",
        prefill="p0",
        decode="d0",
        ttft_ms=20,
    )


class JournalTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / "events.jsonl"
        self.journal = LiveClientEvents(self.path)

    def write(self, rows):
        self.path.write_text("".join(json.dumps(r) + "\n" for r in rows))

    def test_legal_failed_terminals_do_not_drop_following_requests(self):
        statuses = [
            "engine_error",
            "empty_response",
            "incomplete_response",
            "timeout",
            "scheduled",
        ]
        rows = []
        for index, status in enumerate(statuses):
            issue = row(str(index), "issued", 2 * index + 1)
            terminal = row(str(index), "terminal", 2 * index + 2)
            terminal.update(status=status, error="failure evidence")
            rows.extend([issue, terminal])
        self.write(rows)
        self.journal.read()
        self.assertEqual(len(statuses), len(self.journal.terminal))
        self.assertEqual(
            statuses, [r["status"] for r in self.journal.terminal.values()]
        )

    def test_partial_tail_is_retried_and_completed_once(self):
        issue, terminal = row("x", "issued", 1), row("x", "terminal", 2)
        data = json.dumps(terminal)
        self.path.write_text(json.dumps(issue) + "\n" + data[:12])
        self.journal.read()
        self.assertEqual(["x"], list(self.journal.issued))
        self.assertFalse(self.journal.terminal)
        with self.path.open("a") as f:
            f.write(data[12:] + "\n")
        self.journal.read()
        self.journal.read()
        self.assertEqual({"x": terminal}, self.journal.terminal)

    def test_missing_duplicate_or_reordered_evidence_is_rejected(self):
        for rows in (
            [row("x", "terminal", 1)],
            [row("x", "issued", 2)],
            [row("x", "issued", 1), row("x", "issued", 2)],
        ):
            with self.subTest(rows=rows):
                self.write(rows)
                with self.assertRaises(ValueError):
                    LiveClientEvents(self.path).read()

    def test_inflight_before_fault_is_in_transition_not_later_steady_window(self):
        self.write(
            [
                row("crossing", "issued", 1, start=9900, end=9900),
                row("finished", "issued", 2, start=9500, end=9500),
                row("finished", "terminal", 3, start=9500, end=9999),
                row("crossing", "terminal", 4, start=9900, end=10050, target="B"),
                row("blackhole", "issued", 5, start=9990, end=9990),
            ]
        )
        self.journal.read()
        self.assertEqual(
            {"crossing", "blackhole"},
            set(self.journal.cohort(10000, 11000, transition=True)),
        )
        self.assertFalse(self.journal.cohort(10000, 11000))
        self.assertNotIn("blackhole", self.journal.terminal)

    def test_ambient_live_flag_is_cleared_by_harness(self):
        with patch.dict("os.environ", {"LIVE_CLIENT_EVENTS": "true"}):
            self.assertEqual("", ClientOps(None)._base_env({})["LIVE_CLIENT_EVENTS"])
            self.assertEqual(
                "true",
                ClientOps(None)._base_env({"LIVE_CLIENT_EVENTS": "true"})[
                    "LIVE_CLIENT_EVENTS"
                ],
            )

    def test_final_reconciliation_rejects_missing_duplicate_and_mismatched_rows(self):
        self.write([row("x", "issued", 1), row("x", "terminal", 2)])
        self.journal.read()
        for final, want in [
            ([row("x", "terminal", 2)], "PASS"),
            ([], "FAIL"),
            ([row("x", "terminal", 2)] * 2, "FAIL"),
            ([dict(row("x", "terminal", 2), master_target="B")], "FAIL"),
        ]:
            with self.subTest(final=final):
                clock = Clock()
                ctx = RuntimeContext({}, None, self.tmp.name, clock, clock.sleep)
                client = ctx.register_resource("ha_client", object())
                rows = ctx.register_resource("ha_rows", final)
                with patch.object(obs, "_journal", return_value=self.journal):
                    result = obs.reconcile(ctx, dict(client=client, rows=rows), None)
                self.assertEqual(want, result.checks[0].status)


class MeasurementTest(unittest.TestCase):
    def test_absent_engine_is_retained_as_zero_in_distribution(self):
        data = obs.measurements(
            [row(str(i), "terminal", i) for i in range(30)],
            {"prefill": ["p0", "p1"], "decode": ["d0", "d1"]},
        )
        self.assertEqual({"p0": 30, "p1": 0}, data["distribution"]["prefill"]["counts"])
        self.assertEqual(1, data["distribution"]["prefill"]["max_share"])
        with self.assertRaises(ValueError):
            obs.measurements(
                [dict(row("x", "terminal", 1), prefill="foreign")],
                {"prefill": ["p0"], "decode": ["d0"]},
            )

    def test_local_ownership_does_not_include_shared_engine_load(self):
        raw = {
            "scheduler_inflight": 0,
            "prefill_endpoints": [
                dict(inflight_batches=0, inflight_requests=0, inflight_route_requests=0)
            ],
            "decode_endpoints": [
                dict(
                    reserved_total=0,
                    active_dispatch_permits=0,
                    engine_load=20,
                    total_load=20,
                )
            ],
        }
        self.assertFalse(any(obs.owned_counts(raw).values()))
        raw["decode_endpoints"][0]["reserved_total"] = 1
        self.assertEqual(1, obs.owned_counts(raw)["decode.reserved_total"])
        del raw["decode_endpoints"][0]["reserved_total"]
        with self.assertRaises(KeyError):
            obs.owned_counts(raw)

    def run_checkpoint(self, missing=False, biased=False):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        clock = Clock()
        ctx = RuntimeContext(
            {},
            NS(manager=NS(master_instance_target=lambda env, target: target)),
            temp.name,
            clock,
            clock.sleep,
        )
        ctx.env = NS(spec=NS(n_prefill=2, n_decode=2))
        journal = LiveClientEvents(Path(temp.name) / "not-created")
        for i in range(40):
            rid = str(i)
            issue = row(rid, "issued", 2 * i + 1, start=10000 + i, end=10000 + i)
            journal.issued[rid] = issue
            if not missing or i != 39:
                journal.terminal[rid] = dict(
                    row(rid, "terminal", 2 * i + 2, start=10000 + i, end=10100 + i),
                    prefill="p0" if biased else "p" + str(i % 2),
                )
        flow = NS(proc=NS(alive=lambda: True))
        handle = ctx.register_resource("ha_client", NS(flow=flow))
        params = dict(
            client=handle,
            target="A",
            duration_s=2,
            min_samples=30,
            min_success=0.95,
            min_target_share=1,
            max_prefill_share=0.75,
            max_owner_load=128,
        )
        info = {
            "ready": True,
            "worker_summary": {
                role: {"alive": 2, "discovered": 2} for role in ("PREFILL", "DECODE")
            },
        }
        raw = {
            "scheduler_inflight": 8,
            "prefill_endpoints": [
                dict(ip_port="p" + str(i), inflight_batches=2) for i in range(2)
            ],
            "decode_endpoints": [
                dict(ip_port="d" + str(i), total_load=4) for i in range(2)
            ],
        }
        with patch.object(obs, "_journal", return_value=journal), patch.object(
            obs,
            "_pools",
            return_value={"prefill": ["p0", "p1"], "decode": ["d0", "d1"]},
        ), patch.object(
            obs,
            "_master_json",
            side_effect=lambda ctx, target, path, deadline, *args: (
                info if path.endswith("info") else raw
            ),
        ), patch.object(
            obs, "_process"
        ), patch.object(
            obs.time, "time", side_effect=clock
        ):
            result = obs.checkpoint(ctx, params, Deadline(15, clock, clock.sleep))
        self.assertTrue(Path(result.artifacts[0]).is_file())
        return {r.id: r.status for r in result.checks}

    def test_checkpoint_rejects_blackhole_and_one_sided_distribution(self):
        self.assertTrue(all(v == "PASS" for v in self.run_checkpoint().values()))
        self.assertEqual("FAIL", self.run_checkpoint(missing=True)["accounting"])
        self.assertEqual("FAIL", self.run_checkpoint(biased=True)["balance"])


if __name__ == "__main__":
    unittest.main()
