import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from scenario.runtime import Deadline
from runtime.java_flow import JavaFlowGroup
from traffic.realistic import write_trace


class JavaFlowTest(unittest.TestCase):
    def test_large_flow_event_budget_is_explicit_and_bounded(self):
        flow = JavaFlowGroup(
            None,
            "/unused",
            run_id="r",
            group_id="g",
            phase_id="p",
            poll_s=1,
            max_events=201600,
        )
        self.assertEqual(flow.journal.max_events, 201600)
        with self.assertRaises(ValueError):
            JavaFlowGroup(
                None,
                "/unused",
                run_id="r",
                group_id="g",
                phase_id="p",
                poll_s=1,
                max_events=2000001,
            )

    def test_no_process_exit_is_not_drain_even_with_terminal_status(self):
        with tempfile.TemporaryDirectory() as d:
            flow = JavaFlowGroup(
                None, d, run_id="run", group_id="group", phase_id="formal", poll_s=0.01
            )
            flow.control.mkdir()
            flow.proc = SimpleNamespace(proc=SimpleNamespace(poll=lambda: None))
            (flow.control / "status.json").write_text(
                json.dumps(dict(**flow.identity, state="DRAINED", submitted=0))
            )
            with self.assertRaises(TimeoutError):
                flow.drain(Deadline(0))
            self.assertFalse(flow.evidence_snapshot()["complete"])

    def test_stop_is_atomic_and_does_not_terminate_process(self):
        with tempfile.TemporaryDirectory() as d:
            flow = JavaFlowGroup(
                None, d, run_id="run", group_id="group", phase_id="formal", poll_s=0.01
            )
            flow.control.mkdir()
            flow.proc = SimpleNamespace(proc=SimpleNamespace(poll=lambda: None))
            (flow.control / "status.json").write_text(
                json.dumps(dict(**flow.identity, state="SENDING", submitted=1))
            )
            with self.assertRaises(TimeoutError):
                flow.stop_sending(Deadline(0))
            command = json.loads((flow.control / "stop.json").read_text())
            self.assertEqual("stop_sending", command["operation"])
            self.assertEqual(flow.stop_command, command["command_id"])
            self.assertFalse((flow.control / "stop.json.tmp").exists())

    def test_trace_is_reproducible_and_preserves_declared_prefix(self):
        spec = dict(seed=19,count=3,block_size=512,families=1,shared_blocks=0,
            prefix_blocks=0,suffix_blocks=1,zipf_alpha=0,cold_fraction=0,
            pinned_blocks=[[4,5]*256],output_tokens=2,priority=30)
        with tempfile.TemporaryDirectory() as d:
            one=Path(d)/"one.jsonl"
            write_trace(one,spec,"run:group")
            two=Path(d)/"two.jsonl"
            write_trace(two,spec,"run:group")
            self.assertEqual(one.read_bytes(), two.read_bytes())
            rows = [json.loads(line) for line in one.read_text().splitlines()]
            self.assertEqual([0, 1, 2], [r["ts"] for r in rows])
            self.assertTrue(
                all(r["input_token_blocks"][0] == [4,5]*256 and r["il"] == 1024 for r in rows)
            )
            self.assertEqual(3, len({r["rid"] for r in rows}))

    def test_partial_evidence_keeps_unfinished_requests(self):
        with tempfile.TemporaryDirectory() as d:
            flow = JavaFlowGroup(
                None, d, run_id="run", group_id="group", phase_id="formal", poll_s=0.01
            )
            flow.journal.issued = {
                "done": {"request_id": "done"},
                "pending": {"request_id": "pending"},
            }
            flow.journal.terminal = {
                "done": {"request_id": "done", "status": "engine_error"}
            }
            snapshot = flow.evidence_snapshot()
            self.assertFalse(snapshot["complete"])
            self.assertEqual(len(snapshot["records"]), 2)
            self.assertEqual(snapshot["unfinished"], [{"request_id": "pending"}])
            self.assertEqual(snapshot["records"][0]["status"], "engine_error")
