import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from flexlb_test_framework.scenario.runtime import Deadline
from online_eval.java_flow import JavaFlowGroup
from online_eval.synthetic_trace import write_trace


class JavaFlowTest(unittest.TestCase):
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
        spec = dict(
            seed=19,
            count=3,
            interval_ms=10,
            block_size=1024,
            families=[
                dict(
                    name="hot",
                    prefix_tokens=[4, 5],
                    suffix_length=3,
                    token_max=100,
                    output_len=2,
                    priority=30,
                )
            ],
        )
        with tempfile.TemporaryDirectory() as d:
            one = write_trace(Path(d) / "one.jsonl", spec, "run:group")
            two = write_trace(Path(d) / "two.jsonl", spec, "run:group")
            self.assertEqual(one.read_bytes(), two.read_bytes())
            rows = [json.loads(line) for line in one.read_text().splitlines()]
            self.assertEqual([0, 10, 20], [r["ts"] for r in rows])
            self.assertTrue(
                all(r["input_ids"][:2] == [4, 5] and r["il"] == 5 for r in rows)
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
