import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from scenario.runtime import Deadline
from adapters.java_flow import JavaFlowGroup
from traffic.synthetic.generator import write_trace


class JavaFlowTest(unittest.TestCase):
    def test_natural_completion_racing_stop_retains_drain_checks(self):
        from unittest.mock import Mock
        with tempfile.TemporaryDirectory() as d:
            flow = JavaFlowGroup(None, d, run_id="r", group_id="g", phase_id="p", poll_s=0.01)
            flow.control.mkdir()
            flow.proc = SimpleNamespace(proc=SimpleNamespace(poll=lambda: 0))
            sending = dict(**flow.identity, state="SENDING", process_returncode=None)
            drained = dict(**flow.identity, state="DRAINED", sending_end_reason="natural",
                           submitted=1, terminal=1, process_returncode=0,
                           observed_terminal=1, unfinished_ids=[])
            # The producer finishes after the first observation but before reading stop.json.
            flow.control_status = Mock(side_effect=[sending, drained])
            flow.status = Mock(side_effect=AssertionError("stop must not parse the journal"))
            flow.stop_sending(Deadline(1))
            self.assertTrue((flow.control / "stop.json").exists())
            flow.status = Mock(return_value=drained)
            self.assertEqual(flow.drain(Deadline(1))["terminal"], 1)
            for invalid in (dict(drained, observed_terminal=0),
                            dict(drained, unfinished_ids=["pending"]),
                            dict(drained, process_returncode=1)):
                flow.status.return_value = invalid
                with self.assertRaises(RuntimeError):
                    flow.drain(Deadline(1))

    def test_unacknowledged_stop_requires_explicit_natural_completion(self):
        from adapters.flow_control import validate_drain
        identity = dict(run_id="r", group_id="g")
        state = dict(**identity, state="DRAINED", submitted=1, terminal=1)
        for extra in ({}, {"sending_end_reason": "stop_command"},
                      {"sending_end_reason": "natural", "applied_command_id": "other"}):
            with self.assertRaisesRegex(ValueError, "accepting the stop"):
                validate_drain(dict(state, **extra), identity, terminal_count=1, command_id="s")
        validate_drain(dict(state, applied_command_id="s"), identity,
                       terminal_count=1, command_id="s")
        with self.assertRaisesRegex(ValueError, "retain every"):
            validate_drain(dict(state, sending_end_reason="natural", terminal=0), identity,
                           terminal_count=0, command_id="s")

    def test_control_observation_does_not_parse_request_journal(self):
        from unittest.mock import Mock
        with tempfile.TemporaryDirectory() as d:
            flow = JavaFlowGroup(None, d, run_id="r", group_id="g", phase_id="p", poll_s=1)
            flow.control.mkdir()
            path = flow.control / "status.json"
            path.write_text(json.dumps(dict(**flow.identity, state="SENDING")))
            flow.journal.read = Mock(side_effect=ValueError("journal corrupt"))
            self.assertEqual(flow.control_status()["state"], "SENDING")
            flow.journal.read.assert_not_called()
            with self.assertRaisesRegex(ValueError, "journal corrupt"):
                flow.status()
            path.write_text(json.dumps(dict(run_id="other", group_id="g", phase_id="p")))
            with self.assertRaisesRegex(ValueError, "identity mismatch"):
                flow.control_status()

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
