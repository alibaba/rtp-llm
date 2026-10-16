import json
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace

from scenario.runtime import Deadline
from adapters.java_flow import JavaFlowGroup
from traffic.synthetic.generator import write_trace

from adapters.flow_control import FlowObservation, FlowStatus


def control_row(**fields):
    row = dict(schema_version=1, revision=1, recorded_epoch_ms=1,
               submitted=0, started=0, terminal=0, outstanding=0)
    row.update(fields)
    row["started"] = fields.get("started", row["submitted"])
    row["observed_started"] = fields.get("observed_started", row["started"])
    row["outstanding"] = fields.get("outstanding", row["submitted"] - row["terminal"])
    if row.get("state") in {"DRAINING", "DRAINED", "INCOMPLETE"}:
        row.setdefault("sending_end_reason", "natural")
    return row


def observation(row):
    return FlowObservation(FlowStatus.decode(control_row(**row)), row["process_returncode"])


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
            flow.observe_control = Mock(side_effect=[observation(sending), observation(drained)])
            flow.status = Mock(side_effect=AssertionError("stop must not parse the journal"))
            flow.stop_sending(Deadline(time.monotonic() + 1))
            self.assertTrue((flow.control / "stop.json").exists())
            flow.observe_control = Mock(return_value=observation(drained))
            flow.status = Mock(return_value=control_row(**drained))
            self.assertEqual(flow.drain(Deadline(time.monotonic() + 1))["terminal"], 1)
            for invalid in (dict(drained, observed_terminal=0),
                            dict(drained, unfinished_ids=["pending"]),
                            dict(drained, process_returncode=1)):
                flow = JavaFlowGroup(None, d, run_id="r", group_id="g", phase_id="p", poll_s=0.01)
                flow.observe_control = Mock(return_value=observation(drained))
                flow.status = Mock(return_value=control_row(**invalid))
                with self.assertRaises(RuntimeError):
                    flow.drain(Deadline(time.monotonic() + 1))

    def test_drain_waits_for_exit_before_reading_request_journal(self):
        from unittest.mock import Mock
        flow = JavaFlowGroup(None, "/unused", run_id="r", group_id="g", phase_id="p", poll_s=0.001)
        draining = dict(**flow.identity, state="DRAINING", process_returncode=None)
        drained = dict(**flow.identity, state="DRAINED", submitted=1, terminal=1,
                       process_returncode=0, observed_terminal=1, unfinished_ids=[])
        def final_journal():
            self.assertEqual(flow.observe_control.call_count, 2)
            return control_row(**drained)
        flow.observe_control = Mock(side_effect=[observation(draining), observation(drained)])
        flow.status = Mock(side_effect=final_journal)
        self.assertEqual(flow.drain(Deadline(time.monotonic() + 1))["state"], "DRAINED")
        flow.status.assert_called_once()

    def test_process_exit_is_observed_before_reading_final_state(self):
        with tempfile.TemporaryDirectory() as d:
            flow = JavaFlowGroup(None, d, run_id="r", group_id="g", phase_id="p", poll_s=0.01)
            flow.control.mkdir()
            path = flow.control / "status.json"
            path.write_text(json.dumps(control_row(**flow.identity, state="DRAINING")))
            def exit_and_publish_final():
                path.write_text(json.dumps(control_row(**flow.identity, state="DRAINED")))
                return 0
            flow.proc = SimpleNamespace(proc=SimpleNamespace(poll=exit_and_publish_final))
            observed = flow.observe_control()
            self.assertEqual(observed.process_returncode, 0)
            self.assertEqual(observed.status.state.value, "DRAINED")

    def test_successful_drain_is_immutable_and_idempotent(self):
        from unittest.mock import Mock
        flow = JavaFlowGroup(None, "/unused", run_id="r", group_id="g", phase_id="p", poll_s=0.01)
        drained = control_row(**flow.identity, state="DRAINED", process_returncode=0,
                              observed_terminal=0, unfinished_ids=[])
        flow.observe_control = Mock(return_value=observation(drained))
        flow.status = Mock(return_value=drained)
        first = flow.drain(Deadline(time.monotonic() + 1))
        first["state"] = "corrupted"
        first["unfinished_ids"].append("corrupted")
        self.assertEqual(flow.drain(Deadline(0))["state"], "DRAINED")
        self.assertEqual(flow.stop_sending(Deadline(0))["unfinished_ids"], [])
        flow.status.assert_called_once()
        flow.observe_control.assert_called_once()

    def test_failed_drain_can_retry_without_cached_failure(self):
        from unittest.mock import Mock
        flow = JavaFlowGroup(None, "/unused", run_id="r", group_id="g", phase_id="p", poll_s=0.01)
        final = control_row(**flow.identity, state="DRAINED", submitted=1, terminal=1,
                            process_returncode=0, observed_terminal=1, unfinished_ids=[])
        flow.observe_control = Mock(return_value=observation(final))
        flow.status = Mock(side_effect=[dict(final, observed_terminal=0), final])
        with self.assertRaisesRegex(RuntimeError, "journal"):
            flow.drain(Deadline(time.monotonic() + 1))
        self.assertEqual(flow.drain(Deadline(time.monotonic() + 1))["terminal"], 1)
        self.assertEqual(flow.status.call_count, 2)

    def test_protocol_rejects_invalid_wire_status(self):
        row = control_row(state="DRAINED")
        for patch in ({"state": "unknown"}, {"submitted": True}, {"schema_version": 2},
                      {"outstanding": 1}, {"started": 1},
                      {"sending_end_reason": "stop_command"}, {"applied_command_id": "foreign"}):
            with self.assertRaises(ValueError):
                FlowStatus.decode(dict(row, **patch))

    def test_unacknowledged_stop_requires_explicit_natural_completion(self):
        from adapters.flow_control import validate_drain
        identity = dict(run_id="r", group_id="g")
        state = control_row(**identity, state="DRAINED", submitted=1, terminal=1)
        for extra in ({"sending_end_reason": None}, {"sending_end_reason": "stop_command"},
                      {"sending_end_reason": "natural", "applied_command_id": "other"}):
            with self.assertRaises(ValueError):
                validate_drain(dict(state, **extra), identity, terminal_count=1, command_id="s")
        validate_drain(dict(state, applied_command_id="s", sending_end_reason="stop_command"), identity,
                       terminal_count=1, command_id="s")
        with self.assertRaisesRegex(ValueError, "retain every"):
            validate_drain(dict(state, state="DRAINING", sending_end_reason="natural", terminal=0, outstanding=1), identity,
                           terminal_count=0, command_id="s")

    def test_control_observation_does_not_parse_request_journal(self):
        from unittest.mock import Mock
        with tempfile.TemporaryDirectory() as d:
            flow = JavaFlowGroup(None, d, run_id="r", group_id="g", phase_id="p", poll_s=1)
            flow.control.mkdir()
            path = flow.control / "status.json"
            path.write_text(json.dumps(control_row(**flow.identity, state="SENDING")))
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
                json.dumps(control_row(**flow.identity, state="DRAINED", submitted=0))
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
                json.dumps(control_row(**flow.identity, state="SENDING", submitted=1))
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

    def test_live_issued_projection_does_not_require_terminal_drain_or_copy_unselected_data(self):
        from unittest.mock import Mock
        flow = JavaFlowGroup(None, '/unused', run_id='r', group_id='g', phase_id='p', poll_s=1)
        flow.journal = SimpleNamespace(read=Mock(), issued={'a':dict(rid='a',send_start_epoch_ms=123,
            unselected_payload='large')}, terminal={})
        flow.status = Mock(side_effect=AssertionError('terminal status must not be required'))
        rows = flow.issued_snapshot(Deadline(time.monotonic()+1), fields=('send_start_epoch_ms',))
        self.assertEqual(rows, [dict(send_start_epoch_ms=123)])
        rows[0]['send_start_epoch_ms'] = 9
        self.assertEqual(flow.journal.issued['a']['send_start_epoch_ms'], 123)
        flow.collection_profile = 'aggregate'
        with self.assertRaisesRegex(ValueError, 'journal'):
            flow.issued_snapshot(Deadline(time.monotonic()+1), fields=('send_start_epoch_ms',))

    def test_live_issued_snapshot_propagates_invalid_journal_and_deadline(self):
        from unittest.mock import Mock
        flow = JavaFlowGroup(None, '/unused', run_id='r', group_id='g', phase_id='p', poll_s=1)
        flow.journal.read = Mock(side_effect=ValueError('journal was truncated'))
        with self.assertRaisesRegex(ValueError, 'truncated'):
            flow.issued_snapshot(Deadline(time.monotonic()+1), fields=('send_start_epoch_ms',))
        flow.journal.read.reset_mock()
        with self.assertRaises(TimeoutError):
            flow.issued_snapshot(Deadline(0), fields=('send_start_epoch_ms',))
        flow.journal.read.assert_not_called()
