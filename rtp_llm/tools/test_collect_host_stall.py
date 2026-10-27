"""Linux CPU-only regressions; GDB is mocked and no process is attached."""

import argparse
import copy
import importlib.util
import io
import json
import os
import signal
import subprocess
import sys
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from unittest import mock

SPEC = importlib.util.spec_from_file_location(
    "host_stall_under_test", Path(__file__).with_name("collect_host_stall.py")
)
collector = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(collector)
MARKER = "__HOST_STALL_unit_test__"
COMPLETE_EVENTS = [
    {"event": "attach_verified"},
    {"event": "collection_finished", "errors": []},
    {"event": "detach_returned"},
    {"event": "gdb_finished", "detached": True, "errors": []},
]
SELECTED_COMMANDS = (
    "bt full",
    "frame 0",
    "info frame",
    "info args",
    "info registers",
    "x/16gx $sp",
    "x/16i $pc",
)


def complete_events(identity):
    events = [copy.deepcopy(COMPLETE_EVENTS[0])]
    for tid, task in sorted(identity["threads"].items()):
        fields = {"tid": int(tid), "startticks": task["startticks"]}
        events.append({"event": "selected_thread", **fields})
        events.extend(
            {"event": "selected_thread_command_completed", "text": text, **fields}
            for text in SELECTED_COMMANDS
        )
        events.append({"event": "selected_thread_completed", "errors": [], **fields})
    events.extend(copy.deepcopy(COMPLETE_EVENTS[1:]))
    return events


class RecordingInput(io.StringIO):
    def close(self):
        self.saved = self.getvalue()
        super().close()


class FakeGdb:
    """Model only the spawned debugger; never invoke a subprocess or signal."""

    def __init__(self, mode, events, stdout):
        self.pid = 900000000
        self.stdin = RecordingInput()
        self.mode = mode
        self.returncode = None
        self.signals = []
        self.waits = 0
        for event in events:
            stdout.write(MARKER + json.dumps(event) + "\n")
        stdout.flush()

    def poll(self):
        return self.returncode

    def wait(self, timeout):
        self.waits += 1
        if self.returncode is not None:
            return self.returncode
        if self.mode in ("stubborn", "never_exits") or (
            self.mode == "timeout" and self.waits == 1
        ):
            raise subprocess.TimeoutExpired("fake-gdb", timeout)
        self.returncode = 0
        return 0

    def send_signal(self, signum):
        self.signals.append((self.pid, signum))

    def terminate(self):
        self.send_signal(signal.SIGTERM)

    def kill(self):
        self.send_signal(signal.SIGKILL)
        if self.mode != "never_exits":
            self.returncode = -signal.SIGKILL


@unittest.skipUnless(sys.platform == "linux", "requires read-only Linux /proc")
class CollectorTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix="host-stall-test-")
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.target_pid = os.getpid()
        self.identity = collector.read_identity(self.target_pid, [self.target_pid])
        self.args = argparse.Namespace(
            pid=self.target_pid,
            startticks=self.identity["startticks"],
            tid=[self.target_pid],
            output=self.root / "snapshot",
            version_label="unit-test-unverified",
            timeout=5,
            trace_file=[],
            gdb="gdb-mocked-never-executed",
        )
        self.mode = "success"
        self.events = complete_events(self.identity)
        self.gdb = None
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.popen = self.stack.enter_context(
            mock.patch.object(collector.subprocess, "Popen", side_effect=self.spawn)
        )
        self.stack.enter_context(
            mock.patch.object(collector.secrets, "token_hex", return_value="unit_test")
        )
        self.direct_kill = self.stack.enter_context(
            mock.patch.object(collector.os, "kill")
        )
        # The real /proc target is this test process, but the collector identity
        # is mocked separately. Production must reject self-attach.
        self.stack.enter_context(
            mock.patch.object(collector.os, "getpid", return_value=self.target_pid + 1)
        )

    def spawn(self, command, **kwargs):
        self.assertEqual(command[1:4], ["-nx", "-nh", "-q"])
        self.assertTrue((self.args.output / "before" / "maps").exists())
        for index, trace in enumerate(self.args.trace_file):
            snapshot = self.args.output / ("trace-%03d.snapshot" % index)
            self.assertEqual(snapshot.read_bytes(), trace.read_bytes())
        self.gdb = FakeGdb(self.mode, self.events, kwargs["stdout"])
        return self.gdb

    def run_collector(self):
        success, output = collector.collect(self.args)
        self.assertEqual(output, self.args.output)
        self.direct_kill.assert_not_called()
        return success, json.loads((output / "manifest.json").read_text())

    def test_identity_mismatch_prevents_spawn(self):
        for field, value in (
            ("pid", self.target_pid + 1),
            ("tgid", self.target_pid + 1),
            ("startticks", self.args.startticks + 1),
            ("tracer_pid", 777),
        ):
            with self.subTest(field=field):
                self.args.output = self.root / field
                wrong = copy.deepcopy(self.identity)
                wrong[field] = value
                with mock.patch.object(collector, "read_identity", return_value=wrong):
                    success, metadata = self.run_collector()
                self.assertFalse(success)
                self.assertIn("error", metadata)
                self.assertFalse(metadata["detach_verified"])
        self.popen.assert_not_called()

    def test_selected_thread_tracer_prevents_spawn(self):
        wrong = copy.deepcopy(self.identity)
        wrong["threads"][str(self.target_pid)]["tracer_pid"] = 777
        with mock.patch.object(collector, "read_identity", return_value=wrong):
            success, _ = self.run_collector()
        self.assertFalse(success)
        self.popen.assert_not_called()

    def test_identity_change_immediately_before_spawn_is_rejected(self):
        wrong = copy.deepcopy(self.identity)
        wrong["startticks"] += 1
        with mock.patch.object(
            collector,
            "read_identity",
            side_effect=[self.identity, wrong, self.identity, self.identity],
        ):
            success, _ = self.run_collector()
        self.assertFalse(success)
        self.popen.assert_not_called()

    def test_selected_tid_reuse_before_spawn_is_rejected(self):
        wrong = copy.deepcopy(self.identity)
        wrong["threads"][str(self.target_pid)]["startticks"] += 1
        with mock.patch.object(
            collector,
            "read_identity",
            side_effect=[self.identity, wrong, self.identity, self.identity],
        ):
            success, metadata = self.run_collector()
        self.assertFalse(success)
        self.assertIn("selected TID changed before attach", metadata["error"])
        self.popen.assert_not_called()

    def test_success_requires_complete_markers_and_after_identity(self):
        trace = self.root / "trace.jsonl"
        trace.write_text('{"stage":"blocked"}\n')
        self.args.trace_file = [trace]
        success, metadata = self.run_collector()
        self.assertTrue(success)
        self.assertTrue(metadata["detach_verified"])
        self.assertEqual(metadata["after"]["startticks"], self.args.startticks)
        self.assertEqual(metadata["after"]["tracer_pid"], 0)
        self.assertEqual(metadata["final_identity"]["startticks"], self.args.startticks)
        self.assertEqual(metadata["final_identity"]["tracer_pid"], 0)
        self.assertEqual(self.gdb.signals, [])
        self.assertEqual(self.args.output.stat().st_mode & 0o777, 0o700)
        self.assertTrue(metadata["selected_thread_evidence"]["complete"])
        self.assertEqual(
            metadata["selected_thread_evidence"]["completed_tids"], [self.target_pid]
        )
        self.assertEqual(
            metadata["selected_thread_evidence"]["required_commands"],
            list(SELECTED_COMMANDS),
        )
        self.assertIn("historical CUDA stream/event", " ".join(metadata["limitations"]))
        script = (self.args.output / "gdb.commands").read_text()
        self.assertIn('"x/16i $pc"', script)
        self.assertIn('emit("lock_wait_frame"', script)

    def test_generic_clean_gdb_markers_are_not_selected_thread_evidence(self):
        self.events = copy.deepcopy(COMPLETE_EVENTS)
        success, metadata = self.run_collector()
        self.assertFalse(success)
        self.assertEqual(metadata["gdb_returncode"], 0)
        self.assertTrue(metadata["detach_verified"])
        self.assertFalse(metadata["selected_thread_evidence"]["complete"])
        self.assertEqual(metadata["selected_thread_evidence"]["completed_tids"], [])

    def test_all_selected_tids_must_complete_in_same_attach(self):
        identity = copy.deepcopy(self.identity)
        second_tid = self.target_pid + 1000000
        second = copy.deepcopy(identity["threads"][str(self.target_pid)])
        second.update(pid=second_tid, startticks=second["startticks"] + 1)
        identity["threads"][str(second_tid)] = second
        self.args.tid = [second_tid, self.target_pid, second_tid]
        self.events = complete_events(identity)
        with mock.patch.object(collector, "read_identity", return_value=identity):
            success, metadata = self.run_collector()
        self.assertTrue(success)
        self.assertEqual(
            metadata["selected_thread_evidence"]["completed_tids"],
            [self.target_pid, second_tid],
        )
        self.popen.assert_called_once()

    def test_incomplete_second_tid_cannot_be_hidden_by_complete_first_tid(self):
        identity = copy.deepcopy(self.identity)
        second_tid = self.target_pid + 1000000
        second = copy.deepcopy(identity["threads"][str(self.target_pid)])
        second.update(pid=second_tid, startticks=second["startticks"] + 1)
        identity["threads"][str(second_tid)] = second
        self.args.tid = [self.target_pid, second_tid]
        self.events = [
            event
            for event in complete_events(identity)
            if not (event.get("tid") == second_tid and event.get("text") == "bt full")
        ]
        with mock.patch.object(collector, "read_identity", return_value=identity):
            success, metadata = self.run_collector()
        self.assertFalse(success)
        self.assertTrue(metadata["detach_verified"])
        self.assertEqual(
            metadata["selected_thread_evidence"]["completed_tids"], [self.target_pid]
        )
        self.assertEqual(
            metadata["selected_thread_evidence"]["errors"][0]["tid"], second_tid
        )

    def test_selected_evidence_rejects_wrong_generation_commands_and_errors(self):
        for problem in ("generation", "wrong_command", "duplicate", "order", "errors"):
            with self.subTest(problem=problem):
                self.args.output = self.root / problem
                self.events = complete_events(self.identity)
                if problem == "generation":
                    self.events[2]["startticks"] += 1
                elif problem == "wrong_command":
                    self.events[2]["text"] = "bt"
                elif problem == "duplicate":
                    self.events.insert(2, copy.deepcopy(self.events[2]))
                elif problem == "order":
                    self.events[2], self.events[3] = self.events[3], self.events[2]
                else:
                    for event in self.events:
                        if event["event"] == "selected_thread_completed":
                            event["errors"] = ["info args: unavailable"]
                success, metadata = self.run_collector()
                self.assertFalse(success)
                self.assertTrue(metadata["detach_verified"])
                self.assertFalse(metadata["selected_thread_evidence"]["complete"])

    def test_missing_completion_marker_fails_even_when_gdb_exits_zero(self):
        self.events = [
            event for event in self.events if event["event"] != "collection_finished"
        ]
        success, metadata = self.run_collector()
        self.assertFalse(success)
        self.assertEqual(metadata["gdb_returncode"], 0)
        self.assertTrue(metadata["detach_verified"])

    def test_selected_thread_evidence_precedes_all_thread_backtrace(self):
        script = collector.build_gdb_script(
            self.args.pid, self.args.startticks, self.identity["threads"], MARKER
        )
        all_threads = script.index('command("thread apply all bt")')
        for command in SELECTED_COMMANDS:
            self.assertIn('"' + command + '"', script)
        selected_loop = script.index('for text in config["selected_thread_commands"]:')
        self.assertLess(selected_loop, all_threads)
        self.assertLess(script.index('emit("selected_thread_completed"'), all_threads)
        self.assertLess(script.index('emit("lock_wait_frame"'), all_threads)
        self.assertLess(all_threads, script.index('command("info sharedlibrary")'))

    def test_after_identity_or_tracer_mismatch_cannot_report_detach(self):
        for field, value in (
            ("startticks", self.args.startticks + 1),
            ("tracer_pid", 777),
        ):
            with self.subTest(field=field):
                self.args.output = self.root / ("after-" + field)
                wrong = copy.deepcopy(self.identity)
                wrong[field] = value
                with mock.patch.object(
                    collector,
                    "read_identity",
                    side_effect=[self.identity, self.identity, wrong],
                ):
                    success, metadata = self.run_collector()
                self.assertFalse(success)
                self.assertFalse(metadata["detach_verified"])
                self.assertIn("after_verification_error", metadata)

    def test_timeout_signals_only_spawned_gdb_and_remains_failure(self):
        self.mode = "timeout"
        success, metadata = self.run_collector()
        self.assertFalse(success)
        self.assertTrue(metadata["timed_out"])
        self.assertTrue(metadata["detach_verified"])
        self.assertEqual(self.gdb.signals, [(self.gdb.pid, signal.SIGINT)])
        self.assertIn("detach\nquit 1\n", self.gdb.stdin.saved)
        self.assertNotEqual(self.gdb.pid, self.args.pid)

    def test_final_leader_reuse_after_thread_scan_prevents_detach_verification(self):
        reused = copy.deepcopy(self.identity)
        reused["startticks"] += 1
        with mock.patch.object(
            collector,
            "read_identity",
            side_effect=[self.identity, self.identity, self.identity, reused],
        ):
            success, metadata = self.run_collector()
        self.assertFalse(success)
        self.assertFalse(metadata["detach_verified"])
        self.assertEqual(metadata["final_identity"]["startticks"], reused["startticks"])
        self.assertIn("after_verification_error", metadata)

    def test_unselected_traced_thread_prevents_detach_verification(self):
        wrong = copy.deepcopy(self.identity)
        wrong["threads"]["999"] = {
            "pid": 999,
            "startticks": 123,
            "state": "t",
            "tracer_pid": 777,
        }
        with mock.patch.object(
            collector,
            "read_identity",
            side_effect=[self.identity, self.identity, wrong],
        ):
            success, metadata = self.run_collector()
        self.assertFalse(success)
        self.assertFalse(metadata["detach_verified"])
        self.assertIn("after_all_threads_checked", metadata)

    def test_unreaped_gdb_cannot_report_detach_verified(self):
        self.mode = "never_exits"
        success, metadata = self.run_collector()
        self.assertFalse(success)
        self.assertIn("cleanup_error", metadata)
        self.assertIsNone(metadata["gdb_returncode"])
        self.assertFalse(metadata["detach_verified"])

    def test_unresponsive_gdb_cleanup_is_bounded_and_exact_child_only(self):
        self.mode = "stubborn"
        success, metadata = self.run_collector()
        self.assertFalse(success)
        self.assertTrue(metadata["timed_out"])
        self.assertEqual(
            self.gdb.signals,
            [
                (self.gdb.pid, sig)
                for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGKILL)
            ],
        )
        self.assertEqual(self.gdb.waits, 4)

    def test_post_spawn_metadata_error_still_cleans_up_child(self):
        original = collector.write_json

        def fail_post_spawn(path, value):
            if "gdb_pid" in value and "finished_at_utc" not in value:
                raise OSError("injected manifest write failure")
            original(path, value)

        with mock.patch.object(collector, "write_json", side_effect=fail_post_spawn):
            success, metadata = self.run_collector()
        self.assertFalse(success)
        self.assertIn("injected manifest write failure", metadata["error"])
        self.assertEqual(self.gdb.signals, [(self.gdb.pid, signal.SIGINT)])
        self.assertIsNotNone(self.gdb.poll())

    def test_existing_output_is_not_overwritten(self):
        self.args.output.mkdir()
        marker = self.args.output / "keep"
        marker.write_text("original")
        with self.assertRaises(FileExistsError):
            collector.collect(self.args)
        self.assertEqual(marker.read_text(), "original")
        self.assertEqual(list(self.args.output.iterdir()), [marker])
        self.popen.assert_not_called()

    def test_self_attach_is_rejected_before_creating_output(self):
        with mock.patch.object(collector.os, "getpid", return_value=self.target_pid):
            with self.assertRaisesRegex(ValueError, "collector itself"):
                collector.collect(self.args)
        self.assertFalse(self.args.output.exists())
        self.popen.assert_not_called()


if __name__ == "__main__":
    unittest.main(verbosity=2)
