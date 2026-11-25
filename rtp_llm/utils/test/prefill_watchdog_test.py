import io
import os
import select
import shlex
import shutil
import signal
import stat
import struct
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

if sys.platform == "linux":
    import resource

from rtp_llm.utils import prefill_watchdog as watchdog


def _write_progress(path, phase, sequence, tid=10, layer=0):
    word = phase | (layer << 4) | (tid << 14) | (sequence << 36)
    with path.open("r+b") as slot:
        slot.write(struct.pack("<Q", word if phase else 0))


def _elf_core_fixture():
    header = bytearray(64)
    header[:6] = b"\x7fELF\x02\x01"
    struct.pack_into("<H", header, 16, 4)  # ET_CORE
    struct.pack_into("<Q", header, 32, 64)  # e_phoff
    struct.pack_into("<HH", header, 54, 56, 1)  # e_phentsize and e_phnum
    note = struct.pack("<IIQQQQQQ", 4, 0, 120, 0, 0, 4, 4, 4)
    return header + note + b"note"


@unittest.skipUnless(sys.platform == "linux", "requires /proc and mmap slots")
class PrefillWatchdogTest(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.output = Path(temp.name) / "cores"
        self.monitor = watchdog.PrefillWatchdog(
            {"prepare": 1, "layer": 1, "tail": 1, "gpu_wait": 1},
            self.output,
            10,
        )
        self.addCleanup(self.monitor.close)

    def _register(self, rank=0, pid=None):
        path = Path(self.monitor.create_slot(rank))
        self.monitor.register_process(rank, os.getpid() if pid is None else pid)
        return path

    def _poll_at(self, seconds):
        with mock.patch.object(watchdog.time, "monotonic", return_value=seconds):
            self.monitor.poll()

    def _mock_helper(self):
        process_patch = mock.patch.object(watchdog.subprocess, "Popen")
        signal_patch = mock.patch.object(watchdog.os, "killpg")
        fake_process = process_patch.start()
        self.signal_helper = signal_patch.start()
        self.addCleanup(process_patch.stop)
        self.addCleanup(signal_patch.stop)
        self.addCleanup(self.monitor.close)
        fake_process.return_value.pid = 999001
        fake_process.return_value.poll.return_value = None
        fake_process.return_value.wait.return_value = -9
        return fake_process

    def _expire(self, path, phase=1, sequence=1):
        _write_progress(path, phase, sequence)
        self._poll_at(1)
        self._poll_at(2)

    def test_default_disabled_and_invalid_threshold_disabled(self):
        self.assertIsNone(
            watchdog.PrefillWatchdog.from_config("", str(self.output), False)
        )
        with self.assertLogs(watchdog.__name__, "WARNING"):
            self.assertIsNone(
                watchdog.PrefillWatchdog.from_config(
                    "layer=nan", str(self.output), False
                )
            )
        enabled = watchdog.PrefillWatchdog.from_config(
            " prepare = 5,layer=20 ", str(self.output), False
        )
        self.assertIsNotNone(enabled)
        self.addCleanup(enabled.close)

    def test_gpu_capture_is_opt_in_to_enabled_watchdog(self):
        for thresholds, gpu_option, expected in (
            ("", True, None),
            ("layer=2", False, False),
            ("layer=2", True, True),
        ):
            with self.subTest(thresholds=thresholds, gpu_option=gpu_option):
                monitor = watchdog.PrefillWatchdog.from_config(
                    thresholds, str(self.output), gpu_option
                )
                if expected is None:
                    self.assertIsNone(monitor)
                else:
                    self.assertIsNotNone(monitor)
                    self.addCleanup(monitor.close)
                    self.assertEqual(monitor._gpu_core, expected)

    def test_idle_and_real_progress_do_not_trigger(self):
        path = self._register()
        popen = self._mock_helper()
        self._poll_at(100)
        for seq in range(1, 8):
            _write_progress(path, 2, seq, layer=seq)
            self._poll_at(100 + seq)
        popen.assert_not_called()

    def test_unconfigured_stage_warns_without_attaching(self):
        path = self._register()
        popen = self._mock_helper()
        with self.assertLogs(watchdog.__name__, "WARNING") as captured:
            self._expire(path, phase=3)
            self._poll_at(9)
        self.assertEqual(len(captured.output), 1)
        self.assertIn("without a configured deadline", captured.output[0])
        self.assertIn("phase=graph", captured.output[0])
        popen.assert_not_called()

    def test_debugger_preserves_inherited_file_limit_and_detaches(self):
        path = self._register()
        _write_progress(path, 1, 1)
        command = watchdog._gdb_capture_command(self.monitor._ranks[0])
        gdb = mock.Mock()
        disk = mock.Mock(f_bavail=131136, f_frsize=1024)
        with (
            mock.patch.dict(sys.modules, {"gdb": gdb}),
            mock.patch.object(os, "statvfs", return_value=disk),
            mock.patch.object(resource, "getrlimit", return_value=(8192, 16384)),
            mock.patch.object(resource, "setrlimit") as setrlimit,
            mock.patch.object(signal, "signal") as set_signal,
        ):
            exec(command.removeprefix("python "), {})
        setrlimit.assert_called_once_with(resource.RLIMIT_FSIZE, (8192, 16384))
        set_signal.assert_called_once_with(signal.SIGXFSZ, signal.SIG_IGN)
        self.assertEqual(
            gdb.execute.call_args_list,
            [
                mock.call(f"attach {os.getpid()}"),
                mock.call("generate-core-file core.partial"),
                mock.call("detach"),
            ],
        )

    def test_debugger_accepts_progress_change_on_queued_rank(self):
        path = self._register()
        command = watchdog._gdb_capture_command(self.monitor._ranks[0])
        gdb = mock.Mock(GdbError=RuntimeError)

        def execute(instruction):
            if instruction.startswith("attach "):
                _write_progress(path, 2, 2)

        gdb.execute.side_effect = execute
        with (
            mock.patch.dict(sys.modules, {"gdb": gdb}),
            mock.patch.object(resource, "setrlimit"),
            mock.patch.object(signal, "signal"),
        ):
            exec(command.removeprefix("python "), {})
        self.assertEqual(
            gdb.execute.call_args_list,
            [
                mock.call(f"attach {os.getpid()}"),
                mock.call("generate-core-file core.partial"),
                mock.call("detach"),
            ],
        )

    def test_debugger_detaches_if_process_identity_changes_during_attach(self):
        self._register()
        command = watchdog._gdb_capture_command(self.monitor._ranks[0])
        gdb = mock.Mock(GdbError=RuntimeError)
        stat_text = Path(f"/proc/{os.getpid()}/stat").read_text()
        comm, _, data = stat_text.rpartition(")")
        fields = data.split()
        fields[19] = str(int(fields[19]) + 1)
        reused = comm + ") " + " ".join(fields)
        with (
            mock.patch.dict(sys.modules, {"gdb": gdb}),
            mock.patch.object(resource, "setrlimit"),
            mock.patch.object(signal, "signal"),
            mock.patch(
                "builtins.open",
                side_effect=[io.StringIO(stat_text), io.StringIO(reused)],
            ),
            self.assertRaises(RuntimeError),
        ):
            exec(command.removeprefix("python "), {})
        self.assertEqual(
            gdb.execute.call_args_list,
            [mock.call(f"attach {os.getpid()}"), mock.call("detach")],
        )

    def test_missing_gdb_is_attempted_once(self):
        path = self._register()
        launch, (helper,) = self._mock_gdb_batch(1)
        self._expire(path)
        helper.poll.return_value = 1
        self._poll_at(3)
        self._poll_at(20)
        launch.assert_called_once()
        with (
            mock.patch.object(watchdog.shutil, "which", return_value=None),
            self.assertRaisesRegex(FileNotFoundError, "gdb is unavailable"),
        ):
            watchdog._run_capture(self.output / ".rank0", self.monitor._ranks[0], False)

    def _mock_gdb_batch(self, count):
        popen = self._mock_helper()
        children = [mock.Mock() for _ in range(count)]
        for i, child in enumerate(children):
            child.pid = 999001 + i
            child.poll.return_value = None
            child.wait.return_value = -9
        popen.side_effect = children
        return popen, children

    def _complete_dump(self, popen, child, rank, code=0):
        staging = Path(popen.call_args.args[0][2])
        self.assertTrue(staging.name.startswith(f".rank{rank}-"), staging)
        if code == 0:
            staging.mkdir(mode=0o700, parents=True)
            (staging / "core.partial").write_bytes(_elf_core_fixture())
            watchdog._publish_core(staging, code, gpu=False)
        child.poll.return_value = code
        return staging

    def test_gpu_then_cpu_capture_for_each_rank_in_same_incident(self):
        self.monitor._gpu_core = True
        trigger = self._register(7)
        self.monitor.register_process(4, os.getpid())  # No progress slot.
        popen, children = self._mock_gdb_batch(4)
        self._expire(trigger)
        incident = self.monitor._incident

        for index, (rank, gpu) in enumerate(
            ((7, True), (7, False), (4, True), (4, False))
        ):
            self.assertEqual(popen.call_count, index + 1)
            launch = popen.call_args
            staging = Path(launch.args[0][2])
            self.assertTrue(staging.name.startswith(f".rank{rank}-"))
            self.assertIn(f"-incident{incident}-", staging.name)
            if gpu:
                self.assertEqual(launch.args[0][-1], "GPU")
                staging.mkdir(mode=0o700, parents=True)
                (staging / "gpu.nvcudmp.partial").write_bytes(b"gpu core fixture")
                trace = "#0  gpu_wait_leaf ()\n#1  prefill_kernel ()\n"
                (staging / "cuda-gdb.log").write_text(
                    f"{watchdog._GPU_CORE_READABLE}\n{trace}"
                )
                watchdog._publish_core(staging, 0, gpu=True)
                children[index].poll.return_value = 0
            else:
                self._complete_dump(popen, children[index], rank)
            self._poll_at(3 + index)
            self.assertFalse(staging.exists())

        self.assertEqual(popen.call_count, 4)
        for rank in (4, 7):
            (gpu,) = self.output.glob(f"rank{rank}-*incident{incident}-*/gpu.nvcudmp")
            (cpu,) = self.output.glob(f"rank{rank}-*incident{incident}-*/core")
            self.assertEqual(stat.S_IMODE(gpu.stat().st_mode), 0o600)
            self.assertEqual(cpu.read_bytes(), _elf_core_fixture())
            self.assertIn(
                "#0  gpu_wait_leaf", gpu.with_name("cuda-gdb.log").read_text()
            )

    def test_gpu_core_without_offline_readback_is_not_published(self):
        staging = self.output / ".rank0"
        staging.mkdir(parents=True)
        (staging / "gpu.nvcudmp.partial").write_bytes(b"unverified")
        (staging / "cuda-gdb.log").write_text(
            f"Failed to read core file: invalid dump\n{watchdog._GPU_CORE_READABLE}\n"
        )
        with self.assertRaisesRegex(ValueError, "CUDA core could not be read back"):
            watchdog._publish_core(staging, 0, gpu=True)
        self.assertFalse(list(self.output.glob("rank*/gpu.nvcudmp")))

    def test_readable_gpu_core_retained_with_long_backtrace_or_no_active_stack(self):
        for index, (code, tail) in enumerate(
            ((0, "#0  deep_stack()\n" * 800), (1, "No stack.\n"))
        ):
            staging = self.output / f".rank{index}"
            staging.mkdir(parents=True)
            (staging / "gpu.nvcudmp.partial").write_bytes(b"gpu core fixture")
            diagnostic = f"{watchdog._GPU_CORE_READABLE}\n{tail}"
            (staging / "cuda-gdb.log").write_text(diagnostic)
            saved = watchdog._publish_core(staging, code, gpu=True)
            self.assertEqual(saved.with_name("cuda-gdb.log").read_text(), diagnostic)

    def test_gpu_timeout_then_attach_failure_still_collect_cpu_and_peers(self):
        self.monitor._gpu_core = True
        trigger = self._register()
        self._register(1)
        popen, (gpu0, cpu0, gpu1, cpu1) = self._mock_gdb_batch(4)
        self._expire(trigger)
        self._poll_at(12)
        self.signal_helper.assert_called_once_with(gpu0.pid, signal.SIGTERM)
        popen.assert_called_once()
        gpu0.poll.return_value = 1
        self._poll_at(13)
        self._complete_dump(popen, cpu0, 0)
        self._poll_at(14)
        gpu1.poll.return_value = 1
        with self.assertLogs(watchdog.__name__, "WARNING") as captured:
            self._poll_at(15)
        self.assertIn("capture failed", "\n".join(captured.output))
        self._complete_dump(popen, cpu1, 1)
        self._poll_at(16)
        self.assertEqual(popen.call_count, 4)
        self.assertEqual(len(list(self.output.glob("rank*/core"))), 2)
        self.assertFalse(list(self.output.glob("rank*/gpu.nvcudmp")))

    def test_gpu_debugger_without_embedded_python_checks_pid_and_inherits_file_limit(
        self,
    ):
        self._register()
        identity = self.monitor._ranks[0]
        staging = self.output / "rank core"
        commands = watchdog._cuda_gdb_commands(identity, staging)
        identity_guard, post_attach_guard = (
            line
            for line in commands.splitlines()
            if line.startswith("shell ") and "/proc/" in line
        )
        self.assertEqual(identity_guard, post_attach_guard)
        checks = commands.split(identity_guard)
        self.assertIn(f"\nattach {identity.pid}\n", checks[1])
        self.assertIn("\n    detach\n", checks[2])
        self.assertIn("\ngenerate-core-file --no-cpu", checks[2])
        self.assertIn(
            f"--cuda-core-file {shlex.quote(str(staging / 'gpu-core-stream'))}\n",
            commands,
        )
        self.assertNotIn("skip_local_memory", commands)
        self.assertLess(
            watchdog._GPU_READBACK_COMMANDS.index("target cudacore"),
            watchdog._GPU_READBACK_COMMANDS.index(watchdog._GPU_CORE_READABLE),
        )
        self.assertEqual(subprocess.run(shlex.split(identity_guard)[1:]).returncode, 0)

        stale = watchdog._RankIdentity(identity.pid, identity.start_ticks + 1)
        stale_guard = next(
            line
            for line in watchdog._cuda_gdb_commands(stale, staging).splitlines()
            if line.startswith("shell ")
        )
        self.assertNotEqual(subprocess.run(shlex.split(stale_guard)[1:]).returncode, 0)

        with (
            mock.patch.object(resource, "getrlimit", return_value=(8192, 16384)),
            mock.patch.object(resource, "setrlimit") as setrlimit,
        ):
            self.assertEqual(watchdog._limit_capture_files(65536), 8192)
        setrlimit.assert_called_once_with(resource.RLIMIT_FSIZE, (8192, 16384))

    def test_pod_batch_collects_all_ranks_and_rearms_after_recovery(self):
        trigger = self._register(7)
        idle = self._register(4)
        progressing = self._register(6)
        self.monitor.register_process(5, os.getpid())  # No mmap progress slot.
        popen, children = self._mock_gdb_batch(5)

        _write_progress(progressing, 2, 1)
        _write_progress(trigger, 1, 1)
        self._poll_at(1)
        _write_progress(progressing, 2, 2)
        self._poll_at(2)
        popen.assert_called_once()

        # The trigger recovers without cancelling the rest of the admitted Pod batch.
        _write_progress(trigger, 0, 0)
        self._poll_at(3)
        popen.assert_called_once()
        for index, rank in enumerate((7, 4, 5, 6)):
            _write_progress(progressing, 2, index + 3)
            self._complete_dump(popen, children[index], rank)
            self._poll_at(4 + index)
            self.assertEqual(popen.call_count, min(index + 2, 4))

        self.assertEqual(len(list(self.output.glob("rank*/core"))), 4)
        _write_progress(progressing, 0, 0)
        self._poll_at(8)
        _write_progress(idle, 1, 2)
        self._poll_at(9)
        self._poll_at(10)
        self.assertEqual(popen.call_count, 5)

    def test_peer_expiring_before_its_capture_needs_only_one_attempt(self):
        first = self._register(0)
        second = self._register(1)
        popen, children = self._mock_gdb_batch(2)
        self._expire(first)
        first_incident = self.monitor._incident
        _write_progress(second, 1, 1)
        self._poll_at(3)
        _write_progress(first, 0, 0)
        self._poll_at(4)
        self._complete_dump(popen, children[0], 0)
        self._poll_at(5)
        self._complete_dump(popen, children[1], 1)
        self._poll_at(6)
        self.assertEqual(self.monitor._incident, first_incident)
        self.assertEqual(popen.call_count, 2)
        self.assertEqual(len(list(self.output.glob("rank*/core"))), 2)

    def test_peer_progress_can_rearm_while_trigger_is_still_stalled(self):
        first = self._register(0)
        peer = self._register(1)
        popen, children = self._mock_gdb_batch(3)
        self._expire(first)
        self._complete_dump(popen, children[0], 0)
        self._poll_at(3)
        initial = self.monitor._incident
        _write_progress(peer, 1, 2)
        self._complete_dump(popen, children[1], 1)
        self._poll_at(4)
        self._poll_at(5)
        self.assertNotEqual(self.monitor._incident, initial)
        self.assertEqual(popen.call_count, 3)
        self.assertIn(".rank1-", popen.call_args.args[0][2])

    def test_healthy_peer_pause_does_not_trigger_another_capture(self):
        first = self._register(0)
        peer = self._register(1)
        popen, children = self._mock_gdb_batch(3)
        self._expire(first)
        _write_progress(first, 0, 0)
        _write_progress(peer, 1, 2)
        self._complete_dump(popen, children[0], 0)
        self._poll_at(3)
        self._poll_at(6)
        self._complete_dump(popen, children[1], 1)
        self._poll_at(7)
        self.assertEqual(popen.call_count, 2)
        self._poll_at(8)
        self.assertEqual(popen.call_count, 3)  # A fresh full deadline expired.

    def test_collective_waiter_gets_a_fresh_deadline_after_peer_capture(self):
        trigger = self._register(0)
        waiter = self._register(1)
        self._register(2)
        popen, children = self._mock_gdb_batch(3)
        self._expire(trigger)
        _write_progress(trigger, 0, 0)
        self._complete_dump(popen, children[0], 0)
        self._poll_at(3)
        self._complete_dump(popen, children[1], 1)
        self._poll_at(4)

        # Rank 1 already finished its own capture before entering a
        # collective that waits on rank 2 while its debugger is attached.
        _write_progress(waiter, 1, 2)
        self._poll_at(5)
        self._poll_at(7)
        self._complete_dump(popen, children[2], 2)
        self._poll_at(8)
        _write_progress(waiter, 0, 0)
        self._poll_at(9)
        self.assertEqual(popen.call_count, 3)

    def test_new_progress_during_attach_gets_a_fresh_deadline(self):
        trigger = self._register()
        popen, (first, _) = self._mock_gdb_batch(2)
        self._expire(trigger)
        _write_progress(trigger, 2, 2)
        self._poll_at(3)
        self._poll_at(5)
        self._complete_dump(popen, first, 0)
        self._poll_at(6)
        self.assertEqual(popen.call_count, 1)
        self._poll_at(7)
        self.assertEqual(popen.call_count, 2)

    def test_quarantined_owned_tracer_pauses_collective_waiters_until_detach(self):
        with mock.patch.object(watchdog, "_pid_start_ticks", return_value=10):
            trigger = self._register(0, pid=101)
            waiter = self._register(1, pid=102)
            self._register(2, pid=103)
            popen, children = self._mock_gdb_batch(4)
            self._expire(trigger)
            _write_progress(trigger, 0, 0)
            self._complete_dump(popen, children[0], 0)
            self._poll_at(3)
            self._complete_dump(popen, children[1], 1)
            self._poll_at(4)
            _write_progress(waiter, 1, 2)
            self._poll_at(5)

            with mock.patch.object(
                watchdog._Dump, "owns_trace", return_value=True
            ) as trace:
                self._poll_at(14)
                self._poll_at(15)
                self.assertEqual(len(self.monitor._reaping), 1)
                self._poll_at(18)
                self.assertEqual(popen.call_count, 3)
                trace.return_value = False
                self._poll_at(19)
                self.assertEqual(popen.call_count, 3)
                self._poll_at(20)
                self.assertEqual(popen.call_count, 4)
                self.assertIn(".rank1-", popen.call_args.args[0][2])

    def test_tracer_must_belong_to_our_capture_group(self):
        trigger = self._register()
        self._mock_helper()
        self._expire(trigger)
        capture = self.monitor._dump
        status = self.output.parent / "status"
        status.write_text("TracerPid:\t123\n")
        with (
            mock.patch.object(Path, "glob", return_value=[status]),
            mock.patch.object(watchdog.os, "getpgid", return_value=456) as getpgid,
        ):
            self.assertFalse(capture.owns_trace())
            getpgid.return_value = capture.process.pid
            self.assertTrue(capture.owns_trace())

    def test_failed_and_timed_out_dumps_continue_with_other_ranks(self):
        real_ticks = watchdog._pid_start_ticks
        delayed_pid = 101001
        with mock.patch.object(
            watchdog,
            "_pid_start_ticks",
            side_effect=lambda pid: 200 if pid == delayed_pid else real_ticks(pid),
        ):
            first = self._register(0)
            self._register(1, pid=delayed_pid)
            self._register(2)
            popen, children = self._mock_gdb_batch(3)
            self._expire(first)
            self._complete_dump(popen, children[0], 0, code=1)
            self._poll_at(3)
            self.assertEqual(popen.call_count, 2)
            self._poll_at(13)
            self.signal_helper.assert_called_once_with(children[1].pid, signal.SIGTERM)
            self.assertEqual(popen.call_count, 2)
            self._poll_at(15)
            self.signal_helper.assert_called_once_with(children[1].pid, signal.SIGTERM)
            self.assertEqual(popen.call_count, 3)
            children[1].poll.return_value = -9
            self._complete_dump(popen, children[2], 2)
            self._poll_at(16)
            self.assertFalse(list(self.output.glob("rank1*/core")))
            self.assertEqual(len(list(self.output.glob("rank2*/core"))), 1)
            self._poll_at(17)
            self.assertEqual(popen.call_count, 3)

    def test_dead_and_replaced_queued_pids_are_skipped(self):
        first = self._register(0)
        dead, replaced = 101001, 101002
        known_times = {dead: 10, replaced: 20}
        orig_pid_start_ticks = watchdog._pid_start_ticks

        def start_ticks(pid):
            if pid in known_times:
                return known_times[pid]
            return orig_pid_start_ticks(pid)

        popen, children = self._mock_gdb_batch(2)
        with mock.patch.object(watchdog, "_pid_start_ticks", side_effect=start_ticks):
            self._register(1, pid=dead)
            self._register(2, pid=replaced)
            self._register(3)
            self._expire(first)
            known_times[dead] = None
            known_times[replaced] += 1
            self._complete_dump(popen, children[0], 0)
            self._poll_at(3)
            self.assertEqual(popen.call_count, 2)
            self._complete_dump(popen, children[1], 3)
            self._poll_at(4)
        self.assertEqual(len(list(self.output.glob("rank*/core"))), 2)

    def test_close_cancels_pending_cores(self):
        first = self._register(0)
        self._register(1)
        popen = self._mock_helper()
        self._expire(first)
        self.monitor.close()
        self._poll_at(20)
        self.signal_helper.assert_called_once_with(
            popen.return_value.pid, signal.SIGTERM
        )
        popen.assert_called_once()
        self.assertFalse(list(self.output.glob("rank*/core")))

    def test_reused_pid_is_ignored(self):
        path = self._register()
        popen = self._mock_helper()
        with mock.patch.object(watchdog, "_pid_start_ticks", return_value=-1):
            self._expire(path)
        popen.assert_not_called()

    def test_reused_pid_after_storage_precheck_never_attaches(self):
        self._register()
        identity = self.monitor._ranks[0]

        def disk_usage(_output):
            ticks.return_value = identity.start_ticks + 1
            return mock.Mock(free=watchdog._MIN_FREE_BYTES + 8192)

        with (
            mock.patch.object(
                watchdog, "_pid_start_ticks", return_value=identity.start_ticks
            ) as ticks,
            mock.patch.object(watchdog.shutil, "disk_usage", side_effect=disk_usage),
            mock.patch.object(watchdog.shutil, "which", return_value="/fake/gdb"),
            mock.patch.object(watchdog, "_limit_capture_files", return_value=8192),
            mock.patch.object(watchdog.subprocess, "Popen") as attach,
            self.assertRaisesRegex(RuntimeError, "identity changed"),
        ):
            watchdog._run_capture(self.output / ".rank0", identity, False)
        attach.assert_not_called()
        self.assertEqual(list(self.output.iterdir()), [])

    def test_storage_low_never_attaches(self):
        self._register()
        with (
            mock.patch.object(
                watchdog.shutil, "disk_usage", return_value=mock.Mock(free=1)
            ),
            mock.patch.object(watchdog.shutil, "which", return_value="/fake/gdb"),
            mock.patch.object(watchdog.subprocess, "Popen") as attach,
            self.assertRaisesRegex(OSError, "insufficient free space"),
        ):
            watchdog._run_capture(self.output / ".rank0", self.monitor._ranks[0], False)
        attach.assert_not_called()
        self.assertEqual(list(self.output.iterdir()), [])

    def test_health_poll_does_not_check_capture_storage(self):
        path = self._register()
        launch = self._mock_helper()
        with mock.patch.object(
            watchdog.shutil, "disk_usage", side_effect=AssertionError
        ):
            self._expire(path)
        launch.assert_called_once()

    def test_private_core_directory_rejects_symlinks_and_writable_ancestors(self):
        base = self.output.parent
        (base / "real").mkdir(mode=0o700)
        (base / "link").symlink_to(base / "real", target_is_directory=True)
        with self.assertRaisesRegex(OSError, "untrusted"):
            watchdog._ensure_private_core_dir(base / "link" / "cores")
        shared = base / "shared"
        shared.mkdir()
        shared.chmod(0o777)
        with self.assertRaisesRegex(OSError, "writable.*ancestor"):
            watchdog._ensure_private_core_dir(shared / "cores")
        with self.assertRaisesRegex(OSError, "must be private"):
            watchdog._ensure_private_core_dir(shared)
        watchdog._ensure_private_core_dir(base / "real")

    def test_deadline_does_not_block_on_a_mounted_filesystem(self):
        path = self._register()
        launch, (hung, next_helper) = self._mock_gdb_batch(2)
        self._expire(path, phase=5)
        self._poll_at(12)
        self.signal_helper.assert_called_once_with(hung.pid, signal.SIGTERM)
        self._poll_at(13)
        self.signal_helper.assert_called_once_with(hung.pid, signal.SIGTERM)
        hung.wait.assert_not_called()
        self.assertEqual(self.monitor._reaping[0].process, hung)
        _write_progress(path, 5, 2)
        self._poll_at(14)
        self._poll_at(16)
        launch.assert_called_once()  # Do not attach again while the helper is alive.
        hung.poll.return_value = -9
        self._poll_at(17)
        launch.assert_called_once()
        self._poll_at(18)
        self.assertEqual(launch.call_count, 2)
        self.monitor.close()
        self.signal_helper.assert_called_with(next_helper.pid, signal.SIGTERM)
        next_helper.wait.assert_called_once_with(timeout=0.2)

    def test_timed_out_worker_reaps_debugger_and_removes_partial_core(self):
        bin_dir = self.output.parent / "bin"
        bin_dir.mkdir()
        debugger = bin_dir / "gdb"
        debugger.write_text(
            f"#!{sys.executable}\n"
            "from pathlib import Path\n"
            "import time\n"
            "Path('core.partial').write_bytes(b'incomplete')\n"
            "time.sleep(20)\n"
        )
        debugger.chmod(0o700)
        self.monitor._timeout = 1.5
        path = self._register()
        with mock.patch.dict(
            os.environ, {"PATH": str(bin_dir) + ":" + os.environ["PATH"]}
        ):
            self._expire(path)
            capture = self.monitor._dump
            self.assertIsNotNone(capture)
            deadline = time.monotonic() + 5
            partial = capture.staging / "core.partial"
            while not partial.exists() and time.monotonic() < deadline:
                time.sleep(0.01)
            self.assertTrue(partial.exists())
            self.monitor._poll_dump(capture.deadline)
            self.assertNotEqual(capture.process.wait(timeout=5), 0)
            self.assertFalse(capture.staging.exists())

    def test_success_publishes_only_valid_private_core(self):
        staging = self.output / ".rank0"
        staging.mkdir(mode=0o700, parents=True)
        elf = _elf_core_fixture()
        (staging / "core.partial").write_bytes(elf)
        core = watchdog._publish_core(staging, 0, gpu=False)
        self.assertFalse(staging.exists())
        self.assertEqual(core.read_bytes(), elf)
        self.assertEqual(stat.S_IMODE(core.stat().st_mode), 0o600)

    def test_zero_exit_without_complete_elf_does_not_publish(self):
        staging = self.output / ".rank0"
        staging.mkdir(mode=0o700, parents=True)
        (staging / "core.partial").write_bytes(_elf_core_fixture()[:-1])
        with self.assertRaisesRegex(ValueError, "complete ELF core"):
            watchdog._publish_core(staging, 0, gpu=False)
        self.assertFalse(list(self.output.glob("rank*/core")))

    def test_gpu_stream_rejects_oversize_but_allows_writer_to_finish(self):
        staging = self.output / ".rank0"
        staging.mkdir(mode=0o700, parents=True)
        command = [
            sys.executable,
            "-c",
            "from pathlib import Path\n"
            "with open('gpu-core-stream', 'wb') as stream:\n"
            "    stream.write(b'x' * (2 * 1024 * 1024))\n"
            "Path('writer-finished').touch()\n",
        ]
        with (
            (staging / "cuda-gdb.log").open("wb") as log,
            self.assertRaisesRegex(OSError, "exceeds available"),
        ):
            watchdog._collect_gpu_core(command, staging, log, 128 * 1024)
        self.assertTrue((staging / "writer-finished").is_file())
        self.assertLessEqual(
            (staging / "gpu.nvcudmp.partial").stat().st_size, 128 * 1024
        )

    @unittest.skipUnless(
        os.getenv("RTP_LLM_TEST_PREFILL_CORE") == "1" and shutil.which("gdb"),
        "set RTP_LLM_TEST_PREFILL_CORE=1 in a ptrace-capable Linux container",
    )
    def test_real_child_core_and_continued_child_execution(self):
        child = subprocess.Popen(
            [
                sys.executable,
                "-u",
                "-c",
                "import sys\n"
                "print('running')\n"
                "for line in sys.stdin:\n"
                " print(line.strip())\n",
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
        )

        def stop_child():
            child.terminate()
            child.communicate(timeout=5)

        self.addCleanup(stop_child)
        self.assertEqual(child.stdout.readline(), b"running\n")
        path = self._register(pid=child.pid)
        _write_progress(path, 1, 1, tid=child.pid)
        monitor_deadline = time.monotonic() + 13
        while time.monotonic() < monitor_deadline:
            self.monitor.poll()
            cores = list(self.output.glob("rank*/core"))
            if cores:
                break
            time.sleep(0.05)
        self.assertEqual(len(cores), 1)
        self.assertTrue(watchdog._is_elf_core(cores[0]))
        self.assertGreater(cores[0].stat().st_size, 64)
        self.assertIsNone(child.poll())
        state = Path(f"/proc/{child.pid}/status").read_text()
        self.assertIn("TracerPid:\t0\n", state)
        child.stdin.write(b"after-detach\n")
        child.stdin.flush()
        self.assertTrue(select.select([child.stdout], [], [], 2)[0])
        self.assertEqual(child.stdout.readline(), b"after-detach\n")


if __name__ == "__main__":
    unittest.main()
