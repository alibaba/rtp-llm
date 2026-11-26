import os
import shutil
import signal
import stat
import struct
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

if sys.platform == "linux":
    import resource

from rtp_llm.utils import prefill_watchdog as watchdog


def _write_progress(path, phase, sequence, layer=0):
    word = phase | (layer << 4) | (sequence << 36)
    with path.open("r+b") as slot:
        slot.write(struct.pack("<Q", word))


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

    def _register(self, rank=0):
        path = Path(self.monitor.create_slot(rank))
        self.monitor.register_process(rank, os.getpid())
        return path

    def _poll_at(self, *times):
        for seconds in times:
            with mock.patch.object(watchdog.time, "monotonic", return_value=seconds):
                self.monitor.poll()

    def _mock_capture(self):
        popen = mock.patch.object(subprocess, "Popen")
        kill = mock.patch.object(os, "killpg")
        self.launch = popen.start()
        self.kill_group = kill.start()
        self.addCleanup(popen.stop)
        self.addCleanup(kill.stop)
        self.addCleanup(self.monitor.close)
        self.launch.return_value.pid = 999001
        self.launch.return_value.poll.return_value = None

    def _expire(self, path, phase=1):
        _write_progress(path, phase, 1)
        self._poll_at(1, 2)

    def test_thresholds_and_gpu_opt_in(self):
        configure = watchdog.PrefillWatchdog.from_config
        for gpu in (False, True):
            self.assertIsNone(configure("", str(self.output), gpu))
            enabled = configure(" prepare = 5,layer=20 ", str(self.output), gpu)
            self.addCleanup(enabled.close)
            self.assertEqual(enabled._gpu_core, gpu)
        with self.assertLogs(watchdog.__name__, "WARNING"):
            self.assertIsNone(configure("layer=nan", str(self.output), False))

    def test_idle_and_real_progress_do_not_trigger(self):
        path = self._register()
        self._mock_capture()
        self._poll_at(100)
        for seq in range(1, 8):
            _write_progress(path, 2, seq, layer=seq)
            self._poll_at(100 + seq)
        self.launch.assert_not_called()

    def test_unconfigured_stage_warns_without_attaching(self):
        path = self._register()
        self._mock_capture()
        with self.assertLogs(watchdog.__name__, "WARNING") as captured:
            self._expire(path, phase=3)
            self._poll_at(9)
        self.assertEqual(len(captured.output), 1)
        self.assertIn("without a configured deadline", captured.output[0])
        self.assertIn("phase=graph", captured.output[0])
        self.launch.assert_not_called()

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
            self.assertEqual(saved, self.output / f"rank{index}/gpu.nvcudmp")
            self.assertFalse(staging.exists())
            self.assertEqual(saved.read_bytes(), b"gpu core fixture")
            self.assertEqual(stat.S_IMODE(saved.stat().st_mode), 0o600)
            self.assertEqual(saved.with_name("cuda-gdb.log").read_text(), diagnostic)

    def test_close_cancels_pending_cores(self):
        self._mock_capture()
        trigger = self._register()
        self._register(1)
        self._expire(trigger)
        self.monitor.close()
        self._poll_at(20)
        self.kill_group.assert_called_once_with(
            self.launch.return_value.pid, signal.SIGTERM
        )
        self.launch.assert_called_once()

    def test_reused_pid_is_ignored(self):
        path = self._register()
        self._mock_capture()
        with mock.patch.object(watchdog, "_pid_start_ticks", return_value=-1):
            self._expire(path)
        self.launch.assert_not_called()

    def test_low_disk_space_prevents_attach(self):
        self._register()
        with (
            mock.patch.object(shutil, "which", return_value="/fake/gdb"),
            mock.patch.object(shutil, "disk_usage", return_value=mock.Mock(free=1)),
            mock.patch.object(subprocess, "Popen") as attach,
            self.assertRaisesRegex(OSError, "insufficient free space"),
        ):
            watchdog._run_capture(self.output / ".rank0", self.monitor._ranks[0], False)
        attach.assert_not_called()

    def test_capture_respects_inherited_file_limit(self):
        with (
            mock.patch.object(resource, "getrlimit", return_value=(8192, 16384)),
            mock.patch.object(resource, "setrlimit") as set_limit,
        ):
            self.assertEqual(watchdog._limit_capture_files(65536), 8192)
        set_limit.assert_called_once_with(resource.RLIMIT_FSIZE, (8192, 16384))

    def test_health_poll_skips_storage_check_and_does_not_retry_failed_capture(self):
        self._mock_capture()
        path = self._register()
        with mock.patch.object(shutil, "disk_usage", side_effect=AssertionError):
            self._expire(path)
            self.launch.return_value.poll.return_value = 1
            self._poll_at(3, 20)
        self.launch.assert_called_once()

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

    def test_timeout_signals_helper_once(self):
        self._mock_capture()
        self._expire(self._register())
        self._poll_at(11)
        self.kill_group.assert_not_called()
        self._poll_at(12, 12.5)
        self.kill_group.assert_called_once_with(
            self.launch.return_value.pid, signal.SIGTERM
        )
        self.launch.return_value.wait.assert_not_called()

    def test_success_publishes_only_valid_private_core(self):
        staging = self.output / ".rank0"
        staging.mkdir(mode=0o700, parents=True)
        elf = _elf_core_fixture()
        (staging / "core.partial").write_bytes(elf)
        core = watchdog._publish_core(staging, 0, gpu=False)
        self.assertEqual(core, self.output / "rank0/core")
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


if __name__ == "__main__":
    unittest.main()
