import os
import signal
import sys
import time
import unittest
from contextlib import suppress
from unittest.mock import Mock, patch

from rtp_llm.utils.parent_death_signal import install_parent_death_signal


def _spawn_supervisor_with_protected_backend():
    """Return a supervisor PID and its PDEATHSIG-protected backend PID."""
    read_fd, write_fd = os.pipe()
    supervisor_pid = os.fork()
    if supervisor_pid == 0:
        os.close(read_fd)
        expected_parent_pid = os.getpid()
        backend_pid = os.fork()
        if backend_pid == 0:
            try:
                install_parent_death_signal(expected_parent_pid=expected_parent_pid)
                os.write(write_fd, str(os.getpid()).encode())
                os.close(write_fd)
                while True:
                    signal.pause()
            finally:
                os._exit(1)

        os.close(write_fd)
        os.waitpid(backend_pid, 0)
        os._exit(0)

    os.close(write_fd)
    backend_pid_raw = os.read(read_fd, 32)
    os.close(read_fd)
    if not backend_pid_raw:
        os.waitpid(supervisor_pid, 0)
        raise RuntimeError("protected backend exited before reporting its PID")
    return supervisor_pid, int(backend_pid_raw)


def _process_has_exited(pid: int) -> bool:
    try:
        with open(f"/proc/{pid}/stat") as stat_file:
            fields = stat_file.read().split()
    except FileNotFoundError:
        return True
    return len(fields) > 2 and fields[2] == "Z"


class ParentDeathSignalTest(unittest.TestCase):
    @patch("rtp_llm.utils.parent_death_signal.sys.platform", "darwin")
    def test_non_linux_is_a_noop(self):
        with patch("rtp_llm.utils.parent_death_signal.ctypes.CDLL") as cdll:
            self.assertFalse(install_parent_death_signal())
            cdll.assert_not_called()

    @patch("rtp_llm.utils.parent_death_signal.sys.platform", "linux")
    def test_installs_sigkill_for_stable_parent(self):
        libc = Mock()
        libc.prctl.return_value = 0
        with (
            patch("rtp_llm.utils.parent_death_signal.ctypes.CDLL", return_value=libc),
            patch(
                "rtp_llm.utils.parent_death_signal.os.getppid",
                side_effect=[123, 123],
            ),
            patch("rtp_llm.utils.parent_death_signal.os.kill") as kill,
        ):
            self.assertTrue(install_parent_death_signal())

        libc.prctl.assert_called_once_with(1, signal.SIGKILL, 0, 0, 0)
        kill.assert_not_called()

    @patch("rtp_llm.utils.parent_death_signal.sys.platform", "linux")
    def test_kills_self_if_parent_exited_before_setup(self):
        with (
            patch("rtp_llm.utils.parent_death_signal.os.getppid", return_value=1),
            patch("rtp_llm.utils.parent_death_signal.os.getpid", return_value=456),
            patch("rtp_llm.utils.parent_death_signal.os.kill") as kill,
            patch("rtp_llm.utils.parent_death_signal.ctypes.CDLL") as cdll,
        ):
            self.assertTrue(install_parent_death_signal(expected_parent_pid=123))

        kill.assert_called_once_with(456, signal.SIGKILL)
        cdll.assert_not_called()

    @patch("rtp_llm.utils.parent_death_signal.sys.platform", "linux")
    def test_kills_self_if_parent_exits_during_setup(self):
        libc = Mock()
        libc.prctl.return_value = 0
        with (
            patch("rtp_llm.utils.parent_death_signal.ctypes.CDLL", return_value=libc),
            patch(
                "rtp_llm.utils.parent_death_signal.os.getppid",
                side_effect=[123, 1],
            ),
            patch("rtp_llm.utils.parent_death_signal.os.getpid", return_value=456),
            patch("rtp_llm.utils.parent_death_signal.os.kill") as kill,
        ):
            self.assertTrue(install_parent_death_signal())

        kill.assert_called_once_with(456, signal.SIGKILL)

    @patch("rtp_llm.utils.parent_death_signal.sys.platform", "linux")
    def test_surfaces_prctl_failure(self):
        libc = Mock()
        libc.prctl.return_value = -1
        with (
            patch("rtp_llm.utils.parent_death_signal.ctypes.CDLL", return_value=libc),
            patch("rtp_llm.utils.parent_death_signal.os.getppid", return_value=123),
            patch(
                "rtp_llm.utils.parent_death_signal.ctypes.get_errno", return_value=22
            ),
        ):
            with self.assertRaises(OSError) as ctx:
                install_parent_death_signal()

        self.assertEqual(ctx.exception.errno, 22)

    @unittest.skipUnless(sys.platform == "linux", "requires Linux prctl and fork")
    def test_backend_dies_when_supervisor_is_killed(self):
        supervisor_pid, backend_pid = _spawn_supervisor_with_protected_backend()
        try:
            os.kill(supervisor_pid, signal.SIGKILL)
            os.waitpid(supervisor_pid, 0)

            deadline = time.monotonic() + 3
            while time.monotonic() < deadline:
                if _process_has_exited(backend_pid):
                    break
                time.sleep(0.01)

            self.assertTrue(
                _process_has_exited(backend_pid),
                "a backend survived its supervisor",
            )
        finally:
            for pid in (backend_pid, supervisor_pid):
                if not _process_has_exited(pid):
                    with suppress(ProcessLookupError):
                        os.kill(pid, signal.SIGKILL)


if __name__ == "__main__":
    unittest.main()
