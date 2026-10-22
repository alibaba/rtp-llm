"""Unit tests for the process-wide signal guards."""

from __future__ import annotations

import signal
import unittest
from unittest import mock

from rtp_llm.utils import signal_guard


class SignalGuardTest(unittest.TestCase):
    def setUp(self) -> None:
        self._original = signal.getsignal(signal.SIGPIPE)

    def tearDown(self) -> None:
        signal.signal(signal.SIGPIPE, self._original)

    def test_ignore_sigpipe_sets_sig_ign(self) -> None:
        signal.signal(signal.SIGPIPE, signal.SIG_DFL)
        signal_guard.ignore_sigpipe("unit test")
        self.assertIs(signal.getsignal(signal.SIGPIPE), signal.SIG_IGN)

    def test_ignore_sigpipe_is_idempotent(self) -> None:
        signal_guard.ignore_sigpipe()
        signal_guard.ignore_sigpipe()
        self.assertIs(signal.getsignal(signal.SIGPIPE), signal.SIG_IGN)

    def test_ignore_sigpipe_tolerates_off_main_thread(self) -> None:
        with mock.patch.object(
            signal, "signal", side_effect=ValueError("signal only works in main thread")
        ):
            signal_guard.ignore_sigpipe()

    def test_ignore_sigpipe_restores_default_disposition(self) -> None:
        signal.signal(signal.SIGPIPE, signal.SIG_DFL)
        with mock.patch.object(signal_guard, "_sigpipe_warning_logged", False):
            with self.assertLogs(signal_guard.LOGGER, level="WARNING") as logs:
                signal_guard.ignore_sigpipe("because a native runtime reset it")
        self.assertTrue(any("SIGPIPE" in line for line in logs.output))
        self.assertIs(signal.getsignal(signal.SIGPIPE), signal.SIG_IGN)


if __name__ == "__main__":
    unittest.main()
