from __future__ import annotations

import ctypes
import multiprocessing
import os
import signal
import tempfile
import unittest
from multiprocessing.reduction import DupFd
from pathlib import Path
from typing import Any

from rtp_llm.dash_sc.inference.core_dump_control import (
    _XGRAMMAR_SANDBOX_CORE_DUMP_ENV,
    _configure_xgrammar_sandbox_core_dump_for_current_process,
)

_PR_GET_DUMPABLE = 3
_PR_SET_DUMPABLE = 4


def _report_dumpable(conn: Any, core_dump_env: str) -> None:
    try:
        libc = ctypes.CDLL(None, use_errno=True)
        if libc.prctl(_PR_SET_DUMPABLE, 1, 0, 0, 0) != 0:
            error_number = ctypes.get_errno()
            raise OSError(error_number, os.strerror(error_number))
        os.environ[_XGRAMMAR_SANDBOX_CORE_DUMP_ENV] = core_dump_env
        _configure_xgrammar_sandbox_core_dump_for_current_process()
        conn.send(libc.prctl(_PR_GET_DUMPABLE, 0, 0, 0, 0))
    finally:
        conn.close()


def _crash_with_stderr_capture(conn: Any, fault_trace_fd: Any, probe_dir: str) -> None:
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(_PR_SET_DUMPABLE, 1, 0, 0, 0) != 0:
        error_number = ctypes.get_errno()
        raise OSError(error_number, os.strerror(error_number))
    os.environ[_XGRAMMAR_SANDBOX_CORE_DUMP_ENV] = "0"
    _configure_xgrammar_sandbox_core_dump_for_current_process()
    os.chdir(probe_dir)
    with os.fdopen(fault_trace_fd.detach(), "wb", buffering=0) as trace_file:
        os.dup2(trace_file.fileno(), 2)
        conn.send(libc.prctl(_PR_GET_DUMPABLE, 0, 0, 0, 0))
        conn.close()
        os.write(2, b"native stderr trace marker\n")
        os.kill(os.getpid(), signal.SIGSEGV)


class CoreDumpControlTest(unittest.TestCase):
    def _configured_dumpable(self, core_dump_env: str) -> int:
        context = multiprocessing.get_context("spawn")
        parent_conn, child_conn = context.Pipe(duplex=False)
        process = context.Process(
            target=_report_dumpable, args=(child_conn, core_dump_env)
        )
        process.start()
        child_conn.close()
        dumpable = parent_conn.recv()
        parent_conn.close()
        process.join(timeout=10)
        self.assertFalse(process.is_alive())
        self.assertEqual(process.exitcode, 0)
        return dumpable

    def test_environment_controls_worker_dumpability(self) -> None:
        self.assertEqual(self._configured_dumpable("0"), 0)
        self.assertEqual(self._configured_dumpable("1"), 1)

    def test_native_stderr_is_captured_without_a_core_dump(self) -> None:
        context = multiprocessing.get_context("spawn")
        with tempfile.TemporaryDirectory() as probe_dir:
            with tempfile.TemporaryFile(mode="w+b") as fault_file:
                parent_conn, child_conn = context.Pipe(duplex=False)
                process = context.Process(
                    target=_crash_with_stderr_capture,
                    args=(child_conn, DupFd(fault_file.fileno()), probe_dir),
                )
                process.start()
                child_conn.close()
                self.assertTrue(parent_conn.poll(10))
                self.assertEqual(parent_conn.recv(), 0)
                parent_conn.close()
                process.join(timeout=10)

                self.assertFalse(process.is_alive())
                self.assertEqual(process.exitcode, -signal.SIGSEGV)
                fault_file.seek(0)
                fault_trace = fault_file.read().decode("utf-8", errors="replace")
            self.assertIn("native stderr trace marker", fault_trace)
            self.assertEqual(list(Path(probe_dir).glob("core*")), [])


if __name__ == "__main__":
    unittest.main()
