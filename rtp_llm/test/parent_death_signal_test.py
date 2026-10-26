import ctypes
import errno
import multiprocessing
import os
import resource
import signal
import sys
import time
import unittest
from unittest.mock import Mock, patch

from rtp_llm.utils.parent_death_signal import run_with_parent_death_signal


def _idle(writer):
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    writer.send(("ready", os.getpid()))
    while True:
        signal.pause()


def _delayed_worker(parent_pid, writer, gate):
    writer.send(("ready", os.getpid()))
    gate.recv()
    run_with_parent_death_signal(parent_pid, _idle, writer)


def _manager(writer, gate=None):
    # Exercise fatal core-producing signals without writing large core files.
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    ctx = multiprocessing.get_context("spawn")
    workers = []
    for _ in range(1 if gate is not None else 2):
        if gate is None:
            target = run_with_parent_death_signal
            args = (os.getpid(), _idle, writer)
        else:
            target = _delayed_worker
            args = (os.getpid(), writer, gate)
        worker = ctx.Process(target=target, args=args)
        worker.start()
        workers.append(worker)
        writer.send(("spawned", worker.pid))
    _idle(writer)


def _server(writer):
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    ctx = multiprocessing.get_context("spawn")
    manager = ctx.Process(
        target=run_with_parent_death_signal, args=(os.getpid(), _manager, writer)
    )
    manager.start()
    writer.send(("spawned", manager.pid))
    _idle(writer)


def _report(writer, value):
    writer.send(value)


@unittest.skipUnless(sys.platform == "linux", "Linux parent-death signal")
class ParentDeathSignalTest(unittest.TestCase):
    def setUp(self):
        # Adopt and waitpid our orphaned descendants, even if container PID 1
        # does not reap zombies. Restore the test runner's original setting.
        self.libc = ctypes.CDLL(None, use_errno=True)
        self.previous_subreaper = ctypes.c_int()
        self.assertEqual(
            self.libc.prctl(37, ctypes.byref(self.previous_subreaper), 0, 0, 0), 0
        )
        self.assertEqual(self.libc.prctl(36, 1, 0, 0, 0), 0)
        self.addCleanup(self.libc.prctl, 36, self.previous_subreaper.value, 0, 0, 0)
        self.ctx = multiprocessing.get_context("spawn")

    def _waitpid(self, pid):
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            try:
                waited, status = os.waitpid(pid, os.WNOHANG)
                if waited:
                    return status
            except ChildProcessError:
                # A cascading manager death may not have reparented its ranks yet.
                pass
            time.sleep(0.01)
        self.fail(f"child {pid} did not exit and become reapable")

    def _crash(self, signum, cascade=False, delayed=False):
        reader, writer = self.ctx.Pipe(duplex=False)
        gate_reader, gate_writer = self.ctx.Pipe(duplex=False)
        parent = self.ctx.Process(
            target=_server if cascade else _manager,
            args=(writer,) if cascade else (writer, gate_reader if delayed else None),
        )
        descendants = set()
        reaped = set()
        parent.start()
        writer.close()
        gate_reader.close()
        try:
            ready = set()
            expected_children = 3 if cascade else (1 if delayed else 2)
            deadline = time.monotonic() + 30
            while (
                len(ready) != expected_children + 1
                or len(descendants) != expected_children
            ):
                self.assertTrue(reader.poll(max(0, deadline - time.monotonic())))
                event, pid = reader.recv()
                if event == "spawned":
                    descendants.add(pid)
                else:
                    ready.add(pid)
            os.kill(parent.pid, signum)
            parent.join(10)
            self.assertEqual(parent.exitcode, -signum)
            if delayed:
                gate_writer.send(True)
            for pid in descendants:
                status = self._waitpid(pid)
                reaped.add(pid)
                self.assertTrue(os.WIFSIGNALED(status))
                self.assertEqual(os.WTERMSIG(status), signal.SIGKILL)
        finally:
            if parent.is_alive():
                parent.kill()
            parent.join(10)
            for pid in descendants - reaped:
                try:
                    os.kill(pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                self._waitpid(pid)
            reader.close()
            gate_writer.close()

    def test_core_and_sigkill_reap_all_workers(self):
        for signum in (signal.SIGABRT, signal.SIGSEGV, signal.SIGKILL):
            with self.subTest(signal=signum):
                self._crash(signum)

    def test_server_crash_cascades_through_manager_to_workers(self):
        self._crash(signal.SIGABRT, cascade=True)

    def test_parent_exits_before_worker_installs_guard(self):
        self._crash(signal.SIGKILL, delayed=True)

    def test_target_runs_normally_while_parent_is_alive(self):
        reader, writer = self.ctx.Pipe(duplex=False)
        worker = self.ctx.Process(
            target=run_with_parent_death_signal,
            args=(os.getpid(), _report, writer, "healthy"),
        )
        worker.start()
        writer.close()
        try:
            self.assertTrue(reader.poll(30))
            self.assertEqual(reader.recv(), "healthy")
            worker.join(10)
            self.assertEqual(worker.exitcode, 0)
        finally:
            if worker.is_alive():
                worker.kill()
            worker.join(10)
            reader.close()

    def test_registration_failure_does_not_start_worker(self):
        target = Mock()
        with (
            patch("rtp_llm.utils.parent_death_signal.ctypes.CDLL") as library,
            patch(
                "rtp_llm.utils.parent_death_signal.ctypes.get_errno",
                return_value=errno.EPERM,
            ),
        ):
            library.return_value.prctl.return_value = -1
            with self.assertRaises(OSError) as error:
                run_with_parent_death_signal(os.getppid(), target)
        self.assertEqual(error.exception.errno, errno.EPERM)
        target.assert_not_called()


if __name__ == "__main__":
    unittest.main()
