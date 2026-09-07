import multiprocessing
import os
import signal
import sys
import time
import unittest
from unittest.mock import Mock, patch

from rtp_llm.utils.process_lifecycle import arm_parent_death_sigkill
from rtp_llm.utils.process_manager import ProcessManager


def _guarded_child(expected_parent_pid, ready_writer):
    arm_parent_death_sigkill(expected_parent_pid)
    ready_writer.send(os.getpid())
    ready_writer.close()
    while True:
        time.sleep(0.1)


def _child_owner(ready_writer):
    ctx = multiprocessing.get_context("spawn")
    child = ctx.Process(
        target=_guarded_child,
        args=(os.getpid(), ready_writer),
        name="pdeathsig_guarded_child",
    )
    child.start()
    ready_writer.close()
    while True:
        time.sleep(0.1)


def _delayed_guarded_child(expected_parent_pid, gate, child_writer):
    child_writer.send(os.getpid())
    child_writer.close()
    gate.wait()
    arm_parent_death_sigkill(expected_parent_pid)
    while True:
        time.sleep(0.1)


def _delayed_child_owner(gate, child_writer):
    ctx = multiprocessing.get_context("spawn")
    child = ctx.Process(
        target=_delayed_guarded_child,
        args=(os.getpid(), gate, child_writer),
        name="delayed_pdeathsig_guarded_child",
    )
    child.start()
    child_writer.close()
    while True:
        time.sleep(0.1)


def _ignore_sigterm_worker(ready_writer=None):
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    if ready_writer is not None:
        ready_writer.send(True)
        ready_writer.close()
    while True:
        time.sleep(0.1)


def _crash_worker(gate_reader):
    gate_reader.recv()
    gate_reader.close()
    os._exit(7)


def _backend_with_crashing_rank(status_writer):
    ctx = multiprocessing.get_context("spawn")
    crash_reader, crash_writer = ctx.Pipe(duplex=False)
    stubborn_reader, stubborn_writer = ctx.Pipe(duplex=False)
    crashed_rank = ctx.Process(target=_crash_worker, args=(crash_reader,))
    stubborn_rank = ctx.Process(target=_ignore_sigterm_worker, args=(stubborn_writer,))
    crashed_rank.start()
    crash_reader.close()
    stubborn_rank.start()
    stubborn_writer.close()
    if not stubborn_reader.poll(5):
        raise RuntimeError("stubborn rank did not start")
    stubborn_reader.recv()
    stubborn_reader.close()
    status_writer.send({"rank_pids": [crashed_rank.pid, stubborn_rank.pid]})
    status_writer.close()
    crash_writer.send(True)
    crash_writer.close()

    manager = ProcessManager(shutdown_timeout=1, monitor_interval=0.02)
    manager.POST_KILL_REAP_WINDOW = 0.2
    manager.set_processes([crashed_rank, stubborn_rank], shutdown_group="backend")
    manager.monitor_and_release_processes()


def _main_with_failing_backend(status_writer):
    ctx = multiprocessing.get_context("spawn")
    backend_reader, backend_writer = ctx.Pipe(duplex=False)
    frontend_reader, frontend_writer = ctx.Pipe(duplex=False)
    backend = ctx.Process(
        target=_backend_with_crashing_rank,
        args=(backend_writer,),
    )
    frontend = ctx.Process(target=_ignore_sigterm_worker, args=(frontend_writer,))
    frontend.start()
    frontend_writer.close()
    if not frontend_reader.poll(5):
        raise RuntimeError("frontend did not start")
    frontend_reader.recv()
    frontend_reader.close()
    backend.start()
    backend_writer.close()

    if not backend_reader.poll(5):
        raise RuntimeError("backend did not report rank pids")
    status = backend_reader.recv()
    backend_reader.close()
    status_writer.send(
        {
            "backend_pid": backend.pid,
            "frontend_pid": frontend.pid,
            **status,
        }
    )
    status_writer.close()

    manager = ProcessManager(shutdown_timeout=1, monitor_interval=0.02)
    manager.POST_KILL_REAP_WINDOW = 0.2
    manager.add_process(backend, shutdown_group="backend")
    manager.add_process(frontend, shutdown_group="frontend")
    manager.monitor_and_release_processes()


def _pid_is_running(pid):
    try:
        with open(f"/proc/{pid}/stat", "r") as stat_file:
            # A zombie has already obeyed SIGKILL and cannot execute work.
            return stat_file.read().split()[2] != "Z"
    except FileNotFoundError:
        return False


@unittest.skipUnless(sys.platform.startswith("linux"), "requires Linux PDEATHSIG")
class ProcessLifecycleTest(unittest.TestCase):
    @patch("rtp_llm.utils.process_lifecycle.ctypes.CDLL")
    @patch("rtp_llm.utils.process_lifecycle.sys.platform", "darwin")
    def test_non_linux_is_an_explicit_noop(self, cdll_mock):
        self.assertFalse(arm_parent_death_sigkill(123))
        cdll_mock.assert_not_called()

    @patch("rtp_llm.utils.process_lifecycle.ctypes.CDLL")
    @patch("rtp_llm.utils.process_lifecycle.os.getppid", return_value=1)
    def test_pid_one_is_a_valid_parent(self, _getppid_mock, cdll_mock):
        cdll_mock.return_value.prctl = Mock(return_value=0)

        self.assertTrue(arm_parent_death_sigkill(1))

    @patch("rtp_llm.utils.process_lifecycle.logging.error")
    @patch("rtp_llm.utils.process_lifecycle.os.kill")
    @patch("rtp_llm.utils.process_lifecycle.os.getpid", return_value=123)
    @patch("rtp_llm.utils.process_lifecycle.os.getppid", return_value=1)
    @patch("rtp_llm.utils.process_lifecycle.ctypes.CDLL")
    def test_parent_change_kills_without_logging_first(
        self,
        cdll_mock,
        _getppid_mock,
        _getpid_mock,
        kill_mock,
        log_mock,
    ):
        cdll_mock.return_value.prctl = Mock(return_value=0)

        with self.assertRaisesRegex(RuntimeError, "unreachable"):
            arm_parent_death_sigkill(456)

        kill_mock.assert_called_once_with(123, signal.SIGKILL)
        log_mock.assert_not_called()

    def test_parent_dies_before_guard_is_armed(self):
        ctx = multiprocessing.get_context("spawn")
        gate = ctx.Event()
        child_reader, child_writer = ctx.Pipe(duplex=False)
        owner = ctx.Process(target=_delayed_child_owner, args=(gate, child_writer))
        owner.start()
        child_writer.close()
        self.assertTrue(child_reader.poll(5), "guarded child did not start")
        child_pid = child_reader.recv()
        child_reader.close()

        try:
            os.kill(owner.pid, signal.SIGKILL)
            owner.join(timeout=2)
            self.assertFalse(owner.is_alive())
            gate.set()

            deadline = time.monotonic() + 3
            while _pid_is_running(child_pid) and time.monotonic() < deadline:
                time.sleep(0.05)
            self.assertFalse(
                _pid_is_running(child_pid),
                f"guarded child {child_pid} accepted a replacement parent",
            )
        finally:
            gate.set()
            if owner.is_alive():
                owner.kill()
                owner.join(timeout=1)
            if _pid_is_running(child_pid):
                os.kill(child_pid, signal.SIGKILL)

    def test_parent_sigkill_kills_guarded_child(self):
        ctx = multiprocessing.get_context("spawn")
        ready_reader, ready_writer = ctx.Pipe(duplex=False)
        owner = ctx.Process(target=_child_owner, args=(ready_writer,))
        owner.start()
        ready_writer.close()
        self.assertTrue(ready_reader.poll(5), "guarded child did not start")
        child_pid = ready_reader.recv()
        ready_reader.close()

        try:
            os.kill(owner.pid, signal.SIGKILL)
            owner.join(timeout=2)
            self.assertFalse(owner.is_alive())

            deadline = time.monotonic() + 3
            while _pid_is_running(child_pid) and time.monotonic() < deadline:
                time.sleep(0.05)
            self.assertFalse(
                _pid_is_running(child_pid),
                f"guarded child {child_pid} survived parent death",
            )
        finally:
            if owner.is_alive():
                owner.kill()
                owner.join(timeout=1)
            if _pid_is_running(child_pid):
                os.kill(child_pid, signal.SIGKILL)

    def test_rank_crash_cleans_stubborn_sibling_and_whole_instance(self):
        ctx = multiprocessing.get_context("spawn")
        status_reader, status_writer = ctx.Pipe(duplex=False)
        main = ctx.Process(target=_main_with_failing_backend, args=(status_writer,))
        main.start()
        status_writer.close()
        tracked_pids = [main.pid]

        try:
            self.assertTrue(status_reader.poll(5), "nested process tree did not start")
            status = status_reader.recv()
            tracked_pids.extend(
                [status["backend_pid"], status["frontend_pid"], *status["rank_pids"]]
            )
            status_reader.close()

            main.join(timeout=5)
            self.assertFalse(main.is_alive(), "top-level manager did not exit")
            self.assertNotEqual(main.exitcode, 0)

            deadline = time.monotonic() + 3
            while (
                any(_pid_is_running(pid) for pid in tracked_pids)
                and time.monotonic() < deadline
            ):
                time.sleep(0.05)
            self.assertEqual(
                [pid for pid in tracked_pids if _pid_is_running(pid)],
                [],
                "failure cleanup left a tracked process running",
            )
        finally:
            status_reader.close()
            if main.is_alive():
                main.kill()
                main.join(timeout=1)
            for pid in tracked_pids[1:]:
                if _pid_is_running(pid):
                    os.kill(pid, signal.SIGKILL)


if __name__ == "__main__":
    unittest.main()
