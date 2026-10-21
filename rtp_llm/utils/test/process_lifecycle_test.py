import errno
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


def _graceful_child_owner(ready_writer, result_writer):
    ctx = multiprocessing.get_context("spawn")
    manager = ProcessManager(shutdown_timeout=1, monitor_interval=0.02)
    manager.POST_KILL_REAP_WINDOW = 0.2
    child = ctx.Process(
        target=_cooperative_guarded_child, args=(os.getpid(), ready_writer)
    )
    child.start()
    ready_writer.close()
    manager.set_processes([child], shutdown_group="backend")
    manager.monitor_and_release_processes()
    result_writer.send(child.exitcode)
    result_writer.close()


def _cooperative_guarded_child(expected_parent_pid, ready_writer):
    arm_parent_death_sigkill(expected_parent_pid)
    signal.signal(signal.SIGINT, lambda _signum, _frame: sys.exit(0))
    signal.signal(signal.SIGTERM, lambda _signum, _frame: sys.exit(0))
    ready_writer.send(os.getpid())
    ready_writer.close()
    while True:
        time.sleep(0.1)


def _unrelated_worker(stop_event, ready_writer):
    ready_writer.send(os.getpid())
    ready_writer.close()
    stop_event.wait()


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


def _guarded_backend(expected_parent_pid, status_writer):
    arm_parent_death_sigkill(expected_parent_pid)
    ctx = multiprocessing.get_context("spawn")
    rank_reader, rank_writer = ctx.Pipe(duplex=False)
    rank = ctx.Process(target=_guarded_child, args=(os.getpid(), rank_writer))
    rank.start()
    rank_writer.close()
    if not rank_reader.poll(5):
        raise RuntimeError("guarded rank did not start")
    status_writer.send((os.getpid(), rank_reader.recv()))
    rank_reader.close()
    status_writer.close()
    while True:
        time.sleep(0.1)


def _guarded_main(status_writer):
    ctx = multiprocessing.get_context("spawn")
    backend = ctx.Process(
        target=_guarded_backend,
        args=(os.getpid(), status_writer),
    )
    backend.start()
    status_writer.close()
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

    @patch("rtp_llm.utils.process_lifecycle.os.getppid")
    @patch("rtp_llm.utils.process_lifecycle.ctypes.get_errno", return_value=errno.EPERM)
    @patch("rtp_llm.utils.process_lifecycle.ctypes.CDLL")
    def test_prctl_failure_fails_closed(self, cdll_mock, _get_errno_mock, getppid_mock):
        cdll_mock.return_value.prctl = Mock(return_value=-1)

        with self.assertRaises(OSError) as error:
            arm_parent_death_sigkill(os.getpid())

        self.assertEqual(error.exception.errno, errno.EPERM)
        getppid_mock.assert_not_called()

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

    def test_main_sigkill_cascades_to_backend_and_rank(self):
        ctx = multiprocessing.get_context("spawn")
        status_reader, status_writer = ctx.Pipe(duplex=False)
        main = ctx.Process(target=_guarded_main, args=(status_writer,))
        unrelated_stop = ctx.Event()
        unrelated_reader, unrelated_writer = ctx.Pipe(duplex=False)
        unrelated = ctx.Process(
            target=_unrelated_worker, args=(unrelated_stop, unrelated_writer)
        )
        unrelated.start()
        unrelated_writer.close()
        main.start()
        status_writer.close()
        tracked_pids = [main.pid]

        try:
            self.assertTrue(unrelated_reader.poll(5), "unrelated worker did not start")
            unrelated_pid = unrelated_reader.recv()
            self.assertTrue(status_reader.poll(5), "guarded process tree did not start")
            tracked_pids.extend(status_reader.recv())
            status_reader.close()

            os.kill(main.pid, signal.SIGKILL)
            main.join(timeout=2)
            self.assertFalse(main.is_alive())

            deadline = time.monotonic() + 3
            while (
                any(_pid_is_running(pid) for pid in tracked_pids)
                and time.monotonic() < deadline
            ):
                time.sleep(0.05)
            self.assertEqual(
                [pid for pid in tracked_pids if _pid_is_running(pid)],
                [],
                "main parent death left backend or rank running",
            )
            self.assertEqual(unrelated.pid, unrelated_pid)
            self.assertTrue(unrelated.is_alive(), "unrelated worker was killed")
        finally:
            unrelated_reader.close()
            status_reader.close()
            if main.is_alive():
                main.kill()
                main.join(timeout=1)
            for pid in tracked_pids[1:]:
                if _pid_is_running(pid):
                    os.kill(pid, signal.SIGKILL)
            unrelated_stop.set()
            unrelated.join(timeout=2)
            if unrelated.is_alive():
                unrelated.kill()
                unrelated.join(timeout=1)

    def test_normal_shutdown_reaps_guarded_child_without_failure(self):
        ctx = multiprocessing.get_context("spawn")
        ready_reader, ready_writer = ctx.Pipe(duplex=False)
        result_reader, result_writer = ctx.Pipe(duplex=False)
        owner = ctx.Process(
            target=_graceful_child_owner, args=(ready_writer, result_writer)
        )
        owner.start()
        ready_writer.close()
        result_writer.close()
        child_pid = None

        try:
            self.assertTrue(ready_reader.poll(5), "guarded child did not start")
            child_pid = ready_reader.recv()

            os.kill(owner.pid, signal.SIGTERM)
            owner.join(timeout=5)
            self.assertFalse(owner.is_alive(), "normal shutdown did not finish")
            self.assertEqual(owner.exitcode, 0, "normal shutdown was marked as failure")
            self.assertTrue(result_reader.poll(1), "owner did not report child exit")
            self.assertEqual(
                result_reader.recv(), 0, "guarded child was not reaped cleanly"
            )
            self.assertFalse(
                _pid_is_running(child_pid), "guarded child survived shutdown"
            )
        finally:
            ready_reader.close()
            result_reader.close()
            if owner.is_alive():
                owner.kill()
                owner.join(timeout=1)
            if child_pid is not None and _pid_is_running(child_pid):
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
