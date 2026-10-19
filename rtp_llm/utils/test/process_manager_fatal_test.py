"""Regression for shutdown notification after a waiting rank dies."""

import multiprocessing
import os
import resource
import signal
import time
import unittest

from rtp_llm.utils.process_manager import (
    ProcessManager,
    SharedDrainingFlag,
    create_service_draining,
)
from unittest.mock import patch


def _wait_on_event(event, ready):
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    # Model TCPStore's os._exit(1) while another thread waits on the Event.
    signal.signal(signal.SIGUSR1, lambda *_: os._exit(1))
    ready.send("waiting")
    ready.close()
    event.wait()


def _ignore_sigterm(ready):
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    ready.send(os.getpid())
    ready.close()
    while True:
        time.sleep(1)


def _run_manager(result, requested, exit_signal=signal.SIGKILL):
    # A private session lets the test clean up a regressed/hung process tree.
    os.setsid()
    ctx = multiprocessing.get_context("spawn")
    event = SharedDrainingFlag()
    ready_reader, ready_writer = ctx.Pipe(duplex=False)
    waiter = ctx.Process(target=_wait_on_event, args=(event, ready_writer))
    waiter.start()
    ready_writer.close()
    assert ready_reader.poll(5)
    assert ready_reader.recv() == "waiting"
    ready_reader.close()
    os.kill(waiter.pid, exit_signal)
    waiter.join(2)
    assert waiter.exitcode == (1 if exit_signal == signal.SIGUSR1 else -exit_signal)
    reader, writer = ctx.Pipe(duplex=False)
    survivor = ctx.Process(target=_ignore_sigterm, args=(writer,))
    survivor.start()
    writer.close()
    assert reader.poll(5)
    reader.recv()
    reader.close()
    manager = ProcessManager(monitor_interval=0.01, service_draining=event)
    manager.POST_KILL_REAP_WINDOW = 0.1
    manager.set_processes([waiter, survivor])
    if requested:
        manager.request_failure_shutdown()
    result.send(("ready", survivor.pid))
    manager._monitor_processes_health()
    manager._join_all_processes()
    result.send(("done", manager.failure_detected, survivor.exitcode))
    result.close()


def _observe_flag(flag, result):
    result.send("waiting")
    result.send(flag.wait(3))
    result.close()


class ProcessManagerFatalTest(unittest.TestCase):

    def test_draining_switch_default_enabled_and_rollback(self):
        from multiprocessing.synchronize import Event as MultiprocessingEvent

        with patch.dict(os.environ):
            os.environ.pop("RTP_LLM_USE_SHARED_DRAINING_FLAG", None)
            self.assertIsInstance(create_service_draining(), SharedDrainingFlag)
            os.environ["RTP_LLM_USE_SHARED_DRAINING_FLAG"] = "1"
            self.assertIsInstance(create_service_draining(), SharedDrainingFlag)
            os.environ["RTP_LLM_USE_SHARED_DRAINING_FLAG"] = "0"
            event = create_service_draining()
            self.assertIsInstance(event, MultiprocessingEvent)
            self.assertFalse(event.is_set())
            event.set()
            self.assertTrue(event.wait(0))

    def test_shared_flag_wait_and_spawn(self):
        flag = SharedDrainingFlag()
        self.assertFalse(flag.is_set())
        self.assertFalse(flag.wait(0))
        self.assertFalse(flag.wait(-1))
        self.assertFalse(flag.wait(0.01))
        ctx = multiprocessing.get_context("spawn")
        reader, writer = ctx.Pipe(duplex=False)
        proc = ctx.Process(target=_observe_flag, args=(flag, writer))
        proc.start()
        writer.close()
        try:
            self.assertTrue(reader.poll(3))
            self.assertEqual(reader.recv(), "waiting")
            flag.set()
            self.assertTrue(reader.poll(3))
            self.assertTrue(reader.recv())
            flag.set()
            self.assertTrue(flag.wait(0))
            proc.join(2)
            self.assertEqual(proc.exitcode, 0)
        finally:
            if proc.is_alive():
                proc.kill()
            proc.join(2)
            reader.close()

    def test_delayed_waiter_does_not_turn_graceful_shutdown_into_failure(self):
        ctx = multiprocessing.get_context("spawn")
        flag = SharedDrainingFlag()
        reader, writer = ctx.Pipe(duplex=False)
        proc = ctx.Process(target=_observe_flag, args=(flag, writer))
        proc.start()
        writer.close()
        manager = ProcessManager(service_draining=flag)
        manager._defer_first_sigterm = True
        try:
            self.assertTrue(reader.poll(3))
            self.assertEqual(reader.recv(), "waiting")
            os.kill(proc.pid, signal.SIGSTOP)
            start = time.monotonic()
            manager._signal_handler(signal.SIGTERM, None)
            self.assertLess(time.monotonic() - start, 0.5)
            self.assertFalse(manager.failure_detected)
            self.assertFalse(manager.shutdown_requested)
            self.assertTrue(manager.is_deferred_sigterm_pending())
            os.kill(proc.pid, signal.SIGCONT)
            self.assertTrue(reader.poll(3))
            self.assertTrue(reader.recv())
        finally:
            manager._cancel_deferred_sigterm_timer()
            if proc.is_alive():
                os.kill(proc.pid, signal.SIGCONT)
            proc.join(2)
            if proc.is_alive():
                proc.kill()
                proc.join(2)
            reader.close()

    def test_notification_does_not_require_a_new_thread(self):
        flag = SharedDrainingFlag()
        manager = ProcessManager(service_draining=flag)
        with patch(
            "threading.Thread.start", side_effect=RuntimeError("can't start new thread")
        ):
            manager._signal_handler(signal.SIGTERM, None)
        self.assertTrue(manager.shutdown_requested)
        self.assertFalse(manager.failure_detected)
        self.assertTrue(flag.is_set())

    def test_flag_survives_a_killed_waiter(self):
        ctx = multiprocessing.get_context("spawn")
        flag = SharedDrainingFlag()
        reader, writer = ctx.Pipe(duplex=False)
        proc = ctx.Process(target=_observe_flag, args=(flag, writer))
        proc.start()
        writer.close()
        try:
            self.assertTrue(reader.poll(3))
            self.assertEqual(reader.recv(), "waiting")
            proc.kill()
            proc.join(2)
            flag.set()
            self.assertTrue(flag.is_set())
            self.assertTrue(flag.wait(0))
        finally:
            if proc.is_alive():
                proc.kill()
            proc.join(2)
            reader.close()

    def test_killed_event_waiter_does_not_block_failure_cleanup(self):
        cases = [
            (mode, sig)
            for mode in (False, True)
            for sig in (signal.SIGKILL, signal.SIGABRT, signal.SIGUSR1)
        ]
        for requested, exit_signal in cases:
            with self.subTest(requested=requested, exit_signal=exit_signal):
                ctx = multiprocessing.get_context("spawn")
                reader, writer = ctx.Pipe(duplex=False)
                harness = ctx.Process(
                    target=_run_manager, args=(writer, requested, exit_signal)
                )
                harness.start()
                writer.close()
                try:
                    self.assertTrue(reader.poll(10), "reproduction setup failed")
                    phase, _ = reader.recv()
                    self.assertEqual(phase, "ready")
                    self.assertTrue(
                        reader.poll(3),
                        "failure cleanup hung on Event.set after a waiter was SIGKILLed",
                    )
                    self.assertEqual(reader.recv(), ("done", True, -signal.SIGKILL))
                    harness.join(2)
                    self.assertEqual(harness.exitcode, 0)
                finally:
                    # Kill only the isolated session created by this test.
                    try:
                        if os.getpgid(harness.pid) == harness.pid:
                            os.killpg(harness.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    harness.join(2)
                    reader.close()


if __name__ == "__main__":
    unittest.main()
