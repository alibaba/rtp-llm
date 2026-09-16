import logging
import os
import tempfile
import threading
import time
import unittest
from unittest import mock

from rtp_llm.access_logger.async_log_handler import AsyncRotatingFileHandler


def _record(message: str) -> logging.LogRecord:
    return logging.LogRecord(
        name="access",
        level=logging.INFO,
        pathname=__file__,
        lineno=0,
        msg=message,
        args=(),
        exc_info=None,
    )


def _wait_until(predicate, timeout: float = 1.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.001)
    return predicate()


class _CrashOnceHandler(AsyncRotatingFileHandler):
    def __init__(self, *args, **kwargs):
        self._crash_once = True
        super().__init__(*args, **kwargs)

    def _worker_loop(self) -> None:
        if self._crash_once:
            self._crash_once = False
            return
        super()._worker_loop()


class _BackloggedDeadWorkerHandler(AsyncRotatingFileHandler):
    def __init__(self, *args, **kwargs):
        self.first_worker_started = threading.Event()
        self.stop_first_worker = threading.Event()
        self._first_worker = True
        super().__init__(*args, **kwargs)

    def _worker_loop(self) -> None:
        if self._first_worker:
            self.first_worker_started.set()
            self.stop_first_worker.wait()
            self._first_worker = False
            return
        super()._worker_loop()


class AsyncRotatingFileHandlerTest(unittest.TestCase):
    def test_close_drains_queue_and_stops_worker(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            log_path = os.path.join(temp_dir, "access.log")
            handler = AsyncRotatingFileHandler(
                log_path,
                max_queue_size=128,
                flush_interval=0.01,
            )
            handler.setFormatter(logging.Formatter("%(message)s"))

            for index in range(50):
                handler.emit(_record(f"record-{index}"))

            handler.close()

            with open(log_path, encoding="utf-8") as log_file:
                self.assertEqual(
                    log_file.read().splitlines(),
                    [f"record-{index}" for index in range(50)],
                )
            stats = handler.get_stats()
            self.assertEqual(stats["enqueued"], 50)
            self.assertEqual(stats["written"], 50)
            self.assertEqual(stats["dropped"], 0)
            self.assertEqual(stats["write_errors"], 0)
            self.assertEqual(stats["pending"], 0)
            self.assertEqual(stats["queue_depth"], 0)
            self.assertFalse(stats["worker_alive"])

            handler.close()
            handler.emit(_record("ignored-after-close"))
            stats = handler.get_stats()
            self.assertEqual(stats["enqueued"], 50)
            self.assertEqual(stats["rejected_closing"], 1)

    def test_concurrent_worker_start_creates_only_one_worker(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            handler = AsyncRotatingFileHandler(
                os.path.join(temp_dir, "access.log"), flush_interval=0.01
            )
            starters = [threading.Thread(target=handler._start_worker) for _ in range(16)]
            for starter in starters:
                starter.start()
            for starter in starters:
                starter.join()

            workers = [
                thread
                for thread in threading.enumerate()
                if thread.name == f"AsyncLogWorker-{id(handler)}"
            ]
            self.assertEqual(len(workers), 1)
            handler.close()

    def test_worker_start_failure_is_fail_open(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            log_path = os.path.join(temp_dir, "access.log")
            with mock.patch.object(
                threading.Thread,
                "start",
                side_effect=RuntimeError("thread unavailable"),
            ):
                handler = AsyncRotatingFileHandler(
                    log_path,
                    delay=True,
                    flush_interval=0.01,
                    drain_timeout=0.05,
                )
                handler.emit(_record("queued-without-worker"))

                flush_started = time.monotonic()
                handler.flush()
                flush_elapsed = time.monotonic() - flush_started

                close_started = time.monotonic()
                handler.close()
                close_elapsed = time.monotonic() - close_started

            stats = handler.get_stats()
            self.assertGreaterEqual(stats["worker_start_errors"], 3)
            self.assertEqual(stats["enqueued"], 1)
            self.assertEqual(stats["pending"], 1)
            self.assertLess(flush_elapsed, 0.5)
            self.assertLess(close_elapsed, 0.5)

    def test_blocked_target_io_does_not_block_flush_or_close(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            handler = AsyncRotatingFileHandler(
                os.path.join(temp_dir, "access.log"),
                flush_interval=0.01,
                drain_timeout=0.05,
            )
            target_emit = handler._file_handler.emit
            io_started = threading.Event()
            release_io = threading.Event()

            def blocked_emit(record: logging.LogRecord) -> None:
                io_started.set()
                release_io.wait()
                target_emit(record)

            handler._file_handler.emit = blocked_emit
            try:
                handler.emit(_record("blocked"))
                self.assertTrue(io_started.wait(timeout=1.0))

                flush_started = time.monotonic()
                handler.flush()
                flush_elapsed = time.monotonic() - flush_started

                close_started = time.monotonic()
                handler.close()
                close_elapsed = time.monotonic() - close_started

                self.assertLess(flush_elapsed, 0.5)
                self.assertLess(close_elapsed, 0.5)
                self.assertEqual(handler.get_stats()["pending"], 1)
            finally:
                release_io.set()

            self.assertTrue(
                _wait_until(
                    lambda: handler.get_stats()["pending"] == 0
                    and not handler.get_stats()["worker_alive"]
                )
            )

    def test_blocked_target_close_respects_the_close_deadline(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            handler = AsyncRotatingFileHandler(
                os.path.join(temp_dir, "access.log"),
                flush_interval=0.01,
                drain_timeout=0.05,
            )
            target_close = handler._file_handler.close
            close_started = threading.Event()
            release_close = threading.Event()

            def blocked_close() -> None:
                close_started.set()
                release_close.wait()
                target_close()

            handler._file_handler.close = blocked_close
            try:
                started_at = time.monotonic()
                handler.close()
                close_elapsed = time.monotonic() - started_at

                self.assertTrue(close_started.is_set())
                self.assertLess(close_elapsed, 0.5)
            finally:
                release_close.set()

            self.assertTrue(
                _wait_until(lambda: not handler.get_stats()["worker_alive"])
            )

    def test_worker_crash_is_restarted_and_queue_is_drained(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            log_path = os.path.join(temp_dir, "access.log")
            handler = _CrashOnceHandler(log_path, flush_interval=0.01)
            handler.setFormatter(logging.Formatter("%(message)s"))

            deadline = time.monotonic() + 1.0
            while handler.get_stats()["worker_alive"] and time.monotonic() < deadline:
                time.sleep(0.001)
            handler.emit(_record("after-crash"))
            handler.close()

            with open(log_path, encoding="utf-8") as log_file:
                self.assertEqual(log_file.read().splitlines(), ["after-crash"])
            self.assertEqual(handler.get_stats()["worker_restarts"], 1)

    def test_dead_worker_with_large_queue_is_restarted_without_sync_drain(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            log_path = os.path.join(temp_dir, "access.log")
            handler = _BackloggedDeadWorkerHandler(
                log_path,
                max_queue_size=4096,
                flush_interval=0.001,
                drain_timeout=2.0,
            )
            handler.setFormatter(logging.Formatter("%(message)s"))
            self.assertTrue(handler.first_worker_started.wait(timeout=1.0))

            record_count = 2000
            for index in range(record_count):
                handler.emit(_record(f"backlog-{index}"))
            handler.stop_first_worker.set()
            self.assertTrue(
                _wait_until(lambda: not handler.get_stats()["worker_alive"])
            )

            handler.flush()
            stats = handler.get_stats()
            self.assertEqual(stats["pending"], 0)
            self.assertEqual(stats["written"], record_count)
            self.assertEqual(stats["worker_restarts"], 1)
            handler.close()

            with open(log_path, encoding="utf-8") as log_file:
                self.assertEqual(len(log_file.read().splitlines()), record_count)

    def test_close_rejects_concurrent_and_late_producers(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            handler = AsyncRotatingFileHandler(
                os.path.join(temp_dir, "access.log"),
                max_queue_size=4096,
                flush_interval=0.01,
            )
            handler.setFormatter(logging.Formatter("%(message)s"))
            start = threading.Event()

            def produce(producer_id: int) -> None:
                start.wait()
                for index in range(200):
                    handler.emit(_record(f"{producer_id}-{index}"))

            producers = [
                threading.Thread(target=produce, args=(producer_id,))
                for producer_id in range(8)
            ]
            for producer in producers:
                producer.start()
            start.set()
            handler.close()
            for producer in producers:
                producer.join()
            handler.emit(_record("late"))

            stats = handler.get_stats()
            self.assertEqual(stats["pending"], 0)
            self.assertEqual(stats["written"], stats["enqueued"])
            self.assertGreater(stats["rejected_closing"], 0)
            self.assertFalse(stats["worker_alive"])


if __name__ == "__main__":
    unittest.main()
