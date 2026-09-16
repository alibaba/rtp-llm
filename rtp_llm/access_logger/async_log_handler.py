# -*- coding: utf-8 -*-
"""Non-blocking rotating file handler with a bounded, drainable worker queue."""

import logging
import queue
import sys
import threading
import time
from logging.handlers import RotatingFileHandler
from typing import Optional


class AsyncRotatingFileHandler(logging.Handler):
    """Write records on one background worker without blocking producers."""

    def __init__(
        self,
        filename: str,
        mode: str = "a",
        max_bytes: int = 0,
        backup_count: int = 0,
        encoding: Optional[str] = None,
        delay: bool = False,
        max_queue_size: int = 10000,
        flush_interval: float = 1.0,
        drain_timeout: Optional[float] = None,
        **kwargs,
    ):
        super().__init__()
        if max_queue_size <= 0:
            raise ValueError("max_queue_size must be greater than 0")
        if flush_interval <= 0:
            raise ValueError("flush_interval must be greater than 0")
        if drain_timeout is not None and drain_timeout <= 0:
            raise ValueError("drain_timeout must be greater than 0")

        self._file_handler = RotatingFileHandler(
            filename=filename,
            mode=mode,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding=encoding,
            delay=delay,
            **kwargs,
        )
        self._flush_interval = flush_interval
        self._drain_timeout = (
            drain_timeout
            if drain_timeout is not None
            else max(1.0, flush_interval * 2)
        )
        self._queue: queue.Queue[logging.LogRecord] = queue.Queue(
            maxsize=max_queue_size
        )
        self._worker_thread: Optional[threading.Thread] = None
        self._worker_started_once = False
        self._stop_event = threading.Event()
        self._state_lock = threading.RLock()
        self._target_lock = threading.Lock()
        self._pending_condition = threading.Condition()
        self._closing = False
        self._handler_closed = False
        self._target_closed = False
        self._pending = 0

        self._stats_lock = threading.Lock()
        self._stats = {
            "dropped": 0,
            "rejected_closing": 0,
            "enqueued": 0,
            "written": 0,
            "write_errors": 0,
            "worker_restarts": 0,
            "worker_start_errors": 0,
            "max_queue_depth": 0,
        }
        self._start_worker()

    @staticmethod
    def _diagnose(message: str) -> None:
        """Write internal failures directly so this handler cannot recurse."""
        try:
            sys.stderr.write(f"AsyncRotatingFileHandler: {message}\n")
        except Exception:
            pass

    def _start_worker(self) -> bool:
        with self._state_lock:
            return self._start_worker_locked()

    def _start_worker_locked(self) -> bool:
        if self._handler_closed or self._target_closed:
            return False
        if self._worker_thread is not None and self._worker_thread.is_alive():
            return True

        restarting = self._worker_started_once
        worker: Optional[threading.Thread] = None
        try:
            worker = threading.Thread(
                target=self._worker_loop,
                name=f"AsyncLogWorker-{id(self)}",
                daemon=True,
            )
            self._worker_thread = worker
            worker.start()
        except Exception as error:
            worker_alive = False
            if worker is not None:
                try:
                    worker_alive = worker.is_alive()
                except Exception:
                    pass
            if not worker_alive and self._worker_thread is worker:
                self._worker_thread = None
            with self._stats_lock:
                self._stats["worker_start_errors"] += 1
            self._diagnose(f"failed to start worker: {error!r}")
            return worker_alive

        self._worker_started_once = True
        if restarting:
            with self._stats_lock:
                self._stats["worker_restarts"] += 1
        return True

    def _worker_loop(self) -> None:
        try:
            while not self._stop_event.is_set() or not self._queue.empty():
                self._process_batch()
            self._close_target()
        except BaseException as error:
            self._diagnose(f"worker stopped unexpectedly: {error!r}")
        finally:
            with self._state_lock:
                if self._worker_thread is threading.current_thread():
                    self._worker_thread = None
            with self._pending_condition:
                self._pending_condition.notify_all()

    def _process_batch(self) -> None:
        try:
            records = [self._queue.get(timeout=self._flush_interval)]
        except queue.Empty:
            return

        while len(records) < 100:
            try:
                records.append(self._queue.get_nowait())
            except queue.Empty:
                break

        try:
            for record in records:
                self._write_record(record)
            self._flush_target()
        finally:
            for _ in records:
                self._finish_record()

    def _finish_record(self) -> None:
        self._queue.task_done()
        with self._pending_condition:
            self._pending -= 1
            self._pending_condition.notify_all()

    def _write_record(self, record: logging.LogRecord) -> None:
        try:
            with self._target_lock:
                self._file_handler.emit(record)
            with self._stats_lock:
                self._stats["written"] += 1
        except Exception as error:
            with self._stats_lock:
                self._stats["write_errors"] += 1
            self._diagnose(f"failed to write log record: {error!r}")

    def _flush_target(self) -> None:
        try:
            with self._target_lock:
                self._file_handler.flush()
        except Exception as error:
            self._diagnose(f"failed to flush target handler: {error!r}")

    def _close_target(self) -> None:
        with self._target_lock:
            if self._target_closed:
                return
            try:
                self._file_handler.close()
            except Exception as error:
                self._diagnose(f"failed to close target handler: {error!r}")
        with self._state_lock:
            self._target_closed = True
        with self._pending_condition:
            self._pending_condition.notify_all()

    def _ensure_worker(self) -> bool:
        with self._state_lock:
            return self._start_worker_locked()

    def _wait_for_drain(self, deadline: float) -> bool:
        while True:
            with self._pending_condition:
                if self._pending == 0:
                    return True
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False

            worker_started = self._ensure_worker()
            with self._pending_condition:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False
                self._pending_condition.wait(
                    timeout=min(remaining, 0.05 if worker_started else 0.01)
                )

    def _wait_for_target_close(self, deadline: float) -> bool:
        while True:
            with self._state_lock:
                if self._target_closed:
                    return True
            if time.monotonic() >= deadline:
                return False

            worker_started = self._ensure_worker()
            with self._pending_condition:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False
                self._pending_condition.wait(
                    timeout=min(remaining, 0.05 if worker_started else 0.01)
                )

    def emit(self, record: logging.LogRecord) -> None:
        """Enqueue one record without blocking; reject producers once close starts."""
        with self._state_lock:
            if self._closing or self._handler_closed:
                with self._stats_lock:
                    self._stats["rejected_closing"] += 1
                return
            self._start_worker_locked()
            with self._pending_condition:
                self._pending += 1
            try:
                self._queue.put_nowait(record)
            except queue.Full:
                with self._pending_condition:
                    self._pending -= 1
                    self._pending_condition.notify_all()
                with self._stats_lock:
                    self._stats["dropped"] += 1
                    dropped = self._stats["dropped"]
                if dropped % 10 == 1:
                    self._diagnose(
                        f"dropped {dropped} log records because the queue is full"
                    )
                return

            queue_depth = self._queue.qsize()
            with self._stats_lock:
                self._stats["enqueued"] += 1
                self._stats["max_queue_depth"] = max(
                    self._stats["max_queue_depth"], queue_depth
                )

    def flush(self) -> None:
        """Wait at most one drain deadline for all accepted records."""
        if not hasattr(self, "_state_lock") or self._handler_closed:
            return
        deadline = time.monotonic() + self._drain_timeout
        if not self._wait_for_drain(deadline):
            self._diagnose(
                f"flush timed out with {self.get_stats()['pending']} records pending"
            )

    def close(self) -> None:
        if not hasattr(self, "_state_lock"):
            return
        with self._state_lock:
            if self._handler_closed or self._closing:
                return
            self._closing = True
            self._stop_event.set()
            self._start_worker_locked()

        deadline = time.monotonic() + self._drain_timeout
        drained = self._wait_for_drain(deadline)
        target_closed = self._wait_for_target_close(deadline) if drained else False
        if not drained:
            self._diagnose(
                f"close timed out with {self.get_stats()['pending']} records pending"
            )
        elif not target_closed:
            self._diagnose("close timed out waiting for target handler")
        with self._state_lock:
            self._handler_closed = True
        super().close()

    def get_stats(self) -> dict:
        with self._stats_lock:
            stats = dict(self._stats)
        with self._pending_condition:
            stats["pending"] = self._pending
        with self._state_lock:
            stats["queue_depth"] = self._queue.qsize()
            stats["worker_alive"] = bool(
                self._worker_thread and self._worker_thread.is_alive()
            )
            stats["closing"] = self._closing
        return stats

    def setFormatter(self, formatter: logging.Formatter) -> None:
        super().setFormatter(formatter)
        if hasattr(self, "_target_lock"):
            with self._target_lock:
                self._file_handler.setFormatter(formatter)

    def setLevel(self, level) -> None:
        super().setLevel(level)
        if hasattr(self, "_target_lock"):
            with self._target_lock:
                self._file_handler.setLevel(level)

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
