# -*- coding: utf-8 -*-
"""
Async Log Handler

Core Features:
1. Uses queue to buffer log messages, avoiding main thread blocking
2. Dedicated background thread handles actual writing
3. Drop strategy when queue is full to protect main flow
4. Graceful shutdown mechanism ensures important logs are not lost

Working Mechanism:
    主线程调用日志
           ↓
    AsyncRotatingFileHandler.emit()  ← 接收日志，放入队列（非阻塞）
           ↓
       内存队列
           ↓
    后台线程取出日志记录
           ↓
    _write_record()
           ↓
    self._file_handler.emit()  ← 实际写入文件（可能阻塞）

Two emit() Methods:
- AsyncRotatingFileHandler.emit(): Fast reception, puts record into queue
- RotatingFileHandler.emit(): Actual file writing with rotation logic
"""

import logging
import queue
import threading
from logging.handlers import RotatingFileHandler
from typing import Optional


class AsyncRotatingFileHandler(logging.Handler):
    """Async file log handler - runs forever"""

    def __init__(self, filename: str, mode: str = 'a', max_bytes: int = 0,
                 backup_count: int = 0, encoding: Optional[str] = None,
                 delay: bool = False, max_queue_size: int = 10000,
                 flush_interval: float = 1.0, **kwargs):
        """
        Initialize async log handler

        Args:
            filename: Log file name
            mode: File open mode
            max_bytes: Maximum file size in bytes
            backup_count: Number of backup files
            encoding: File encoding
            delay: Whether to delay file creation
            max_queue_size: Maximum memory queue size
            flush_interval: Flush interval in seconds
            **kwargs: Other parameters
        """
        super().__init__()

        # Create underlying file handler
        self._file_handler = RotatingFileHandler(
            filename=filename,
            mode=mode,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding=encoding,
            delay=delay,
            **kwargs
        )

        # Queue configuration
        self._max_queue_size = max_queue_size
        self._flush_interval = flush_interval

        # Create queue and thread
        self._queue = queue.Queue(maxsize=max_queue_size)
        self._worker_thread = None

        # Statistics
        self._stats = {
            'dropped': 0,
        }

        # Start background thread
        self._start_worker()
        logging.info(f"AsyncRotatingFileHandler init complete, worker thread alive: {self._worker_thread.is_alive() if self._worker_thread else 'None'}")


    def _start_worker(self) -> None:
        """Start background worker thread"""
        if self._worker_thread is None or not self._worker_thread.is_alive():
            self._worker_thread = threading.Thread(
                target=self._worker_loop,
                name=f"AsyncLogWorker-{id(self)}",
                daemon=True
            )
            self._worker_thread.start()

    def _worker_loop(self) -> None:
        """Background thread work loop"""
        try:
            while True:
                try:
                    self._process_batch()
                except Exception as e:
                    logging.error(f"AsyncLogWorker error: {e}")

        except Exception as e:
            logging.error(f"AsyncLogWorker fatal error: {e}")
            self._drain_queue()

    def _process_batch(self) -> None:
        """Process a batch of log records from the queue"""
        records_batch = []

        # Get first record (blocking wait)
        try:
            record = self._queue.get(timeout=self._flush_interval)
            if record is not None:
                records_batch.append(record)
        except queue.Empty:
            return

        # Collect up to 100 records for batch processing
        while len(records_batch) < 100:
            try:
                record = self._queue.get_nowait()
                if record is not None:
                    records_batch.append(record)
            except queue.Empty:
                break

        # Batch write logs
        for record in records_batch:
            self._write_record(record)

        # Force flush file buffer
        self._file_handler.flush()

    def _write_record(self, record: logging.LogRecord) -> None:
        """Write single log record"""
        try:
            # Synchronous operation - will block until write is completed
            self._file_handler.emit(record)
        except Exception as e:
            # Log error but don't raise exception
            logging.error(f"Failed to write log record: {e}")

    def _drain_queue(self) -> None:
        """When the program exits, processes the remaining log records in the queue/"""
        try:
            while True:
                try:
                    record = self._queue.get_nowait()
                    if record is not None:
                        self._write_record(record)
                except queue.Empty:
                    break
        except Exception as e:
            logging.error(f"Error draining queue: {e}")

    def emit(self, record: logging.LogRecord) -> None:
        """Send log record to async queue - non-blocking"""
        # Auto-restart dead worker thread
        if not self._worker_thread or not self._worker_thread.is_alive():
            logging.warning("AsyncLogHandler worker thread died, restarting...")
            self._start_worker()

        try:
            self._queue.put_nowait(record)
        except queue.Full:
            # Drop logs when queue is full to protect main flow from blocking
            self._stats['dropped'] += 1
            if self._stats['dropped'] % 10 == 1:  # Reduce logging frequency
                logging.warning(f"AsyncLogHandler: dropped {self._stats['dropped']} log records (queue full)")

    def flush(self) -> None:
        """Flush log buffer - NOOP"""
        pass

    def close(self) -> None:
        pass

    def setFormatter(self, formatter: logging.Formatter) -> None:
        """Set formatter"""
        super().setFormatter(formatter)
        if self._file_handler:
            self._file_handler.setFormatter(formatter)

    def setLevel(self, level) -> None:
        """Set log level"""
        super().setLevel(level)
        if self._file_handler:
            self._file_handler.setLevel(level)

    def __del__(self):
        """Destructor"""
        pass