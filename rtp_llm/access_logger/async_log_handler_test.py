import logging
import tempfile
import threading
import unittest
from pathlib import Path

from rtp_llm.access_logger.async_log_handler import AsyncRotatingFileHandler
from rtp_llm.access_logger.log_utils import get_process_log_filename


class AsyncLogHandlerTest(unittest.TestCase):
    def test_close_drains_and_process_files_are_distinct(self):
        self.assertNotEqual(
            get_process_log_filename("access.log", 0, 1),
            get_process_log_filename("access.log", 0, 2),
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "access.log"
            handler = AsyncRotatingFileHandler(str(path), flush_interval=0.01)
            handler.setFormatter(logging.Formatter("%(message)s"))
            for i in range(100):
                handler.handle(
                    logging.LogRecord("test", logging.INFO, "", 0, "row %d", (i,), None)
                )
            handler.close()
            handler.close()
            self.assertEqual(
                path.read_text().splitlines(), [f"row {i}" for i in range(100)]
            )
            stats = handler.get_stats()
            self.assertEqual(stats["written"], 100)
            self.assertEqual(stats["queue_depth"], 0)
            self.assertFalse(stats["worker_alive"])

    def test_full_queue_drops_without_blocking_request(self):
        with tempfile.TemporaryDirectory() as tmp:
            handler = AsyncRotatingFileHandler(
                str(Path(tmp) / "access.log"), max_queue_size=1, flush_interval=0.01
            )
            entered, release = threading.Event(), threading.Event()
            write = handler._write_record

            def blocked_write(record):
                entered.set()
                release.wait(5)
                write(record)

            handler._write_record = blocked_write
            record = logging.LogRecord("test", logging.INFO, "", 0, "record", (), None)
            try:
                handler.handle(record)
                self.assertTrue(entered.wait(5))
                handler.handle(record)
                handler.handle(record)
                self.assertEqual(handler.get_stats()["dropped"], 1)
            finally:
                release.set()
                handler.close()
            self.assertEqual(handler.get_stats()["written"], 2)


if __name__ == "__main__":
    unittest.main()
