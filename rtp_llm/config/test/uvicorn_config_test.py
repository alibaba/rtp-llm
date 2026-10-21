import logging
import tempfile
import unittest
from pathlib import Path

from uvicorn import Config

from rtp_llm.access_logger.async_log_handler import AsyncRotatingFileHandler
from rtp_llm.config.uvicorn_config import configure_uvicorn_logging


class UvicornConfigTest(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.log_path = Path(self.temp_dir.name)
        self.access_logger = logging.getLogger("uvicorn.access")
        self.saved_state = (
            self.access_logger.handlers[:],
            self.access_logger.level,
            self.access_logger.propagate,
            self.access_logger.disabled,
        )
        for handler in self.access_logger.handlers[:]:
            self.access_logger.removeHandler(handler)
        self.addCleanup(self.restore_access_logger)

        self.mm_loggers = []
        for name in ("mm_access", "mm_query_access"):
            logger = logging.Logger(name, level=logging.INFO)
            handler = AsyncRotatingFileHandler(
                str(self.log_path / f"{name}.log"), flush_interval=0.01
            )
            logger.addHandler(handler)
            self.addCleanup(handler.close)
            self.mm_loggers.append((logger, handler))

    def restore_access_logger(self):
        for handler in self.access_logger.handlers[:]:
            self.access_logger.removeHandler(handler)
            handler.close()
        handlers, level, propagate, disabled = self.saved_state
        for handler in handlers:
            self.access_logger.addHandler(handler)
        self.access_logger.setLevel(level)
        self.access_logger.propagate = propagate
        self.access_logger.disabled = disabled

    def configure_http_logging(self):
        configure_uvicorn_logging(str(self.log_path))
        # Exercise real Uvicorn logging initialization, as in VitEndpointApp.
        Config(app=lambda: None, log_config=None)

    def test_mm_logs_survive_http_startup(self):
        for logger, _ in self.mm_loggers:
            logger.info("before HTTP startup")
        self.configure_http_logging()
        for logger, handler in self.mm_loggers:
            logger.info("after HTTP startup")
            handler.flush()
            self.assertEqual(
                (self.log_path / f"{logger.name}.log").read_text().splitlines(),
                ["before HTTP startup", "after HTTP startup"],
            )
            self.assertEqual(handler.get_stats()["rejected_closing"], 0)
            self.assertFalse(handler.get_stats()["closing"])

    def test_reconfigure_keeps_http_logging_without_duplicates(self):
        self.configure_http_logging()
        self.configure_http_logging()
        self.access_logger.info(
            '%s - "%s %s HTTP/%s" %d',
            "127.0.0.1:1234",
            "GET",
            "/health",
            "1.1",
            200,
        )
        lines = (self.log_path / "uvicorn_access.log").read_text().splitlines()
        self.assertEqual(len(lines), 1)
        self.assertIn('127.0.0.1:1234 - "GET /health HTTP/1.1" 200', lines[0])
        for logger, handler in self.mm_loggers:
            logger.info("after reconfiguration")
            handler.flush()
            self.assertIn(
                "after reconfiguration",
                (self.log_path / f"{logger.name}.log").read_text(),
            )
            self.assertEqual(handler.get_stats()["rejected_closing"], 0)


if __name__ == "__main__":
    unittest.main()
