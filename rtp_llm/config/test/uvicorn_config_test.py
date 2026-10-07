import logging
import tempfile
import unittest
from pathlib import Path

from rtp_llm.access_logger.async_log_handler import AsyncRotatingFileHandler
from rtp_llm.config.uvicorn_config import configure_uvicorn_access_logging


class UvicornAccessLoggingTest(unittest.TestCase):
    def test_http_configuration_preserves_async_logs(self):
        logger = logging.getLogger("test.mm_access")
        http_logger = logging.getLogger("uvicorn.access")
        old_state = (
            list(http_logger.handlers),
            http_logger.level,
            http_logger.propagate,
            http_logger.disabled,
        )
        http_logger.handlers = []
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            handler = AsyncRotatingFileHandler(str(path / "mm.log"))
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
            logger.propagate = False
            try:
                configure_uvicorn_access_logging(directory)
                logger.info("first completed embedding")
                handler.flush()
                self.assertIn(
                    "first completed embedding", (path / "mm.log").read_text()
                )
                configure_uvicorn_access_logging(directory)
                logger.info("second completed embedding")
                handler.flush()
                self.assertIn(
                    "second completed embedding", (path / "mm.log").read_text()
                )
                self.assertEqual(len(http_logger.handlers), 1)
                http_logger.info(
                    '%s - "%s %s HTTP/%s" %d', "127.0.0.1", "GET", "/health", "1.1", 200
                )
                for access_handler in http_logger.handlers:
                    access_handler.flush()
                self.assertIn("/health", (path / "uvicorn_access.log").read_text())
            finally:
                logger.removeHandler(handler)
                handler.close()
                for access_handler in http_logger.handlers:
                    access_handler.close()
                (
                    http_logger.handlers,
                    http_logger.level,
                    http_logger.propagate,
                    http_logger.disabled,
                ) = old_state


if __name__ == "__main__":
    unittest.main()
