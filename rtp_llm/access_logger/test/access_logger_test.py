import logging
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from rtp_llm.access_logger.access_logger import AccessLogger, MMAccessLogger
from rtp_llm.structure.request_constants import request_id_field_name


class AccessLoggerTest(unittest.TestCase):
    def test_disable_access_log(self):
        for logger_type in (AccessLogger, MMAccessLogger):
            for disabled in ("0", "1"):
                with self.subTest(logger=logger_type, disabled=disabled):
                    with tempfile.TemporaryDirectory() as directory, patch.dict(
                        os.environ, {"DISABLE_ACCESS_LOG": disabled}
                    ):
                        logger = logger_type(directory, 1, async_mode=True)
                        logger.logger.setLevel(logging.INFO)
                        logger.query_logger.setLevel(logging.INFO)
                        request = (
                            {request_id_field_name: 1, "prompt": "hello"}
                            if logger_type is AccessLogger
                            else []
                        )
                        logger.log_query_access(request)
                        logger.log_success_access(request, {"response": "hello"})
                        logger.log_exception_access(request, RuntimeError("failure"))
                        for sink in (logger.logger, logger.query_logger):
                            for handler in sink.handlers:
                                handler.close()
                        files = list(Path(directory).glob("*.log"))
                        if disabled == "1":
                            self.assertEqual(files, [])
                        else:
                            contents = "".join(p.read_text() for p in files)
                            self.assertIn("hello", contents)
                            self.assertIn("failure", contents)
                        for sink in (logger.logger, logger.query_logger):
                            for handler in sink.handlers[:]:
                                sink.removeHandler(handler)
                                handler.close()
                            sink.disabled = False


if __name__ == "__main__":
    unittest.main()
