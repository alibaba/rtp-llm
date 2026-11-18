import contextlib
import importlib.util
import io
import json
import logging
import logging.config
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

path = Path(__file__).resolve().parents[2] / "adapters/vllm/VLLMLogging.py"
spec = importlib.util.spec_from_file_location("file_logging_under_test", path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class VLLMLoggingTest(unittest.TestCase):
    def test_file_only_and_bounded_rotation(self):
        with tempfile.TemporaryDirectory() as directory:
            handler = module.ProcessRotatingFileHandler(directory, maxBytes=128, backupCount=2)
            logger = logging.Logger("test", logging.INFO)
            logger.addHandler(handler)
            logger.propagate = False
            stdout, stderr = io.StringIO(), io.StringIO()
            with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                for i in range(100):
                    logger.warning("record=%03d %s", i, "x" * 48)
            handler.close()
            files = list(Path(directory).glob("*.log*"))
            self.assertEqual(len(files), 3)
            self.assertIn("record=099", Path(handler.baseFilename).read_text())
            self.assertEqual(stdout.getvalue() + stderr.getvalue(), "")

    @unittest.skipUnless(hasattr(os, "fork"), "requires fork")
    def test_fork_uses_distinct_pid_files(self):
        with tempfile.TemporaryDirectory() as directory:
            handler = module.ProcessRotatingFileHandler(directory)
            logger = logging.Logger("fork", logging.INFO)
            logger.addHandler(handler)
            logger.warning("parent-only")
            child = os.fork()
            if child == 0:
                try:
                    logger.warning("child-only")
                    handler.close()
                finally:
                    os._exit(0)
            _, status = os.waitpid(child, 0)
            self.assertEqual(status, 0)
            handler.close()
            parent = Path(directory, f"navi_sched_adapter.{os.getpid()}.log").read_text()
            child_text = Path(directory, f"navi_sched_adapter.{child}.log").read_text()
            self.assertIn("parent-only", parent)
            self.assertNotIn("child-only", parent)
            self.assertIn("child-only", child_text)

    def test_io_failure_does_not_write_stderr_or_raise(self):
        with tempfile.TemporaryDirectory() as directory:
            handler = module.ProcessRotatingFileHandler(directory)
            logger = logging.Logger("failure", logging.INFO)
            logger.addHandler(handler)
            stderr = io.StringIO()
            with mock.patch.object(handler, "_open", side_effect=OSError("disk unavailable")), contextlib.redirect_stderr(stderr):
                logger.warning("still must not interrupt inference")
            self.assertEqual(handler.error_count, 1)
            self.assertEqual(stderr.getvalue(), "")
            handler.close()

    def test_unwritable_directory_falls_back_to_regular_file(self):
        with tempfile.TemporaryDirectory() as directory:
            blocker = Path(directory, "not-a-directory")
            blocker.write_text("file")
            handler = module.ProcessRotatingFileHandler(blocker)
            record = logging.LogRecord("fallback", logging.INFO, __file__, 1, "fallback-file-only", (), None)
            handler.handle(record)
            handler.close()
            self.assertIn("fallback-file-only", Path(handler.baseFilename).read_text())
            Path(handler.baseFilename).unlink()

    def test_template_environment_controls_retention(self):
        with tempfile.TemporaryDirectory() as directory, mock.patch.dict(os.environ, {
            "NAVI_SCHED_LOG_MAX_BYTES": "209715200",
            "NAVI_SCHED_LOG_BACKUP_COUNT": "20",
        }):
            output = Path(directory, "logging.json")
            module.write_vllm_logging_config(output)
            handler = json.loads(output.read_text())["handlers"]["file"]
            self.assertEqual(handler["maxBytes"], 209715200)
            self.assertEqual(handler["backupCount"], 20)

    def test_generated_native_config_is_file_only(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory, "logging.json")
            module.write_vllm_logging_config(output)
            config = json.loads(output.read_text())
            self.assertEqual(set(config["handlers"]), {"file"})
            self.assertNotIn("stream", config["handlers"]["file"])
            self.assertTrue(all(not v["propagate"] for v in config["loggers"].values()))
            self.assertGreater(config["handlers"]["file"]["maxBytes"], 0)
            self.assertGreater(config["handlers"]["file"]["backupCount"], 0)


if __name__ == "__main__":
    unittest.main()
