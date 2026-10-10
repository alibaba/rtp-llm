import sys
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from rtp_llm.utils import startup_timing


class StartupTimingTest(unittest.TestCase):
    def setUp(self):
        self.origin = patch.object(startup_timing, "_restore_origin", None)
        self.origin.start()
        self.addCleanup(self.origin.stop)

    def test_stage_records_duration_in_python_and_native_logs(self):
        native = Mock()
        with patch.dict(
            sys.modules,
            {"libth_transformer": SimpleNamespace(log_startup_event=native)},
        ):
            with patch.object(
                startup_timing.time, "monotonic", side_effect=[10, 10.125]
            ):
                with self.assertLogs(startup_timing.LOGGER, "INFO") as logs:
                    with startup_timing.startup_stage("backend.test", rank=2):
                        pass
        self.assertEqual(len(logs.output), 2)
        self.assertIn("event=begin", logs.output[0])
        self.assertIn("event=end", logs.output[1])
        self.assertIn("elapsed_ms=125.000", logs.output[1])
        self.assertIn("rank=2", logs.output[1])
        self.assertEqual(native.call_args_list[1].args[0], logs.records[1].getMessage())

    def test_failure_preserves_original_exception(self):
        failure = KeyboardInterrupt("original")
        with self.assertLogs(startup_timing.LOGGER, "INFO") as logs:
            with self.assertRaises(KeyboardInterrupt) as caught:
                with startup_timing.startup_stage("restore.test"):
                    raise failure
        self.assertIs(caught.exception, failure)
        self.assertIn("event=failed", logs.output[-1])
        self.assertIn("error_type=KeyboardInterrupt", logs.output[-1])

    def test_resume_resets_checkpoint_clock_for_each_generation(self):
        startup_timing._restore_origin = (123, -1000000, "old")
        with patch.object(startup_timing.os, "getpid", return_value=123):
            with patch.object(
                startup_timing.time,
                "monotonic",
                side_effect=[100, 100, 100.25, 200, 200],
            ):
                with self.assertLogs(startup_timing.LOGGER, "INFO") as logs:
                    startup_timing.mark_restore_resumed("g1", "backend-0")
                    startup_timing.startup_event("backend.test", "ready")
                    startup_timing.mark_restore_resumed("g2", "backend-0")
        self.assertIn("since_resume_ms=0.000", logs.output[0])
        self.assertIn("since_resume_ms=250.000", logs.output[1])
        self.assertIn("generation=g1", logs.output[1])
        self.assertIn("since_resume_ms=0.000", logs.output[2])
        self.assertIn("generation=g2", logs.output[2])

    def test_forked_child_does_not_inherit_parent_timing(self):
        startup_timing._restore_origin = (-1, 0, "parent")
        with self.assertLogs(startup_timing.LOGGER, "INFO") as logs:
            startup_timing.startup_event("child.test", "ready")
        self.assertNotIn("since_resume_ms", logs.output[0])

    def test_native_logging_failure_does_not_fail_startup(self):
        native = Mock(side_effect=RuntimeError("unavailable"))
        with patch.dict(
            sys.modules,
            {"libth_transformer": SimpleNamespace(log_startup_event=native)},
        ):
            with self.assertLogs(startup_timing.LOGGER, "INFO"):
                with startup_timing.startup_stage("backend.test"):
                    pass
        self.assertEqual(native.call_count, 2)

    def test_python_logging_without_native_runtime(self):
        with patch.dict(sys.modules, {"libth_transformer": None}):
            with self.assertLogs(startup_timing.LOGGER, "INFO") as logs:
                startup_timing.startup_event("launcher.test", "ready")
            self.assertIsNone(sys.modules["libth_transformer"])
        self.assertIn("stage=launcher.test event=ready", logs.output[0])


if __name__ == "__main__":
    unittest.main()
