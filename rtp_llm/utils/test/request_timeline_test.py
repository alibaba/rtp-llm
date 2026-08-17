import json
import os
import unittest
from unittest.mock import Mock, patch

from rtp_llm.utils.request_timeline import (
    TIMELINE_LOG_PREFIX,
    configure_request_trace,
    log_timeline_event,
    parse_duration_seconds,
    request_trace_status,
    reset_request_trace_override,
    timeline_enabled,
    timeline_phase,
)


class RequestTimelineTest(unittest.TestCase):
    def tearDown(self):
        reset_request_trace_override()

    def _event(self, logger: Mock, call_index: int = 0) -> dict:
        args = logger.info.call_args_list[call_index].args
        self.assertEqual(args[0], "%s%s")
        self.assertEqual(args[1], TIMELINE_LOG_PREFIX)
        return json.loads(args[2])

    @patch.dict(os.environ, {}, clear=True)
    def test_disabled_by_default(self):
        logger = Mock()
        log_timeline_event("pg", "request_arrive", request_id=7, logger=logger)
        logger.info.assert_not_called()

    @patch.dict(os.environ, {"ENABLE_REQUEST_TIMELINE_LOG": "true"}, clear=True)
    def test_emits_compact_structured_event(self):
        logger = Mock()
        log_timeline_event(
            "pg",
            "request_arrive",
            request_id=7,
            ts_us=123,
            logger=logger,
            input_token_len=19,
        )

        event = self._event(logger)
        self.assertEqual(
            event,
            {
                "schema_version": 1,
                "component": "pg",
                "event": "request_arrive",
                "ts_us": 123,
                "request_id": 7,
                "input_token_len": 19,
            },
        )

    @patch.dict(os.environ, {"ENABLE_REQUEST_TIMELINE_LOG": "1"}, clear=True)
    def test_phase_emits_matching_start_and_end(self):
        logger = Mock()
        with timeline_phase("pg", "feature_generate", request_id=11, logger=logger):
            pass

        self.assertEqual(logger.info.call_count, 2)
        start = self._event(logger, 0)
        end = self._event(logger, 1)
        self.assertEqual(start["event"], "phase_start")
        self.assertEqual(end["event"], "phase_end")
        self.assertEqual(start["phase"], end["phase"])
        self.assertEqual(end["status"], "ok")
        self.assertGreaterEqual(end["duration_us"], 0)

    @patch.dict(os.environ, {"ENABLE_REQUEST_TIMELINE_LOG": "yes"}, clear=True)
    def test_phase_marks_exception(self):
        logger = Mock()
        with self.assertRaisesRegex(RuntimeError, "boom"):
            with timeline_phase("backend", "execute", logger=logger):
                raise RuntimeError("boom")
        self.assertEqual(self._event(logger, 1)["status"], "error")

    def test_duration_parser(self):
        self.assertEqual(parse_duration_seconds("30s"), 30)
        self.assertEqual(parse_duration_seconds("1.5m"), 90)
        self.assertEqual(parse_duration_seconds(2), 2)
        with self.assertRaisesRegex(ValueError, "duration"):
            parse_duration_seconds("forever")

    @patch.dict(os.environ, {}, clear=True)
    def test_runtime_window_expires_and_can_be_disabled(self):
        status = configure_request_trace(True, duration="30s", now_us=1_000_000)
        self.assertTrue(status["enabled"])
        self.assertEqual(status["start_ts_us"], 1_000_000)
        self.assertEqual(status["expires_ts_us"], 31_000_000)
        self.assertTrue(request_trace_status(now_us=30_000_000)["enabled"])
        self.assertFalse(request_trace_status(now_us=31_000_000)["enabled"])

        status = configure_request_trace(False, now_us=2_000_000)
        self.assertFalse(status["enabled"])

    @patch.dict(os.environ, {"ENABLE_REQUEST_TIMELINE_LOG": "1"}, clear=True)
    def test_api_disable_overrides_environment_default(self):
        self.assertTrue(request_trace_status(now_us=1)["enabled"])
        configure_request_trace(False, now_us=1)
        self.assertFalse(request_trace_status(now_us=1)["enabled"])


if __name__ == "__main__":
    unittest.main()
