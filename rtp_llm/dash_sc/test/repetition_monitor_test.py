from __future__ import annotations

import types
from contextlib import contextmanager
from unittest import TestCase, main
from unittest.mock import patch

import rtp_llm.dash_sc.repetition_monitor as repetition_monitor
from rtp_llm.dash_sc.repetition_monitor import (
    NativeModuleStatus,
    RequestRepetitionMonitor,
    ToolCallLoopConfig,
    ToolCallMarkerConfig,
    detect_tool_call_loop,
)


@contextmanager
def _native_status(status):
    """Pin the process-wide native status singleton for the duration of a test."""
    previous = repetition_monitor._NATIVE_STATUS
    repetition_monitor._NATIVE_STATUS = status
    try:
        yield
    finally:
        repetition_monitor._NATIVE_STATUS = previous


def _patched_native_module(fake_native):
    return _native_status(
        NativeModuleStatus(available=True, module=fake_native, module_name="fake")
    )


def _patched_native_unavailable(error: str = "missing native module"):
    return _native_status(NativeModuleStatus(available=False, error=error))


def _fresh_native_status():
    """Reset the singleton to None so the real resolver runs on next access."""
    return _native_status(None)


class NativeAvailabilityTest(TestCase):
    """Native availability is resolved once and reused, not probed per request."""

    def test_import_failure_warns_once_and_is_cached(self) -> None:
        with _fresh_native_status(), patch.object(
            repetition_monitor.importlib,
            "import_module",
            side_effect=ImportError("no libonline_repetition_tracker"),
        ):
            with self.assertLogs(
                "rtp_llm.dash_sc.repetition_monitor", level="WARNING"
            ) as logs:
                first = repetition_monitor.native_online_repetition_status()
                second = repetition_monitor.native_online_repetition_status()

        self.assertFalse(first.available)
        self.assertIs(first, second)  # cached: same object, resolver ran once
        self.assertIn("no libonline_repetition_tracker", first.error)
        self.assertEqual(len(logs.output), 1)

    def test_module_without_required_api_is_unavailable(self) -> None:
        with _fresh_native_status(), patch.object(
            repetition_monitor.importlib,
            "import_module",
            return_value=types.SimpleNamespace(),
        ):
            with self.assertLogs(
                "rtp_llm.dash_sc.repetition_monitor", level="WARNING"
            ) as logs:
                status = repetition_monitor.native_online_repetition_status()

        self.assertFalse(status.available)
        self.assertIn("missing API check_tool_call_loop", status.error)
        self.assertEqual(len(logs.output), 1)


class RepetitionMonitorTest(TestCase):
    def test_native_unavailable_surfaces_in_record_fields(self) -> None:
        marker = ToolCallMarkerConfig(begin_ids=(1,), end_ids=(2,))
        with _patched_native_unavailable("no libonline_repetition_tracker"):
            monitor = RequestRepetitionMonitor(
                tool_loop_config=ToolCallLoopConfig(enabled=True),
                tool_markers=(marker,),
            )
            fields = monitor.record_fields()

        self.assertFalse(fields["repetition_monitor_available"])
        self.assertEqual(
            fields["repetition_monitor_impl"], "tool=online_cpp_pybind_unavailable"
        )
        self.assertIn(
            "no libonline_repetition_tracker",
            fields["repetition_monitor_unavailable_reason"],
        )

    def test_tool_runtime_error_is_visible(self) -> None:
        def check_tool_call_loop(*_args):
            raise RuntimeError("tool loop check failed")

        fake_native = types.SimpleNamespace(check_tool_call_loop=check_tool_call_loop)
        marker = ToolCallMarkerConfig(begin_ids=(1,), end_ids=(2,))
        with _patched_native_module(fake_native):
            monitor = RequestRepetitionMonitor(
                tool_loop_config=ToolCallLoopConfig(enabled=True),
                tool_markers=(marker,),
            )
            monitor.check_tool_call_loop([1, 2, 3])
            fields = monitor.record_fields()

        self.assertFalse(fields["repetition_monitor_available"])
        self.assertEqual(fields["tool_call_loop_impl"], "online_cpp_pybind_unavailable")
        self.assertIn(
            "RuntimeError: tool loop check failed", fields["tool_call_loop_error"]
        )
        self.assertIn(
            "RuntimeError: tool loop check failed",
            fields["repetition_monitor_unavailable_reason"],
        )

    def test_detect_tool_call_loop_uses_request_level_native_function(self) -> None:
        class FakeNativeResult:
            hit = True
            repeat_count = 5
            current_span_tokens = 5
            marker_index = 0

        captured = {}

        def check_tool_call_loop(
            input_ids,
            output_ids,
            marker_begin_ids,
            marker_end_ids,
            repeat_threshold,
            max_span_tokens,
        ):
            captured["input_ids"] = tuple(input_ids)
            captured["output_ids"] = tuple(output_ids)
            captured["marker_begin_ids"] = tuple(marker_begin_ids[0])
            captured["marker_end_ids"] = tuple(marker_end_ids[0])
            captured["repeat_threshold"] = repeat_threshold
            captured["max_span_tokens"] = max_span_tokens
            return FakeNativeResult()

        fake_native = types.SimpleNamespace(check_tool_call_loop=check_tool_call_loop)
        with _patched_native_module(fake_native):
            result = detect_tool_call_loop(
                [1, 2, 10, 3, 4] * 4,
                [1, 2, 10, 3, 4],
                [ToolCallMarkerConfig(begin_ids=[1, 2], end_ids=[3, 4])],
                ToolCallLoopConfig(
                    enabled=True, repeat_threshold=5, max_span_tokens=16
                ),
            )

        self.assertIsNotNone(result)
        assert result is not None
        self.assertTrue(result.hit)
        self.assertEqual(result.repeat_count, 5)
        self.assertEqual(captured["input_ids"], tuple([1, 2, 10, 3, 4] * 4))
        self.assertEqual(captured["output_ids"], (1, 2, 10, 3, 4))
        self.assertEqual(captured["marker_begin_ids"], (1, 2))
        self.assertEqual(captured["marker_end_ids"], (3, 4))
        self.assertEqual(captured["repeat_threshold"], 5)
        self.assertEqual(captured["max_span_tokens"], 16)


if __name__ == "__main__":
    main()
