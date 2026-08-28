from __future__ import annotations

import types
from contextlib import contextmanager
from unittest import TestCase, main
from unittest.mock import patch

import rtp_llm.dash_sc.repetition_monitor as repetition_monitor
from rtp_llm.dash_sc.repetition_monitor import (
    NativeModuleStatus,
    OutputRepetitionConfig,
    RequestRepetitionMonitor,
    RequestRepetitionMonitorConfig,
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


def _fake_output_native(result):
    class FakeConfig:
        pass

    class FakeTracker:
        def __init__(self, _config):
            self.result = result

        def update_many(self, _token_ids):
            return self.result

        def finalize(self):
            return self.result

    return types.SimpleNamespace(
        OnlineRepetitionConfig=FakeConfig,
        OnlineRepetitionTracker=FakeTracker,
        check_tool_call_loop=lambda *_args: (False, 0, 0, -1),
    )


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
        self.assertIn("check_tool_call_loop", status.error)
        self.assertEqual(len(logs.output), 1)


class RepetitionMonitorTest(TestCase):
    def test_streaming_output_repetition_detects_same_token_run(self) -> None:
        config = RequestRepetitionMonitorConfig(
            output_config=OutputRepetitionConfig(
                enabled=True, min_repeats=3, min_duplicate_tokens=8
            )
        )
        result = types.SimpleNamespace(
            hit=True, repeat_unit_size=1, repeat_count=10,
            covered_token_count=10, duplicate_token_count=9,
            start_index=0, end_index=10, first_detect_index=8,
            non_contiguous=False, occurrence_count=10,
        )
        with _patched_native_module(_fake_output_native(result)):
            monitor = RequestRepetitionMonitor(monitor_config=config)
            monitor.update_output_delta([42] * 5)
            monitor.update_output_delta([42] * 5)
            monitor.finalize_output()
            fields = monitor.record_fields()
        self.assertTrue(fields["output_repetition"])
        self.assertEqual(fields["output_repetition_kind"], "same_token_run")
        self.assertEqual(fields["output_repetition_period"], 1)

    def test_streaming_output_repetition_detects_non_contiguous_span(self) -> None:
        config = RequestRepetitionMonitorConfig(
            output_config=OutputRepetitionConfig(
                enabled=True,
                min_repeats=3,
                min_duplicate_tokens=64,
                non_contiguous_min_span=32,
                non_contiguous_min_occurrences=3,
                non_contiguous_max_span=32,
            )
        )
        repeated = list(range(100, 132))
        result = types.SimpleNamespace(
            hit=True, repeat_unit_size=32, repeat_count=3,
            covered_token_count=96, duplicate_token_count=64,
            start_index=0, end_index=131, first_detect_index=131,
            non_contiguous=True, occurrence_count=3,
        )
        with _patched_native_module(_fake_output_native(result)):
            monitor = RequestRepetitionMonitor(monitor_config=config)
            monitor.update_output_delta(repeated + list(range(1000, 1017)))
            monitor.update_output_delta(repeated + list(range(2000, 2018)))
            monitor.update_output_delta(repeated)
            monitor.finalize_output()
            fields = monitor.record_fields()
        self.assertTrue(fields["output_repetition"])
        self.assertEqual(
            fields["output_repetition_kind"], "non_contiguous_span_repeat"
        )
        self.assertEqual(fields["output_repetition_occurrence_count"], 3)

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
            fields["repetition_monitor_impl"],
            "output=disabled,tool=online_cpp_pybind_unavailable",
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
