from __future__ import annotations

import random
import time
import types
from contextlib import contextmanager
from unittest import TestCase, main

import rtp_llm.dash_sc.repetition_monitor as repetition_monitor
from rtp_llm.dash_sc.repetition_monitor import (
    NativeToolCallLoopGuard,
    NativeToolCallSpanRecorder,
    OnlineRepetitionConfig,
    OutputRepetitionClassification,
    RequestRepetitionMonitor,
    RepetitionClassificationConfig,
    TailRepetitionConfig,
    TailRepetitionResult,
    ToolCallLoopConfig,
    ToolCallMarkerConfig,
    classify_output_repetition,
    detect_online_repetition,
    detect_tail_repetition,
)


@contextmanager
def _patched_native_module(fake_native):
    previous = (
        repetition_monitor._NATIVE_ONLINE_REPETITION_MODULE,
        repetition_monitor._NATIVE_ONLINE_REPETITION_LOAD_ATTEMPTED,
        repetition_monitor._NATIVE_ONLINE_REPETITION_MODULE_NAME,
        repetition_monitor._NATIVE_ONLINE_REPETITION_LOAD_ERROR,
        set(repetition_monitor._NATIVE_ONLINE_REPETITION_WARNED_FEATURES),
    )
    repetition_monitor._NATIVE_ONLINE_REPETITION_MODULE = fake_native
    repetition_monitor._NATIVE_ONLINE_REPETITION_LOAD_ATTEMPTED = True
    repetition_monitor._NATIVE_ONLINE_REPETITION_MODULE_NAME = "fake_native"
    repetition_monitor._NATIVE_ONLINE_REPETITION_LOAD_ERROR = None
    repetition_monitor._NATIVE_ONLINE_REPETITION_WARNED_FEATURES.clear()
    try:
        yield
    finally:
        (
            repetition_monitor._NATIVE_ONLINE_REPETITION_MODULE,
            repetition_monitor._NATIVE_ONLINE_REPETITION_LOAD_ATTEMPTED,
            repetition_monitor._NATIVE_ONLINE_REPETITION_MODULE_NAME,
            repetition_monitor._NATIVE_ONLINE_REPETITION_LOAD_ERROR,
            warned_features,
        ) = previous
        repetition_monitor._NATIVE_ONLINE_REPETITION_WARNED_FEATURES.clear()
        repetition_monitor._NATIVE_ONLINE_REPETITION_WARNED_FEATURES.update(
            warned_features
        )


@contextmanager
def _patched_native_unavailable(error: str = "missing native module"):
    previous = (
        repetition_monitor._NATIVE_ONLINE_REPETITION_MODULE,
        repetition_monitor._NATIVE_ONLINE_REPETITION_LOAD_ATTEMPTED,
        repetition_monitor._NATIVE_ONLINE_REPETITION_MODULE_NAME,
        repetition_monitor._NATIVE_ONLINE_REPETITION_LOAD_ERROR,
        set(repetition_monitor._NATIVE_ONLINE_REPETITION_WARNED_FEATURES),
    )
    repetition_monitor._NATIVE_ONLINE_REPETITION_MODULE = None
    repetition_monitor._NATIVE_ONLINE_REPETITION_LOAD_ATTEMPTED = True
    repetition_monitor._NATIVE_ONLINE_REPETITION_MODULE_NAME = None
    repetition_monitor._NATIVE_ONLINE_REPETITION_LOAD_ERROR = error
    repetition_monitor._NATIVE_ONLINE_REPETITION_WARNED_FEATURES.clear()
    try:
        yield
    finally:
        (
            repetition_monitor._NATIVE_ONLINE_REPETITION_MODULE,
            repetition_monitor._NATIVE_ONLINE_REPETITION_LOAD_ATTEMPTED,
            repetition_monitor._NATIVE_ONLINE_REPETITION_MODULE_NAME,
            repetition_monitor._NATIVE_ONLINE_REPETITION_LOAD_ERROR,
            warned_features,
        ) = previous
        repetition_monitor._NATIVE_ONLINE_REPETITION_WARNED_FEATURES.clear()
        repetition_monitor._NATIVE_ONLINE_REPETITION_WARNED_FEATURES.update(
            warned_features
        )


class TailRepetitionMonitorTest(TestCase):
    def test_detects_truncated_periodic_tail(self) -> None:
        tokens = [10, 20, 30, 99] * 8 + [10, 20]

        result = detect_tail_repetition(
            tokens,
            TailRepetitionConfig(min_duplicate_tokens=0),
        )

        self.assertIsNotNone(result)
        assert result is not None
        self.assertEqual(result.repeat_unit_size, 4)
        self.assertEqual(result.repeat_count, 8)
        self.assertEqual(result.partial_tail_tokens, 2)
        self.assertEqual(result.covered_token_count, 34)
        self.assertEqual(result.duplicate_token_count, 30)
        self.assertEqual(result.start_index, 0)
        self.assertEqual(result.end_index, len(tokens))
        self.assertFalse(result.window_capped)

    def test_ignores_two_complete_repeats_by_default(self) -> None:
        tokens = [10, 20, 30, 99] * 2

        self.assertIsNone(
            detect_tail_repetition(tokens, TailRepetitionConfig(min_duplicate_tokens=0))
        )

    def test_random_tail_has_no_signal(self) -> None:
        rng = random.Random(20260610)
        tokens = [rng.randrange(1, 10_000_000) for _ in range(4096)]

        self.assertIsNone(detect_tail_repetition(tokens))

    def test_repeated_long_tail_is_capped(self) -> None:
        tokens = [95553] * 10000

        result = detect_tail_repetition(
            tokens,
            TailRepetitionConfig(tail_window=4096, min_duplicate_tokens=0),
        )

        self.assertIsNotNone(result)
        assert result is not None
        self.assertEqual(result.repeat_unit_size, 1)
        self.assertEqual(result.repeat_count, 4096)
        self.assertEqual(result.covered_token_count, 4096)
        self.assertTrue(result.window_capped)

    def test_16k_smoke_stays_bounded(self) -> None:
        tokens = [10, 20, 30, 99] * 4096

        begin = time.perf_counter()
        result = detect_tail_repetition(tokens)
        elapsed_ms = (time.perf_counter() - begin) * 1000.0

        self.assertIsNotNone(result)
        # Keep this loose enough for CI variance; local benchmark should be
        # sub-millisecond to low-millisecond.
        self.assertLess(elapsed_ms, 20.0)

    def test_online_repetition_enabled_by_default_when_native_available(self) -> None:
        tokens = [10, 20, 30, 99] * 8 + [10, 20]

        class FakeNativeResult:
            hit = True
            repeat_unit_size = 4
            repeat_count = 8
            partial_tail_tokens = 2
            covered_token_count = 34
            duplicate_token_count = 30
            start_index = 0
            end_index = 34
            first_detect_index = 33

        fake_native = types.SimpleNamespace(
            detect_repetition_hit_only=lambda *args: FakeNativeResult()
        )
        with _patched_native_module(fake_native):
            result = detect_online_repetition(tokens)

        self.assertIsNotNone(result)
        assert result is not None
        self.assertEqual(result.repeat_unit_size, 4)

    def test_token_only_fallback_is_conservative(self) -> None:
        tokens = [10, 20, 30, 99] * 8 + [10, 20]
        repetition = detect_tail_repetition(
            tokens,
            TailRepetitionConfig(min_duplicate_tokens=0),
        )

        result = classify_output_repetition(
            repetition,
            config=RepetitionClassificationConfig(
                strong_min_duplicate_tokens=512,
                strong_min_repeat_count=8,
            ),
        )

        self.assertIsNotNone(result)
        assert result is not None
        self.assertEqual(result.level, "structural_repeat")
        self.assertEqual(result.source, "token_only")

    def test_token_only_extreme_repeat_is_strong(self) -> None:
        tokens = [95553] * 2048
        repetition = detect_tail_repetition(
            tokens,
            TailRepetitionConfig(min_duplicate_tokens=0),
        )

        result = classify_output_repetition(
            repetition,
            config=RepetitionClassificationConfig(
                strong_min_duplicate_tokens=512,
                strong_min_repeat_count=8,
            ),
        )

        self.assertIsNotNone(result)
        assert result is not None
        self.assertEqual(result.level, "strong_loop")
        self.assertEqual(result.source, "token_only")

    def test_online_repetition_uses_native_module_when_enabled(self) -> None:
        class FakeNativeResult:
            hit = True
            repeat_unit_size = 4
            repeat_count = 8
            partial_tail_tokens = 2
            covered_token_count = 34
            duplicate_token_count = 30
            start_index = 0
            end_index = 34
            first_detect_index = 33

        fake_native = types.SimpleNamespace(
            detect_repetition_hit_only=lambda *args: FakeNativeResult()
        )
        with _patched_native_module(fake_native):
            result = detect_online_repetition(
                [10, 20, 30, 99] * 8 + [10, 20],
                OnlineRepetitionConfig(enabled=True, min_duplicate_tokens=0),
            )

        self.assertIsNotNone(result)
        assert result is not None
        self.assertEqual(result.repeat_unit_size, 4)
        self.assertEqual(result.first_detect_index, 33)

    def test_request_monitor_final_output_uses_max_native_detector(self) -> None:
        calls: list[str] = []

        class FakeNativeResult:
            hit = True
            repeat_unit_size = 4
            repeat_count = 10
            partial_tail_tokens = 2
            covered_token_count = 42
            duplicate_token_count = 38
            start_index = 0
            end_index = 42
            first_detect_index = 11

        def detect_max(*_args):
            calls.append("max")
            return FakeNativeResult()

        def detect_hit_only(*_args):
            calls.append("hit_only")
            raise AssertionError("final access-log detection must not use hit_only")

        fake_native = types.SimpleNamespace(
            detect_repetition_hit_only=detect_hit_only,
            detect_repetition_max=detect_max,
        )
        with _patched_native_module(fake_native):
            monitor = RequestRepetitionMonitor(
                output_config=TailRepetitionConfig(enabled=True),
                online_config=OnlineRepetitionConfig(
                    enabled=True, min_duplicate_tokens=0, hit_only=True
                ),
                tool_loop_config=ToolCallLoopConfig(enabled=False),
            )
            result = monitor.finalize_output([10, 20, 30, 99] * 10 + [10, 20])
            fields = monitor.record_fields()

        self.assertIsNotNone(result)
        self.assertEqual(calls, ["max"])
        self.assertEqual(fields["output_repetition_impl"], "online_cpp_pybind_max")
        self.assertEqual(
            fields["repetition_monitor_impl"], "output=online_cpp_pybind_max"
        )
        self.assertTrue(fields["repetition_monitor_available"])
        self.assertEqual(fields["output_repetition_token_len"], 38)

    def test_request_monitor_primary_token_len_prefers_alert_source(self) -> None:
        monitor = RequestRepetitionMonitor(
            output_config=TailRepetitionConfig(enabled=False),
            tool_loop_config=ToolCallLoopConfig(enabled=False),
        )
        monitor.output_repetition = TailRepetitionResult(
            repeat_unit_size=4,
            repeat_count=10,
            partial_tail_tokens=2,
            covered_token_count=42,
            duplicate_token_count=38,
            start_index=0,
            end_index=42,
            window_capped=False,
        )
        monitor.output_repetition_classification = OutputRepetitionClassification(
            level="structural_repeat",
            reason="token_only_low_priority",
            source="token_only",
        )
        monitor.tool_call_loop_result = repetition_monitor.ToolCallLoopResult(
            hit=True,
            repeat_count=5,
            threshold=5,
            current_span_tokens=7,
            marker_index=0,
            history_suffix_count=4,
            current_suffix_count=1,
            span_overflow=False,
        )

        summary = monitor.diagnostic_summary()
        self.assertEqual(summary["output_repetition_token_len"], 38)
        self.assertEqual(summary["tool_call_loop_token_len"], 7)
        self.assertEqual(summary["repetition_token_len"], 7)
        self.assertEqual(summary["repetition_primary_source"], "tool_call_loop")

        monitor.output_repetition_classification = OutputRepetitionClassification(
            level="strong_loop",
            reason="token_only_long_repeat",
            source="token_only",
        )
        summary = monitor.diagnostic_summary()
        self.assertEqual(summary["repetition_token_len"], 38)
        self.assertEqual(summary["repetition_primary_source"], "output_repetition")

    def test_native_unavailable_is_visible_and_warned_once(self) -> None:
        with _patched_native_unavailable("no libonline_repetition_tracker"):
            with self.assertLogs(
                "rtp_llm.dash_sc.repetition_monitor", level="WARNING"
            ) as logs:
                first = repetition_monitor.warn_native_unavailable_once(
                    "tool_call_loop"
                )
                second = repetition_monitor.warn_native_unavailable_once(
                    "tool_call_loop"
                )
            monitor = RequestRepetitionMonitor(
                output_config=TailRepetitionConfig(enabled=True),
                online_config=OnlineRepetitionConfig(enabled=True),
                tool_loop_config=ToolCallLoopConfig(enabled=False),
            )
            fields = monitor.record_fields()

        self.assertFalse(first.available)
        self.assertFalse(second.available)
        self.assertEqual(len(logs.output), 1)
        self.assertFalse(fields["repetition_monitor_available"])
        self.assertIn(
            "no libonline_repetition_tracker",
            fields["repetition_monitor_unavailable_reason"],
        )

    def test_missing_native_tool_api_is_visible_before_delta(self) -> None:
        fake_native = types.SimpleNamespace()
        marker = ToolCallMarkerConfig(begin_ids=(1,), end_ids=(2,), name="fake_tool")
        with _patched_native_module(fake_native):
            monitor = RequestRepetitionMonitor(
                output_config=TailRepetitionConfig(enabled=False),
                tool_loop_config=ToolCallLoopConfig(enabled=True),
                tool_markers=(marker,),
            )
            with self.assertLogs(
                "rtp_llm.dash_sc.repetition_monitor", level="WARNING"
            ) as logs:
                fields = monitor.record_fields()

        self.assertFalse(fields["repetition_monitor_available"])
        self.assertEqual(
            fields["repetition_monitor_impl"],
            "tool=online_cpp_pybind_unavailable",
        )
        self.assertIn(
            "missing native API for tool_call_loop",
            fields["repetition_monitor_unavailable_reason"],
        )
        self.assertEqual(len(logs.output), 1)

    def test_tool_runtime_error_is_visible(self) -> None:
        class FakeNativeMarker:
            begin_ids = ()
            end_ids = ()
            name = ""

        class FakeRecorder:
            def __init__(self, *_args):
                pass

            def update_many(self, _token_ids):
                raise RuntimeError("recorder failed")

        class FakeGuard:
            def __init__(self, *_args):
                pass

        fake_native = types.SimpleNamespace(
            ToolCallMarkerIds=FakeNativeMarker,
            ToolCallSpanRecorder=FakeRecorder,
            TokenToolCallLoopGuard=FakeGuard,
        )
        marker = ToolCallMarkerConfig(begin_ids=(1,), end_ids=(2,), name="fake_tool")
        with _patched_native_module(fake_native):
            monitor = RequestRepetitionMonitor(
                output_config=TailRepetitionConfig(enabled=False),
                tool_loop_config=ToolCallLoopConfig(enabled=True),
                tool_markers=(marker,),
            )
            with self.assertLogs(
                "rtp_llm.dash_sc.repetition_monitor", level="WARNING"
            ) as logs:
                monitor.update_output_delta([1, 2, 3])
            fields = monitor.record_fields()

        self.assertFalse(fields["repetition_monitor_available"])
        self.assertEqual(
            fields["tool_call_loop_impl"], "online_cpp_pybind_unavailable"
        )
        self.assertIn("RuntimeError: recorder failed", fields["tool_call_loop_error"])
        self.assertIn(
            "RuntimeError: recorder failed",
            fields["repetition_monitor_unavailable_reason"],
        )
        self.assertEqual(len(logs.output), 1)

    def test_packaged_native_module_imports_from_runfiles(self) -> None:
        previous = (
            repetition_monitor._NATIVE_ONLINE_REPETITION_MODULE,
            repetition_monitor._NATIVE_ONLINE_REPETITION_LOAD_ATTEMPTED,
            repetition_monitor._NATIVE_ONLINE_REPETITION_MODULE_NAME,
            repetition_monitor._NATIVE_ONLINE_REPETITION_LOAD_ERROR,
        )
        repetition_monitor._NATIVE_ONLINE_REPETITION_MODULE = None
        repetition_monitor._NATIVE_ONLINE_REPETITION_LOAD_ATTEMPTED = False
        repetition_monitor._NATIVE_ONLINE_REPETITION_MODULE_NAME = None
        repetition_monitor._NATIVE_ONLINE_REPETITION_LOAD_ERROR = None
        try:
            native = repetition_monitor._load_native_online_repetition_module()
            self.assertIsNotNone(native)
            self.assertTrue(hasattr(native, "detect_repetition_max"))
            self.assertTrue(hasattr(native, "ToolCallMarkerIds"))
            self.assertTrue(hasattr(native, "ToolCallSpanRecorder"))
            self.assertTrue(hasattr(native, "TokenToolCallLoopGuard"))
            result = native.detect_repetition_max([1, 2, 1, 2, 1, 2], 3, 0, 8)
            marker = native.ToolCallMarkerIds()
            marker.begin_ids = [1]
            marker.end_ids = [2]
            marker.name = "smoke_tool"
            recorder = native.ToolCallSpanRecorder([marker], 16)
            spans = recorder.update_many([1, 42, 2])
            guard = native.TokenToolCallLoopGuard([marker], 5, 16)
            guard_result = guard.check_completed_span(
                [1, 42, 2] * 4,
                spans[0].token_ids,
                spans[0].marker_index,
                spans[0].overflow,
            )
        finally:
            (
                repetition_monitor._NATIVE_ONLINE_REPETITION_MODULE,
                repetition_monitor._NATIVE_ONLINE_REPETITION_LOAD_ATTEMPTED,
                repetition_monitor._NATIVE_ONLINE_REPETITION_MODULE_NAME,
                repetition_monitor._NATIVE_ONLINE_REPETITION_LOAD_ERROR,
            ) = previous

        self.assertTrue(result.hit)
        self.assertEqual(result.repeat_unit_size, 2)
        self.assertEqual(len(spans), 1)
        self.assertTrue(guard_result.hit)
        self.assertEqual(guard_result.repeat_count, 5)

    def test_tool_call_span_recorder_uses_native_module(self) -> None:
        class FakeNativeMarker:
            begin_ids = ()
            end_ids = ()
            name = ""

        class FakeNativeSpan:
            marker_index = 0
            token_ids = (1, 2, 10, 3, 4)
            overflow = False

        class FakeNativeRecorder:
            def __init__(self, markers, max_span_tokens):
                self.markers = markers
                self.max_span_tokens = max_span_tokens

            def update_many(self, token_ids):
                self.token_ids = token_ids
                return [FakeNativeSpan()]

            def reset(self):
                pass

        fake_native = types.SimpleNamespace(
            ToolCallMarkerIds=FakeNativeMarker,
            ToolCallSpanRecorder=FakeNativeRecorder,
        )
        with _patched_native_module(fake_native):
            recorder = NativeToolCallSpanRecorder(
                [ToolCallMarkerConfig(begin_ids=[1, 2], end_ids=[3, 4])],
                ToolCallLoopConfig(enabled=True, max_span_tokens=16),
            )
            spans = recorder.update_tokens([1, 2, 10, 3, 4])

        self.assertEqual(len(spans), 1)
        self.assertEqual(spans[0].marker_index, 0)
        self.assertEqual(tuple(spans[0].token_ids), (1, 2, 10, 3, 4))
        self.assertFalse(spans[0].overflow)

    def test_tool_call_loop_guard_uses_native_module(self) -> None:
        class FakeNativeMarker:
            begin_ids = ()
            end_ids = ()
            name = ""

        class FakeNativeResult:
            hit = True
            repeat_count = 5
            threshold = 5
            current_span_tokens = 5
            marker_index = 0
            history_suffix_count = 4
            current_suffix_count = 1
            span_overflow = False

        class FakeNativeGuard:
            def __init__(self, markers, repeat_threshold, max_span_tokens):
                self.markers = markers
                self.repeat_threshold = repeat_threshold
                self.max_span_tokens = max_span_tokens

            def check_completed_span(
                self, input_ids, current_span_ids, marker_index, span_overflow
            ):
                self.input_ids = input_ids
                self.current_span_ids = current_span_ids
                self.marker_index = marker_index
                self.span_overflow = span_overflow
                return FakeNativeResult()

            def reset(self):
                pass

        fake_native = types.SimpleNamespace(
            ToolCallMarkerIds=FakeNativeMarker,
            TokenToolCallLoopGuard=FakeNativeGuard,
        )
        with _patched_native_module(fake_native):
            guard = NativeToolCallLoopGuard(
                [ToolCallMarkerConfig(begin_ids=[1, 2], end_ids=[3, 4])],
                ToolCallLoopConfig(enabled=True, repeat_threshold=5, max_span_tokens=16),
            )
            result = guard.check_completed_span(
                [1, 2, 10, 3, 4] * 4, [1, 2, 10, 3, 4], 0
            )

        self.assertIsNotNone(result)
        assert result is not None
        self.assertTrue(result.hit)
        self.assertEqual(result.repeat_count, 5)
        self.assertEqual(result.history_suffix_count, 4)


if __name__ == "__main__":
    main()
