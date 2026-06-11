"""Low-overhead token repetition monitor for frontend access logging.

The production path calls a native C++/pybind detector once per completed
request for output repetition, and incrementally records completed tool-call
spans from streamed output chunks. Python owns token-id policy and logging;
C++ owns token-id matching.
"""

from __future__ import annotations

import dataclasses
import importlib
import logging
import os
import threading
import time
from typing import Any, Optional, Sequence


_LOGGER = logging.getLogger(__name__)


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _env_token_ids(name: str, default: Sequence[int]) -> tuple[int, ...]:
    value = os.environ.get(name)
    if value is None or value.strip() == "":
        return tuple(int(token_id) for token_id in default)
    parts = value.replace(";", ",").replace(" ", ",").split(",")
    token_ids = []
    try:
        for part in parts:
            if part.strip():
                token_ids.append(int(part.strip()))
    except ValueError:
        return tuple(int(token_id) for token_id in default)
    return tuple(token_ids)


@dataclasses.dataclass(frozen=True)
class TailRepetitionConfig:
    enabled: bool = os.environ.get("RTP_LLM_OUTPUT_REPETITION_MONITOR", "1") != "0"
    min_repeats: int = _env_int("RTP_LLM_OUTPUT_REPETITION_MIN_REPEATS", 3)
    min_duplicate_tokens: int = _env_int(
        "RTP_LLM_OUTPUT_REPETITION_MIN_DUP_TOKENS", 32
    )
    max_period: int = _env_int("RTP_LLM_OUTPUT_REPETITION_MAX_PERIOD", 512)
    tail_window: int = _env_int("RTP_LLM_OUTPUT_REPETITION_TAIL_WINDOW", 4096)


@dataclasses.dataclass(frozen=True)
class TailRepetitionResult:
    repeat_unit_size: int
    repeat_count: int
    partial_tail_tokens: int
    covered_token_count: int
    duplicate_token_count: int
    start_index: int
    end_index: int
    window_capped: bool


DEFAULT_TAIL_REPETITION_CONFIG = TailRepetitionConfig()


@dataclasses.dataclass(frozen=True)
class RepetitionClassificationConfig:
    strong_min_duplicate_tokens: int = _env_int(
        "RTP_LLM_OUTPUT_REPETITION_STRONG_MIN_DUP_TOKENS", 512
    )
    strong_min_repeat_count: int = _env_int(
        "RTP_LLM_OUTPUT_REPETITION_STRONG_MIN_REPEATS", 8
    )


@dataclasses.dataclass(frozen=True)
class OutputRepetitionClassification:
    level: str
    reason: str
    source: str


DEFAULT_REPETITION_CLASSIFICATION_CONFIG = RepetitionClassificationConfig()

_STRONG_LOOP = "strong_loop"
_STRUCTURAL_REPEAT = "structural_repeat"
_TOKEN_ONLY = "token_only"


def classify_output_repetition(
    repetition,
    config: RepetitionClassificationConfig = DEFAULT_REPETITION_CLASSIFICATION_CONFIG,
) -> Optional[OutputRepetitionClassification]:
    """Classify a repetition candidate without decoding token ids.

    Production monitoring intentionally stays token-only. It marks only very
    large/high-repeat candidates as ``strong_loop``; everything else is a
    low-priority ``structural_repeat`` signal for offline sampling.
    """

    if repetition is None:
        return None

    duplicate_tokens = int(getattr(repetition, "duplicate_token_count", 0))
    repeat_count = int(getattr(repetition, "repeat_count", 0))
    if (
        duplicate_tokens >= config.strong_min_duplicate_tokens
        and repeat_count >= config.strong_min_repeat_count
    ):
        return OutputRepetitionClassification(
            level=_STRONG_LOOP,
            reason="token_only_long_repeat",
            source=_TOKEN_ONLY,
        )
    return OutputRepetitionClassification(
        level=_STRUCTURAL_REPEAT,
        reason="token_only_low_priority",
        source=_TOKEN_ONLY,
    )


@dataclasses.dataclass(frozen=True)
class OnlineRepetitionConfig:
    enabled: bool = (
        os.environ.get("RTP_LLM_OUTPUT_REPETITION_ONLINE_MONITOR", "1") != "0"
    )
    min_repeats: int = _env_int("RTP_LLM_OUTPUT_REPETITION_MIN_REPEATS", 3)
    min_duplicate_tokens: int = _env_int(
        "RTP_LLM_OUTPUT_REPETITION_MIN_DUP_TOKENS", 32
    )
    max_period: int = _env_int("RTP_LLM_OUTPUT_REPETITION_MAX_PERIOD", 512)
    hit_only: bool = (
        os.environ.get("RTP_LLM_OUTPUT_REPETITION_ONLINE_HIT_ONLY", "1") != "0"
    )


@dataclasses.dataclass(frozen=True)
class OnlineRepetitionResult:
    repeat_unit_size: int
    repeat_count: int
    partial_tail_tokens: int
    covered_token_count: int
    duplicate_token_count: int
    start_index: int
    end_index: int
    first_detect_index: int
    window_capped: bool = False


DEFAULT_ONLINE_REPETITION_CONFIG = OnlineRepetitionConfig()
_NATIVE_ONLINE_REPETITION_MODULE = None
_NATIVE_ONLINE_REPETITION_LOAD_ATTEMPTED = False
_NATIVE_ONLINE_REPETITION_MODULE_NAME: Optional[str] = None
_NATIVE_ONLINE_REPETITION_LOAD_ERROR: Optional[str] = None
_NATIVE_ONLINE_REPETITION_WARNED_FEATURES: set[str] = set()
_NATIVE_ONLINE_REPETITION_LOAD_LOCK = threading.Lock()


@dataclasses.dataclass(frozen=True)
class NativeModuleStatus:
    available: bool
    module_name: Optional[str] = None
    error: Optional[str] = None


def _load_native_online_repetition_module():
    global _NATIVE_ONLINE_REPETITION_LOAD_ATTEMPTED
    global _NATIVE_ONLINE_REPETITION_MODULE
    global _NATIVE_ONLINE_REPETITION_MODULE_NAME
    global _NATIVE_ONLINE_REPETITION_LOAD_ERROR
    if _NATIVE_ONLINE_REPETITION_LOAD_ATTEMPTED:
        return _NATIVE_ONLINE_REPETITION_MODULE
    with _NATIVE_ONLINE_REPETITION_LOAD_LOCK:
        if _NATIVE_ONLINE_REPETITION_LOAD_ATTEMPTED:
            return _NATIVE_ONLINE_REPETITION_MODULE
        errors = []
        for module_name in (
            "rtp_llm.cpp.repetition.libonline_repetition_tracker",
            "libonline_repetition_tracker",
        ):
            try:
                _NATIVE_ONLINE_REPETITION_MODULE = importlib.import_module(module_name)
                _NATIVE_ONLINE_REPETITION_MODULE_NAME = module_name
                _NATIVE_ONLINE_REPETITION_LOAD_ERROR = None
                break
            except Exception as e:
                errors.append(f"{module_name}: {type(e).__name__}: {e}")
                continue
        if _NATIVE_ONLINE_REPETITION_MODULE is None:
            _NATIVE_ONLINE_REPETITION_MODULE_NAME = None
            _NATIVE_ONLINE_REPETITION_LOAD_ERROR = (
                "; ".join(errors) or "module not found"
            )
        _NATIVE_ONLINE_REPETITION_LOAD_ATTEMPTED = True
    return _NATIVE_ONLINE_REPETITION_MODULE


def native_online_repetition_status(load: bool = False) -> NativeModuleStatus:
    if load:
        _load_native_online_repetition_module()
    return NativeModuleStatus(
        available=_NATIVE_ONLINE_REPETITION_MODULE is not None,
        module_name=_NATIVE_ONLINE_REPETITION_MODULE_NAME,
        error=_NATIVE_ONLINE_REPETITION_LOAD_ERROR,
    )


def warn_native_unavailable_once(feature: str) -> NativeModuleStatus:
    status = native_online_repetition_status(load=True)
    if status.available:
        return status
    if feature not in _NATIVE_ONLINE_REPETITION_WARNED_FEATURES:
        _NATIVE_ONLINE_REPETITION_WARNED_FEATURES.add(feature)
        _LOGGER.warning(
            "native repetition monitor unavailable for %s: %s",
            feature,
            status.error or "unknown import error",
        )
    return status


def _warn_native_feature_unavailable_once(feature: str, reason: str) -> None:
    if feature in _NATIVE_ONLINE_REPETITION_WARNED_FEATURES:
        return
    _NATIVE_ONLINE_REPETITION_WARNED_FEATURES.add(feature)
    _LOGGER.warning(
        "native repetition monitor unavailable for %s: %s",
        feature,
        reason or "unknown native API error",
    )


def _missing_native_api(native, required_names: Sequence[str]) -> tuple[str, ...]:
    return tuple(name for name in required_names if not hasattr(native, name))


def _native_feature_unavailable_reason(
    feature: str,
    required_names: Sequence[str],
) -> Optional[str]:
    status = native_online_repetition_status(load=True)
    if not status.available:
        warn_native_unavailable_once(feature)
        return status.error or "native module unavailable"
    native = _NATIVE_ONLINE_REPETITION_MODULE
    missing = _missing_native_api(native, required_names)
    if missing:
        reason = f"missing native API for {feature}: {','.join(missing)}"
        _warn_native_feature_unavailable_once(feature, reason)
        return reason
    return None


def detect_online_repetition(
    tokens: Sequence[int],
    config: OnlineRepetitionConfig = DEFAULT_ONLINE_REPETITION_CONFIG,
) -> Optional[OnlineRepetitionResult]:
    """Detect output repetition using the native C++ online tracker.

    This is the production-oriented path: Python calls C++ once per completed
    request (or per chunk once the native token path owns the tracker). It does
    not fall back to an expensive Python full-output scan when the extension is
    unavailable.
    """

    if not config.enabled:
        return None
    native = _load_native_online_repetition_module()
    if native is None:
        warn_native_unavailable_once("output_repetition")
        return None

    min_repeats = max(3, int(config.min_repeats))
    min_duplicate_tokens = max(0, int(config.min_duplicate_tokens))
    max_period = max(1, int(config.max_period))
    if config.hit_only:
        missing = _missing_native_api(native, ("detect_repetition_hit_only",))
        if missing:
            _warn_native_feature_unavailable_once(
                "output_repetition",
                f"missing native API for output_repetition: {','.join(missing)}",
            )
            return None
        result = native.detect_repetition_hit_only(
            tokens, min_repeats, min_duplicate_tokens, max_period
        )
    else:
        missing = _missing_native_api(native, ("detect_repetition_max",))
        if missing:
            _warn_native_feature_unavailable_once(
                "output_repetition",
                f"missing native API for output_repetition: {','.join(missing)}",
            )
            return None
        result = native.detect_repetition_max(
            tokens, min_repeats, min_duplicate_tokens, max_period
        )
    if not result.hit:
        return None
    return OnlineRepetitionResult(
        repeat_unit_size=result.repeat_unit_size,
        repeat_count=result.repeat_count,
        partial_tail_tokens=result.partial_tail_tokens,
        covered_token_count=result.covered_token_count,
        duplicate_token_count=result.duplicate_token_count,
        start_index=result.start_index,
        end_index=result.end_index,
        first_detect_index=result.first_detect_index,
    )


@dataclasses.dataclass(frozen=True)
class ToolCallMarkerConfig:
    begin_ids: Sequence[int]
    end_ids: Sequence[int]
    name: str = ""


@dataclasses.dataclass(frozen=True)
class ToolCallLoopConfig:
    enabled: bool = os.environ.get("RTP_LLM_TOOL_CALL_LOOP_MONITOR", "1") != "0"
    repeat_threshold: int = _env_int("RTP_LLM_TOOL_CALL_LOOP_THRESHOLD", 5)
    max_span_tokens: int = _env_int("RTP_LLM_TOOL_CALL_LOOP_MAX_SPAN_TOKENS", 16384)


@dataclasses.dataclass(frozen=True)
class CompletedToolCallSpan:
    marker_index: int
    token_ids: Sequence[int]
    overflow: bool = False


@dataclasses.dataclass(frozen=True)
class ToolCallLoopResult:
    hit: bool
    repeat_count: int
    threshold: int
    current_span_tokens: int
    marker_index: int
    history_suffix_count: int
    current_suffix_count: int
    span_overflow: bool


DEFAULT_TOOL_CALL_LOOP_CONFIG = ToolCallLoopConfig()
DEFAULT_TOOL_CALL_MARKERS = (
    ToolCallMarkerConfig(
        begin_ids=_env_token_ids(
            "RTP_LLM_TOOL_CALL_LOOP_BEGIN_IDS", (30, 128825, 40148, 5406)
        ),
        end_ids=_env_token_ids(
            "RTP_LLM_TOOL_CALL_LOOP_END_IDS", (1718, 128825, 40148, 5406, 1018)
        ),
        name=os.environ.get("RTP_LLM_TOOL_CALL_LOOP_MARKER_NAME", "dsv4_dsml_invoke"),
    ),
)


def _build_native_tool_call_markers(native, markers: Sequence[ToolCallMarkerConfig]):
    native_markers = []
    for marker in markers:
        native_marker = native.ToolCallMarkerIds()
        native_marker.begin_ids = [int(token_id) for token_id in marker.begin_ids]
        native_marker.end_ids = [int(token_id) for token_id in marker.end_ids]
        native_marker.name = marker.name
        native_markers.append(native_marker)
    return native_markers


def _tool_call_loop_result_from_native(result) -> ToolCallLoopResult:
    return ToolCallLoopResult(
        hit=bool(result.hit),
        repeat_count=int(result.repeat_count),
        threshold=int(result.threshold),
        current_span_tokens=int(result.current_span_tokens),
        marker_index=int(result.marker_index),
        history_suffix_count=int(result.history_suffix_count),
        current_suffix_count=int(result.current_suffix_count),
        span_overflow=bool(result.span_overflow),
    )


class NativeToolCallSpanRecorder:
    """Chunk-level wrapper around the native token-id tool_call span recorder."""

    def __init__(
        self,
        markers: Sequence[ToolCallMarkerConfig],
        config: ToolCallLoopConfig = DEFAULT_TOOL_CALL_LOOP_CONFIG,
    ):
        self._native = (
            _load_native_online_repetition_module() if config.enabled else None
        )
        self._recorder = None
        if self._native is not None and markers:
            self._recorder = self._native.ToolCallSpanRecorder(
                _build_native_tool_call_markers(self._native, markers),
                max(1, int(config.max_span_tokens)),
            )

    @property
    def available(self) -> bool:
        return self._recorder is not None

    def update_tokens(self, token_ids: Sequence[int]) -> Sequence[CompletedToolCallSpan]:
        if self._recorder is None:
            return ()
        spans = self._recorder.update_many(token_ids)
        return tuple(
            CompletedToolCallSpan(
                marker_index=int(span.marker_index),
                token_ids=tuple(int(token_id) for token_id in span.token_ids),
                overflow=bool(span.overflow),
            )
            for span in spans
        )

    def reset(self) -> None:
        if self._recorder is not None:
            self._recorder.reset()


class NativeToolCallLoopGuard:
    """Request-scoped wrapper that scans input_ids once and caches history spans."""

    def __init__(
        self,
        markers: Sequence[ToolCallMarkerConfig],
        config: ToolCallLoopConfig = DEFAULT_TOOL_CALL_LOOP_CONFIG,
    ):
        self._enabled = bool(config.enabled)
        self._native = (
            _load_native_online_repetition_module() if self._enabled else None
        )
        self._guard = None
        self._history_sent = False
        self._marker_count = len(markers)
        if self._native is not None and markers:
            self._guard = self._native.TokenToolCallLoopGuard(
                _build_native_tool_call_markers(self._native, markers),
                max(2, int(config.repeat_threshold)),
                max(1, int(config.max_span_tokens)),
            )

    @property
    def available(self) -> bool:
        return self._guard is not None

    def check_completed_span(
        self,
        input_ids: Sequence[int],
        current_span_ids: Sequence[int],
        marker_index: int,
        span_overflow: bool = False,
    ) -> Optional[ToolCallLoopResult]:
        if self._guard is None:
            return None
        input_arg = input_ids if not self._history_sent else ()
        result = self._guard.check_completed_span(
            input_arg,
            current_span_ids,
            int(marker_index),
            bool(span_overflow),
        )
        if (
            not span_overflow
            and current_span_ids
            and 0 <= int(marker_index) < self._marker_count
        ):
            self._history_sent = True
        return _tool_call_loop_result_from_native(result)

    def reset(self) -> None:
        if self._guard is not None:
            self._guard.reset()
        self._history_sent = False


class RequestRepetitionMonitor:
    """Request-scoped facade used by dash-sc access logging.

    The access-log interceptor owns transport and serialization. This object owns
    the repetition detectors, native availability state, and the flat fields
    emitted into the access log. Keeping that boundary explicit prevents the
    interceptor from depending on pybind wrapper details.
    """

    def __init__(
        self,
        *,
        raw_mode: bool = False,
        input_ids: Optional[Sequence[int]] = None,
        output_config: Optional[TailRepetitionConfig] = None,
        online_config: Optional[OnlineRepetitionConfig] = None,
        classify_config: Optional[RepetitionClassificationConfig] = None,
        tool_loop_config: Optional[ToolCallLoopConfig] = None,
        tool_markers: Optional[Sequence[ToolCallMarkerConfig]] = None,
    ) -> None:
        self.raw_mode = raw_mode
        self.input_ids: Sequence[int] = input_ids or ()
        self.output_config = output_config or DEFAULT_TAIL_REPETITION_CONFIG
        self.online_config = online_config or DEFAULT_ONLINE_REPETITION_CONFIG
        self.classify_config = (
            classify_config or DEFAULT_REPETITION_CLASSIFICATION_CONFIG
        )
        self.tool_loop_config = tool_loop_config or DEFAULT_TOOL_CALL_LOOP_CONFIG
        self.tool_markers = tuple(tool_markers or DEFAULT_TOOL_CALL_MARKERS)

        self.output_repetition: Optional[
            TailRepetitionResult | OnlineRepetitionResult
        ] = None
        self.output_repetition_classification: Optional[
            OutputRepetitionClassification
        ] = None
        self.output_repetition_check_ms: Optional[float] = None
        self.output_repetition_impl: Optional[str] = None

        self.tool_call_loop_result: Optional[ToolCallLoopResult] = None
        self.tool_call_loop_check_ms: Optional[float] = None
        self.tool_call_loop_output_span_count = 0
        self.tool_call_loop_impl: Optional[str] = None
        self._output_error: Optional[str] = None
        self._tool_error: Optional[str] = None

        self._tool_call_loop_init_attempted = False
        self._tool_call_marker_names: tuple[str, ...] = tuple(
            marker.name for marker in self.tool_markers
        )
        self._tool_call_span_recorder: Optional[NativeToolCallSpanRecorder] = None
        self._tool_call_loop_guard: Optional[NativeToolCallLoopGuard] = None

    def set_input_ids(self, input_ids: Optional[Sequence[int]]) -> None:
        self.input_ids = input_ids or ()

    def _native_needed(self) -> bool:
        if self.raw_mode:
            return False
        output_native = bool(self.output_config.enabled and self.online_config.enabled)
        tool_native = bool(self.tool_loop_config.enabled and self.tool_markers)
        return output_native or tool_native

    def native_status(self) -> NativeModuleStatus:
        return native_online_repetition_status(load=self._native_needed())

    def _output_impl_name(self) -> str:
        if self.raw_mode:
            return "disabled_raw_mode"
        if not self.output_config.enabled:
            return "disabled"
        if self.online_config.enabled:
            return "online_cpp_pybind_max"
        return "tail_python"

    def _tool_impl_name(self) -> str:
        if self.raw_mode:
            return "disabled_raw_mode"
        if not self.tool_loop_config.enabled or not self.tool_markers:
            return "disabled"
        return self.tool_call_loop_impl or "online_cpp_pybind"

    def finalize_output(
        self,
        generated_ids: Sequence[int],
    ) -> Optional[TailRepetitionResult | OnlineRepetitionResult]:
        if self.output_repetition_check_ms is not None:
            return self.output_repetition
        if self.raw_mode or not self.output_config.enabled:
            self.output_repetition_impl = self._output_impl_name()
            return None

        begin = time.perf_counter()
        self.output_repetition_impl = self._output_impl_name()
        try:
            if self.online_config.enabled:
                final_config = dataclasses.replace(self.online_config, hit_only=False)
                self.output_repetition = detect_online_repetition(
                    generated_ids, final_config
                )
            else:
                self.output_repetition = detect_tail_repetition(
                    generated_ids, self.output_config
                )
            self.output_repetition_classification = classify_output_repetition(
                self.output_repetition,
                self.classify_config,
            )
        except Exception as e:
            _LOGGER.debug("output repetition detection failed: %s", e)
            self.output_repetition = None
            self.output_repetition_classification = None
            self._output_error = f"{type(e).__name__}: {e}"
            if self.online_config.enabled:
                self.output_repetition_impl = "online_cpp_pybind_unavailable"
                _warn_native_feature_unavailable_once(
                    "output_repetition", self._output_error
                )
        finally:
            self.output_repetition_check_ms = (time.perf_counter() - begin) * 1000.0
        return self.output_repetition

    def _ensure_tool_call_loop_monitor(self) -> bool:
        if self.raw_mode or not self.tool_loop_config.enabled or not self.tool_markers:
            self.tool_call_loop_impl = (
                "disabled_raw_mode" if self.raw_mode else "disabled"
            )
            return False
        if not self._tool_call_loop_init_attempted:
            self._tool_call_loop_init_attempted = True
            self.tool_call_loop_impl = "online_cpp_pybind"
            try:
                self._tool_call_span_recorder = NativeToolCallSpanRecorder(
                    self.tool_markers,
                    self.tool_loop_config,
                )
                self._tool_call_loop_guard = NativeToolCallLoopGuard(
                    self.tool_markers,
                    self.tool_loop_config,
                )
            except Exception as e:
                _LOGGER.debug("tool_call loop monitor init failed: %s", e)
                self._tool_error = f"{type(e).__name__}: {e}"
                _warn_native_feature_unavailable_once(
                    "tool_call_loop", self._tool_error
                )
                self._tool_call_span_recorder = None
                self._tool_call_loop_guard = None
        available = bool(
            self._tool_call_span_recorder is not None
            and self._tool_call_loop_guard is not None
            and getattr(self._tool_call_span_recorder, "available", True)
            and getattr(self._tool_call_loop_guard, "available", True)
        )
        if not available:
            self.tool_call_loop_impl = "online_cpp_pybind_unavailable"
            status = warn_native_unavailable_once("tool_call_loop")
            self._tool_error = status.error or self._tool_error or "unavailable"
        return available

    def update_output_delta(self, delta_ids: Sequence[int]) -> None:
        if not delta_ids or not self._ensure_tool_call_loop_monitor():
            return
        begin = time.perf_counter()
        try:
            assert self._tool_call_span_recorder is not None
            assert self._tool_call_loop_guard is not None
            spans = self._tool_call_span_recorder.update_tokens(delta_ids)
            self.tool_call_loop_output_span_count += len(spans)
            for span in spans:
                result = self._tool_call_loop_guard.check_completed_span(
                    self.input_ids,
                    span.token_ids,
                    span.marker_index,
                    span.overflow,
                )
                if result is None:
                    continue
                previous = self.tool_call_loop_result
                if previous is None or (
                    bool(result.hit),
                    int(result.repeat_count),
                ) > (
                    bool(previous.hit),
                    int(previous.repeat_count),
                ):
                    self.tool_call_loop_result = result
        except Exception as e:
            _LOGGER.debug("tool_call loop detection failed: %s", e)
            self._tool_error = f"{type(e).__name__}: {e}"
            self.tool_call_loop_impl = "online_cpp_pybind_unavailable"
            self._tool_call_span_recorder = None
            self._tool_call_loop_guard = None
            _warn_native_feature_unavailable_once("tool_call_loop", self._tool_error)
        finally:
            elapsed_ms = (time.perf_counter() - begin) * 1000.0
            self.tool_call_loop_check_ms = (
                elapsed_ms
                if self.tool_call_loop_check_ms is None
                else self.tool_call_loop_check_ms + elapsed_ms
            )

    def tool_call_loop_marker_name(self) -> Optional[str]:
        result = self.tool_call_loop_result
        if result is None:
            return None
        marker_index = int(result.marker_index)
        if 0 <= marker_index < len(self._tool_call_marker_names):
            return self._tool_call_marker_names[marker_index]
        return None

    def output_repeat_kind(self) -> str:
        if self.output_repetition_impl == "tail_python":
            return "tail_adjacent"
        if self.output_repetition_impl == "online_cpp_pybind_max":
            return "online_adjacent_max"
        if self.output_repetition_impl == "online_cpp_pybind_hit_only":
            return "online_adjacent_hit_only"
        return self.output_repetition_impl or "unknown"

    def monitor_impl(self) -> str:
        if self.raw_mode:
            return "disabled_raw_mode"
        parts = []
        if self.output_config.enabled:
            output_impl = self.output_repetition_impl or self._output_impl_name()
            parts.append(f"output={output_impl}")
        if self.tool_loop_config.enabled and self.tool_markers:
            parts.append(f"tool={self._tool_impl_name()}")
        if not parts:
            return "disabled"
        return ",".join(parts)

    def monitor_available(self) -> tuple[bool, Optional[str]]:
        if self.raw_mode:
            return False, "raw_mode"
        reasons = []
        if self.output_config.enabled and self.online_config.enabled:
            reason = _native_feature_unavailable_reason(
                "output_repetition", ("detect_repetition_max",)
            )
            if reason:
                self.output_repetition_impl = "online_cpp_pybind_unavailable"
                reasons.append(reason)
            if self._output_error:
                reasons.append(f"output: {self._output_error}")
        if self.tool_loop_config.enabled and self.tool_markers:
            tool_initialized = bool(
                self._tool_call_span_recorder is not None
                and self._tool_call_loop_guard is not None
                and getattr(self._tool_call_span_recorder, "available", True)
                and getattr(self._tool_call_loop_guard, "available", True)
            )
            if not tool_initialized:
                reason = _native_feature_unavailable_reason(
                    "tool_call_loop",
                    (
                        "ToolCallMarkerIds",
                        "ToolCallSpanRecorder",
                        "TokenToolCallLoopGuard",
                    ),
                )
                if reason:
                    self.tool_call_loop_impl = "online_cpp_pybind_unavailable"
                    reasons.append(reason)
            if self._tool_error:
                self.tool_call_loop_impl = "online_cpp_pybind_unavailable"
                reasons.append(f"tool_call: {self._tool_error}")
        if reasons:
            return False, "; ".join(dict.fromkeys(reasons))
        return True, None

    def diagnostic_summary(self) -> dict[str, Any]:
        rep = self.output_repetition
        rep_cls = self.output_repetition_classification
        tool_loop = self.tool_call_loop_result
        output_repeated = rep is not None
        function_tool_repeated = bool(tool_loop.hit) if tool_loop is not None else False

        reasons: list[str] = []
        if output_repeated:
            level = rep_cls.level if rep_cls is not None else "unknown"
            reason = rep_cls.reason if rep_cls is not None else "unknown"
            reasons.append(f"output_repetition:{level}:{reason}")
        if function_tool_repeated and tool_loop is not None:
            marker_name = self.tool_call_loop_marker_name() or "unknown"
            reasons.append(
                "tool_call_loop:"
                f"marker={marker_name}:"
                f"repeat_count={tool_loop.repeat_count}:"
                f"threshold={tool_loop.threshold}"
            )

        output_level = rep_cls.level if rep_cls is not None else None
        output_token_len = rep.duplicate_token_count if rep is not None else None
        tool_token_len = (
            tool_loop.current_span_tokens
            if function_tool_repeated and tool_loop is not None
            else None
        )
        if output_level == _STRONG_LOOP:
            repeated_token_len = output_token_len
            primary_source = "output_repetition"
        elif function_tool_repeated:
            repeated_token_len = tool_token_len
            primary_source = "tool_call_loop"
        elif output_repeated:
            repeated_token_len = output_token_len
            primary_source = "output_repetition"
        else:
            repeated_token_len = None
            primary_source = None

        monitor_available, unavailable_reason = self.monitor_available()
        return {
            "repetition_detected": output_repeated or function_tool_repeated,
            "repetition_alert": output_level == _STRONG_LOOP
            or function_tool_repeated,
            "repetition_reason": ";".join(reasons) if reasons else None,
            "repetition_primary_source": primary_source,
            "repetition_token_len": repeated_token_len,
            "output_repetition_token_len": output_token_len,
            "tool_call_loop_token_len": tool_token_len,
            "function_tool_repeated": function_tool_repeated,
            "repetition_monitor_impl": self.monitor_impl(),
            "repetition_monitor_available": monitor_available,
            "repetition_monitor_unavailable_reason": unavailable_reason,
        }

    def record_fields(self) -> dict[str, Any]:
        rep = self.output_repetition
        rep_cls = self.output_repetition_classification
        tool_loop = self.tool_call_loop_result

        fields = self.diagnostic_summary()
        fields.update(
            {
                "output_repetition": rep is not None,
                "output_repetition_check_ms": (
                    round(self.output_repetition_check_ms, 6)
                    if self.output_repetition_check_ms is not None
                    else None
                ),
                "output_repetition_impl": self.output_repetition_impl,
                "output_repetition_error": self._output_error,
                "output_repetition_level": (
                    rep_cls.level if rep_cls is not None else None
                ),
                "output_repetition_reason": (
                    rep_cls.reason if rep_cls is not None else None
                ),
                "output_repetition_classification_source": (
                    rep_cls.source if rep_cls is not None else None
                ),
                "output_repetition_unit_size": (
                    rep.repeat_unit_size if rep is not None else None
                ),
                "output_repetition_repeat_count": (
                    rep.repeat_count if rep is not None else None
                ),
                "output_repetition_partial_tail_tokens": (
                    rep.partial_tail_tokens if rep is not None else None
                ),
                "output_repetition_covered_token_count": (
                    rep.covered_token_count if rep is not None else None
                ),
                "output_repetition_duplicate_token_count": (
                    rep.duplicate_token_count if rep is not None else None
                ),
                "output_repetition_start_index": (
                    rep.start_index if rep is not None else None
                ),
                "output_repetition_end_index": (
                    rep.end_index if rep is not None else None
                ),
                "output_repetition_window_capped": (
                    rep.window_capped if rep is not None else None
                ),
                "tool_call_loop_hit": (
                    bool(tool_loop.hit) if tool_loop is not None else False
                ),
                "tool_call_loop_check_ms": (
                    round(self.tool_call_loop_check_ms, 6)
                    if self.tool_call_loop_check_ms is not None
                    else None
                ),
                "tool_call_loop_impl": self._tool_impl_name(),
                "tool_call_loop_error": self._tool_error,
                "tool_call_loop_output_span_count": (
                    self.tool_call_loop_output_span_count
                ),
                "tool_call_loop_repeat_count": (
                    tool_loop.repeat_count if tool_loop is not None else None
                ),
                "tool_call_loop_threshold": (
                    tool_loop.threshold if tool_loop is not None else None
                ),
                "tool_call_loop_current_span_tokens": (
                    tool_loop.current_span_tokens if tool_loop is not None else None
                ),
                "tool_call_loop_marker_index": (
                    tool_loop.marker_index if tool_loop is not None else None
                ),
                "tool_call_loop_marker_name": self.tool_call_loop_marker_name(),
                "tool_call_loop_history_suffix_count": (
                    tool_loop.history_suffix_count if tool_loop is not None else None
                ),
                "tool_call_loop_current_suffix_count": (
                    tool_loop.current_suffix_count if tool_loop is not None else None
                ),
                "tool_call_loop_span_overflow": (
                    tool_loop.span_overflow if tool_loop is not None else None
                ),
            }
        )
        return fields


def detect_tail_repetition(
    tokens: Sequence[int],
    config: TailRepetitionConfig = DEFAULT_TAIL_REPETITION_CONFIG,
) -> Optional[TailRepetitionResult]:
    """Detect a repeated suffix in bounded time.

    The check is candidate-driven:

    1. Candidate periods are only distances where the last token equals the
       token one period earlier.
    2. For each candidate period, walk backward while ``tokens[i] ==
       tokens[i-period]``.
    3. Stop after ``tail_window`` covered tokens, so cost is bounded even for
       extremely long looping outputs.

    The result may therefore be a lower-bound when ``window_capped`` is true.
    That is deliberate for production monitoring; offline reports can do exact
    expensive scans if needed.
    """

    if not config.enabled:
        return None
    token_count = len(tokens)
    min_repeats = max(3, int(config.min_repeats))
    min_duplicate_tokens = max(0, int(config.min_duplicate_tokens))
    max_period = max(1, int(config.max_period))
    tail_window = max(1, int(config.tail_window))

    max_period = min(max_period, token_count // min_repeats)
    if max_period < 1:
        return None

    best: Optional[TailRepetitionResult] = None
    last_index = token_count - 1
    last_token = int(tokens[last_index])

    for period in range(1, max_period + 1):
        if best is not None and tail_window - period <= best.duplicate_token_count:
            break
        if int(tokens[last_index - period]) != last_token:
            continue

        max_matched = min(token_count - period, max(0, tail_window - period))
        matched = 0
        while matched < max_matched:
            index = last_index - matched
            if int(tokens[index]) != int(tokens[index - period]):
                break
            matched += 1

        covered = matched + period
        repeat_count = covered // period
        if repeat_count < min_repeats:
            continue
        partial_tail = covered % period
        duplicate_tokens = period * (repeat_count - 1) + partial_tail
        if duplicate_tokens < min_duplicate_tokens:
            continue

        result = TailRepetitionResult(
            repeat_unit_size=period,
            repeat_count=repeat_count,
            partial_tail_tokens=partial_tail,
            covered_token_count=covered,
            duplicate_token_count=duplicate_tokens,
            start_index=token_count - covered,
            end_index=token_count,
            window_capped=covered == tail_window and covered < token_count,
        )
        if best is None or (
            result.duplicate_token_count,
            result.covered_token_count,
            -result.repeat_unit_size,
        ) > (
            best.duplicate_token_count,
            best.covered_token_count,
            -best.repeat_unit_size,
        ):
            best = result

    return best
