"""Low-overhead output and tool-call repetition monitoring for frontend logging.

The production path uses a native C++/pybind streaming detector per request.
Python owns token-id policy and access-log fields; C++ owns token-id matching.
"""

from __future__ import annotations

import dataclasses
import importlib
import logging
import time
from typing import Any, Optional, Sequence

_LOGGER = logging.getLogger(__name__)

# Whether the native tracker ``.so`` is packaged and what it exports are
# build-time facts, fixed for the life of the process. Resolve them exactly once
# into ``_NATIVE_STATUS`` below; the per-request path only reads that cached
# global, it never re-imports or re-probes.
_NATIVE_TRACKER_MODULES = (
    "rtp_llm.cpp.repetition.libonline_repetition_tracker",
    "libonline_repetition_tracker",
)
_NATIVE_TRACKER_REQUIRED_APIS = (
    "OnlineRepetitionConfig",
    "OnlineRepetitionTracker",
    "check_tool_call_loop",
)


@dataclasses.dataclass(frozen=True)
class NativeModuleStatus:
    available: bool
    module: Optional[Any] = None
    module_name: Optional[str] = None
    error: Optional[str] = None


# Process-wide singleton: None until first resolved, then reused forever.
_NATIVE_STATUS: Optional[NativeModuleStatus] = None


def _resolve_native_status() -> NativeModuleStatus:
    errors: list[str] = []
    module = None
    module_name = None
    for name in _NATIVE_TRACKER_MODULES:
        try:
            module = importlib.import_module(name)
            module_name = name
            break
        except Exception as e:
            errors.append(f"{name}: {type(e).__name__}: {e}")
    missing = [
        name
        for name in _NATIVE_TRACKER_REQUIRED_APIS
        if module is not None and not hasattr(module, name)
    ]
    if module is not None and missing:
        errors.append(f"{module_name}: missing API {','.join(missing)}")
        module = None
        module_name = None
    if module is None:
        error = "; ".join(errors) or "module not found"
        _LOGGER.warning("native repetition monitor unavailable: %s", error)
        return NativeModuleStatus(available=False, error=error)
    return NativeModuleStatus(available=True, module=module, module_name=module_name)


def native_online_repetition_status() -> NativeModuleStatus:
    """Return the once-resolved native tracker status, cached for the process.

    The import + API probe runs on the first call only and reuses the result
    thereafter. It is lazy (not at import time) on purpose: CPU/test hosts
    without the ``.so`` — and the production case where the native lib path is
    configured during runtime startup — must resolve on first use, not when this
    module is imported. The per-request path just reads the cached object.
    """
    global _NATIVE_STATUS
    if _NATIVE_STATUS is None:
        _NATIVE_STATUS = _resolve_native_status()
    return _NATIVE_STATUS


@dataclasses.dataclass(frozen=True)
class ToolCallMarkerConfig:
    begin_ids: Sequence[int]
    end_ids: Sequence[int]


@dataclasses.dataclass(frozen=True)
class OutputRepetitionConfig:
    enabled: bool = True
    min_repeats: int = 3
    min_duplicate_tokens: int = 32
    max_period: int = 512
    non_contiguous_min_span: int = 32
    non_contiguous_min_occurrences: int = 3
    non_contiguous_max_span: int = 256


@dataclasses.dataclass(frozen=True)
class OutputRepetitionResult:
    hit: bool
    repeat_unit_size: int
    repeat_count: int
    covered_token_count: int
    duplicate_token_count: int
    start_index: int
    end_index: int
    first_detect_index: int
    non_contiguous: bool
    occurrence_count: int


@dataclasses.dataclass(frozen=True)
class ToolCallLoopConfig:
    enabled: bool = True
    repeat_threshold: int = 5
    max_span_tokens: int = 16384


@dataclasses.dataclass(frozen=True)
class ToolCallLoopResult:
    hit: bool
    repeat_count: int
    current_span_tokens: int
    marker_index: int


DEFAULT_TOOL_CALL_LOOP_CONFIG = ToolCallLoopConfig()


@dataclasses.dataclass(frozen=True)
class RequestRepetitionMonitorConfig:
    # Keep ad-hoc/default records backward compatible; production startup passes
    # an explicit enabled policy from RepetitionDetectionConfig.
    output_config: OutputRepetitionConfig = dataclasses.field(
        default_factory=lambda: OutputRepetitionConfig(enabled=False)
    )
    tool_loop_config: ToolCallLoopConfig = dataclasses.field(
        default_factory=ToolCallLoopConfig
    )
    # Tool-call begin/end markers are a service-level constant (fixed by the
    # model's chat template), tokenized once at startup. Empty means tool-call
    # loop monitoring stays off.
    tool_markers: tuple[ToolCallMarkerConfig, ...] = dataclasses.field(
        default_factory=tuple
    )


DEFAULT_REQUEST_REPETITION_MONITOR_CONFIG = RequestRepetitionMonitorConfig()


def _build_native_tool_call_marker_ids(
    markers: Sequence[ToolCallMarkerConfig],
) -> tuple[list[list[int]], list[list[int]]]:
    begin_ids = []
    end_ids = []
    for marker in markers:
        begin_ids.append([int(token_id) for token_id in marker.begin_ids])
        end_ids.append([int(token_id) for token_id in marker.end_ids])
    return begin_ids, end_ids


def _tool_call_loop_result_from_native(result) -> ToolCallLoopResult:
    if isinstance(result, (tuple, list)):
        hit, repeat_count, current_span_tokens, marker_index = result[:4]
        return ToolCallLoopResult(
            hit=bool(hit),
            repeat_count=int(repeat_count),
            current_span_tokens=int(current_span_tokens),
            marker_index=int(marker_index),
        )
    return ToolCallLoopResult(
        hit=bool(result.hit),
        repeat_count=int(result.repeat_count),
        current_span_tokens=int(result.current_span_tokens),
        marker_index=int(result.marker_index),
    )


def detect_tool_call_loop(
    input_ids: Sequence[int],
    output_ids: Sequence[int],
    markers: Sequence[ToolCallMarkerConfig],
    config: ToolCallLoopConfig = DEFAULT_TOOL_CALL_LOOP_CONFIG,
) -> Optional[ToolCallLoopResult]:
    if not config.enabled or not markers or not output_ids:
        return None
    status = native_online_repetition_status()
    if not status.available:
        return None

    marker_begin_ids, marker_end_ids = _build_native_tool_call_marker_ids(markers)
    result = status.module.check_tool_call_loop(
        input_ids,
        output_ids,
        marker_begin_ids,
        marker_end_ids,
        max(2, int(config.repeat_threshold)),
        max(1, int(config.max_span_tokens)),
    )
    loop_result = _tool_call_loop_result_from_native(result)
    if not loop_result.hit:
        return None
    return loop_result


class RequestRepetitionMonitor:
    """Request-scoped facade used by dash-sc access logging.

    The access-log emit layer owns transport and serialization. This object owns
    the tool-call loop detector, native availability state, and the flat fields
    emitted into the access log. Keeping that boundary explicit prevents the
    access-log layer from depending on pybind wrapper details.
    """

    def __init__(
        self,
        *,
        raw_mode: bool = False,
        input_ids: Optional[Sequence[int]] = None,
        tool_loop_config: Optional[ToolCallLoopConfig] = None,
        tool_markers: Optional[Sequence[ToolCallMarkerConfig]] = None,
        monitor_config: Optional[RequestRepetitionMonitorConfig] = None,
    ) -> None:
        config = monitor_config or DEFAULT_REQUEST_REPETITION_MONITOR_CONFIG
        self.raw_mode = raw_mode
        self.input_ids: Sequence[int] = input_ids or ()
        self.tool_loop_config = tool_loop_config or config.tool_loop_config
        self.output_config = config.output_config
        # Markers come from the service-level config unless a caller (tests)
        # passes them explicitly.
        self.tool_markers = (
            tuple(tool_markers)
            if tool_markers is not None
            else tuple(config.tool_markers)
        )

        self.tool_call_loop_result: Optional[ToolCallLoopResult] = None
        self.tool_call_loop_check_ms: Optional[float] = None
        self.tool_call_loop_impl: Optional[str] = None
        self._tool_error: Optional[str] = None
        self.output_repetition_result: Optional[OutputRepetitionResult] = None
        self.output_repetition_check_ms: float = 0.0
        self.output_repetition_impl: Optional[str] = None
        self._output_error: Optional[str] = None
        self._output_tracker = None
        self._output_finalized = False

    def set_input_ids(self, input_ids: Optional[Sequence[int]]) -> None:
        self.input_ids = input_ids or ()

    def is_active(self) -> bool:
        return not self.raw_mode and (
            self.output_config.enabled
            or (self.tool_loop_config.enabled and bool(self.tool_markers))
        )

    def _ensure_output_tracker(self):
        if self._output_tracker is not None:
            return self._output_tracker
        status = native_online_repetition_status()
        if not status.available:
            self.output_repetition_impl = "online_cpp_pybind_unavailable"
            return None
        native_config = status.module.OnlineRepetitionConfig()
        native_config.min_repeats = max(3, int(self.output_config.min_repeats))
        native_config.min_duplicate_tokens = max(
            0, int(self.output_config.min_duplicate_tokens)
        )
        native_config.max_period = max(1, int(self.output_config.max_period))
        native_config.non_contiguous_min_span = max(
            8, int(self.output_config.non_contiguous_min_span)
        )
        native_config.non_contiguous_min_occurrences = max(
            2, int(self.output_config.non_contiguous_min_occurrences)
        )
        native_config.non_contiguous_max_span = max(
            native_config.non_contiguous_min_span,
            int(self.output_config.non_contiguous_max_span),
        )
        self._output_tracker = status.module.OnlineRepetitionTracker(native_config)
        self.output_repetition_impl = "online_cpp_pybind"
        return self._output_tracker

    def update_output_delta(self, delta_ids: Sequence[int]) -> None:
        if (
            self.raw_mode
            or not self.output_config.enabled
            or not delta_ids
            or self._output_finalized
        ):
            return
        begin = time.perf_counter()
        try:
            tracker = self._ensure_output_tracker()
            if tracker is not None:
                tracker.update_many([int(token_id) for token_id in delta_ids])
        except Exception as e:
            _LOGGER.debug("output repetition update failed: %s", e)
            self._output_error = f"{type(e).__name__}: {e}"
            self.output_repetition_impl = "online_cpp_pybind_unavailable"
        finally:
            self.output_repetition_check_ms += (
                time.perf_counter() - begin
            ) * 1000.0

    def finalize_output(self) -> None:
        if self._output_finalized:
            return
        self._output_finalized = True
        if self.raw_mode or not self.output_config.enabled:
            self.output_repetition_impl = (
                "disabled_raw_mode" if self.raw_mode else "disabled"
            )
            return
        begin = time.perf_counter()
        try:
            tracker = self._ensure_output_tracker()
            if tracker is None:
                return
            tracker.finalize()
            result = tracker.result
            if result.hit:
                self.output_repetition_result = OutputRepetitionResult(
                    hit=True,
                    repeat_unit_size=int(result.repeat_unit_size),
                    repeat_count=int(result.repeat_count),
                    covered_token_count=int(result.covered_token_count),
                    duplicate_token_count=int(result.duplicate_token_count),
                    start_index=int(result.start_index),
                    end_index=int(result.end_index),
                    first_detect_index=int(result.first_detect_index),
                    non_contiguous=bool(result.non_contiguous),
                    occurrence_count=int(result.occurrence_count),
                )
        except Exception as e:
            _LOGGER.debug("output repetition finalization failed: %s", e)
            self._output_error = f"{type(e).__name__}: {e}"
            self.output_repetition_impl = "online_cpp_pybind_unavailable"
        finally:
            self.output_repetition_check_ms += (
                time.perf_counter() - begin
            ) * 1000.0

    def _tool_impl_name(self) -> str:
        if self.raw_mode:
            return "disabled_raw_mode"
        if not self.tool_loop_config.enabled or not self.tool_markers:
            return "disabled"
        return self.tool_call_loop_impl or "online_cpp_pybind"

    def check_tool_call_loop(self, generated_ids: Sequence[int]) -> None:
        if self.tool_call_loop_check_ms is not None:
            return
        if self.raw_mode or not self.tool_loop_config.enabled or not self.tool_markers:
            self.tool_call_loop_impl = (
                "disabled_raw_mode" if self.raw_mode else "disabled"
            )
            return
        begin = time.perf_counter()
        self.tool_call_loop_impl = "online_cpp_pybind"
        try:
            self.tool_call_loop_result = detect_tool_call_loop(
                self.input_ids,
                generated_ids,
                self.tool_markers,
                self.tool_loop_config,
            )
        except Exception as e:
            _LOGGER.debug("tool_call loop detection failed: %s", e)
            self._tool_error = f"{type(e).__name__}: {e}"
            self.tool_call_loop_impl = "online_cpp_pybind_unavailable"
        finally:
            self.tool_call_loop_check_ms = (time.perf_counter() - begin) * 1000.0

    def check_generated_ids(self, generated_ids: Sequence[int]) -> None:
        self.finalize_output()
        self.check_tool_call_loop(generated_ids)

    def monitor_impl(self) -> str:
        if self.raw_mode:
            return "disabled_raw_mode"
        output_impl = self.output_repetition_impl or (
            "online_cpp_pybind" if self.output_config.enabled else "disabled"
        )
        return f"output={output_impl},tool={self._tool_impl_name()}"

    def monitor_available(self) -> tuple[bool, Optional[str]]:
        if self.raw_mode:
            return False, "raw_mode"
        if self.output_config.enabled:
            status = native_online_repetition_status()
            if not status.available:
                return False, status.error or "native module unavailable"
            if self._output_error:
                return False, f"output: {self._output_error}"
        if self.tool_loop_config.enabled and self.tool_markers:
            status = native_online_repetition_status()
            if not status.available:
                self.tool_call_loop_impl = "online_cpp_pybind_unavailable"
                return False, status.error or "native module unavailable"
            if self._tool_error:
                self.tool_call_loop_impl = "online_cpp_pybind_unavailable"
                return False, f"tool_call: {self._tool_error}"
        return True, None

    def record_fields(self) -> dict[str, Any]:
        available, unavailable_reason = self.monitor_available()
        # Monitor didn't run (raw mode / native missing / check errored): only
        # report why. Emitting all-clear detection results would be a lie.
        if not available:
            return {
                "repetition_monitor_impl": self.monitor_impl(),
                "repetition_monitor_available": False,
                "repetition_monitor_unavailable_reason": unavailable_reason,
                "tool_call_loop_impl": self._tool_impl_name(),
                "tool_call_loop_error": self._tool_error,
            }

        tool_loop = self.tool_call_loop_result
        tool_hit = bool(tool_loop.hit) if tool_loop is not None else False
        output = self.output_repetition_result
        output_hit = bool(output.hit) if output is not None else False
        hit = output_hit or tool_hit
        repetition_tokens = (
            output.duplicate_token_count
            if output_hit and output is not None
            else tool_loop.current_span_tokens
            if tool_hit and tool_loop is not None
            else None
        )
        tool_span = (
            tool_loop.current_span_tokens
            if tool_hit and tool_loop is not None
            else None
        )
        output_kind = None
        if output_hit and output is not None:
            if output.non_contiguous:
                output_kind = "non_contiguous_span_repeat"
            elif output.repeat_unit_size == 1:
                output_kind = "same_token_run"
            elif output.repeat_unit_size <= 64:
                output_kind = "exact_ngram_loop"
            else:
                output_kind = "long_span_repeat"
        return {
            "repetition_detected": hit,
            "repetition_alert": hit,
            "repetition_reason": (
                f"output_repetition:{output_kind}:repeat_count={output.repeat_count}"
                if output_hit and output is not None
                else (
                    f"tool_call_loop:repeat_count={tool_loop.repeat_count}"
                    if tool_hit and tool_loop is not None
                    else None
                )
            ),
            "repetition_primary_source": (
                "output_repetition"
                if output_hit
                else ("tool_call_loop" if tool_hit else None)
            ),
            "repetition_token_len": repetition_tokens,
            "tool_call_loop_token_len": tool_span,
            "function_tool_repeated": tool_hit,
            "repetition_monitor_impl": self.monitor_impl(),
            "repetition_monitor_available": True,
            "repetition_monitor_unavailable_reason": None,
            "tool_call_loop_impl": self._tool_impl_name(),
            "tool_call_loop_error": self._tool_error,
            "tool_call_loop_check_ms": (
                round(self.tool_call_loop_check_ms, 6)
                if self.tool_call_loop_check_ms is not None
                else None
            ),
            "tool_call_loop_hit": tool_hit,
            "tool_call_loop_repeat_count": (
                tool_loop.repeat_count if tool_loop is not None else None
            ),
            "tool_call_loop_current_span_tokens": (
                tool_loop.current_span_tokens if tool_loop is not None else None
            ),
            "tool_call_loop_marker_index": (
                tool_loop.marker_index if tool_loop is not None else None
            ),
            "output_repetition": output_hit,
            "output_repetition_kind": output_kind,
            "output_repetition_impl": self.output_repetition_impl,
            "output_repetition_error": self._output_error,
            "output_repetition_check_ms": round(
                self.output_repetition_check_ms, 6
            ),
            "output_repetition_period": (
                output.repeat_unit_size if output is not None else None
            ),
            "output_repetition_repeat_count": (
                output.repeat_count if output is not None else None
            ),
            "output_repetition_occurrence_count": (
                output.occurrence_count if output is not None else None
            ),
            "output_repetition_duplicate_token_count": (
                output.duplicate_token_count if output is not None else None
            ),
            "output_repetition_covered_token_count": (
                output.covered_token_count if output is not None else None
            ),
            "output_repetition_start_index": (
                output.start_index if output is not None else None
            ),
            "output_repetition_end_index": (
                output.end_index if output is not None else None
            ),
            "output_repetition_first_detect_index": (
                output.first_detect_index if output is not None else None
            ),
        }
