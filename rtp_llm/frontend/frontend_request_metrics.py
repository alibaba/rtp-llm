import logging
import threading
import time
from dataclasses import dataclass, replace
from typing import Any, Callable, Dict, Iterable, Optional, Tuple

from rtp_llm.metrics import GaugeMetrics, kmonitor


def _field(value: Any, name: str, default: Any = None) -> Any:
    if value is None:
        return default
    if isinstance(value, dict):
        return value.get(name, default)
    return getattr(value, name, default)


def _as_nonnegative_int(value: Any) -> Optional[int]:
    if value is None or isinstance(value, bool):
        return None
    try:
        return max(int(value), 0)
    except (TypeError, ValueError):
        return None


def _iter_aux_info(response: Any) -> Iterable[Any]:
    response_batch = _field(response, "response_batch")
    if response_batch:
        for item in response_batch:
            yield from _iter_aux_info(item)
        return

    aux_info = _field(response, "aux_info")
    if isinstance(aux_info, (list, tuple)):
        yield from (item for item in aux_info if item is not None)
    elif aux_info:
        yield aux_info


def _usage_snapshot(response: Any) -> Optional["_RequestSnapshot"]:
    usage = _field(response, "usage")
    if not usage:
        return None
    prompt_tokens = _as_nonnegative_int(_field(usage, "prompt_tokens"))
    output_tokens = _as_nonnegative_int(_field(usage, "completion_tokens"))
    prompt_details = _field(usage, "prompt_tokens_details")
    cached_tokens = _as_nonnegative_int(_field(prompt_details, "cached_tokens"))
    if prompt_tokens is None and output_tokens is None:
        return None
    return _RequestSnapshot(
        input_len=prompt_tokens,
        output_len=output_tokens,
        reuse_len=cached_tokens,
    )


@dataclass(frozen=True)
class _RequestSnapshot:
    input_len: Optional[int] = None
    output_len: Optional[int] = None
    generate_token_num: int = 0
    context_token_num: Optional[int] = None
    context_token_num_with_cache: Optional[int] = None
    reuse_len: Optional[int] = None
    speculative_verify_rounds: int = 0
    speculative_accepted_token_num: int = 0
    speculative_proposed_draft_tokens: int = 0
    context_execute_time_us: int = 0
    context_execute_time_with_cache_us: int = 0
    generate_execute_time_us: int = 0


@dataclass(frozen=True)
class _TpsCounters:
    input_tokens_with_cache: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    context_time_with_cache_us: int = 0
    context_time_us: int = 0
    generate_time_us: int = 0


def _tps_counter_delta(
    current: _TpsCounters, previous: _TpsCounters
) -> Tuple[_TpsCounters, _TpsCounters]:
    delta_values = {}
    high_water_values = {}
    for field in _TpsCounters.__dataclass_fields__:
        current_value = getattr(current, field)
        previous_value = getattr(previous, field)
        delta_values[field] = max(current_value - previous_value, 0)
        high_water_values[field] = max(current_value, previous_value)
    return _TpsCounters(**delta_values), _TpsCounters(**high_water_values)


def _tps_counters(snapshot: _RequestSnapshot) -> _TpsCounters:
    input_tokens_with_cache = snapshot.context_token_num_with_cache
    if input_tokens_with_cache is None:
        input_tokens_with_cache = snapshot.input_len or 0

    input_tokens = snapshot.context_token_num
    if input_tokens is None:
        if snapshot.input_len is not None and snapshot.reuse_len is not None:
            input_tokens = max(snapshot.input_len - snapshot.reuse_len, 0)
        else:
            input_tokens = 0

    context_time_with_cache_us = snapshot.context_execute_time_with_cache_us
    if context_time_with_cache_us <= 0:
        # Rolling-upgrade fallback for backends that only expose the
        # non-cache execution-time allocation.
        context_time_with_cache_us = snapshot.context_execute_time_us

    return _TpsCounters(
        input_tokens_with_cache=input_tokens_with_cache,
        input_tokens=input_tokens,
        output_tokens=snapshot.generate_token_num,
        context_time_with_cache_us=context_time_with_cache_us,
        context_time_us=snapshot.context_execute_time_us,
        generate_time_us=snapshot.generate_execute_time_us,
    )


def _request_snapshot(response: Any) -> Optional[_RequestSnapshot]:
    usage_snapshot = _usage_snapshot(response)
    aux_infos = list(_iter_aux_info(response))
    if not aux_infos:
        return usage_snapshot

    # Prompt/cache and cumulative speculative counters belong to the
    # GenerateStream and can be repeated for num_return_sequences. Output
    # length is per sequence and must be summed.
    input_len = _as_nonnegative_int(_field(aux_infos[0], "input_len"))
    reuse_len = _as_nonnegative_int(_field(aux_infos[0], "reuse_len"))
    output_len = 0
    has_output_len = False
    generate_token_num = 0
    explicit_generate_token_num = _as_nonnegative_int(
        _field(response, "generate_token_num")
    )
    speculative_verify_rounds = 0
    speculative_accepted_token_num = 0
    speculative_proposed_draft_tokens = 0
    context_execute_time_us = 0
    context_execute_time_with_cache_us = 0
    generate_execute_time_us = 0
    for aux in aux_infos:
        sequence_output_len = _as_nonnegative_int(_field(aux, "output_len"))
        if sequence_output_len is not None:
            has_output_len = True
            output_len += sequence_output_len
            generate_token_num += max(sequence_output_len - 1, 0)
        sequence_generate_token_num = _as_nonnegative_int(
            _field(aux, "generate_token_num")
        )
        if sequence_generate_token_num is not None:
            # This is a GenerateStream-level cumulative counter repeated on
            # every returned sequence, so keep one authoritative copy.
            explicit_generate_token_num = max(
                explicit_generate_token_num or 0,
                sequence_generate_token_num,
            )
        speculative_verify_rounds = max(
            speculative_verify_rounds,
            _as_nonnegative_int(_field(aux, "speculative_verify_rounds")) or 0,
        )
        speculative_accepted_token_num = max(
            speculative_accepted_token_num,
            _as_nonnegative_int(_field(aux, "speculative_accepted_token_num")) or 0,
        )
        speculative_proposed_draft_tokens = max(
            speculative_proposed_draft_tokens,
            _as_nonnegative_int(_field(aux, "speculative_proposed_draft_tokens")) or 0,
        )
        context_execute_time_us = max(
            context_execute_time_us,
            _as_nonnegative_int(_field(aux, "context_execute_time_us")) or 0,
        )
        context_execute_time_with_cache_us = max(
            context_execute_time_with_cache_us,
            _as_nonnegative_int(_field(aux, "context_execute_time_with_cache_us")) or 0,
        )
        generate_execute_time_us = max(
            generate_execute_time_us,
            _as_nonnegative_int(_field(aux, "generate_execute_time_us")) or 0,
        )

    # New backends expose authoritative private cumulative counters on the
    # GenerateOutputs envelope. They represent complete engine-step
    # contributions and must override the per-request AuxInfo allocations.
    explicit_context_execute_time_us = _as_nonnegative_int(
        _field(response, "context_execute_time_us")
    )
    if explicit_context_execute_time_us is not None:
        context_execute_time_us = explicit_context_execute_time_us
    explicit_context_execute_time_with_cache_us = _as_nonnegative_int(
        _field(response, "context_execute_time_with_cache_us")
    )
    if explicit_context_execute_time_with_cache_us is not None:
        context_execute_time_with_cache_us = explicit_context_execute_time_with_cache_us
    explicit_generate_execute_time_us = _as_nonnegative_int(
        _field(response, "generate_execute_time_us")
    )
    if explicit_generate_execute_time_us is not None:
        generate_execute_time_us = explicit_generate_execute_time_us

    # Renderers may expose one AuxInfo object for a multi-choice response while
    # usage contains request-wide lengths. Prefer those aggregate request
    # lengths without replacing the backend execution/speculative counters.
    request_input_len = (
        usage_snapshot.input_len
        if usage_snapshot is not None and usage_snapshot.input_len is not None
        else input_len
    )
    request_output_len = (
        usage_snapshot.output_len
        if usage_snapshot is not None and usage_snapshot.output_len is not None
        else (output_len if has_output_len else None)
    )
    request_reuse_len = (
        usage_snapshot.reuse_len
        if usage_snapshot is not None and usage_snapshot.reuse_len is not None
        else reuse_len
    )

    context_token_num = _as_nonnegative_int(_field(response, "context_token_num"))
    context_token_num_with_cache = _as_nonnegative_int(
        _field(response, "context_token_num_with_cache")
    )
    for aux in aux_infos:
        aux_context_token_num = _as_nonnegative_int(_field(aux, "context_token_num"))
        if aux_context_token_num is not None:
            context_token_num = max(
                context_token_num or 0,
                aux_context_token_num,
            )
        aux_context_token_num_with_cache = _as_nonnegative_int(
            _field(aux, "context_token_num_with_cache")
        )
        if aux_context_token_num_with_cache is not None:
            context_token_num_with_cache = max(
                context_token_num_with_cache or 0,
                aux_context_token_num_with_cache,
            )

    return _RequestSnapshot(
        input_len=request_input_len,
        output_len=request_output_len,
        generate_token_num=(
            explicit_generate_token_num
            if explicit_generate_token_num is not None
            else (
                speculative_accepted_token_num
                if speculative_verify_rounds > 0
                else generate_token_num
            )
        ),
        context_token_num=context_token_num,
        context_token_num_with_cache=context_token_num_with_cache,
        reuse_len=request_reuse_len,
        speculative_verify_rounds=speculative_verify_rounds,
        speculative_accepted_token_num=speculative_accepted_token_num,
        speculative_proposed_draft_tokens=speculative_proposed_draft_tokens,
        context_execute_time_us=context_execute_time_us,
        context_execute_time_with_cache_us=context_execute_time_with_cache_us,
        generate_execute_time_us=generate_execute_time_us,
    )


def _request_units(response: Any) -> Iterable[Tuple[Any, Any]]:
    """Yield independently-accounted prompts from an outward response frame."""
    metric_unit_id = _as_nonnegative_int(_field(response, "_frontend_metric_unit_id"))
    metric_attempt = _as_nonnegative_int(_field(response, "_frontend_metric_attempt"))
    if metric_unit_id is not None:
        unit_key: Any = metric_unit_id
        if metric_attempt is not None:
            unit_key = (metric_unit_id, metric_attempt)
        yield unit_key, response
        return

    response_batch = _field(response, "response_batch")
    if response_batch:
        yield from enumerate(response_batch)
    else:
        yield (0, metric_attempt) if metric_attempt is not None else 0, response


def _merge_snapshots(
    snapshots: Iterable[_RequestSnapshot],
) -> Optional[_RequestSnapshot]:
    snapshots = list(snapshots)
    if not snapshots:
        return None

    def optional_sum(name: str) -> Optional[int]:
        values = [getattr(snapshot, name) for snapshot in snapshots]
        known_values = [value for value in values if value is not None]
        return sum(known_values) if known_values else None

    return _RequestSnapshot(
        input_len=optional_sum("input_len"),
        output_len=optional_sum("output_len"),
        generate_token_num=sum(snapshot.generate_token_num for snapshot in snapshots),
        context_token_num=optional_sum("context_token_num"),
        context_token_num_with_cache=optional_sum("context_token_num_with_cache"),
        reuse_len=optional_sum("reuse_len"),
        speculative_verify_rounds=sum(
            snapshot.speculative_verify_rounds for snapshot in snapshots
        ),
        speculative_accepted_token_num=sum(
            snapshot.speculative_accepted_token_num for snapshot in snapshots
        ),
        speculative_proposed_draft_tokens=sum(
            snapshot.speculative_proposed_draft_tokens for snapshot in snapshots
        ),
        context_execute_time_us=sum(
            snapshot.context_execute_time_us for snapshot in snapshots
        ),
        context_execute_time_with_cache_us=sum(
            snapshot.context_execute_time_with_cache_us for snapshot in snapshots
        ),
        generate_execute_time_us=sum(
            snapshot.generate_execute_time_us for snapshot in snapshots
        ),
    )


def _response_has_output_payload(response: Any) -> bool:
    response_batch = _field(response, "response_batch")
    if response_batch:
        return any(_response_has_output_payload(item) for item in response_batch)

    raw_response = _field(response, "response")
    if isinstance(raw_response, str):
        return bool(raw_response)
    if isinstance(raw_response, (list, tuple)):
        return any(bool(item) for item in raw_response)

    choices = _field(response, "choices")
    if choices:
        for choice in choices:
            delta = _field(choice, "delta")
            message = _field(choice, "message")
            for payload in (delta, message):
                if payload is None:
                    continue
                for name in (
                    "content",
                    "reasoning_content",
                    "function_call",
                    "tool_calls",
                ):
                    if _field(payload, name):
                        return True

    output_ids = _field(response, "output_ids")
    if output_ids is not None:
        try:
            if len(output_ids) > 0:
                return True
        except TypeError:
            return True

    # Raw pipeline response types are not required to expose rendered payload
    # fields. In that compatibility path, a positive cumulative token delta is
    # the best available indication of a non-empty outward frame.
    return raw_response is None and choices is None


class FrontendRequestMetricState:
    def __init__(
        self,
        owner: "FrontendRequestMetrics",
        tags: Dict[str, str],
        container_tags: Dict[str, str],
        streaming: bool,
        speculative_steps: int,
        start_ms: float,
    ):
        self._owner = owner
        self._tags = tags
        self._container_tags = container_tags
        self._streaming = streaming
        self._start_ms = start_ms
        self._first_output_ms: Optional[float] = None
        self._first_output_len: Optional[int] = None
        self._last_observed_output_len: Dict[Any, int] = {}
        self._unit_snapshots: Dict[Any, _RequestSnapshot] = {}
        self._tps_unit_snapshots: Dict[Any, _RequestSnapshot] = {}
        self._last_backend_batch_size: Dict[Any, int] = {}
        self._last_backend_output_len: Dict[Any, int] = {}
        self._fallback_generate_token_num: Dict[Any, int] = {}
        self._snapshot: Optional[_RequestSnapshot] = None
        self._last_tps_counters = _TpsCounters()
        self._finished = False

    def observe(self, response: Any, now_ms: Optional[float] = None) -> None:
        try:
            self._observe(response, now_ms)
        except Exception:
            # Metrics are best-effort. Malformed or partially-upgraded AuxInfo
            # must not break an otherwise valid inference response.
            logging.exception("failed to observe frontend request metrics")

    def _observe(self, response: Any, now_ms: Optional[float]) -> None:
        if self._finished:
            return
        first_frame_delta = 0
        for unit_id, unit_response in _request_units(response):
            snapshot = _request_snapshot(unit_response)
            if snapshot is None:
                continue
            self._unit_snapshots[unit_id] = snapshot
            if snapshot.output_len is None:
                continue

            has_payload = _response_has_output_payload(unit_response)
            previous_output_len = self._last_observed_output_len.get(unit_id, 0)
            output_delta = max(snapshot.output_len - previous_output_len, 0)
            if has_payload:
                # Advance only when the renderer exposes a payload. Empty
                # tool-call buffering frames remain part of the next delta.
                self._last_observed_output_len[unit_id] = max(
                    previous_output_len,
                    snapshot.output_len,
                )
                first_frame_delta += output_delta

        self._snapshot = _merge_snapshots(self._unit_snapshots.values())
        # Once the raw backend side-channel has produced a snapshot it is the
        # sole TPS source. Outward frames may carry newer request-level MTP or
        # beam counters and must not advance the engine-aligned high-water.
        if not self._tps_unit_snapshots:
            self._record_tps_progress(self._snapshot)
        if self._first_output_ms is None and first_frame_delta > 0:
            self._first_output_ms = self._owner.now_ms() if now_ms is None else now_ms
            self._first_output_len = first_frame_delta

    def observe_tps(self, response: Any) -> None:
        """Record backend cumulative TPS counters without changing TTFT state."""
        try:
            self._observe_tps(response)
        except Exception:
            # Keep the raw backend generator isolated from metric parsing.
            logging.exception("failed to observe frontend TPS metrics")

    def _observe_tps(self, response: Any) -> None:
        if self._finished:
            return
        for unit_id, unit_response in _request_units(response):
            snapshot = _request_snapshot(unit_response)
            if snapshot is not None:
                self._tps_unit_snapshots[unit_id] = self._backend_tps_snapshot(
                    unit_id,
                    unit_response,
                    snapshot,
                )
        self._record_tps_progress(_merge_snapshots(self._tps_unit_snapshots.values()))

    def _backend_tps_snapshot(
        self,
        unit_id: Any,
        response: Any,
        snapshot: _RequestSnapshot,
    ) -> _RequestSnapshot:
        updates: Dict[str, Any] = {}

        # A non-beam num_return_sequences request executes its prompt once per
        # sequence in the engine. Keep request-level input_len unchanged, but
        # use the actual context execution batch for TPS numerators.
        context_batch_size = _as_nonnegative_int(
            _field(response, "_frontend_context_batch_size")
        )
        if context_batch_size is not None:
            context_batch_size = max(context_batch_size, 1)
            if (
                snapshot.context_token_num_with_cache is None
                and snapshot.input_len is not None
            ):
                updates["context_token_num_with_cache"] = (
                    snapshot.input_len * context_batch_size
                )
            if (
                snapshot.context_token_num is None
                and snapshot.input_len is not None
                and snapshot.reuse_len is not None
            ):
                updates["context_token_num"] = (
                    max(snapshot.input_len - snapshot.reuse_len, 0) * context_batch_size
                )

        # Older backends do not expose cumulative generate_token_num. Raw
        # frames still let us reproduce NormalExecutor's decode numerator: a
        # step executes the previous frame's batch width. This remains
        # monotonic when variable beam width grows or shrinks.
        output_batch_size = _as_nonnegative_int(
            _field(response, "_frontend_output_batch_size")
        )
        if output_batch_size is not None:
            aux_infos = list(_iter_aux_info(response))
            current_output_len = max(
                (
                    value
                    for value in (
                        _as_nonnegative_int(_field(aux, "output_len"))
                        for aux in aux_infos
                    )
                    if value is not None
                ),
                default=None,
            )
            previous_output_len = self._last_backend_output_len.get(unit_id)
            has_explicit_generate_counter = _as_nonnegative_int(
                _field(response, "generate_token_num")
            ) is not None or any(
                _as_nonnegative_int(_field(aux, "generate_token_num")) is not None
                for aux in aux_infos
            )
            if (
                not has_explicit_generate_counter
                and snapshot.speculative_verify_rounds == 0
            ):
                generate_token_num = self._fallback_generate_token_num.get(
                    unit_id,
                    # Rolling-upgrade compatibility: an old non-streaming
                    # backend may expose only one final frame. Preserve its
                    # output_len-derived numerator instead of replacing it
                    # with zero merely because no previous raw frame exists.
                    snapshot.generate_token_num,
                )
                previous_batch_size = self._last_backend_batch_size.get(unit_id)
                if previous_batch_size is not None:
                    step_output_len = (
                        max(current_output_len - previous_output_len, 0)
                        if current_output_len is not None
                        and previous_output_len is not None
                        else max(
                            (
                                _as_nonnegative_int(_field(aux, "step_output_len")) or 0
                                for aux in aux_infos
                            ),
                            default=0,
                        )
                    )
                    generate_token_num += previous_batch_size * step_output_len
                self._fallback_generate_token_num[unit_id] = generate_token_num
                updates["generate_token_num"] = generate_token_num
            if current_output_len is not None:
                # A stale/duplicate frame must not roll the beam width back;
                # the next decode step is charged using the batch width from
                # the most recent cumulative output position.
                if (
                    previous_output_len is None
                    or current_output_len > previous_output_len
                ):
                    self._last_backend_output_len[unit_id] = current_output_len
                    self._last_backend_batch_size[unit_id] = output_batch_size
            else:
                # Compatibility with old raw frames that expose only a step
                # length and therefore have no cumulative stale-frame signal.
                self._last_backend_batch_size[unit_id] = output_batch_size

        return replace(snapshot, **updates) if updates else snapshot

    def _record_tps_progress(self, snapshot: Optional[_RequestSnapshot]) -> None:
        if snapshot is None:
            return
        counters = _tps_counters(snapshot)
        delta, high_water = _tps_counter_delta(counters, self._last_tps_counters)
        self._owner._record_tps_delta(delta)
        # Token/time high-water marks advance independently. The container
        # window retains an unmatched side until its pair arrives; this is
        # required when integer execution-time allocation gives one active
        # stream 0us while the engine still counts its executed tokens.
        self._last_tps_counters = high_water

    def finish(self, now_ms: Optional[float] = None) -> None:
        if self._finished:
            return
        self._finished = True
        end_ms = self._owner.now_ms() if now_ms is None else now_ms
        try:
            # Normally the final response has already been observed, making
            # this a zero delta. Keep the call here to capture any residual
            # cumulative counters before the request-level report is emitted.
            tps_snapshot = _merge_snapshots(self._tps_unit_snapshots.values())
            self._record_tps_progress(
                tps_snapshot if tps_snapshot is not None else self._snapshot
            )
            self._report(end_ms)
        except Exception:
            logging.exception("failed to report frontend request metrics")
        finally:
            self._owner._request_finished(self._container_tags)

    def _report(self, end_ms: float) -> None:
        report = self._owner.report
        tags = self._tags
        duration_ms = max(end_ms - self._start_ms, 0.0)
        report(GaugeMetrics.FRONTEND_REQUEST_RT_MS_METRIC, duration_ms, tags)

        ttft_ms: Optional[float] = None
        if self._first_output_ms is not None:
            ttft_ms = max(self._first_output_ms - self._start_ms, 0.0)
            report(GaugeMetrics.FRONTEND_TTFT_MS_METRIC, ttft_ms, tags)

        snapshot = self._snapshot
        if snapshot is None:
            return
        tpot_ms: Optional[float] = None
        if (
            self._first_output_ms is not None
            and self._first_output_len is not None
            and snapshot.output_len is not None
        ):
            remaining_tokens = snapshot.output_len - self._first_output_len
            if remaining_tokens > 0:
                tpot_ms = max(end_ms - self._first_output_ms, 0.0) / remaining_tokens
        if snapshot.input_len is not None:
            report(
                GaugeMetrics.FRONTEND_INPUT_LENGTH_METRIC,
                snapshot.input_len,
                tags,
            )
        if snapshot.output_len is not None:
            report(
                GaugeMetrics.FRONTEND_OUTPUT_LENGTH_METRIC,
                snapshot.output_len,
                tags,
            )
        if snapshot.reuse_len is not None:
            report(
                GaugeMetrics.FRONTEND_CACHED_TOKEN_LENGTH_METRIC,
                snapshot.reuse_len,
                tags,
            )
        if snapshot.input_len is not None and snapshot.reuse_len is not None:
            if snapshot.input_len > 0:
                report(
                    GaugeMetrics.FRONTEND_CACHE_HIT_RATIO_METRIC,
                    min(snapshot.reuse_len * 100.0 / snapshot.input_len, 100.0),
                    tags,
                )

        if (
            self._streaming
            and self._first_output_len is not None
            and self._first_output_len > 0
        ):
            report(
                GaugeMetrics.FRONTEND_STREAM_FIRST_OUTPUT_TOKEN_LENGTH_METRIC,
                self._first_output_len,
                tags,
            )

        if tpot_ms is not None:
            report(GaugeMetrics.FRONTEND_TPOT_MS_METRIC, tpot_ms, tags)

        if snapshot.speculative_verify_rounds > 0:
            avg_accept_len = (
                snapshot.speculative_accepted_token_num
                / snapshot.speculative_verify_rounds
            )
            report(
                GaugeMetrics.FRONTEND_SPECULATIVE_AVG_ACCEPT_LENGTH_METRIC,
                avg_accept_len,
                tags,
            )

        if snapshot.speculative_proposed_draft_tokens > 0:
            accepted_draft_tokens = max(
                snapshot.speculative_accepted_token_num
                - snapshot.speculative_verify_rounds,
                0,
            )
            accept_rate = (
                accepted_draft_tokens / snapshot.speculative_proposed_draft_tokens
            )
            report(
                GaugeMetrics.FRONTEND_SPECULATIVE_ACCEPT_RATE_METRIC,
                min(max(accept_rate, 0.0), 1.0),
                tags,
            )


class FrontendRequestMetrics:
    def __init__(
        self,
        metric_reporter: Any = kmonitor,
        clock: Callable[[], float] = time.monotonic,
        concurrency_report_interval_s: Optional[float] = None,
    ):
        self._metric_reporter = metric_reporter
        self._clock = clock
        self._active_requests = 0
        self._active_lock = threading.Lock()
        self._container_tags: Optional[Dict[str, str]] = None
        self._pending_tps = _TpsCounters()
        self._close_lock = threading.Lock()
        self._closed = False
        self._heartbeat_stop = threading.Event()
        self._heartbeat_thread: Optional[threading.Thread] = None
        if (
            concurrency_report_interval_s is not None
            and concurrency_report_interval_s > 0
        ):
            self._heartbeat_thread = threading.Thread(
                target=self._report_concurrency_periodically,
                args=(concurrency_report_interval_s,),
                name="frontend-metric-heartbeat",
                daemon=True,
            )
            self._heartbeat_thread.start()

    def now_ms(self) -> float:
        return self._clock() * 1000.0

    def report(self, metric: Any, value: float, tags: Dict[str, str]) -> None:
        try:
            self._metric_reporter.report(metric, value, tags)
        except Exception:
            logging.exception("failed to report frontend metric %s", metric)

    def begin(
        self,
        *,
        rank_id: str,
        server_id: str,
        source: str,
        streaming: bool,
        speculative_steps: int,
        priority: str = "0",
    ) -> FrontendRequestMetricState:
        tags = {
            "rank_id": rank_id,
            "server_id": server_id,
            "source": source,
            "streaming": str(streaming).lower(),
            "priority": priority,
        }
        container_tags = {
            "rank_id": rank_id,
            "server_id": server_id,
        }
        with self._active_lock:
            self._active_requests += 1
            concurrency = self._active_requests
            self._container_tags = container_tags
        self.report(
            GaugeMetrics.FRONTEND_CONCURRENCY_METRIC,
            concurrency,
            container_tags,
        )
        return FrontendRequestMetricState(
            self,
            tags,
            container_tags,
            streaming,
            speculative_steps,
            self.now_ms(),
        )

    def _request_finished(self, tags: Dict[str, str]) -> None:
        with self._active_lock:
            self._active_requests = max(self._active_requests - 1, 0)
            concurrency = self._active_requests
            heartbeat_running = self._heartbeat_thread is not None
        self.report(GaugeMetrics.FRONTEND_CONCURRENCY_METRIC, concurrency, tags)
        if not heartbeat_running:
            # Tests and embedders may intentionally disable the periodic
            # reporter. Flush the accumulated container window at request end
            # rather than falling back to a per-request TPS formula.
            self._report_tps_window()

    def _record_tps_delta(self, delta: _TpsCounters) -> None:
        with self._active_lock:
            pending = self._pending_tps
            values = {
                field: getattr(pending, field)
                for field in _TpsCounters.__dataclass_fields__
            }
            for field in _TpsCounters.__dataclass_fields__:
                values[field] += max(getattr(delta, field), 0)
            self._pending_tps = _TpsCounters(**values)

    def _report_tps_window(self, *, emit_idle_zero: bool = True) -> None:
        with self._active_lock:
            counters = self._pending_tps
            report_values = {field: 0 for field in _TpsCounters.__dataclass_fields__}
            pending_values = {field: 0 for field in _TpsCounters.__dataclass_fields__}
            for token_field, time_field in (
                ("input_tokens_with_cache", "context_time_with_cache_us"),
                ("input_tokens", "context_time_us"),
                ("output_tokens", "generate_time_us"),
            ):
                token_value = getattr(counters, token_field)
                time_value = getattr(counters, time_field)
                target = (
                    report_values
                    if token_value > 0 and time_value > 0
                    else pending_values
                )
                target[token_field] = token_value
                target[time_field] = time_value
            report_counters = _TpsCounters(**report_values)
            self._pending_tps = _TpsCounters(**pending_values)
            active_requests = self._active_requests
            tags = (
                dict(self._container_tags) if self._container_tags is not None else None
            )
        if tags is None:
            return

        has_context_with_cache = report_counters.context_time_with_cache_us > 0
        has_context = report_counters.context_time_us > 0
        has_generate = report_counters.generate_time_us > 0
        if has_context_with_cache:
            self.report(
                GaugeMetrics.FRONTEND_INPUT_TOKEN_TPS_METRIC,
                report_counters.input_tokens_with_cache
                * 1_000_000.0
                / report_counters.context_time_with_cache_us,
                tags,
            )
        if has_context:
            self.report(
                GaugeMetrics.FRONTEND_NONCACHE_INPUT_TOKEN_TPS_METRIC,
                report_counters.input_tokens
                * 1_000_000.0
                / report_counters.context_time_us,
                tags,
            )
        if has_generate:
            output_tps = (
                report_counters.output_tokens
                * 1_000_000.0
                / report_counters.generate_time_us
            )
            self.report(
                GaugeMetrics.FRONTEND_OUTPUT_TOKEN_TPS_METRIC,
                output_tps,
                tags,
            )
            self.report(
                GaugeMetrics.FRONTEND_NONCACHE_OUTPUT_TOKEN_TPS_METRIC,
                output_tps,
                tags,
            )
        if (
            not (has_context_with_cache or has_context or has_generate)
            and self._pending_tps == _TpsCounters()
            and active_requests == 0
            and emit_idle_zero
        ):
            for metric in (
                GaugeMetrics.FRONTEND_INPUT_TOKEN_TPS_METRIC,
                GaugeMetrics.FRONTEND_NONCACHE_INPUT_TOKEN_TPS_METRIC,
                GaugeMetrics.FRONTEND_OUTPUT_TOKEN_TPS_METRIC,
                GaugeMetrics.FRONTEND_NONCACHE_OUTPUT_TOKEN_TPS_METRIC,
            ):
                self.report(metric, 0.0, tags)

    def _report_concurrency_periodically(self, interval_s: float) -> None:
        while not self._heartbeat_stop.wait(interval_s):
            with self._active_lock:
                concurrency = self._active_requests
                tags = (
                    dict(self._container_tags)
                    if self._container_tags is not None
                    else None
                )
            if tags is not None:
                self.report(
                    GaugeMetrics.FRONTEND_CONCURRENCY_METRIC,
                    concurrency,
                    tags,
                )
            self._report_tps_window()

    def close(self) -> None:
        with self._close_lock:
            if self._closed:
                return
            self._heartbeat_stop.set()
            heartbeat_thread = self._heartbeat_thread
            if heartbeat_thread is not None:
                # A real termination barrier is required here: the heartbeat
                # may already have taken the pending counters for reporting.
                heartbeat_thread.join()
            with self._active_lock:
                self._heartbeat_thread = None
            # Requests completed after the last heartbeat remain in this
            # partial window. Flush it without creating an extra idle sample.
            self._report_tps_window(emit_idle_zero=False)
            self._closed = True
