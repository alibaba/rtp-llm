import logging
import threading
import time
from dataclasses import dataclass
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


def _request_snapshot(response: Any) -> Optional[_RequestSnapshot]:
    aux_infos = list(_iter_aux_info(response))
    if not aux_infos:
        return _usage_snapshot(response)

    # Prompt and cache reuse belong to the request and must not be multiplied by
    # num_return_sequences. Output and speculative rounds are per sequence.
    input_len = _as_nonnegative_int(_field(aux_infos[0], "input_len"))
    reuse_len = _as_nonnegative_int(_field(aux_infos[0], "reuse_len"))
    output_len = 0
    has_output_len = False
    generate_token_num = 0
    explicit_generate_token_num: Optional[int] = None
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
            explicit_generate_token_num = (
                explicit_generate_token_num or 0
            ) + sequence_generate_token_num
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

    return _RequestSnapshot(
        input_len=input_len,
        output_len=output_len if has_output_len else None,
        generate_token_num=(
            explicit_generate_token_num
            if explicit_generate_token_num is not None
            else generate_token_num
        ),
        context_token_num=_as_nonnegative_int(
            _field(aux_infos[0], "context_token_num")
        ),
        context_token_num_with_cache=_as_nonnegative_int(
            _field(aux_infos[0], "context_token_num_with_cache")
        ),
        reuse_len=reuse_len,
        speculative_verify_rounds=speculative_verify_rounds,
        speculative_accepted_token_num=speculative_accepted_token_num,
        speculative_proposed_draft_tokens=speculative_proposed_draft_tokens,
        context_execute_time_us=context_execute_time_us,
        context_execute_time_with_cache_us=context_execute_time_with_cache_us,
        generate_execute_time_us=generate_execute_time_us,
    )


def _request_units(response: Any) -> Iterable[Tuple[int, Any]]:
    """Yield independently-accounted prompts from an outward response frame."""
    response_batch = _field(response, "response_batch")
    if response_batch:
        yield from enumerate(response_batch)
    else:
        yield 0, response


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
        self._last_observed_output_len: Dict[int, int] = {}
        self._unit_snapshots: Dict[int, _RequestSnapshot] = {}
        self._snapshot: Optional[_RequestSnapshot] = None
        self._finished = False

    def observe(self, response: Any, now_ms: Optional[float] = None) -> None:
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
        if self._first_output_ms is None and first_frame_delta > 0:
            self._first_output_ms = self._owner.now_ms() if now_ms is None else now_ms
            self._first_output_len = first_frame_delta

    def finish(self, now_ms: Optional[float] = None) -> None:
        if self._finished:
            return
        self._finished = True
        end_ms = self._owner.now_ms() if now_ms is None else now_ms
        try:
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
        self._report_token_tps(snapshot)

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

    def _report_token_tps(self, snapshot: _RequestSnapshot) -> None:
        report = self._owner.report
        tags = self._tags
        input_tokens_with_cache = snapshot.context_token_num_with_cache
        if input_tokens_with_cache is None:
            input_tokens_with_cache = snapshot.input_len
        input_tokens = snapshot.context_token_num
        if (
            input_tokens is None
            and snapshot.input_len is not None
            and snapshot.reuse_len is not None
        ):
            input_tokens = max(snapshot.input_len - snapshot.reuse_len, 0)

        context_time_with_cache_us = snapshot.context_execute_time_with_cache_us
        if context_time_with_cache_us <= 0:
            # Rolling-upgrade fallback for backends that only expose the
            # non-cache execution-time allocation.
            context_time_with_cache_us = snapshot.context_execute_time_us
        if input_tokens_with_cache is not None and context_time_with_cache_us > 0:
            report(
                GaugeMetrics.FRONTEND_INPUT_TOKEN_TPS_METRIC,
                input_tokens_with_cache * 1_000_000.0 / context_time_with_cache_us,
                tags,
            )
        if input_tokens is not None and snapshot.context_execute_time_us > 0:
            report(
                GaugeMetrics.FRONTEND_NONCACHE_INPUT_TOKEN_TPS_METRIC,
                input_tokens * 1_000_000.0 / snapshot.context_execute_time_us,
                tags,
            )

        if snapshot.output_len is not None and snapshot.generate_execute_time_us > 0:
            # Match rtp_llm_generate_tps: the first token is produced by the
            # context step; decode execution time covers subsequent tokens.
            decode_tokens = snapshot.generate_token_num
            if decode_tokens > 0:
                output_tps = (
                    decode_tokens * 1_000_000.0 / snapshot.generate_execute_time_us
                )
                report(
                    GaugeMetrics.FRONTEND_OUTPUT_TOKEN_TPS_METRIC,
                    output_tps,
                    tags,
                )
                report(
                    GaugeMetrics.FRONTEND_NONCACHE_OUTPUT_TOKEN_TPS_METRIC,
                    output_tps,
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
    ) -> FrontendRequestMetricState:
        tags = {
            "rank_id": rank_id,
            "server_id": server_id,
            "source": source,
            "streaming": str(streaming).lower(),
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
        self.report(GaugeMetrics.FRONTEND_CONCURRENCY_METRIC, concurrency, tags)

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

    def close(self) -> None:
        self._heartbeat_stop.set()
        if self._heartbeat_thread is not None:
            self._heartbeat_thread.join(timeout=1.0)
