"""Read the Native query's input without consuming its asynchronous result.

A missing local prefix stays unknown: the planned request cursor and the
predicted total cache hit are different quantities. Capture runs on the
EngineCore scheduler thread and never consumes the asynchronous query result.
"""

from concurrent.futures import Future
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class NativeQueryInput:
    generation: int
    source: str
    local_prefix_tokens: Optional[int]

    def expected_external_tokens(self, predicted_total_hit_tokens):
        if (
            type(predicted_total_hit_tokens) is not int
            or predicted_total_hit_tokens < 0
        ):
            raise ValueError("predicted total hit tokens must be a nonnegative integer")
        if self.local_prefix_tokens is None:
            return None
        return max(0, predicted_total_hit_tokens - self.local_prefix_tokens)


def capture_native_query_input(state):
    """Call on the EngineCore scheduler thread while taking its workload snapshot."""
    if state is None:
        return NativeQueryInput(0, "not_observed", None)
    generation = state.match_generation
    if type(generation) is not int or generation < 0:
        raise ValueError("invalid Native match generation")

    # Native clears match_signature when it consumes the Future. Until then,
    # the signature holds the original local-prefix argument, even when the
    # request's planned cursor has advanced. No Future method is called here.
    if state.match_future is not None:
        signature = state.match_signature
        source = "active_query"
    elif state.resolved_match is not None:
        signature = state.resolved_match_signature
        source = "resolved_query"
    else:
        return NativeQueryInput(generation, "not_observed", None)
    if not isinstance(signature, tuple) or len(signature) != 2:
        raise ValueError("invalid Native match signature")
    prefix = signature[0]
    if type(prefix) is not int or prefix < -1:
        raise ValueError("invalid Native query local prefix")
    if prefix == -1:
        # Arrival prequery precedes Native's local-prefix selection. Its -1
        # marker is not a zero-token HBM miss or a reusable suffix argument.
        return NativeQueryInput(generation, "arrival_prequery", None)
    return NativeQueryInput(generation, source, prefix)


@dataclass(frozen=True)
class NativeQueryOutcome:
    status: str
    completed_prefix_tokens: Optional[int] = None
    exception_type: Optional[str] = None


def completed_native_query_outcome(state, query_input, *, future_ready):
    """Observe cached completion without re-raising a stored Native exception.

    Future.result raises the same stored exception object on every read and
    appends observer frames to its traceback. Inspect exception() first so an
    unsuccessful observation cannot change Native's later error handling.
    """
    if state is None or state.match_future is None:
        return NativeQueryOutcome("unobserved")
    if future_ready is not True:
        return NativeQueryOutcome("pending")
    future = state.match_future
    if type(future) is not Future:
        return NativeQueryOutcome("unsupported_future")
    if Future.cancelled(future):
        return NativeQueryOutcome("cancelled")
    try:
        error = Future.exception(future, timeout=0)
        if error is not None:
            return NativeQueryOutcome("exception", exception_type=type(error).__name__)
        result = Future.result(future, timeout=0)
    except Exception:
        # A stale readiness hint or an unsupported state is not a cache miss.
        return NativeQueryOutcome("unavailable")
    if (
        getattr(result, "generation", None) != query_input.generation
        or state.match_generation != query_input.generation
        or getattr(result, "match_signature", None) is not state.match_signature
    ):
        return NativeQueryOutcome("stale_result")
    if getattr(result, "fetch_failed", None) is True:
        return NativeQueryOutcome("fetch_failed")
    if getattr(result, "fetch_failed", None) is not False:
        return NativeQueryOutcome("invalid_result")
    if query_input.source != "active_query" or query_input.local_prefix_tokens is None:
        return NativeQueryOutcome("prefix_unobserved")
    hit = getattr(result, "hit_length", None)
    if type(hit) is not int or hit < 0:
        return NativeQueryOutcome("invalid_result")
    return NativeQueryOutcome("success", query_input.local_prefix_tokens + hit)


def completed_native_query_prefix(state, query_input, *, future_ready):
    """Retain the existing scalar API while reporting failures separately."""
    return completed_native_query_outcome(
        state, query_input, future_ready=future_ready
    ).completed_prefix_tokens


QUERY_INPUT_SOURCES = {
    "not_observed": 0,
    "active_query": 1,
    "arrival_prequery": 2,
    "resolved_query": 3,
}


def make_native_cache_query_report(request_id, state, received, failed, query_type):
    """Convert one scheduler-thread observation without another Native lookup."""
    done = state.get("native_v6d_match_future_done")
    inflight = state.get("native_v6d_load_inflight")
    future_state = 0 if done is None else 2 if done else 1
    if done is True and state.get("native_v6d_match_completion_status") in (
        "exception", "cancelled", "fetch_failed"
    ):
        future_state = 3
    load_state = (
        3
        if state.get("native_v6d_load_failed")
        else 2 if inflight else 1 if inflight is not None else 0
    )
    receive_state = (
        3
        if failed
        else 2 if received else 1 if failed is not None and received is not None else 0
    )
    local_prefix = state.get("native_v6d_match_local_prefix_tokens")
    return query_type(
        request_id=request_id,
        generation=state.get("native_v6d_match_generation") or 0,
        local_prefix_tokens=-1 if local_prefix is None else local_prefix,
        input_source=state.get("native_v6d_match_input_source") or 0,
        future_state=future_state,
        load_state=load_state,
        receive_state=receive_state,
    )
