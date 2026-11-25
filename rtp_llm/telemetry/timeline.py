"""Request timelines and paired processing stages shared by serving entry points.

The caller retains ownership of its SERVER span. This module owns event
naming, pre-span timestamps, first-response state and stage failure recording.
It neither consumes streams nor holds a tracing scope across a generator yield.
"""

import functools
import inspect
import time
from contextlib import contextmanager

from rtp_llm.telemetry import request_event


@contextmanager
def stage(name, **attributes):
    """Record one operation; report failure/cancellation without changing it."""
    request_event(name + "_started", **attributes)
    result = {}
    try:
        yield result
    except BaseException as error:
        request_event(name + "_failed", **{"error.type": type(error).__name__})
        raise
    else:
        request_event(name + "_completed", **result)


def traced_stage(name):
    """Instrument a complete sync function or coroutine, never a generator."""

    def decorate(function):
        if inspect.isgeneratorfunction(function) or inspect.isasyncgenfunction(
            function
        ):
            raise TypeError(
                "Use stage() around processing, not across generator yields"
            )
        if inspect.iscoroutinefunction(function):

            @functools.wraps(function)
            async def async_call(*args, **kwargs):
                with stage(name):
                    return await function(*args, **kwargs)

            return async_call

        @functools.wraps(function)
        def call(*args, **kwargs):
            with stage(name):
                return function(*args, **kwargs)

        return call

    return decorate


class RequestTimeline:
    """One request's bounded ingress timestamps and response milestones.

    Entry points may learn their parent after receiving a request body. Keep
    ingress boundaries until attach(); first-response flags belong to this
    instance, so concurrent requests and executor reuse cannot share them.
    Ending the SERVER span remains the caller's responsibility.
    """

    def __init__(self, prefix="rtp_llm"):
        self.started_at = time.time_ns()
        self.started_ns = time.monotonic_ns()
        self._prefix = prefix
        self._query_logged_at = None
        self._frame_received_at = None
        self._state = None
        self._response_seen = False
        self._token_response_seen = False

    def query_logged(self):
        self._query_logged_at = time.time_ns()

    def frame_received(self):
        self._frame_received_at = time.time_ns()

    def attach(self, state):
        if state is None or self._state is not None:
            return
        self._state = state
        for name, timestamp in (
            ("request_received", self.started_at),
            ("query_log_completed", self._query_logged_at),
            ("request_frame_received", self._frame_received_at),
        ):
            if timestamp is not None:
                self._emit(name, timestamp=timestamp)

    def response_yielded(self, *, has_tokens, finished, finish_reason):
        if not self._response_seen:
            self._emit("first_response_yielded")
            self._response_seen = True
        if has_tokens and not self._token_response_seen:
            self._emit("first_token_response_yielded")
            self._token_response_seen = True
        if finished:
            self._emit("finished_response_yielded", {"finish_reason": finish_reason})

    def finished(self, status):
        self._emit("request_finished", {"status": status})

    def _emit(self, name, attributes=None, timestamp=None):
        try:
            if self._state is not None:
                self._state.add_event(
                    self._prefix + "." + name, attributes, timestamp=timestamp
                )
        except Exception:
            pass
