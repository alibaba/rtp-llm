"""Opt-in, trace-independent model-server latency histograms."""

import asyncio
import logging
import os
import time
from contextvars import ContextVar
from pathlib import Path

CURRENT_SERVING_METRICS = ContextVar("rtp_llm_serving_metrics", default=None)

# OpenTelemetry GenAI server metric recommendations (seconds).
DURATION_BUCKETS = tuple(0.01 * 2**i for i in range(14))
TTFT_BUCKETS = (
    0.001,
    0.005,
    0.01,
    0.02,
    0.04,
    0.06,
    0.08,
    0.1,
    0.25,
    0.5,
    0.75,
    1.0,
    2.5,
    5.0,
    7.5,
    10.0,
)
TOKEN_BUCKETS = (
    0.01,
    0.025,
    0.05,
    0.075,
    0.1,
    0.15,
    0.2,
    0.3,
    0.4,
    0.5,
    0.75,
    1.0,
    2.5,
)


def create_serving_metrics():
    if os.environ.get("RTP_LLM_METRICS_ENABLED", "0") != "1":
        return None
    # Frontends share a listening port via SO_REUSEPORT. A process-local
    # registry would return a different subset of requests on each scrape.
    directory = os.environ.get("PROMETHEUS_MULTIPROC_DIR")
    if not directory or not Path(directory).is_dir():
        raise ValueError(
            "RTP_LLM_METRICS_ENABLED=1 requires an existing "
            "PROMETHEUS_MULTIPROC_DIR, set before starting the server"
        )
    from prometheus_client import values

    if not values.ValueClass._multiprocess:
        raise ValueError(
            "Set PROMETHEUS_MULTIPROC_DIR before importing prometheus_client"
        )
    return ServingMetrics(multiprocess=True)


class ServingMetrics:
    def __init__(self, multiprocess=False):
        from prometheus_client import CollectorRegistry, Histogram

        self.multiprocess = multiprocess
        self.registry = CollectorRegistry()
        labels = ("gen_ai_operation_name", "gen_ai_provider_name")
        self.duration = Histogram(
            "rtp_llm_server_request_duration_seconds",
            "Frontend request duration until the last response byte, in seconds.",
            labels + ("error_type",),
            buckets=DURATION_BUCKETS,
            registry=self.registry,
        )
        self.ttft = Histogram(
            "rtp_llm_server_time_to_first_token_seconds",
            "Frontend time to first visible output token for successful streams.",
            labels,
            buckets=TTFT_BUCKETS,
            registry=self.registry,
        )
        self.tpot = Histogram(
            "rtp_llm_server_time_per_output_token_seconds",
            "Per successful request mean time per output token after the first.",
            labels,
            buckets=TOKEN_BUCKETS,
            registry=self.registry,
        )

    def render(self):
        from prometheus_client import CollectorRegistry, generate_latest, multiprocess

        registry = self.registry
        if self.multiprocess:
            registry = CollectorRegistry()
            multiprocess.MultiProcessCollector(registry)
        return generate_latest(registry)


class RequestMetrics:
    def __init__(self, metrics, clock=time.perf_counter):
        self.metrics = metrics
        self.clock = clock
        self.start = clock()
        self.first_token = None
        self.last_token = None
        self.output_tokens = 0
        self.exact_tokens = True
        self.token_timings_enabled = False
        self.error = ""
        self.finished = False

    def observe_tokens(self, visible, cumulative_tokens):
        if not self.token_timings_enabled or not visible:
            return
        now = self.clock()
        if self.first_token is None:
            self.first_token = now
            # If the first chunk contains multiple tokens, its timestamp
            # cannot locate the first token within that chunk for mean TPOT.
            if cumulative_tokens != 1:
                self.exact_tokens = False
        self.last_token = now
        if (
            isinstance(cumulative_tokens, bool)
            or not isinstance(cumulative_tokens, int)
            or cumulative_tokens <= self.output_tokens
        ):
            self.exact_tokens = False
        else:
            self.output_tokens = cumulative_tokens

    def finish(self, end, error=""):
        if self.finished:
            return
        self.finished = True
        error = error or self.error
        self.metrics.duration.labels("chat", "rtp_llm", error).observe(
            max(0.0, end - self.start)
        )
        if error or self.first_token is None:
            return
        self.metrics.ttft.labels("chat", "rtp_llm").observe(
            max(0.0, self.first_token - self.start)
        )
        if (
            self.exact_tokens
            and self.output_tokens > 1
            and self.last_token > self.first_token
        ):
            self.metrics.tpot.labels("chat", "rtp_llm").observe(
                (self.last_token - self.first_token) / (self.output_tokens - 1)
            )


class ServingMetricsMiddleware:
    """Track actual ASGI body delivery, excluding post-response cleanup."""

    def __init__(self, app, metrics, clock=time.perf_counter):
        self.app = app
        self.metrics = metrics
        self.clock = clock

    async def __call__(self, scope, receive, send):
        if (
            scope["type"] != "http"
            or scope["method"] != "POST"
            or scope["path"] not in ("/chat/completions", "/v1/chat/completions")
        ):
            return await self.app(scope, receive, send)
        state = RequestMetrics(self.metrics, self.clock)
        token = CURRENT_SERVING_METRICS.set(state)
        status = 500
        last_body = None
        complete = False
        error = ""

        async def send_with_metrics(message):
            nonlocal status, last_body, complete
            await send(message)
            if message["type"] == "http.response.start":
                status = message["status"]
            elif message["type"] == "http.response.body":
                if message.get("body") or last_body is None:
                    last_body = self.clock()
                if not message.get("more_body", False):
                    complete = True

        try:
            await self.app(scope, receive, send_with_metrics)
        except (asyncio.CancelledError, GeneratorExit):
            error = "cancelled"
            raise
        except BaseException:
            error = "internal_error"
            raise
        finally:
            try:
                if not error and not complete:
                    error = "cancelled"
                if not error and status >= 400:
                    error = "client_error" if status < 500 else "server_error"
                state.finish(
                    last_body if complete and last_body is not None else self.clock(),
                    error,
                )
            except Exception:
                logging.exception("Failed to record serving latency metrics")
            finally:
                CURRENT_SERVING_METRICS.reset(token)
