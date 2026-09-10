"""Process-local HTTP admission and lifetime observations (schema v1).

The ASGI call owns the token. Generators and routing tasks may update it, but
only the HTTP owner closes it. No concurrency-controller resources live here.
"""

import logging
import threading
import time
from contextlib import contextmanager
from contextvars import ContextVar
from types import MappingProxyType
from typing import Optional

from rtp_llm.metrics import kmonitor
from rtp_llm.metrics.kmonitor_metric_reporter import (
    AccMetrics,
    GaugeMetrics,
    frontend_metric_identity,
)

CURRENT_FRONTEND_REQUEST: ContextVar[Optional["FrontendRequestToken"]] = ContextVar(
    "frontend_request_metrics", default=None
)
ROUTES = {
    "/": "inference",
    "/chat/completions": "openai",
    "/v1/chat/completions": "openai",
    "/batch_infer": "batch",
    "/v1/batch/chat/completions": "batch",
}
STAGES = ("prepare", "route", "backend")
_last_report_error = 0.0


def _report(reporter, metric, value, tags):
    global _last_report_error
    try:
        reporter.report(metric, value, tags)
    except Exception:
        now = time.monotonic()
        if now - _last_report_error >= 60:
            _last_report_error = now
            logging.exception("Failed to report frontend request metrics")


def admit_current_request():
    token = CURRENT_FRONTEND_REQUEST.get()
    if token is not None:
        token.outcome_once("admitted")


def reject_current_request(event):
    token = CURRENT_FRONTEND_REQUEST.get()
    if token is not None:
        token.outcome_once(event)


def report_concurrency_rejection(error, rank_id, server_id):
    """Both catch layers can see the same exception, including non-target routes."""
    if getattr(error, "_frontend_conflict_reported", False):
        return
    error._frontend_conflict_reported = True
    token = CURRENT_FRONTEND_REQUEST.get()
    identity = (
        token.registry.identity
        if token is not None
        else frontend_metric_identity(rank_id, server_id)
    )
    _report(kmonitor, AccMetrics.CONFLICT_QPS_METRIC, 1, dict(identity))
    reject_current_request("reject_concurrency")


class FrontendRequestRegistry:
    def __init__(self, rank_id, server_id, is_embedding=False, reporter=kmonitor):
        self.identity = MappingProxyType(frontend_metric_identity(rank_id, server_id))
        self.routes = dict(ROUTES)
        if is_embedding:
            self.routes.pop("/")
        self.reporter = reporter
        self.lock = threading.Lock()
        self.counts = {
            (route, stage): 0 for route in set(self.routes.values()) for stage in STAGES
        }
        self._stop = threading.Event()
        self._thread = None

    def event(self, route, event):
        _report(
            self.reporter,
            AccMetrics.FRONTEND_ADMISSION_QPS,
            1,
            {**self.identity, "admission_probe": "v1", "route": route, "event": event},
        )

    def snapshot(self):
        with self.lock:
            return dict(self.counts)

    def sample(self):
        for (route, stage), value in self.snapshot().items():
            _report(
                self.reporter,
                GaugeMetrics.FRONTEND_INFLIGHT,
                value,
                {
                    **self.identity,
                    "admission_probe": "v1",
                    "route": route,
                    "stage": stage,
                },
            )

    def start(self):
        if self._thread is not None:
            return
        self._stop.clear()

        def run():
            while not self._stop.wait(1):
                self.sample()

        self._thread = threading.Thread(
            target=run, name="frontend-metrics", daemon=True
        )
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2)
        self.sample()


class FrontendRequestToken:
    def __init__(self, registry, route):
        self.registry = registry
        self.route = route
        self.outcome = None
        self.stage = "prepare"
        self.closed = False
        self._route_scopes = 0
        self._backend_started = False
        with registry.lock:
            registry.counts[route, self.stage] += 1
        registry.event(route, "received")

    def outcome_once(self, event):
        with self.registry.lock:
            if self.closed or self.outcome is not None:
                return
            self.outcome = event
        self.registry.event(self.route, event)

    def _update_stage_locked(self):
        stage = (
            "route"
            if self._route_scopes
            else "backend" if self._backend_started else "prepare"
        )
        if stage != self.stage:
            self.registry.counts[self.route, self.stage] -= 1
            self.registry.counts[self.route, stage] += 1
            self.stage = stage

    def route_scope_delta(self, delta):
        with self.registry.lock:
            if not self.closed:
                self._route_scopes += delta
                self._update_stage_locked()

    def begin_backend(self):
        with self.registry.lock:
            if not self.closed:
                self._backend_started = True
                self._update_stage_locked()

    def close_once(self):
        with self.registry.lock:
            if self.closed:
                return
            missing_outcome = self.outcome is None
            if missing_outcome:
                self.outcome = "reject_other"
            self.closed = True
            self.registry.counts[self.route, self.stage] -= 1
        if missing_outcome:
            self.registry.event(self.route, "reject_other")


@contextmanager
def frontend_route_scope(token):
    if token is not None:
        token.route_scope_delta(1)
    try:
        yield
    finally:
        if token is not None:
            token.route_scope_delta(-1)


class FrontendRequestMetricsMiddleware:
    def __init__(self, app, registry):
        self.app = app
        self.registry = registry

    async def __call__(self, scope, receive, send):
        route = self.registry.routes.get(scope.get("path"))
        if scope["type"] != "http" or scope.get("method") != "POST" or route is None:
            return await self.app(scope, receive, send)
        token = FrontendRequestToken(self.registry, route)
        scope.setdefault("state", {})["frontend_request_token"] = token
        context = CURRENT_FRONTEND_REQUEST.set(token)
        trailers = False

        async def observed_send(message):
            nonlocal trailers
            if message["type"] == "http.response.start":
                trailers = message.get("trailers", False)
            await send(message)
            if (
                message["type"] == "http.response.body"
                and not message.get("more_body", False)
                and not trailers
            ) or (
                message["type"] == "http.response.trailers"
                and not message.get("more_trailers", False)
            ):
                token.close_once()

        try:
            await self.app(scope, receive, observed_send)
        finally:
            token.close_once()
            CURRENT_FRONTEND_REQUEST.reset(context)
