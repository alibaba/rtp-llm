import asyncio
import threading
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Dict, Iterator, Optional

from rtp_llm.metrics.kmonitor_metric_reporter import AccMetrics, GaugeMetrics

ADMISSION_PROBE = "v1"
STAGES = ("prepare", "route", "backend")
ROUTE_NORMAL = "normal"
ROUTE_OPENAI_CHAT = "openai_chat"
ROUTE_BATCH = "batch"
ROUTE_EMBEDDING = "embedding"
ROUTE_UNKNOWN = "unknown"
ROUTES = (
    ROUTE_NORMAL,
    ROUTE_OPENAI_CHAT,
    ROUTE_BATCH,
    ROUTE_EMBEDDING,
)

_ROUTE_BY_PATH = {
    "/chat/completions": ROUTE_OPENAI_CHAT,
    "/v1/chat/completions": ROUTE_OPENAI_CHAT,
    "/v1/batch/chat/completions": ROUTE_BATCH,
    "/batch_infer": ROUTE_BATCH,
    "/v1/embeddings/similarity": ROUTE_EMBEDDING,
    "/v1/reranker": ROUTE_EMBEDDING,
    "/v1/classifier": ROUTE_EMBEDDING,
    "/v1/embeddings": ROUTE_EMBEDDING,
    "/v1/embeddings/dense": ROUTE_EMBEDDING,
    "/v1/embeddings/sparse": ROUTE_EMBEDDING,
    "/v1/embeddings/colbert": ROUTE_EMBEDDING,
}

_current_token: ContextVar[Optional["FrontendRequestToken"]] = ContextVar(
    "frontend_request_metrics_token", default=None
)


def get_current_frontend_request_token() -> Optional["FrontendRequestToken"]:
    return _current_token.get()


@contextmanager
def bind_frontend_request_token(
    token: "FrontendRequestToken",
) -> Iterator["FrontendRequestToken"]:
    context_token = _current_token.set(token)
    try:
        yield token
    finally:
        _current_token.reset(context_token)


def current_frontend_route(default: str = ROUTE_UNKNOWN) -> str:
    token = get_current_frontend_request_token()
    return token.route if token is not None else default


def frontend_metric_tags(
    rank_id: Any, server_id: Any, route: Optional[str] = None
) -> Dict[str, str]:
    return {
        "route": route or current_frontend_route(),
        "rank_id": str(rank_id),
        "server_id": str(server_id),
    }


def mark_master_route_error_reported(error: BaseException) -> None:
    setattr(error, "_frontend_master_route_error_reported", True)


def master_route_error_reported(error: BaseException) -> bool:
    return bool(getattr(error, "_frontend_master_route_error_reported", False))


class FrontendRequestToken:
    def __init__(self, tracker: "FrontendRequestMetrics", route: str):
        self._tracker = tracker
        self.route = route
        self.stage = "prepare"
        self.outcome: Optional[str] = None
        self.closed = False

    def admit(self) -> None:
        self._set_outcome("admitted")

    def reject_concurrency(self) -> None:
        self._set_outcome("reject_concurrency")

    def reject_unavailable(self) -> None:
        self._set_outcome("reject_unavailable")

    def reject_invalid(self) -> None:
        self._set_outcome("reject_invalid")

    def reject_other(self) -> None:
        self._set_outcome("reject_other")

    def enter_route(self) -> None:
        self._set_stage("route")

    def enter_backend(self) -> None:
        self._set_stage("backend")

    def close(self) -> None:
        self._tracker._close(self)

    def _set_outcome(self, outcome: str) -> None:
        self._tracker._set_outcome(self, outcome)

    def _set_stage(self, stage: str) -> None:
        self._tracker._set_stage(self, stage)


class FrontendRequestMetrics:
    def __init__(self, reporter: Any, rank_id: Any, server_id: Any):
        self._reporter = reporter
        self.rank_id = str(rank_id)
        self.server_id = str(server_id)
        self._lock = threading.RLock()
        self._inflight = {
            route: {stage: 0 for stage in STAGES} for route in ROUTES
        }
        self._report_task: Optional[asyncio.Task] = None
        self._stop_event: Optional[asyncio.Event] = None

    def begin(self, route: str) -> FrontendRequestToken:
        token = FrontendRequestToken(self, route)
        with self._lock:
            self._inflight[route]["prepare"] += 1
        self._report_admission(route, "received")
        return token

    def snapshot(self) -> Dict[str, Dict[str, int]]:
        with self._lock:
            return {
                route: dict(stage_counts)
                for route, stage_counts in self._inflight.items()
            }

    def report_inflight(self) -> None:
        snapshot = self.snapshot()
        for route in ROUTES:
            for stage in STAGES:
                tags = self._tags(route)
                tags["stage"] = stage
                self._reporter.report(
                    GaugeMetrics.FRONTEND_INFLIGHT_METRIC,
                    snapshot[route][stage],
                    tags,
                )

    def start(self) -> None:
        if self._report_task is not None and not self._report_task.done():
            return
        self._stop_event = asyncio.Event()
        self._report_task = asyncio.create_task(self._report_loop())

    async def stop(self) -> None:
        task = self._report_task
        stop_event = self._stop_event
        if task is None or stop_event is None:
            return
        stop_event.set()
        await task
        self._report_task = None
        self._stop_event = None

    async def _report_loop(self) -> None:
        assert self._stop_event is not None
        self.report_inflight()
        while True:
            try:
                await asyncio.wait_for(self._stop_event.wait(), timeout=1.0)
                return
            except asyncio.TimeoutError:
                self.report_inflight()

    def _tags(self, route: str) -> Dict[str, str]:
        tags = frontend_metric_tags(self.rank_id, self.server_id, route)
        tags["admission_probe"] = ADMISSION_PROBE
        return tags

    def _report_admission(self, route: str, event: str) -> None:
        tags = self._tags(route)
        tags["event"] = event
        self._reporter.report(AccMetrics.FRONTEND_ADMISSION_QPS_METRIC, 1, tags)

    def _set_outcome(self, token: FrontendRequestToken, outcome: str) -> None:
        with self._lock:
            if token.closed or token.outcome is not None:
                return
            token.outcome = outcome
        self._report_admission(token.route, outcome)

    def _set_stage(self, token: FrontendRequestToken, stage: str) -> None:
        if stage not in STAGES:
            raise ValueError(f"invalid frontend request stage: {stage}")
        with self._lock:
            if token.closed or token.stage == stage:
                return
            self._inflight[token.route][token.stage] -= 1
            self._inflight[token.route][stage] += 1
            token.stage = stage

    def _close(self, token: FrontendRequestToken) -> None:
        with self._lock:
            if token.closed:
                return
            token.closed = True
            self._inflight[token.route][token.stage] -= 1


class FrontendRequestMetricsMiddleware:
    def __init__(
        self,
        app,
        request_metrics: FrontendRequestMetrics,
        root_is_embedding: bool = False,
    ):
        self.app = app
        self.request_metrics = request_metrics
        self.root_is_embedding = root_is_embedding

    def _route(self, scope) -> Optional[str]:
        if scope.get("type") != "http" or scope.get("method") != "POST":
            return None
        path = scope.get("path", "")
        if path == "/":
            return ROUTE_EMBEDDING if self.root_is_embedding else ROUTE_NORMAL
        return _ROUTE_BY_PATH.get(path)

    async def __call__(self, scope, receive, send) -> None:
        route = self._route(scope)
        if route is None:
            await self.app(scope, receive, send)
            return

        token = self.request_metrics.begin(route)
        status_code: Optional[int] = None
        context_token = _current_token.set(token)

        async def tracked_receive():
            message = await receive()
            if message.get("type") == "http.disconnect":
                if token.outcome is None:
                    token.reject_other()
                token.close()
            return message

        async def tracked_send(message) -> None:
            nonlocal status_code
            if message.get("type") == "http.response.start":
                status_code = message.get("status")
            try:
                await send(message)
            finally:
                if (
                    message.get("type") == "http.response.body"
                    and not message.get("more_body", False)
                ):
                    if token.outcome is None:
                        if status_code in (400, 422):
                            token.reject_invalid()
                        else:
                            token.reject_other()
                    token.close()

        try:
            await self.app(scope, tracked_receive, tracked_send)
        except BaseException:
            if token.outcome is None:
                token.reject_other()
            token.close()
            raise
        finally:
            _current_token.reset(context_token)
