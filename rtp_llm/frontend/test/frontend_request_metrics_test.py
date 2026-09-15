import asyncio
import threading
import unittest
from unittest.mock import Mock, patch

from rtp_llm.frontend.frontend_request_metrics import (
    CURRENT_FRONTEND_REQUEST,
    FrontendRequestMetricsMiddleware,
    FrontendRequestRegistry,
    FrontendRequestToken,
    admit_current_request,
    frontend_route_scope,
    get_frontend_request_registry,
    reject_current_request,
    report_concurrency_rejection,
)
from rtp_llm.metrics.kmonitor_metric_reporter import AccMetrics, GaugeMetrics
from rtp_llm.utils.concurrency_controller import ConcurrencyException


class FrontendRequestMetricsTest(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.reporter = Mock()
        self.registry = FrontendRequestRegistry(2, 3, reporter=self.reporter)

    def events(self):
        return [
            call.args[2]["event"]
            for call in self.reporter.report.call_args_list
            if call.args[0] == AccMetrics.FRONTEND_ADMISSION_QPS
        ]

    async def drive(self, app, path="/", send=None, method="POST"):
        async def receive():
            return {"type": "http.request", "body": b"{}", "more_body": False}

        async def discard(message):
            pass

        scope = {"type": "http", "method": method, "path": path}
        middleware = FrontendRequestMetricsMiddleware(app, self.registry)
        await middleware(scope, receive, send or discard)
        self.assertIsNone(CURRENT_FRONTEND_REQUEST.get())
        self.assertEqual(sum(self.registry.snapshot().values()), 0)
        return scope

    async def test_streaming_lives_until_last_send_completes(self):
        tokens = []

        async def app(scope, receive, send):
            token = CURRENT_FRONTEND_REQUEST.get()
            tokens.append(token)
            admit_current_request()
            with frontend_route_scope(token):
                self.assertEqual(
                    self.registry.snapshot()["http", "inference", "route"], 1
                )
                token.begin_backend()
            await send({"type": "http.response.start", "status": 200})
            await send({"type": "http.response.body", "body": b"a", "more_body": True})
            self.assertFalse(token.closed)
            await send({"type": "http.response.body", "body": b"b"})
            self.assertTrue(token.closed)
            token.close_once()

        async def send(message):
            self.assertEqual(
                self.registry.snapshot()["http", "inference", "backend"], 1
            )
            self.assertFalse(tokens[0].closed)

        await self.drive(app, send=send)
        self.assertEqual(self.events(), ["received", "admitted"])
        for call in self.reporter.report.call_args_list:
            self.assertEqual(call.args[2]["protocol"], "http")
            self.assertEqual(call.args[2]["rank_id"], "2")
            self.assertEqual(call.args[2]["server_id"], "3")

    async def test_trailers_hold_lifetime(self):
        async def app(scope, receive, send):
            token = CURRENT_FRONTEND_REQUEST.get()
            admit_current_request()
            await send({"type": "http.response.start", "status": 200, "trailers": True})
            await send({"type": "http.response.body", "body": b""})
            self.assertFalse(token.closed)
            await send({"type": "http.response.trailers", "headers": []})
            self.assertTrue(token.closed)

        await self.drive(app)

    async def test_cancel_send_failure_and_body_never_started_close(self):
        for failure in ("cancel_before", "cancel_after", "send", "no_body"):
            self.reporter.reset_mock()

            async def app(scope, receive, send):
                if failure == "cancel_before":
                    raise asyncio.CancelledError()
                admit_current_request()
                if failure == "cancel_after":
                    raise asyncio.CancelledError()
                if failure == "send":
                    await send({"type": "http.response.body", "body": b""})

            async def send(message):
                raise OSError("transport failed")

            try:
                await self.drive(app, send=send)
            except (asyncio.CancelledError, OSError):
                pass
            self.assertEqual(sum(self.registry.snapshot().values()), 0)
            self.assertIsNone(CURRENT_FRONTEND_REQUEST.get())
            outcome = "reject_other" if failure == "cancel_before" else "admitted"
            self.assertEqual(self.events(), ["received", outcome])

    async def test_admitted_backend_429_and_semantic_failure_are_not_local_rejections(
        self,
    ):
        async def app(scope, receive, send):
            admit_current_request()
            reject_current_request("reject_invalid")
            await send({"type": "http.response.start", "status": 429})
            await send({"type": "http.response.body", "body": b""})

        await self.drive(app)
        self.assertEqual(self.events(), ["received", "admitted"])

    async def test_concurrency_helper_is_once_for_target_and_non_target_routes(self):
        for targeted in (False, True):
            self.reporter.reset_mock()
            error = ConcurrencyException("full")

            async def app(scope, receive, send):
                report_concurrency_rejection(error, 2, 3)
                report_concurrency_rejection(error, 2, 3)

            with patch(
                "rtp_llm.frontend.frontend_request_metrics.kmonitor", self.reporter
            ):
                await self.drive(app, path="/" if targeted else "/v1/embeddings")
            conflicts = [
                c
                for c in self.reporter.report.call_args_list
                if c.args[0] == AccMetrics.CONFLICT_QPS_METRIC
            ]
            self.assertEqual(len(conflicts), 1)
            self.assertEqual(conflicts[0].args[2], {"rank_id": "2", "server_id": "3"})
            self.assertEqual(
                self.events(), ["received", "reject_concurrency"] if targeted else []
            )

    async def test_aliases_batch_and_whitelist(self):
        async def app(scope, receive, send):
            admit_current_request()

        for path in (
            "/chat/completions",
            "/v1/chat/completions",
            "/batch_infer",
            "/v1/batch/chat/completions",
        ):
            scope = await self.drive(app, path=path)
            route = "batch" if "batch" in path else "openai"
            self.assertEqual(scope["state"]["frontend_request_token"].route, route)
        self.assertEqual(self.events().count("received"), 4)
        self.reporter.reset_mock()
        for path in (
            "/health",
            "/chat/render",
            "/v1/embeddings",
            "/unknown",
            "/chat/completions/",
        ):
            await self.drive(app, path=path)
        await self.drive(app, method="OPTIONS")
        self.assertEqual(self.events(), [])
        self.registry = FrontendRequestRegistry(
            2, 3, is_embedding=True, reporter=self.reporter
        )
        await self.drive(app)
        self.assertEqual(self.events(), [])

    async def test_parallel_route_scopes_and_retry_have_one_http_owner(self):
        token = FrontendRequestToken(self.registry, "batch")
        token.outcome_once("admitted")
        with frontend_route_scope(token):
            with frontend_route_scope(token):
                token.begin_backend()
                self.assertEqual(sum(self.registry.snapshot().values()), 1)
            self.assertEqual(self.registry.snapshot()["http", "batch", "route"], 1)
        self.assertEqual(self.registry.snapshot()["http", "batch", "backend"], 1)
        with frontend_route_scope(token):
            self.assertEqual(self.registry.snapshot()["http", "batch", "route"], 1)
            token.close_once()
        token.begin_backend()
        self.assertEqual(sum(self.registry.snapshot().values()), 0)
        self.assertEqual(self.events(), ["received", "admitted"])

    async def test_sampler_reports_only_positive_values_outside_event_loop(self):
        self.registry = get_frontend_request_registry(2, 3)
        self.assertIs(self.registry, get_frontend_request_registry(2, 3))
        self.registry.reporter = self.reporter
        called = threading.Event()
        thread_ids = []

        def report(metric, value, tags):
            if metric == GaugeMetrics.FRONTEND_INFLIGHT:
                thread_ids.append(threading.get_ident())
                self.assertEqual(value, 1)
                called.set()

        self.reporter.report.side_effect = report
        http = FrontendRequestToken(self.registry, "inference")
        token = FrontendRequestToken(self.registry, "inference", protocol="grpc")
        self.registry.start()
        thread = self.registry._thread
        self.registry.start()
        self.assertIs(thread, self.registry._thread)
        try:
            # Intentionally block this loop; sampler must still run.
            self.assertTrue(called.wait(timeout=3))
            self.assertNotEqual(thread_ids[0], threading.get_ident())
        finally:
            http.close_once()
            token.close_once()
            self.registry.stop()
            self.assertTrue(thread.is_alive())
            self.registry.stop()
        self.assertFalse(thread.is_alive())
        gauges = [
            c
            for c in self.reporter.report.call_args_list
            if c.args[0] == GaugeMetrics.FRONTEND_INFLIGHT
        ]
        self.assertGreaterEqual(len(gauges), 1)
        self.reporter.reset_mock()
        self.registry.sample()
        self.assertEqual(sum(self.registry.snapshot().values()), 0)
        self.reporter.report.assert_not_called()
        self.assertEqual({c.args[2]["protocol"] for c in gauges}, {"http", "grpc"})

    async def test_report_failures_do_not_break_request_or_accounting(self):
        self.reporter.report.side_effect = RuntimeError("sink unavailable")

        async def app(scope, receive, send):
            admit_current_request()
            CURRENT_FRONTEND_REQUEST.get().begin_backend()
            await send({"type": "http.response.body", "body": b""})

        await self.drive(app)
        self.registry.sample()


if __name__ == "__main__":
    unittest.main()
