import unittest

from rtp_llm.metrics.frontend_request_metrics import (
    ROUTE_NORMAL,
    ROUTES,
    STAGES,
    FrontendRequestMetrics,
    FrontendRequestMetricsMiddleware,
    get_current_frontend_request_token,
)
from rtp_llm.metrics.kmonitor_metric_reporter import AccMetrics, GaugeMetrics


class _Reporter:
    def __init__(self):
        self.calls = []

    def report(self, metric, value=1, tags=None):
        self.calls.append((metric, value, dict(tags or {})))


async def _run_middleware(app, reporter):
    metrics = FrontendRequestMetrics(reporter, rank_id=2, server_id=3)
    middleware = FrontendRequestMetricsMiddleware(app, metrics)
    sent = []

    async def receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(message):
        sent.append(message)

    await middleware(
        {"type": "http", "method": "POST", "path": "/"}, receive, send
    )
    return metrics, sent


class FrontendRequestMetricsTest(unittest.IsolatedAsyncioTestCase):
    async def test_tracker_balances_stages_and_reports_one_outcome(self):
        reporter = _Reporter()
        metrics = FrontendRequestMetrics(reporter, rank_id=2, server_id=3)
        token = metrics.begin(ROUTE_NORMAL)

        self.assertEqual(metrics.snapshot()[ROUTE_NORMAL]["prepare"], 1)
        token.enter_route()
        token.enter_route()
        self.assertEqual(metrics.snapshot()[ROUTE_NORMAL]["route"], 1)
        token.enter_backend()
        token.admit()
        token.reject_other()
        token.close()
        token.close()

        self.assertEqual(metrics.snapshot()[ROUTE_NORMAL], dict.fromkeys(STAGES, 0))
        events = [
            tags["event"]
            for metric, _, tags in reporter.calls
            if metric == AccMetrics.FRONTEND_ADMISSION_QPS_METRIC
        ]
        self.assertEqual(events, ["received", "admitted"])

    async def test_inflight_report_includes_zero_for_every_route_and_stage(self):
        reporter = _Reporter()
        metrics = FrontendRequestMetrics(reporter, rank_id=2, server_id=3)

        metrics.report_inflight()

        gauges = [
            (value, tags)
            for metric, value, tags in reporter.calls
            if metric == GaugeMetrics.FRONTEND_INFLIGHT_METRIC
        ]
        self.assertEqual(len(gauges), len(ROUTES) * len(STAGES))
        self.assertTrue(all(value == 0 for value, _ in gauges))
        self.assertEqual(
            {(tags["route"], tags["stage"]) for _, tags in gauges},
            {(route, stage) for route in ROUTES for stage in STAGES},
        )

    async def test_normal_request_is_received_before_app_and_admitted(self):
        reporter = _Reporter()

        async def app(scope, receive, send):
            token = get_current_frontend_request_token()
            self.assertIsNotNone(token)
            token.admit()
            await send({"type": "http.response.start", "status": 200})
            await send({"type": "http.response.body", "body": b"ok"})

        metrics, sent = await _run_middleware(app, reporter)

        self.assertEqual(sent[-1]["body"], b"ok")
        self.assertEqual(metrics.snapshot()[ROUTE_NORMAL], dict.fromkeys(STAGES, 0))
        events = [tags["event"] for _, _, tags in reporter.calls]
        self.assertEqual(events, ["received", "admitted"])
        self.assertEqual(
            reporter.calls[0][2],
            {
                "admission_probe": "v1",
                "route": ROUTE_NORMAL,
                "rank_id": "2",
                "server_id": "3",
                "event": "received",
            },
        )

    async def test_validation_and_admission_reject_have_one_terminal_event(self):
        invalid_reporter = _Reporter()

        async def invalid_app(scope, receive, send):
            await send({"type": "http.response.start", "status": 422})
            await send({"type": "http.response.body", "body": b"invalid"})

        await _run_middleware(invalid_app, invalid_reporter)
        self.assertEqual(
            [tags["event"] for _, _, tags in invalid_reporter.calls],
            ["received", "reject_invalid"],
        )

        reject_reporter = _Reporter()

        async def reject_app(scope, receive, send):
            token = get_current_frontend_request_token()
            token.reject_concurrency()
            await send({"type": "http.response.start", "status": 429})
            await send({"type": "http.response.body", "body": b"busy"})

        await _run_middleware(reject_app, reject_reporter)
        self.assertEqual(
            [tags["event"] for _, _, tags in reject_reporter.calls],
            ["received", "reject_concurrency"],
        )

    async def test_stream_stays_inflight_until_final_body(self):
        reporter = _Reporter()
        observed = []

        async def app(scope, receive, send):
            token = get_current_frontend_request_token()
            token.admit()
            token.enter_backend()
            await send({"type": "http.response.start", "status": 200})
            await send(
                {
                    "type": "http.response.body",
                    "body": b"first",
                    "more_body": True,
                }
            )
            observed.append(token.closed)
            await send({"type": "http.response.body", "body": b"last"})
            observed.append(token.closed)

        metrics, _ = await _run_middleware(app, reporter)

        self.assertEqual(observed, [False, True])
        self.assertEqual(metrics.snapshot()[ROUTE_NORMAL]["backend"], 0)


if __name__ == "__main__":
    unittest.main()
