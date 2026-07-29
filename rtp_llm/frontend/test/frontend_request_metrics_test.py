import unittest
from threading import Event
from types import SimpleNamespace

from rtp_llm.frontend.frontend_request_metrics import FrontendRequestMetrics
from rtp_llm.metrics import GaugeMetrics


class _MetricSink:
    def __init__(self):
        self.calls = []

    def report(self, metric, value=1, tags=None):
        self.calls.append((metric, value, tags or {}))

    def values(self, metric):
        return [
            value for called_metric, value, _ in self.calls if called_metric == metric
        ]


def _response(response, **aux):
    return SimpleNamespace(response=response, aux_info=aux)


def _batch_response(*responses):
    return SimpleNamespace(response_batch=list(responses))


class FrontendRequestMetricsTest(unittest.TestCase):
    def setUp(self):
        self.sink = _MetricSink()
        self.metrics = FrontendRequestMetrics(self.sink, clock=lambda: 1.0)

    def begin(self, *, streaming=True, speculative_steps=0):
        return self.metrics.begin(
            rank_id="0",
            server_id="1",
            source="test",
            streaming=streaming,
            speculative_steps=speculative_steps,
        )

    def test_reports_token_tps_lengths_cache_and_latency(self):
        state = self.begin()
        state.observe(
            _response(
                "first",
                input_len=100,
                output_len=10,
                reuse_len=40,
                iter_count=4,
                context_execute_time_us=50_000,
                context_execute_time_with_cache_us=25_000,
                generate_execute_time_us=90_000,
            ),
            now_ms=1100,
        )
        state.observe(
            _response(
                "rest",
                input_len=100,
                output_len=20,
                reuse_len=40,
                iter_count=8,
                context_execute_time_us=100_000,
                context_execute_time_with_cache_us=50_000,
                generate_execute_time_us=190_000,
            ),
            now_ms=1300,
        )
        state.finish(now_ms=1500)

        self.assertEqual(
            [2000.0],
            self.sink.values(GaugeMetrics.FRONTEND_INPUT_TOKEN_TPS_METRIC),
        )
        self.assertEqual(
            [600.0],
            self.sink.values(GaugeMetrics.FRONTEND_NONCACHE_INPUT_TOKEN_TPS_METRIC),
        )
        self.assertEqual(
            [100.0],
            self.sink.values(GaugeMetrics.FRONTEND_OUTPUT_TOKEN_TPS_METRIC),
        )
        self.assertEqual(
            [40.0], self.sink.values(GaugeMetrics.FRONTEND_CACHE_HIT_RATIO_METRIC)
        )
        self.assertEqual(
            [100.0], self.sink.values(GaugeMetrics.FRONTEND_TTFT_MS_METRIC)
        )
        self.assertEqual([40.0], self.sink.values(GaugeMetrics.FRONTEND_TPOT_MS_METRIC))
        self.assertEqual(
            [10],
            self.sink.values(
                GaugeMetrics.FRONTEND_STREAM_FIRST_OUTPUT_TOKEN_LENGTH_METRIC
            ),
        )

    def test_tool_call_buffering_counts_all_tokens_in_first_payload(self):
        state = self.begin()
        state.observe(
            _response(
                "",
                input_len=10,
                output_len=8,
                reuse_len=0,
                iter_count=3,
            ),
            now_ms=1050,
        )
        state.observe(
            _response(
                "complete tool call",
                input_len=10,
                output_len=25,
                reuse_len=0,
                iter_count=7,
            ),
            now_ms=1200,
        )
        state.finish(now_ms=1300)

        self.assertEqual(
            [25],
            self.sink.values(
                GaugeMetrics.FRONTEND_STREAM_FIRST_OUTPUT_TOKEN_LENGTH_METRIC
            ),
        )
        self.assertEqual(
            [200.0], self.sink.values(GaugeMetrics.FRONTEND_TTFT_MS_METRIC)
        )

    def test_speculative_metrics_match_backend_formula(self):
        state = self.begin(speculative_steps=4)
        state.observe(
            _response(
                "tokens",
                input_len=20,
                output_len=10,
                reuse_len=0,
                speculative_verify_rounds=3,
                speculative_accepted_token_num=9,
                speculative_proposed_draft_tokens=12,
            ),
            now_ms=1100,
        )
        state.finish(now_ms=1200)

        self.assertEqual(
            [3.0],
            self.sink.values(
                GaugeMetrics.FRONTEND_SPECULATIVE_AVG_ACCEPT_LENGTH_METRIC
            ),
        )
        self.assertEqual(
            [0.5],
            self.sink.values(GaugeMetrics.FRONTEND_SPECULATIVE_ACCEPT_RATE_METRIC),
        )

    def test_repeated_sequence_aux_does_not_duplicate_stream_counters(self):
        state = self.begin()
        repeated_counters = {
            "input_len": 10,
            "reuse_len": 2,
            "context_execute_time_us": 20_000,
            "context_execute_time_with_cache_us": 10_000,
            "generate_execute_time_us": 100_000,
            "speculative_verify_rounds": 3,
            "speculative_accepted_token_num": 9,
            "speculative_proposed_draft_tokens": 12,
        }
        state.observe(
            SimpleNamespace(
                response="tokens",
                aux_info=[
                    SimpleNamespace(output_len=5, **repeated_counters),
                    SimpleNamespace(output_len=6, **repeated_counters),
                ],
            ),
            now_ms=1100,
        )
        state.finish(now_ms=1200)

        self.assertEqual(
            [1000.0],
            self.sink.values(GaugeMetrics.FRONTEND_INPUT_TOKEN_TPS_METRIC),
        )
        self.assertEqual(
            [400.0],
            self.sink.values(GaugeMetrics.FRONTEND_NONCACHE_INPUT_TOKEN_TPS_METRIC),
        )
        self.assertEqual(
            [90.0],
            self.sink.values(GaugeMetrics.FRONTEND_OUTPUT_TOKEN_TPS_METRIC),
        )
        self.assertEqual(
            [3.0],
            self.sink.values(
                GaugeMetrics.FRONTEND_SPECULATIVE_AVG_ACCEPT_LENGTH_METRIC
            ),
        )
        self.assertEqual(
            [0.5],
            self.sink.values(GaugeMetrics.FRONTEND_SPECULATIVE_ACCEPT_RATE_METRIC),
        )

    def test_finish_is_idempotent_and_concurrency_returns_to_zero(self):
        state = self.begin()
        state.finish(now_ms=1100)
        state.finish(now_ms=1200)

        self.assertEqual(
            [1, 0], self.sink.values(GaugeMetrics.FRONTEND_CONCURRENCY_METRIC)
        )
        self.assertEqual(
            [100.0], self.sink.values(GaugeMetrics.FRONTEND_REQUEST_RT_MS_METRIC)
        )

    def test_batch_keeps_completed_units_and_reports_each_prompt_once(self):
        state = self.begin()
        state.observe(
            _batch_response(
                _response(
                    "first done",
                    input_len=100,
                    output_len=100,
                    reuse_len=40,
                    context_execute_time_us=100_000,
                    generate_execute_time_us=990_000,
                ),
                _response(
                    "second partial",
                    input_len=200,
                    output_len=10,
                    reuse_len=50,
                    context_execute_time_us=100_000,
                    generate_execute_time_us=90_000,
                ),
            ),
            now_ms=1100,
        )
        # FrontendWorker replaces a completed incremental batch item with an
        # empty response while the remaining item continues.
        state.observe(
            _batch_response(
                _response(""),
                _response(
                    "second done",
                    input_len=200,
                    output_len=20,
                    reuse_len=50,
                    context_execute_time_us=100_000,
                    generate_execute_time_us=190_000,
                ),
            ),
            now_ms=1200,
        )
        state.finish(now_ms=1300)

        self.assertEqual(
            [1500.0],
            self.sink.values(GaugeMetrics.FRONTEND_INPUT_TOKEN_TPS_METRIC),
        )
        self.assertEqual(
            [1050.0],
            self.sink.values(GaugeMetrics.FRONTEND_NONCACHE_INPUT_TOKEN_TPS_METRIC),
        )
        self.assertEqual(
            [100.0],
            self.sink.values(GaugeMetrics.FRONTEND_OUTPUT_TOKEN_TPS_METRIC),
        )
        self.assertEqual(
            [300], self.sink.values(GaugeMetrics.FRONTEND_INPUT_LENGTH_METRIC)
        )
        self.assertEqual(
            [120], self.sink.values(GaugeMetrics.FRONTEND_OUTPUT_LENGTH_METRIC)
        )
        self.assertEqual(
            [90],
            self.sink.values(GaugeMetrics.FRONTEND_CACHED_TOKEN_LENGTH_METRIC),
        )
        self.assertEqual(
            [30.0], self.sink.values(GaugeMetrics.FRONTEND_CACHE_HIT_RATIO_METRIC)
        )

    def test_concurrency_heartbeat_repeats_current_value(self):
        heartbeat_seen = Event()

        class _HeartbeatSink(_MetricSink):
            def report(inner_self, metric, value=1, tags=None):
                super().report(metric, value, tags)
                if (
                    metric == GaugeMetrics.FRONTEND_CONCURRENCY_METRIC
                    and value == 1
                    and len(inner_self.values(metric)) >= 2
                ):
                    heartbeat_seen.set()

        sink = _HeartbeatSink()
        metrics = FrontendRequestMetrics(
            sink,
            clock=lambda: 1.0,
            concurrency_report_interval_s=0.01,
        )
        state = metrics.begin(
            rank_id="0",
            server_id="1",
            source="test",
            streaming=True,
            speculative_steps=0,
        )
        try:
            self.assertTrue(heartbeat_seen.wait(timeout=0.5))
        finally:
            state.finish(now_ms=1100)
            metrics.close()


if __name__ == "__main__":
    unittest.main()
