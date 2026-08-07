import unittest
from threading import Event, Thread
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

    def test_reports_container_tps_lengths_cache_and_latency(self):
        state = self.begin()
        state.observe(
            _response(
                "first",
                input_len=100,
                output_len=10,
                reuse_len=40,
                iter_count=4,
                # Prefill has completed before the first output frame, so its
                # cumulative token/time counters are already final here.
                context_execute_time_us=100_000,
                context_execute_time_with_cache_us=50_000,
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
            [100.0],
            self.sink.values(GaugeMetrics.FRONTEND_NONCACHE_OUTPUT_TOKEN_TPS_METRIC),
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

    def test_speculative_tps_uses_accepted_token_counter(self):
        state = self.begin(speculative_steps=4)
        state.observe(
            _response(
                "tokens",
                input_len=20,
                output_len=5,
                reuse_len=0,
                speculative_verify_rounds=2,
                speculative_accepted_token_num=6,
                speculative_proposed_draft_tokens=8,
                generate_execute_time_us=40_000,
            ),
            now_ms=1100,
        )
        state.finish(now_ms=1200)

        # Match MTP's rtp_llm_generate_tps numerator. output_len - 1 would
        # produce 100 TPS here, while the authoritative accepted count is 6.
        self.assertEqual(
            [150.0],
            self.sink.values(GaugeMetrics.FRONTEND_OUTPUT_TOKEN_TPS_METRIC),
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

    def test_observer_failure_is_isolated_from_inference(self):
        class _BrokenResponse:
            @property
            def response_batch(self):
                raise RuntimeError("broken aux info")

        state = self.begin()
        with self.assertLogs(level="ERROR"):
            state.observe(_BrokenResponse())
            state.observe_tps(_BrokenResponse())
        state.finish(now_ms=1100)

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

    def test_container_tps_uses_ratio_of_window_sums(self):
        sink = _MetricSink()
        metrics = FrontendRequestMetrics(
            sink,
            clock=lambda: 1.0,
            # Keep automatic reporting out of the deterministic test window.
            concurrency_report_interval_s=10.0,
        )
        first = metrics.begin(
            rank_id="0",
            server_id="1",
            source="test",
            streaming=True,
            speculative_steps=0,
        )
        second = metrics.begin(
            rank_id="0",
            server_id="1",
            source="test",
            streaming=True,
            speculative_steps=0,
        )
        first.observe(
            _response(
                "first",
                input_len=100,
                output_len=11,
                reuse_len=0,
                context_execute_time_us=100_000,
                context_execute_time_with_cache_us=100_000,
                generate_execute_time_us=100_000,
            )
        )
        second.observe(
            _response(
                "second",
                input_len=900,
                output_len=91,
                reuse_len=0,
                context_execute_time_us=300_000,
                context_execute_time_with_cache_us=300_000,
                generate_execute_time_us=300_000,
            )
        )
        try:
            first.finish(now_ms=1100)
            second.finish(now_ms=1100)
            metrics._report_tps_window()

            # Ratio of sums: (100 + 900) / (0.1 + 0.3), not the arithmetic
            # mean of the two request TPS values (1000 and 3000).
            self.assertEqual(
                [2500.0],
                sink.values(GaugeMetrics.FRONTEND_INPUT_TOKEN_TPS_METRIC),
            )
            self.assertEqual(
                [2500.0],
                sink.values(GaugeMetrics.FRONTEND_NONCACHE_INPUT_TOKEN_TPS_METRIC),
            )
            self.assertEqual(
                [250.0],
                sink.values(GaugeMetrics.FRONTEND_OUTPUT_TOKEN_TPS_METRIC),
            )
        finally:
            metrics.close()

    def test_streaming_tps_is_accumulated_from_observe_deltas(self):
        sink = _MetricSink()
        metrics = FrontendRequestMetrics(
            sink,
            clock=lambda: 1.0,
            # Manually cut deterministic windows while the request is active.
            concurrency_report_interval_s=10.0,
        )
        state = metrics.begin(
            rank_id="0",
            server_id="1",
            source="test",
            streaming=True,
            speculative_steps=0,
        )
        try:
            state.observe(
                _response(
                    "first",
                    input_len=100,
                    output_len=11,
                    reuse_len=0,
                    context_execute_time_us=100_000,
                    context_execute_time_with_cache_us=100_000,
                    generate_execute_time_us=100_000,
                )
            )
            metrics._report_tps_window()

            # The first TPS point is visible before finish().
            self.assertEqual(
                [100.0],
                sink.values(GaugeMetrics.FRONTEND_OUTPUT_TOKEN_TPS_METRIC),
            )

            state.observe(
                _response(
                    "rest",
                    input_len=100,
                    output_len=31,
                    reuse_len=0,
                    context_execute_time_us=100_000,
                    context_execute_time_with_cache_us=100_000,
                    generate_execute_time_us=300_000,
                )
            )
            metrics._report_tps_window()
            state.finish(now_ms=1100)

            # Second window uses only the cumulative delta: 20 / 0.2s.
            self.assertEqual(
                [100.0, 100.0],
                sink.values(GaugeMetrics.FRONTEND_OUTPUT_TOKEN_TPS_METRIC),
            )
            self.assertEqual(
                [1000.0],
                sink.values(GaugeMetrics.FRONTEND_INPUT_TOKEN_TPS_METRIC),
            )
        finally:
            metrics.close()

    def test_repeated_snapshot_does_not_duplicate_tps_delta(self):
        sink = _MetricSink()
        metrics = FrontendRequestMetrics(
            sink,
            clock=lambda: 1.0,
            concurrency_report_interval_s=10.0,
        )
        state = metrics.begin(
            rank_id="0",
            server_id="1",
            source="test",
            streaming=True,
            speculative_steps=0,
        )
        response = _response(
            "tokens",
            input_len=100,
            output_len=11,
            reuse_len=0,
            context_execute_time_us=100_000,
            context_execute_time_with_cache_us=100_000,
            generate_execute_time_us=100_000,
        )
        try:
            state.observe(response)
            state.observe(response)
            state.finish(now_ms=1100)
            metrics._report_tps_window()

            self.assertEqual(
                [1000.0],
                sink.values(GaugeMetrics.FRONTEND_INPUT_TOKEN_TPS_METRIC),
            )
            self.assertEqual(
                [100.0],
                sink.values(GaugeMetrics.FRONTEND_OUTPUT_TOKEN_TPS_METRIC),
            )
        finally:
            metrics.close()

    def test_tps_waits_until_cumulative_token_and_time_are_paired(self):
        state = self.begin()
        state.observe_tps(
            {
                "aux_info": {
                    "input_len": 100,
                    "output_len": 11,
                    "reuse_len": 0,
                    "context_execute_time_us": 0,
                    "context_execute_time_with_cache_us": 0,
                    "generate_execute_time_us": 0,
                }
            }
        )
        self.metrics._report_tps_window()
        self.assertEqual(
            [],
            self.sink.values(GaugeMetrics.FRONTEND_OUTPUT_TOKEN_TPS_METRIC),
        )
        state.observe_tps(
            {
                "aux_info": {
                    "input_len": 100,
                    "output_len": 11,
                    "reuse_len": 0,
                    "context_execute_time_us": 100_000,
                    "context_execute_time_with_cache_us": 100_000,
                    "generate_execute_time_us": 100_000,
                }
            }
        )
        state.finish(now_ms=1100)

        self.assertEqual(
            [1000.0],
            self.sink.values(GaugeMetrics.FRONTEND_INPUT_TOKEN_TPS_METRIC),
        )
        self.assertEqual(
            [100.0],
            self.sink.values(GaugeMetrics.FRONTEND_OUTPUT_TOKEN_TPS_METRIC),
        )

    def test_container_window_pairs_zero_time_stream_tokens_with_peer_time(self):
        first = self.begin()
        second = self.begin()
        first.observe_tps(
            {
                "aux_info": {"generate_execute_time_us": 0},
                "generate_token_num": 1,
            }
        )
        second.observe_tps(
            {
                "aux_info": {"generate_execute_time_us": 100},
                "generate_token_num": 9,
            }
        )
        first.finish(now_ms=1100)
        second.finish(now_ms=1100)

        # Engine window: (1 + 9) tokens / 100us. The first stream's integer
        # time allocation is zero, but its executed token must not disappear.
        self.assertEqual(
            [100_000.0, 0.0],
            self.sink.values(GaugeMetrics.FRONTEND_OUTPUT_TOKEN_TPS_METRIC),
        )

    def test_backend_tps_observer_sums_all_return_sequences(self):
        state = self.begin()
        state.observe_tps(
            {
                "aux_info": [
                    {
                        "input_len": 100,
                        "output_len": 11,
                        "reuse_len": 0,
                        "context_execute_time_us": 100_000,
                        "context_execute_time_with_cache_us": 100_000,
                        "generate_execute_time_us": 400_000,
                    },
                    {
                        "input_len": 100,
                        "output_len": 91,
                        "reuse_len": 0,
                        "context_execute_time_us": 100_000,
                        "context_execute_time_with_cache_us": 100_000,
                        "generate_execute_time_us": 400_000,
                    },
                ]
            }
        )
        state.finish(now_ms=1100)

        # Decode tokens are summed per sequence: (11 - 1) + (91 - 1).
        self.assertEqual(
            [250.0],
            self.sink.values(GaugeMetrics.FRONTEND_OUTPUT_TOKEN_TPS_METRIC),
        )

    def test_raw_private_tps_counters_are_not_advanced_by_outward_mtp_aux(self):
        state = self.begin(speculative_steps=4)
        state.observe_tps(
            {
                "aux_info": {
                    "output_len": 4,
                    "speculative_verify_rounds": 1,
                    "speculative_accepted_token_num": 3,
                    "generate_execute_time_us": 200,
                },
                # The private envelope intentionally lags one MTP round, just
                # like rtp_llm_generate_tps.
                "generate_token_num": 3,
                "generate_execute_time_us": 100,
            }
        )
        state.observe(
            SimpleNamespace(
                response="visible",
                aux_info={
                    "output_len": 8,
                    "speculative_verify_rounds": 2,
                    "speculative_accepted_token_num": 7,
                    "generate_execute_time_us": 200,
                },
            )
        )
        state.finish(now_ms=1100)

        self.assertEqual(
            [30_000.0],
            self.sink.values(GaugeMetrics.FRONTEND_OUTPUT_TOKEN_TPS_METRIC),
        )

    def test_backend_tps_observer_matches_multi_sequence_context_batch(self):
        state = self.begin()
        repeated_aux = {
            "input_len": 100,
            "output_len": 1,
            "step_output_len": 1,
            "reuse_len": 20,
            "context_execute_time_us": 200_000,
            "context_execute_time_with_cache_us": 100_000,
        }
        state.observe_tps(
            {
                "aux_info": [repeated_aux, repeated_aux],
                "_frontend_context_batch_size": 2,
                "_frontend_output_batch_size": 2,
            }
        )
        state.finish(now_ms=1100)

        # NormalExecutor executes the prompt once per non-beam return sequence,
        # while request-level input length remains one prompt.
        self.assertEqual(
            [2000.0],
            self.sink.values(GaugeMetrics.FRONTEND_INPUT_TOKEN_TPS_METRIC),
        )
        self.assertEqual(
            [800.0],
            self.sink.values(GaugeMetrics.FRONTEND_NONCACHE_INPUT_TOKEN_TPS_METRIC),
        )

    def test_backend_tps_observer_handles_variable_beam_width(self):
        sink = _MetricSink()
        metrics = FrontendRequestMetrics(
            sink,
            clock=lambda: 1.0,
            concurrency_report_interval_s=10.0,
        )
        state = metrics.begin(
            rank_id="0",
            server_id="1",
            source="test",
            streaming=True,
            speculative_steps=0,
        )

        def observe_frame(batch_size, output_len, generate_time_us):
            state.observe_tps(
                {
                    "aux_info": [
                        {
                            "output_len": output_len,
                            "step_output_len": 1,
                            "generate_execute_time_us": generate_time_us,
                        }
                        for _ in range(batch_size)
                    ],
                    "_frontend_context_batch_size": 1,
                    "_frontend_output_batch_size": batch_size,
                }
            )

        try:
            # Prefill produces the first token and establishes a width of four;
            # it is not part of NormalExecutor's decode TPS numerator.
            observe_frame(4, 1, 0)
            # Decode executes previous widths 4, then 2, then 3. The current
            # output list widths deliberately shrink/grow to catch regressions.
            observe_frame(2, 2, 40_000)
            metrics._report_tps_window()
            observe_frame(3, 3, 60_000)
            metrics._report_tps_window()
            observe_frame(1, 4, 90_000)
            metrics._report_tps_window()
            state.finish(now_ms=1100)

            self.assertEqual(
                [100.0, 100.0, 100.0],
                sink.values(GaugeMetrics.FRONTEND_OUTPUT_TOKEN_TPS_METRIC),
            )
        finally:
            metrics.close()

    def test_backend_tps_ignores_stale_variable_beam_width(self):
        sink = _MetricSink()
        metrics = FrontendRequestMetrics(
            sink,
            clock=lambda: 1.0,
            concurrency_report_interval_s=10.0,
        )
        state = metrics.begin(
            rank_id="0",
            server_id="1",
            source="test",
            streaming=True,
            speculative_steps=0,
        )

        def observe_frame(batch_size, output_len, generate_time_us):
            state.observe_tps(
                {
                    "aux_info": [
                        {
                            "output_len": output_len,
                            "step_output_len": 1,
                            "generate_execute_time_us": generate_time_us,
                        }
                        for _ in range(batch_size)
                    ],
                    "_frontend_output_batch_size": batch_size,
                }
            )

        try:
            observe_frame(4, 1, 0)
            observe_frame(2, 2, 40_000)
            metrics._report_tps_window()

            # A delayed frame from output position 1 must not restore width 4.
            observe_frame(4, 1, 0)
            observe_frame(3, 3, 60_000)
            metrics._report_tps_window()
            state.finish(now_ms=1100)

            self.assertEqual(
                [100.0, 100.0],
                sink.values(GaugeMetrics.FRONTEND_OUTPUT_TOKEN_TPS_METRIC),
            )
        finally:
            metrics.close()

    def test_backend_tps_retry_attempts_have_independent_high_water_marks(self):
        sink = _MetricSink()
        metrics = FrontendRequestMetrics(
            sink,
            clock=lambda: 1.0,
            concurrency_report_interval_s=10.0,
        )
        state = metrics.begin(
            rank_id="0",
            server_id="1",
            source="test",
            streaming=False,
            speculative_steps=0,
        )

        def observe_frame(attempt, output_len, generate_time_us):
            state.observe_tps(
                {
                    "aux_info": {
                        "output_len": output_len,
                        "step_output_len": 1,
                        "generate_execute_time_us": generate_time_us,
                    },
                    "_frontend_metric_unit_id": 0,
                    "_frontend_metric_attempt": attempt,
                    "_frontend_output_batch_size": 1,
                }
            )

        try:
            observe_frame(0, 1, 0)
            observe_frame(0, 2, 10_000)
            metrics._report_tps_window()

            # The retried backend restarts cumulative counters at zero, but
            # its real work must be added rather than hidden by attempt 0's
            # high-water mark.
            observe_frame(1, 1, 0)
            observe_frame(1, 2, 20_000)
            metrics._report_tps_window()
            state.finish(now_ms=1100)

            self.assertEqual(
                [100.0, 50.0],
                sink.values(GaugeMetrics.FRONTEND_OUTPUT_TOKEN_TPS_METRIC),
            )
        finally:
            metrics.close()

    def test_old_backend_nonstream_final_frame_keeps_output_numerator(self):
        state = self.begin(streaming=False)
        state.observe_tps(
            {
                "aux_info": {
                    "output_len": 11,
                    "step_output_len": 11,
                    "generate_execute_time_us": 100_000,
                },
                # Production projectors include this metadata even when the
                # old backend does not support the internal side-channel.
                "_frontend_output_batch_size": 1,
                "generate_token_num": None,
            }
        )
        state.finish(now_ms=1100)

        self.assertEqual(
            [100.0],
            self.sink.values(GaugeMetrics.FRONTEND_OUTPUT_TOKEN_TPS_METRIC),
        )

    def test_request_lengths_prefer_multi_choice_usage_over_single_aux(self):
        state = self.begin()

        def response(payload, aux_output_len, total_output_len):
            return SimpleNamespace(
                response=payload,
                aux_info={
                    "input_len": 100,
                    "output_len": aux_output_len,
                    "reuse_len": 40,
                },
                usage=SimpleNamespace(
                    prompt_tokens=100,
                    completion_tokens=total_output_len,
                    prompt_tokens_details=SimpleNamespace(cached_tokens=40),
                ),
            )

        state.observe(response("first", 10, 20), now_ms=1100)
        state.observe(response("rest", 20, 120), now_ms=1300)
        state.finish(now_ms=1500)

        self.assertEqual(
            [120], self.sink.values(GaugeMetrics.FRONTEND_OUTPUT_LENGTH_METRIC)
        )
        self.assertEqual([4.0], self.sink.values(GaugeMetrics.FRONTEND_TPOT_MS_METRIC))

    def test_request_output_length_sums_independent_sequence_terminal_lengths(self):
        state = self.begin(streaming=False)
        state.observe(
            SimpleNamespace(
                response=["first", "second"],
                aux_info=[
                    {"input_len": 10, "output_len": 1, "reuse_len": 0},
                    {"input_len": 10, "output_len": 2, "reuse_len": 0},
                ],
            )
        )
        state.finish(now_ms=1100)

        self.assertEqual(
            [3], self.sink.values(GaugeMetrics.FRONTEND_OUTPUT_LENGTH_METRIC)
        )

    def test_close_flushes_last_tps_window_once(self):
        sink = _MetricSink()
        metrics = FrontendRequestMetrics(
            sink,
            clock=lambda: 1.0,
            concurrency_report_interval_s=60.0,
        )
        state = metrics.begin(
            rank_id="0",
            server_id="1",
            source="test",
            streaming=True,
            speculative_steps=0,
        )
        state.observe(
            _response(
                "tokens",
                input_len=100,
                output_len=11,
                reuse_len=0,
                context_execute_time_us=100_000,
                context_execute_time_with_cache_us=100_000,
                generate_execute_time_us=100_000,
            )
        )
        state.finish(now_ms=1100)

        self.assertEqual([], sink.values(GaugeMetrics.FRONTEND_OUTPUT_TOKEN_TPS_METRIC))
        metrics.close()
        metrics.close()

        self.assertEqual(
            [1000.0],
            sink.values(GaugeMetrics.FRONTEND_INPUT_TOKEN_TPS_METRIC),
        )
        self.assertEqual(
            [100.0],
            sink.values(GaugeMetrics.FRONTEND_OUTPUT_TOKEN_TPS_METRIC),
        )

    def test_close_waits_for_inflight_heartbeat_without_duplicate_tps(self):
        heartbeat_reporting = Event()
        release_heartbeat = Event()
        close_done = Event()

        class _BlockingSink(_MetricSink):
            def report(inner_self, metric, value=1, tags=None):
                super().report(metric, value, tags)
                if (
                    metric == GaugeMetrics.FRONTEND_OUTPUT_TOKEN_TPS_METRIC
                    and value > 0
                ):
                    heartbeat_reporting.set()
                    release_heartbeat.wait(timeout=1.0)

        sink = _BlockingSink()
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
        state.observe(
            _response(
                "tokens",
                input_len=100,
                output_len=11,
                reuse_len=0,
                context_execute_time_us=100_000,
                context_execute_time_with_cache_us=100_000,
                generate_execute_time_us=100_000,
            )
        )

        def close_metrics():
            metrics.close()
            close_done.set()

        close_thread = Thread(target=close_metrics)
        try:
            self.assertTrue(heartbeat_reporting.wait(timeout=0.5))
            close_thread.start()
            self.assertFalse(close_done.wait(timeout=0.05))
            # The heartbeat has already swapped the first window and is
            # blocked in report(). This new delta must remain pending until
            # close() joins the heartbeat and performs its final flush.
            state.observe(
                _response(
                    "tail",
                    input_len=100,
                    output_len=21,
                    reuse_len=0,
                    context_execute_time_us=100_000,
                    context_execute_time_with_cache_us=100_000,
                    generate_execute_time_us=200_000,
                )
            )
            state.finish(now_ms=1100)
        finally:
            release_heartbeat.set()
            if close_thread.ident is not None:
                close_thread.join(timeout=1.0)
            else:
                state.finish(now_ms=1100)
                metrics.close()

        self.assertTrue(close_done.is_set())
        self.assertEqual(
            [100.0, 100.0],
            sink.values(GaugeMetrics.FRONTEND_OUTPUT_TOKEN_TPS_METRIC),
        )


if __name__ == "__main__":
    unittest.main()
