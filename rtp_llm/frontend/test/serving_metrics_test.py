import asyncio
import os
import subprocess
import sys
import tempfile
from unittest import TestCase, main
from unittest.mock import patch

from prometheus_client.parser import text_string_to_metric_families

from rtp_llm.frontend.serving_metrics import (
    CURRENT_SERVING_METRICS,
    RequestMetrics,
    ServingMetrics,
    ServingMetricsMiddleware,
    create_serving_metrics,
)

PREFIX = "rtp_llm_server_"


def samples(metrics):
    return {
        (s.name, tuple(sorted(s.labels.items()))): s.value
        for family in text_string_to_metric_families(metrics.render().decode())
        for s in family.samples
    }


def total(metrics, suffix):
    return sum(
        v for (name, _), v in samples(metrics).items() if name == PREFIX + suffix
    )


class ServingMetricsTest(TestCase):
    def test_observations_units_buckets_and_idempotence(self):
        metrics = ServingMetrics()
        now = [10.0]
        state = RequestMetrics(metrics, lambda: now[0])
        state.token_timings_enabled = True
        now[0] = 10.1
        state.observe_tokens(False, 0)  # role/control chunk
        now[0] = 10.5
        state.observe_tokens(True, 1)
        now[0] = 11.5
        state.observe_tokens(True, 3)  # two more tokens, not one observation each
        state.finish(12.0)
        state.finish(13.0)
        self.assertEqual(total(metrics, "request_duration_seconds_count"), 1)
        self.assertEqual(total(metrics, "request_duration_seconds_sum"), 2)
        self.assertEqual(total(metrics, "time_to_first_token_seconds_sum"), 0.5)
        self.assertEqual(total(metrics, "time_per_output_token_seconds_sum"), 0.5)
        families = list(text_string_to_metric_families(metrics.render().decode()))
        self.assertTrue(all(f.type in ("histogram", "gauge") for f in families))
        buckets = [
            (dict(labels)["le"], value)
            for (name, labels), value in samples(metrics).items()
            if name == PREFIX + "request_duration_seconds_bucket"
        ]
        self.assertEqual(buckets[-1], ("+Inf", 1))
        self.assertEqual([v for _, v in buckets], sorted(v for _, v in buckets))

    def test_missing_timings_are_not_zero_observations(self):
        for counts in ([1], [None, 3], [2, 3], [1, 1], [1, 3, 2]):
            metrics = ServingMetrics()
            now = [0.0]
            state = RequestMetrics(metrics, lambda: now[0])
            state.token_timings_enabled = True
            for count in counts:
                now[0] += 1
                state.observe_tokens(True, count)
            state.finish(now[0] + 1)
            self.assertEqual(total(metrics, "time_to_first_token_seconds_count"), 1)
            self.assertEqual(total(metrics, "time_per_output_token_seconds_count"), 0)
        metrics = ServingMetrics()
        state = RequestMetrics(metrics, lambda: 0)
        state.observe_tokens(True, 32)  # nonstreaming / multiple-choice disabled
        state.finish(1)
        self.assertEqual(total(metrics, "request_duration_seconds_count"), 1)
        self.assertEqual(total(metrics, "time_to_first_token_seconds_count"), 0)

    def test_middleware_body_boundary_errors_and_context(self):
        for outcome in ("success", "stream_error", "http_error", "cancel", "raise"):
            metrics = ServingMetrics()
            now = [0.0]

            async def app(scope, receive, send):
                state = CURRENT_SERVING_METRICS.get()
                state.token_timings_enabled = True
                await send(
                    {
                        "type": "http.response.start",
                        "status": 500 if outcome == "http_error" else 200,
                    }
                )
                now[0] = 1
                state.observe_tokens(True, 1)
                now[0] = 2
                state.observe_tokens(True, 2)
                if outcome == "stream_error":
                    state.error = "internal_error"
                if outcome == "cancel":
                    raise asyncio.CancelledError()
                if outcome == "raise":
                    raise RuntimeError("failed generation")
                await send(
                    {"type": "http.response.body", "body": b"result", "more_body": True}
                )
                now[0] = 9  # access log / cleanup must not extend time-to-last-byte
                await send(
                    {"type": "http.response.body", "body": b"", "more_body": False}
                )

            async def noop(message=None):
                pass

            middleware = ServingMetricsMiddleware(app, metrics, lambda: now[0])
            try:
                asyncio.run(
                    middleware(
                        {
                            "type": "http",
                            "method": "POST",
                            "path": "/v1/chat/completions",
                        },
                        noop,
                        noop,
                    )
                )
            except (RuntimeError, asyncio.CancelledError):
                self.assertIn(outcome, ("cancel", "raise"))
            self.assertIsNone(CURRENT_SERVING_METRICS.get())
            self.assertEqual(total(metrics, "request_duration_seconds_count"), 1)
            self.assertEqual(total(metrics, "request_duration_seconds_sum"), 2)
            self.assertEqual(
                total(metrics, "time_to_first_token_seconds_count"),
                int(outcome == "success"),
            )
            self.assertEqual(
                total(metrics, "time_per_output_token_seconds_count"),
                int(outcome == "success"),
            )

    def test_disabled_and_missing_multiprocess_directory(self):
        with patch.dict(os.environ, {"RTP_LLM_METRICS_ENABLED": "0"}):
            self.assertIsNone(create_serving_metrics())
        with patch.dict(
            os.environ,
            {
                "RTP_LLM_METRICS_ENABLED": "1",
                "PROMETHEUS_MULTIPROC_DIR": "/nonexistent/metrics",
            },
        ):
            with self.assertRaises(ValueError):
                create_serving_metrics()

    def test_multiprocess_scrapes_aggregate_exited_workers_without_duplicates(self):
        with tempfile.TemporaryDirectory() as directory:
            env = dict(
                os.environ,
                RTP_LLM_METRICS_ENABLED="1",
                PROMETHEUS_MULTIPROC_DIR=directory,
            )
            script = """
from rtp_llm.frontend.serving_metrics import create_serving_metrics
metrics = create_serving_metrics()
metrics.duration.labels("chat", "rtp_llm", "").observe(0.5)
"""
            for _ in range(2):
                subprocess.run([sys.executable, "-c", script], env=env, check=True)
            output = subprocess.check_output(
                [
                    sys.executable,
                    "-c",
                    """
from rtp_llm.frontend.serving_metrics import create_serving_metrics
metrics = create_serving_metrics()
assert metrics.render() == metrics.render()
print(metrics.render().decode())
""",
                ],
                env=env,
                text=True,
            )
            family = next(
                f
                for f in text_string_to_metric_families(output)
                if f.name == PREFIX + "request_duration_seconds"
            )
            self.assertEqual(
                [s.value for s in family.samples if s.name.endswith("_count")], [2]
            )
            self.assertEqual(
                [s.value for s in family.samples if s.name.endswith("_sum")], [1]
            )


if __name__ == "__main__":
    main()
