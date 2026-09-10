import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from online_eval.telemetry import SharedMetricSource, http_text


class SharedTelemetryTest(unittest.TestCase):
    def test_bounded_history_replays_old_samples_and_errors_without_scraping(self):
        from online_eval.telemetry import shared_samples_since

        calls = []
        with tempfile.TemporaryDirectory() as d:

            def fetch():
                calls.append(len(calls) + 1)
                if len(calls) == 5:
                    source.stop_event.set()
                if len(calls) == 2:
                    raise ValueError("second sample unavailable")
                return f'value{{label="中文"}} {len(calls)}\n'

            source = SharedMetricSource(
                "http://mock/replay", d, 0.001, fetch, history_limit=2
            )
            source.start()
            try:
                source.thread.join(2)
                self.assertFalse(source.thread.is_alive())
                self.assertEqual(len(source.samples), 2)
                rows = list(shared_samples_since(source.url, 0))
                self.assertEqual([r["sequence"] for r in rows], [1, 2, 3, 4, 5])
                self.assertIn("second sample unavailable", rows[1]["error"])
                self.assertIsNone(rows[1]["body"])
                self.assertEqual(rows[0]["body"], 'value{label="中文"} 1\n')
                self.assertEqual(rows[4]["body"], 'value{label="中文"} 5\n')
                self.assertEqual(len(calls), 5)
            finally:
                source.stop(2)

    def test_consumers_do_not_drain_endpoint_again(self):
        calls = []

        def fetch():
            calls.append(1)
            return "tokens 42\n"

        with tempfile.TemporaryDirectory() as d:
            source = SharedMetricSource("http://mock/metrics", d, 60, fetch)
            source.start()
            try:
                with ThreadPoolExecutor(8) as pool:
                    values = list(
                        pool.map(lambda _: http_text(source.url, 2), range(30))
                    )
                self.assertEqual(values, ["tokens 42\n"] * 30)
                self.assertEqual(len(calls), 1)
                self.assertIn("tokens 42", (Path(d) / "mock.prom").read_text())
                with self.assertRaises(RuntimeError):
                    SharedMetricSource(source.url, d, 60, fetch).start()
            finally:
                source.stop(2)

    def test_source_error_never_falls_back_to_http(self):
        def fetch():
            raise ValueError("injected source failure")

        with tempfile.TemporaryDirectory() as d:
            source = SharedMetricSource("http://not-a-server/metrics", d, 60, fetch)
            source.start()
            try:
                with self.assertRaisesRegex(RuntimeError, "injected source failure"):
                    http_text(source.url, 2)
            finally:
                source.stop(2)

    def test_balance_consumer_uses_source_timestamp_and_deduplicates(self):
        from flexlb_test_framework.harness import BalanceSampler
        from online_eval.telemetry import shared_samples_since

        with tempfile.TemporaryDirectory() as d:
            source = SharedMetricSource(
                "http://127.0.0.1:12345/metrics?per_engine=true",
                d,
                60,
                lambda: 'rtp_llm_context_tps{engine_name="P",role="prefill"} 42\n',
            )
            source.start()
            try:
                source.read(2)
                sample = shared_samples_since(source.url, 0)[0]
                sampler = BalanceSampler(12345, 12346)
                sampler._t0 = sample["monotonic_s"] - 1
                sampler._poll_mock(999)
                sampler._poll_mock(1000)
                self.assertEqual(sampler._series["P"]["rtp_llm_context_tps"], [(1, 42)])
            finally:
                source.stop(2)

    def test_fatal_collector_exit_is_cleanup_failure_and_releases_ownership(self):
        def fatal():
            raise SystemExit("collector terminated")

        with tempfile.TemporaryDirectory() as d:
            source = SharedMetricSource("http://mock/fatal", d, 60, fatal)
            source.start()
            with self.assertRaisesRegex(RuntimeError, "collector terminated"):
                source.read(2)
            with self.assertRaisesRegex(RuntimeError, "collector terminated"):
                source.stop(2)
            replacement = SharedMetricSource(source.url, d, 60, lambda: "value 1\n")
            replacement.start()
            replacement.stop(2)
