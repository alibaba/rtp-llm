import os
from unittest import TestCase, main
from unittest.mock import Mock, patch

from rtp_llm.aios.kmonitor.python_client.kmonitor.kmonitor import KMonitor
from rtp_llm.aios.kmonitor.python_client.kmonitor.metrics.gauge_metric import (
    GaugeMetric,
)
from rtp_llm.aios.kmonitor.python_client.kmonitor.metrics.histogram_metric import (
    HistogramMetric,
)
from rtp_llm.metrics.kmonitor_metric_reporter import (
    AccMetrics,
    GaugeMetrics,
    MetricReporter,
)


class MetricReporterTest(TestCase):
    def setUp(self):
        self.env = patch.dict(os.environ)
        self.env.start()
        self.addCleanup(self.env.stop)
        os.environ.pop("RTP_LLM_LATENCY_HISTOGRAM_ENABLED", None)

    def make_reporter(self):
        monitor = Mock(spec=KMonitor)
        monitor.register_gauge_metric.side_effect = lambda name: GaugeMetric(name, {})
        monitor.register_histogram_metric.side_effect = (
            lambda name, buckets: HistogramMetric(name, {}, buckets)
        )
        reporter = MetricReporter(monitor)
        reporter.init()
        return reporter, monitor

    def test_same_samples_and_tags_are_written_to_both_metrics(self):
        reporter, monitor = self.make_reporter()
        pairs = {
            GaugeMetrics.LANTENCY_METRIC: "py_rtp_framework_rt_seconds",
            GaugeMetrics.RESPONSE_FIRST_TOKEN_RT_METRIC: "py_rtp_response_first_token_rt_seconds",
            GaugeMetrics.RESPONSE_ITER_RT_METRIC: "py_rtp_response_iterate_rt_seconds",
        }
        tags = {"protocol": "grpc", "rank_id": "1"}
        for gauge, name in pairs.items():
            reporter.report(gauge, 125, tags)
            reporter.report(gauge, 375, tags)
            points = reporter._matic_map[gauge.value].fetch_reported_data()
            self.assertEqual([p.value for p in points], [125, 375])
            self.assertEqual([p.tags for p in points], [tags, tags])
            histogram = reporter._histogram_map[gauge.value]
            self.assertEqual(histogram.name, name)
            snapshots = histogram.fetch_reported_data()
            self.assertEqual(snapshots[-2].value, 0.5)
            self.assertEqual(snapshots[-1].value, 2)
            for point in snapshots:
                self.assertEqual(point.tags["protocol"], "grpc")
                self.assertEqual(point.tags["rank_id"], "1")
            self.assertEqual(snapshots[-3].tags["le"], "+Inf")
            self.assertEqual(snapshots[-3].value, 2)
        self.assertEqual(tags, {"protocol": "grpc", "rank_id": "1"})
        self.assertEqual(monitor.register_histogram_metric.call_count, 3)

    def test_default_registration_does_not_invent_observations(self):
        reporter, monitor = self.make_reporter()
        self.assertEqual(monitor.register_histogram_metric.call_count, 3)
        self.assertTrue(
            all(not h.fetch_reported_data() for h in reporter._histogram_map.values())
        )

    def test_explicit_enablement(self):
        os.environ["RTP_LLM_LATENCY_HISTOGRAM_ENABLED"] = "1"
        self.test_same_samples_and_tags_are_written_to_both_metrics()

    def test_disabled_histograms_preserve_existing_metrics(self):
        os.environ["RTP_LLM_LATENCY_HISTOGRAM_ENABLED"] = "0"
        reporter, monitor = self.make_reporter()
        reporter.init()
        monitor.register_histogram_metric.assert_not_called()
        self.assertEqual(reporter._histogram_map, {})
        tags = {"protocol": "http"}
        for gauge in (
            GaugeMetrics.LANTENCY_METRIC,
            GaugeMetrics.RESPONSE_FIRST_TOKEN_RT_METRIC,
            GaugeMetrics.RESPONSE_ITER_RT_METRIC,
        ):
            reporter.report(gauge, 125, tags)
            points = reporter._matic_map[gauge.value].fetch_reported_data()
            self.assertEqual([p.value for p in points], [125])
            self.assertEqual([p.tags for p in points], [tags])
        reporter.report(AccMetrics.QPS_METRIC, 1, tags)
        reporter._matic_map[AccMetrics.QPS_METRIC.value].report.assert_called_once_with(
            1, tags
        )

    def test_unrelated_metrics_and_repeated_init(self):
        reporter, monitor = self.make_reporter()
        reporter.init()
        self.assertEqual(monitor.register_histogram_metric.call_count, 3)
        reporter.report(GaugeMetrics.INPUT_TOKEN_SIZE_METRIC, 42)
        reporter.report(AccMetrics.QPS_METRIC, 1, {"source": "test"})
        self.assertEqual(
            reporter._matic_map[GaugeMetrics.INPUT_TOKEN_SIZE_METRIC.value]
            .fetch_reported_data()[0]
            .value,
            42,
        )
        reporter._matic_map[AccMetrics.QPS_METRIC.value].report.assert_called_once_with(
            1, {"source": "test"}
        )
        self.assertTrue(
            all(not h.fetch_reported_data() for h in reporter._histogram_map.values())
        )

    def test_histogram_failure_does_not_change_existing_gauge(self):
        reporter, _ = self.make_reporter()
        with self.assertLogs(level="ERROR"):
            reporter.report(GaugeMetrics.LANTENCY_METRIC, -5)
        self.assertEqual(
            reporter._matic_map[GaugeMetrics.LANTENCY_METRIC.value]
            .fetch_reported_data()[0]
            .value,
            -5,
        )
        self.assertEqual(
            reporter._histogram_map[
                GaugeMetrics.LANTENCY_METRIC.value
            ].fetch_reported_data(),
            [],
        )
        # Recording continues after a bad observation.
        reporter.report(GaugeMetrics.LANTENCY_METRIC, 25)
        self.assertEqual(
            reporter._histogram_map[GaugeMetrics.LANTENCY_METRIC.value]
            .fetch_reported_data()[-1]
            .value,
            1,
        )


if __name__ == "__main__":
    main()
