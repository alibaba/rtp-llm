import unittest
from unittest import mock

from rtp_llm.aios.kmonitor.python_client.kmonitor.metrics.metric_factory import (
    MetricFactory,
)
from rtp_llm.metrics.kmonitor_metric_reporter import (
    AccMetrics,
    GaugeMetrics,
    MetricReporter,
)


class MetricReporterTest(unittest.TestCase):
    def make_reporter(self):
        backend = mock.Mock()
        backend.register_acc_metric.side_effect = lambda name, tags: (
            MetricFactory.create_metric("AccMetric", name, tags)
        )
        backend.register_gauge_metric.side_effect = lambda name, tags: (
            MetricFactory.create_metric("GaugeMetric", name, tags)
        )
        return MetricReporter(backend), backend

    def test_default_registration_is_unchanged_and_idempotent(self):
        reporter, backend = self.make_reporter()
        reporter.init()
        reporter.init()
        self.assertEqual(
            backend.register_acc_metric.call_args_list,
            [mock.call(metric.value, {}) for metric in AccMetrics],
        )
        self.assertEqual(
            backend.register_gauge_metric.call_args_list,
            [mock.call(metric.value, {}) for metric in GaugeMetrics],
        )

    def test_workers_have_distinct_inspection_series_and_no_vit_counters(self):
        success = AccMetrics.GREENNET_SUCCESS_QPS_METRIC
        error = AccMetrics.GREENNET_ERROR_QPS_METRIC
        rt = GaugeMetrics.GREENNET_RT_METRIC
        points = []
        with mock.patch(
            "rtp_llm.aios.kmonitor.python_client.kmonitor.metrics.acc_metric.time"
        ) as clock:
            clock.time.return_value = 100.0
            workers = []
            for pid in ("1", "2"):
                reporter, _ = self.make_reporter()
                reporter.init(
                    acc_metrics=(success, error),
                    gauge_metrics=(rt,),
                    tags={"source": "greennet_worker", "worker_pid": pid},
                )
                self.assertEqual(
                    set(reporter._matic_map), {success.value, error.value, rt.value}
                )
                reporter.report(success, 1)
                reporter.report(rt, 100)
                workers.append(reporter)
            clock.time.return_value = 101.0
            for reporter in workers:
                points.extend(reporter._matic_map[success.value].fetch_reported_data())
                self.assertEqual(
                    reporter._matic_map[error.value].fetch_reported_data(), []
                )
                self.assertEqual(
                    reporter._matic_map[rt.value].fetch_reported_data()[0].tags,
                    points[-1].tags,
                )
        self.assertEqual(sum(point.value for point in points), 2.0)
        self.assertEqual({point.tags["worker_pid"] for point in points}, {"1", "2"})

    def test_emb_lifecycle_and_delivery_metrics_reach_registered_backend(self):
        reporter, _ = self.make_reporter()
        counters = (
            AccMetrics.VIT_REMOTE_EMB_CLIENT_CREATE_QPS_METRIC,
            AccMetrics.VIT_REMOTE_EMB_CLIENT_CREATE_ERROR_QPS_METRIC,
            AccMetrics.VIT_REMOTE_EMB_METADATA_QPS_METRIC,
            AccMetrics.VIT_REMOTE_EMB_RESULT_ACCEPTED_QPS_METRIC,
        )
        gauges = (
            GaugeMetrics.VIT_REMOTE_EMB_CLIENT_CREATE_RT_METRIC,
            GaugeMetrics.VIT_REMOTE_EMB_QUEUE_WAIT_METRIC,
        )
        with mock.patch(
            "rtp_llm.aios.kmonitor.python_client.kmonitor.metrics.acc_metric.time"
        ) as clock:
            clock.time.return_value = 100.0
            reporter.init()
            for metric in counters:
                reporter.report(metric, 1, {"status": "success"})
            for metric in gauges:
                reporter.report(metric, 25, {"operation": "read"})
            clock.time.return_value = 101.0
            for metric in counters:
                points = reporter._matic_map[metric.value].fetch_reported_data()
                self.assertEqual(len(points), 1)
                self.assertEqual(points[0].value, 1.0)
            for metric in gauges:
                points = reporter._matic_map[metric.value].fetch_reported_data()
                self.assertEqual(len(points), 1)
                self.assertEqual(points[0].value, 25)

    def test_empty_selection_does_not_register_all_metrics(self):
        reporter, backend = self.make_reporter()
        reporter.init(acc_metrics=(), gauge_metrics=())
        backend.register_acc_metric.assert_not_called()
        backend.register_gauge_metric.assert_not_called()


if __name__ == "__main__":
    unittest.main()
