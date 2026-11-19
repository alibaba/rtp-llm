import unittest
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier
from unittest import mock

from rtp_llm.aios.kmonitor.python_client.kmonitor.metrics.metric_factory import (
    MetricFactory,
)
from rtp_llm.metrics.emb_interval_metrics import EmbLatencyMetric, EmbOutcomeMetric
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
        backend.register_custom_metric.side_effect = lambda metric: metric
        return MetricReporter(backend), backend

    def test_default_registration_is_unchanged_and_idempotent(self):
        reporter, backend = self.make_reporter()
        reporter.init()
        reporter.init()
        self.assertEqual(
            backend.register_acc_metric.call_args_list,
            [
                mock.call(metric.value, {})
                for metric in AccMetrics
                if metric
                not in (
                    AccMetrics.VIT_REMOTE_EMB_OPERATION_QPS_METRIC,
                    AccMetrics.VIT_REMOTE_EMB_STAGE_QPS_METRIC,
                    AccMetrics.VIT_REMOTE_EMB_STAGE_ERROR_QPS_METRIC,
                )
            ],
        )
        self.assertEqual(
            backend.register_gauge_metric.call_args_list,
            [
                mock.call(metric.value, {})
                for metric in GaugeMetrics
                if metric
                not in (
                    GaugeMetrics.VIT_REMOTE_EMB_STAGE_RT_METRIC,
                    GaugeMetrics.VIT_REMOTE_EMB_QUEUE_WAIT_METRIC,
                    GaugeMetrics.VIT_REMOTE_EMB_READ_RT_METRIC,
                    GaugeMetrics.VIT_REMOTE_EMB_WRITE_RT_METRIC,
                )
            ],
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
        with mock.patch(
            "rtp_llm.aios.kmonitor.python_client.kmonitor.metrics.acc_metric.time"
        ) as clock:
            clock.time.return_value = 100.0
            reporter.init()
            for metric in counters:
                reporter.report(metric, 1, {"status": "success"})
            clock.time.return_value = 101.0
            for metric in counters:
                points = reporter._matic_map[metric.value].fetch_reported_data()
                self.assertEqual(len(points), 1)
                self.assertEqual(points[0].value, 1.0)

    def test_emb_stage_distribution_and_qps_use_same_samples(self):
        rt = GaugeMetrics.VIT_REMOTE_EMB_STAGE_RT_METRIC.value
        with mock.patch("rtp_llm.metrics.emb_interval_metrics.time.monotonic") as clock:
            clock.return_value = 10.0
            metric = EmbLatencyMetric(rt, {"worker": "a"})
            metric.report(
                6, {"operation": "read", "stage": "metadata", "status": "success"}
            )
            clock.return_value = 12.0
            first = metric.fetch_reported_data()
            tags = {"operation": "read", "stage": "metadata", "status": "timeout"}
            for value in (1, 5, 7):
                metric.report(value, tags)
            clock.return_value = 13.0
            second = metric.fetch_reported_data()
            clock.return_value = 14.0
            idle = metric.fetch_reported_data()
            self.assertTrue(idle)
            self.assertTrue(
                all(p.metric_name.endswith("_qps") and p.value == 0 for p in idle)
            )

        self.assertEqual(
            sum(p.value for p in first + second if p.metric_name == rt + "_count"), 4
        )
        self.assertEqual(
            sum(p.value for p in first + second if p.metric_name == rt + "_sum"), 19
        )
        bucket_name = rt + "_bucket"
        self.assertEqual(
            [
                p.value
                for p in second
                if p.metric_name == bucket_name and p.tags["le"] in ("1", "5", "+Inf")
            ],
            [1, 2, 3],
        )
        qps = AccMetrics.VIT_REMOTE_EMB_STAGE_QPS_METRIC.value
        self.assertEqual(
            [
                (p.tags["status"], p.value)
                for p in first + second
                if p.metric_name == qps and p.value
            ],
            [("success", 0.5), ("timeout", 3)],
        )
        self.assertEqual(
            [
                p.value
                for p in first + second
                if p.metric_name
                == AccMetrics.VIT_REMOTE_EMB_STAGE_ERROR_QPS_METRIC.value
            ],
            [0, 3],
        )
        rates = {
            p.tags["status"]: p.value
            for p in second
            if p.metric_name == AccMetrics.VIT_REMOTE_EMB_OPERATION_QPS_METRIC.value
        }
        self.assertEqual(rates, dict(total=3, success=0, error=3, skipped=0))
        self.assertTrue(all(p.tags["worker"] == "a" for p in first + second))

    def test_emb_concurrent_flush_keeps_every_outcome_and_wait_sample(self):
        qps = AccMetrics.VIT_REMOTE_EMB_OPERATION_QPS_METRIC.value
        rt = GaugeMetrics.VIT_REMOTE_EMB_QUEUE_WAIT_METRIC.value
        workers = 8
        iterations = 300
        barrier = Barrier(workers + 1)
        with mock.patch("rtp_llm.metrics.emb_interval_metrics.time.monotonic") as clock:
            clock.return_value = 0.0
            counters = EmbOutcomeMetric(qps, {})
            waits = EmbLatencyMetric(rt, {})

            def produce():
                barrier.wait(timeout=10)
                for i in range(iterations):
                    status = ("success", "error", "skipped")[i % 3]
                    counters.report(
                        1, {"operation": "read", "stage": "all", "status": status}
                    )
                    waits.report(i, {"operation": "read_transfer"})

            outcomes = []
            latencies = []
            with ThreadPoolExecutor(workers) as executor:
                futures = [executor.submit(produce) for _ in range(workers)]
                barrier.wait(timeout=10)
                for tick in range(1, 5):
                    clock.return_value = float(tick)
                    outcomes.extend(counters.fetch_reported_data())
                    latencies.extend(waits.fetch_reported_data())
                for future in futures:
                    future.result(timeout=10)
            clock.return_value = 5.0
            outcomes.extend(counters.fetch_reported_data())
            latencies.extend(waits.fetch_reported_data())
            clock.return_value = 6.0
            idle = counters.fetch_reported_data()
            self.assertEqual(len(idle), 4)
            self.assertTrue(all(p.value == 0 for p in idle))
            self.assertFalse(waits.fetch_reported_data())
        for status in ("total", "success", "error", "skipped"):
            self.assertEqual(
                sum(p.value for p in outcomes if p.tags["status"] == status),
                workers * iterations // (1 if status == "total" else 3),
            )
        self.assertEqual(
            sum(p.value for p in latencies if p.metric_name == rt + "_count"),
            workers * iterations,
        )
        self.assertEqual(
            sum(p.value for p in latencies if p.metric_name == rt + "_sum"),
            workers * sum(range(iterations)),
        )
        self.assertEqual(
            sum(
                p.value
                for p in latencies
                if p.metric_name == rt + "_bucket" and p.tags["le"] == "+Inf"
            ),
            workers * iterations,
        )

    def test_empty_selection_does_not_register_all_metrics(self):
        reporter, backend = self.make_reporter()
        reporter.init(acc_metrics=(), gauge_metrics=())
        backend.register_acc_metric.assert_not_called()
        backend.register_gauge_metric.assert_not_called()


if __name__ == "__main__":
    unittest.main()
