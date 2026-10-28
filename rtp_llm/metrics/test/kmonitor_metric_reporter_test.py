import unittest
from unittest.mock import Mock, patch

from rtp_llm.aios.kmonitor.python_client.kmonitor.kmonitor import MetricTypes
from rtp_llm.aios.kmonitor.python_client.kmonitor.metrics.metric_factory import (
    MetricFactory,
)
from rtp_llm.metrics.kmonitor_metric_reporter import (
    AccMetrics,
    GaugeMetrics,
    MetricReporter,
)


def make_reporter():
    metrics = {}

    def register(kind, name, tags):
        if name in metrics:
            raise AssertionError(f"duplicate registration: {name}")
        metrics[name] = MetricFactory.create_metric(kind, name, tags)
        return metrics[name]

    kmon = Mock()
    kmon.register_acc_metric.side_effect = lambda name, tags: register(
        MetricTypes.ACC_METRIC, name, tags
    )
    kmon.register_gauge_metric.side_effect = lambda name, tags: register(
        MetricTypes.GAUGE_METRIC, name, tags
    )
    return MetricReporter(kmon), metrics


class MetricReporterTest(unittest.TestCase):
    def test_default_registration_is_idempotent(self):
        reporter, metrics = make_reporter()
        reporter.init()
        reporter.init()
        self.assertEqual(set(metrics), {m.value for m in (*AccMetrics, *GaugeMetrics)})
        self.assertEqual(metrics[AccMetrics.VIT_SUCCESS_QPS_METRIC.value].tags, {})

    def test_scoped_registration_preserves_tags_and_empty_selection(self):
        reporter, metrics = make_reporter()
        reporter.init(
            acc_metrics=(),
            gauge_metrics=(GaugeMetrics.VIT_PREPROCESS_RT_METRIC,),
            tags={"dp_rank": "2"},
        )
        reporter.report(GaugeMetrics.VIT_PREPROCESS_RT_METRIC, 12, {"stage": "resize"})
        self.assertEqual(set(metrics), {GaugeMetrics.VIT_PREPROCESS_RT_METRIC.value})
        point = metrics[
            GaugeMetrics.VIT_PREPROCESS_RT_METRIC.value
        ].fetch_reported_data()[0]
        self.assertEqual(point.value, 12)
        self.assertEqual(point.tags, {"dp_rank": "2", "stage": "resize"})

    def test_four_dp_success_series_survive_without_tp_duplicates(self):
        for tp_size in (1, 2):
            with self.subTest(tp_size=tp_size), patch(
                "rtp_llm.aios.kmonitor.python_client.kmonitor.metrics.acc_metric.time.time",
                return_value=10.0,
            ) as clock:
                success_metrics = []
                for world_rank in range(4 * tp_size):
                    reporter, metrics = make_reporter()
                    tp_rank, dp_rank = world_rank % tp_size, world_rank // tp_size
                    reporter.init_backend(
                        world_rank=world_rank, tp_rank=tp_rank, dp_rank=dp_rank
                    )
                    if tp_rank:
                        self.assertEqual(metrics, {})
                        continue
                    self.assertEqual(
                        AccMetrics.QPS_METRIC.value in metrics, world_rank == 0
                    )
                    reporter.report(AccMetrics.VIT_SUCCESS_QPS_METRIC, dp_rank + 1)
                    reporter.report(GaugeMetrics.VIT_IMAGE_COUNT_METRIC, 3)
                    image_point = metrics[
                        GaugeMetrics.VIT_IMAGE_COUNT_METRIC.value
                    ].fetch_reported_data()[0]
                    self.assertEqual(image_point.tags, {"dp_rank": str(dp_rank)})
                    success_metrics.append(
                        metrics[AccMetrics.VIT_SUCCESS_QPS_METRIC.value]
                    )
                clock.return_value = 11.0
                points = [
                    p
                    for metric in success_metrics
                    for p in metric.fetch_reported_data()
                ]
                self.assertEqual(
                    {p.tags["dp_rank"]: p.value for p in points},
                    {"0": 1, "1": 2, "2": 3, "3": 4},
                )
                self.assertEqual(sum(p.value for p in points), 10)


if __name__ == "__main__":
    unittest.main()
