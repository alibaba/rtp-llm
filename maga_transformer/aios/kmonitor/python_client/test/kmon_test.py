import unittest
import logging

from maga_transformer.aios.kmonitor.python_client.kmonitor.metrics.acc_metric import AccMetric
from maga_transformer.aios.kmonitor.python_client.kmonitor.kmonitor import KMonitor, MetricTypes
from maga_transformer.aios.kmonitor.python_client.kmonitor.report_worker import report_worker
from maga_transformer.aios.kmonitor.python_client.kmonitor.metrics.metric_factory import MetricFactory

class KmonTest(unittest.TestCase):
    def setUp(self) -> None:
        pass

    def tearDown(self):
        pass

    def test_qps_metric(self) -> None:
        metric = MetricFactory.create_metric(MetricTypes.ACC_METRIC, 'test', {})
        metric.report(1, {'key': 'v1'})
        metric.report(1, {'key': 'v2'})
        metric.report(1, {'key': 'v1'})
        metric.report(2, {'key': 'v2'})
        points = metric.fetch_reported_data()
        self.assertEqual(len(points), 3)
        self.assertDictEqual(points[0].tags, {})
        self.assertDictEqual(points[1].tags, {'key': 'v1'})
        self.assertDictEqual(points[2].tags, {'key': 'v2'})
        self.assertAlmostEqual(points[1].value * 1.5, points[2].value)
        report_worker.stop()

    def test_report(self) -> None:
        kmon = KMonitor({
            'tag_a': 'aa'
        })
        metric_name = 'test_metric'
        kmon.register_metric(MetricTypes.GAUGE_METRIC, metric_name)
        kmon.report_metric(metric_name, 2.5)
        kmon.report_metric(metric_name, 3.6)
        kmon.report_metric(metric_name, 4.7)
        kmon.report_metric(metric_name, 5.8)
        self.assertFalse(report_worker.started)
        events = report_worker.get_report_events()
        self.assertEqual(4, len(events))
        for i in range(0, 4):
            print(events[i])
            tokens = events[i].body.decode('utf-8').split(' ')
            self.assertEqual(tokens[0], metric_name)
            self.assertAlmostEqual(float(tokens[2]), 2.5 + 1.1 * i)
            self.assertEqual(tokens[3], 'tag_a=aa')

unittest.main()
