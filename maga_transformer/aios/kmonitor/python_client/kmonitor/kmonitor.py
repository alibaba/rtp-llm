import logging
from threading import Lock
from typing import Dict

from maga_transformer.aios.kmonitor.python_client.kmonitor.report_worker import report_worker
from maga_transformer.aios.kmonitor.python_client.kmonitor.metrics.metric_base import MetricBase
from maga_transformer.aios.kmonitor.python_client.kmonitor.metrics.metric_factory import MetricFactory


class MetricTypes:
    GAUGE_METRIC = "GaugeMetric"
    ACC_METRIC = "AccMetric"

# this is the only class that user should import.
class KMonitor(object):
    def __init__(self, default_tags: Dict[str, str] = {}):
        super(KMonitor, self).__init__()
        self.report_worker = report_worker
        self.default_tags : Dict[str, str] = report_worker.init_tags
        self.default_tags.update(default_tags)
        self.metrics : Dict[str, MetricBase] = {}
        self.lock : Lock = Lock()

    def register_metric(self, type: str, name: str, tags: Dict[str, str] = {}) -> MetricBase:
        metric_tags = self.default_tags.copy()
        metric_tags.update(tags)
        metric = MetricFactory.create_metric(type, name, metric_tags)
        self.report_worker.register_metric(metric)
        with self.lock:
            self.metrics[name] = metric
        logging.debug(f'kmonitor registered metric [{type}][{name}]')
        return metric

    def register_gauge_metric(self, name: str, tags: Dict[str, str] = {}) -> MetricBase:
        return self.register_metric(MetricTypes.GAUGE_METRIC, name, tags)

    def register_acc_metric(self, name: str, tags: Dict[str, str] = {}) -> MetricBase:
        return self.register_metric(MetricTypes.ACC_METRIC, name, tags)

    def report_metric(self, metric_name: str, value: float = 1, extra_tags: Dict[str, str] = {}) -> None:
        with self.lock:
            metric = self.metrics.get(metric_name, None)
        if metric is None:
            raise Exception(f'metric {metric_name} not registered')
        metric.report(value, extra_tags)

    def flush(self) -> None:
        self.report_worker.do_report()
