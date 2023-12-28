from typing import Dict, List
from maga_transformer.aios.kmonitor.python_client.kmonitor.metrics.metric_base import MetricBase, MetricDataPoint
from threading import Lock

class GaugeMetric(MetricBase):
    def __init__(self, *args):
        super(GaugeMetric, self).__init__(*args)
        self.lock: Lock = Lock()
        self.report_queue: List[MetricDataPoint] = []

    def report(self, value: float = 1, tags: Dict[str, str] = {}) -> None:
        report_tags = self.tags.copy()
        report_tags.update(tags)
        data_point = MetricDataPoint(value, report_tags)
        with self.lock:
            self.report_queue.append(data_point)

    def fetch_reported_data(self) -> List[MetricDataPoint]:
        with self.lock:
            report_data = self.report_queue
            self.report_queue = []
        return report_data
