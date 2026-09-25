from threading import Lock
from typing import Dict, List

from rtp_llm.aios.kmonitor.python_client.kmonitor.metrics.metric_base import (
    MetricBase,
    MetricDataPoint,
)
from rtp_llm.aios.kmonitor.python_client.kmonitor.reporting import (
    reporting_epoch,
    reporting_lock,
)


class GaugeMetric(MetricBase):
    def __init__(self, *args):
        super(GaugeMetric, self).__init__(*args)
        self.lock: Lock = Lock()
        self.report_queue: List[MetricDataPoint] = []
        self.reporting_epoch = reporting_epoch()

    def _sync_reporting_epoch(self) -> bool:
        epoch = reporting_epoch()
        if epoch != self.reporting_epoch:
            self.report_queue = []
            self.reporting_epoch = epoch
        return epoch % 2 == 0

    def report(self, value: float = 1, tags: Dict[str, str] = {}) -> None:
        report_tags = self.tags.copy()
        report_tags.update(tags)
        data_point = MetricDataPoint(value, report_tags)
        with reporting_lock(), self.lock:
            if self._sync_reporting_epoch():
                self.report_queue.append(data_point)

    def fetch_reported_data(self) -> List[MetricDataPoint]:
        with reporting_lock(), self.lock:
            self._sync_reporting_epoch()
            report_data = self.report_queue
            self.report_queue = []
        return report_data
