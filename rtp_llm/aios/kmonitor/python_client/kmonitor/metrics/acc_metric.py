import time
from threading import Lock
from typing import Dict, List

from rtp_llm.aios.kmonitor.python_client.kmonitor.metrics.metric_base import (
    MetricBase,
    MetricDataPoint,
)
from rtp_llm.aios.kmonitor.python_client.kmonitor.reporting import (
    reporting_epoch,
    reporting_lock,
    reporting_resume_time,
)


# there should be a lock at metric level,
# thus no lock in this class.
class ValueAggregation(object):
    def __init__(self, tags: Dict[str, str]):
        super(ValueAggregation, self).__init__()
        self.acc_value: float = 0
        self.tags = tags

    def accumulate(self, value: float) -> None:
        self.acc_value += value

    def fetch_report_data(self, time_interval: float) -> MetricDataPoint:
        value = self.acc_value / time_interval
        self.acc_value = 0
        return MetricDataPoint(value, self.tags)


class AccMetric(MetricBase):
    def __init__(self, *args):
        super(AccMetric, self).__init__(*args)
        self.last_report_time: float = time.time()
        self.tag_value_map: Dict[int, ValueAggregation] = {
            hash(frozenset(self.tags.items())): ValueAggregation(self.tags)
        }
        self.lock = Lock()
        self.reporting_epoch = reporting_epoch()

    def _sync_reporting_epoch(self) -> bool:
        epoch = reporting_epoch()
        if epoch != self.reporting_epoch:
            for value in self.tag_value_map.values():
                value.acc_value = 0
            # A late first sample still belongs to the entire awake window.
            self.last_report_time = (
                reporting_resume_time() if epoch % 2 == 0 else time.time()
            )
            self.reporting_epoch = epoch
        return epoch % 2 == 0

    def report(self, value: float = 1, tags: Dict[str, str] = {}) -> None:
        report_tags = self.tags.copy()
        report_tags.update(tags)
        tag_hash = hash(frozenset(report_tags.items()))

        with reporting_lock(), self.lock:
            if not self._sync_reporting_epoch():
                return
            value_agg = self.tag_value_map.get(tag_hash, ValueAggregation(report_tags))
            value_agg.accumulate(value)
            self.tag_value_map[tag_hash] = value_agg

    def fetch_reported_data(self) -> List[MetricDataPoint]:
        data_list = []
        with reporting_lock(), self.lock:
            if not self._sync_reporting_epoch():
                return []
            current_time = time.time()
            report_time_interval = current_time - self.last_report_time
            if report_time_interval <= 0:
                return []
            self.last_report_time = current_time
            for value_agg in self.tag_value_map.values():
                data_list.append(value_agg.fetch_report_data(report_time_interval))
        return data_list
