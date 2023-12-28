from maga_transformer.aios.kmonitor.python_client.kmonitor.metrics.metric_base import MetricBase, MetricDataPoint

from threading import Lock
from typing import Dict, List
import time

# there should be a lock at metric level,
# thus no lock in this class.
class ValueAggregation(object):
    def __init__(self, tags: Dict[str, str]):
        super(ValueAggregation, self).__init__()
        self.acc_value : float = 0
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
        self.last_report_time : float = time.time()
        self.tag_value_map : Dict[int, ValueAggregation] = {
            hash(frozenset(self.tags.items())): ValueAggregation(self.tags)
        }
        self.lock = Lock()

    def report(self, value: float = 1, tags: Dict[str, str] = {}) -> None:
        report_tags = self.tags.copy()
        report_tags.update(tags)
        tag_hash = hash(frozenset(report_tags.items()))

        with self.lock:
            value_agg = self.tag_value_map.get(tag_hash, ValueAggregation(report_tags))
            value_agg.accumulate(value)
            self.tag_value_map[tag_hash] = value_agg

    def fetch_reported_data(self) -> List[MetricDataPoint]:
        data_list = []
        with self.lock:
            current_time = time.time()
            report_time_interval = current_time - self.last_report_time
            self.last_report_time = current_time
            for value_agg in self.tag_value_map.values():
                data_list.append(value_agg.fetch_report_data(report_time_interval))
        return data_list

