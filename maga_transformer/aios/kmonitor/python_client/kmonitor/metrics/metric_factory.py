from maga_transformer.aios.kmonitor.python_client.kmonitor.metrics.metric_base import MetricBase
from maga_transformer.aios.kmonitor.python_client.kmonitor.metrics.gauge_metric import GaugeMetric
from maga_transformer.aios.kmonitor.python_client.kmonitor.metrics.acc_metric import AccMetric

from typing import Dict

class MetricFactory:
    metric_types = [GaugeMetric, AccMetric]
    metric_class_map: dict = { clazz.__name__: clazz for clazz in metric_types}

    @staticmethod
    def create_metric(type: str, name: str, tags: Dict[str, str]) -> MetricBase:
        metric_clazz = MetricFactory.metric_class_map.get(type, None)
        if metric_clazz == None:
            raise Exception(f'unknown metric type {type}. '
                             'available types: {MetricFactory.metric_class_map.keys()}')
        return metric_clazz(name, tags)

