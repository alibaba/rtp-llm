__all__ = ["GaugeMetrics", "AccMetrics", "kmonitor"]

from rtp_llm.aios.kmonitor.python_client.kmonitor.kmonitor import KMonitor

from .kmonitor_metric_reporter import AccMetrics, GaugeMetrics, MetricReporter

_kmonitor = KMonitor()
kmonitor = MetricReporter(_kmonitor)
