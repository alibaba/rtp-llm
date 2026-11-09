import contextlib
import contextvars
import time
from dataclasses import dataclass, field
from typing import Dict, Iterator, List, Optional, Union

from rtp_llm.metrics.kmonitor_metric_reporter import AccMetrics, GaugeMetrics


@dataclass
class VitMetricSample:
    metric: Union[AccMetrics, GaugeMetrics]
    value: float
    tags: Dict[str, str] = field(default_factory=dict)


@dataclass
class VitPreprocessMetrics:
    samples: List[VitMetricSample] = field(default_factory=list)

    def report(
        self,
        metric: Union[AccMetrics, GaugeMetrics],
        value: float,
        tags: Optional[Dict[str, str]] = None,
    ) -> None:
        self.samples.append(VitMetricSample(metric, value, tags or {}))


_current_metrics: contextvars.ContextVar[Optional[VitPreprocessMetrics]] = (
    contextvars.ContextVar("vit_preprocess_metrics", default=None)
)


@contextlib.contextmanager
def collect_vit_preprocess_metrics() -> Iterator[VitPreprocessMetrics]:
    metrics = VitPreprocessMetrics()
    token = _current_metrics.set(metrics)
    try:
        yield metrics
    finally:
        _current_metrics.reset(token)


@contextlib.contextmanager
def vit_preprocess_timer(
    metric: GaugeMetrics,
    tags: Optional[Dict[str, str]] = None,
) -> Iterator[None]:
    start_ns = time.monotonic_ns()
    try:
        yield
    finally:
        metrics = _current_metrics.get()
        if metrics is not None:
            metrics.report(metric, (time.monotonic_ns() - start_ns) / 1000.0, tags)


def record_vit_preprocess_value(
    metric: Union[AccMetrics, GaugeMetrics],
    value: Union[int, float],
    tags: Optional[Dict[str, str]] = None,
) -> bool:
    """Return whether the sample was captured for parent-side reporting."""
    metrics = _current_metrics.get()
    if metrics is not None:
        metrics.report(metric, float(value), tags)
        return True
    return False


def video_resized_pixel_count(
    frame_count: int, resized_height: int, resized_width: int
) -> int:
    return int(frame_count) * int(resized_height) * int(resized_width)
