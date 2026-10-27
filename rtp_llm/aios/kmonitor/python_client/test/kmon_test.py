import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from typing import List
from unittest.mock import patch

from rtp_llm.aios.kmonitor.python_client.kmonitor import (
    report_worker as report_worker_module,
)
from rtp_llm.aios.kmonitor.python_client.kmonitor.kmonitor import KMonitor, MetricTypes
from rtp_llm.aios.kmonitor.python_client.kmonitor.metrics.metric_base import (
    MetricBase,
    MetricDataPoint,
)
from rtp_llm.aios.kmonitor.python_client.kmonitor.metrics.metric_factory import (
    MetricFactory,
)
from rtp_llm.aios.kmonitor.python_client.kmonitor.report_worker import (
    ReportWorker,
    report_worker,
)


class UnprintableValue:
    def __str__(self) -> str:
        raise ValueError("unprintable value")


class FixedPointsMetric(MetricBase):
    """Returns the same data points every cycle, like a stuck bad series."""

    def __init__(self, name: str, points: List[MetricDataPoint]):
        super().__init__(name, {})
        self.points = points

    def report(self, value: float = 1, tags={}) -> None:
        pass

    def fetch_reported_data(self) -> List[MetricDataPoint]:
        return self.points


def _make_idle_worker() -> ReportWorker:
    # Build the worker without its background thread so tests drive it directly.
    with patch.object(ReportWorker, "start"):
        return ReportWorker()


class KmonTest(unittest.TestCase):
    def setUp(self) -> None:
        pass

    def tearDown(self):
        pass

    def test_qps_metric(self) -> None:
        metric = MetricFactory.create_metric(MetricTypes.ACC_METRIC, "test", {})
        metric.report(1, {"key": "v1"})
        metric.report(1, {"key": "v2"})
        metric.report(1, {"key": "v1"})
        metric.report(2, {"key": "v2"})
        points = metric.fetch_reported_data()
        self.assertEqual(len(points), 3)
        self.assertDictEqual(points[0].tags, {})
        self.assertDictEqual(points[1].tags, {"key": "v1"})
        self.assertDictEqual(points[2].tags, {"key": "v2"})
        self.assertAlmostEqual(points[1].value * 1.5, points[2].value)
        report_worker.stop()

    def test_request_tags_are_normalized(self) -> None:
        metric = MetricFactory.create_metric(
            MetricTypes.ACC_METRIC, "test_tags", {"app": "a", "empty": ""}
        )
        metric.report(1, {"source": None})
        metric.report(1, {"source": 7})
        metric.report(1, {"source": " tpm gateway=1 "})
        metric.report(1, {"source": ""})
        metric.report(1, {"source": {"request_id": "r1"}})
        metric.report(1, {"source": ["a"]})
        metric.report(1, {"source": object()})
        metric.report(1, {"source": "bad\ud800"})
        points = metric.fetch_reported_data()
        # The pre-created default-tag series comes first and has no source.
        self.assertNotIn("source", points[0].tags)
        sources = [point.tags["source"] for point in points[1:]]
        # None, "", dict, list and objects all collapse into one "unknown" series.
        self.assertEqual(sources, ["unknown", "7", "tpm_gateway_1", "bad?"])
        for point in points:
            self.assertEqual(point.tags["app"], "a")
            self.assertEqual(point.tags["empty"], "")

    def test_render_event_accepts_non_string_tags(self) -> None:
        event = report_worker.render_event(
            "m", 1, MetricDataPoint(1.0, {"a": None, "b": 2, "c": "x y", "d": "\udc80"})
        )
        self.assertEqual(event.body.decode("utf-8"), "m 1 1.0 a=unknown b=2 c=x_y d=?")

    def test_bad_point_does_not_hide_other_series(self) -> None:
        worker = _make_idle_worker()
        good = MetricDataPoint(1.0, {"source": "a"})
        bad = MetricDataPoint(UnprintableValue(), {"source": "b"})
        tail = MetricDataPoint(2.0, {"source": "c"})
        worker.register_metric(FixedPointsMetric("m_bad", [good, bad, tail]))
        worker.register_metric(FixedPointsMetric("m_next", [good]))
        with patch.object(report_worker_module.logging, "error") as log_error:
            for _ in range(3):
                bodies = [
                    event.body.decode("utf-8") for event in worker.get_report_events()
                ]
                self.assertEqual(
                    [body.split(" ", 2)[0] for body in bodies],
                    ["m_bad", "m_bad", "m_next"],
                )
                self.assertTrue(bodies[0].endswith(" 1.0 source=a"))
                self.assertTrue(bodies[1].endswith(" 2.0 source=c"))
        # Errors repeat every cycle but produce one log record per interval.
        self.assertEqual(log_error.call_count, 1)

    def test_error_log_limit_is_shared_by_concurrent_callers(self) -> None:
        worker = _make_idle_worker()
        start = threading.Barrier(8)

        def log_error():
            start.wait(timeout=10)
            worker.log_error_limited("report or flush failed")

        def record_log(message):
            # Logging must not hold the timestamp reservation lock.
            self.assertTrue(worker.error_log_lock.acquire(timeout=1))
            worker.error_log_lock.release()

        with (
            patch.object(
                report_worker_module.time, "monotonic", return_value=100.0
            ) as now,
            patch.object(
                report_worker_module.logging, "error", side_effect=record_log
            ) as log,
            # get_report_events() can call the limiter while holding this lock.
            worker.metric_lock,
            ThreadPoolExecutor(max_workers=8) as pool,
        ):
            calls = [pool.submit(log_error) for _ in range(8)]
            for call in calls:
                call.result(timeout=10)
            self.assertEqual(log.call_count, 1)
            now.return_value = 159.0
            worker.log_error_limited("still failing")
            self.assertEqual(log.call_count, 1)
            now.return_value = 160.0
            worker.log_error_limited("next interval")
            self.assertEqual(log.call_count, 2)

    def test_surrogate_source_keeps_reporting_over_cycles(self) -> None:
        worker = _make_idle_worker()
        metric = MetricFactory.create_metric(MetricTypes.ACC_METRIC, "m_qps", {})
        worker.register_metric(metric)
        for _ in range(3):
            metric.report(1, {"source": "\ud800"})
            metric.report(1, {"source": "ok"})
            events = worker.get_report_events()
            tags = [event.body.decode("utf-8").split(" ")[3:] for event in events]
            # The default-tag series has no tags, so its line ends with a space.
            self.assertEqual(tags, [[""], ["source=?"], ["source=ok"]])

    def test_report_cycle_survives_errors(self) -> None:
        worker = _make_idle_worker()
        calls = []

        def do_report():
            calls.append(1)
            if len(calls) == 1:
                raise TypeError("bad tag")
            if len(calls) == 3:
                worker.started = False

        worker.started = True
        with (
            patch.object(
                report_worker_module, "_ReportWorker__REPORT_INTERVAL_SECOND", 0
            ),
            patch.object(worker, "do_report", side_effect=do_report),
            patch.object(report_worker_module.logging, "error") as log_error,
        ):
            worker.report_cycle()
        self.assertEqual(len(calls), 3)
        self.assertEqual(log_error.call_count, 1)

    def test_report_thread_survives_errors(self) -> None:
        worker = _make_idle_worker()
        calls = []
        reported_after_error = threading.Event()

        def do_report():
            calls.append(1)
            if len(calls) <= 2:
                raise TypeError("bad tag")
            reported_after_error.set()

        with (
            patch.object(
                report_worker_module, "_ReportWorker__REPORT_INTERVAL_SECOND", 0.001
            ),
            patch.object(worker, "do_report", side_effect=do_report),
            patch.object(report_worker_module.logging, "error"),
        ):
            worker.start()
            try:
                self.assertTrue(reported_after_error.wait(timeout=10))
                self.assertTrue(worker.report_thread.is_alive())
            finally:
                worker.stop()
                worker.report_thread.join(timeout=10)
        self.assertFalse(worker.report_thread.is_alive())
        self.assertGreaterEqual(len(calls), 3)

    def test_report(self) -> None:
        kmon = KMonitor({"tag_a": "aa"})
        metric_name = "test_metric"
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
            tokens = events[i].body.decode("utf-8").split(" ")
            self.assertEqual(tokens[0], metric_name)
            self.assertAlmostEqual(float(tokens[2]), 2.5 + 1.1 * i)
            self.assertEqual(tokens[3], "tag_a=aa")


unittest.main()
