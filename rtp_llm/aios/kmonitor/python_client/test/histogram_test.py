import math
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from unittest import TestCase, main
from unittest.mock import patch

from rtp_llm.aios.kmonitor.python_client.kmonitor.kmonitor import KMonitor
from rtp_llm.aios.kmonitor.python_client.kmonitor.metrics.histogram_metric import (
    HistogramMetric,
)
from rtp_llm.aios.kmonitor.python_client.kmonitor.report_worker import ReportWorker


class HistogramTest(TestCase):
    def test_boundaries_cumulative_snapshots_and_labels(self):
        metric = HistogramMetric("latency", {"model": "test"}, (0, 0.1, 1))
        for value in (0, 0.1, 0.5, 2):
            metric.report(value)
        points = metric.fetch_reported_data()
        self.assertEqual([p.value for p in points], [1, 2, 3, 4, 2.6, 4])
        self.assertEqual(
            [p.tags["le"] for p in points[:4]], ["0.0", "0.1", "1.0", "+Inf"]
        )
        self.assertEqual(
            [p.value for p in metric.fetch_reported_data()], [p.value for p in points]
        )
        metric.report(1, {"model": "other"})
        by_model = {
            p.tags["model"]: p.value
            for p in metric.fetch_reported_data()
            if p.metric_name == "latency_count"
        }
        self.assertEqual(by_model, {"test": 4, "other": 1})
        self.assertEqual(
            len({p.tags["process_instance_id"] for p in metric.fetch_reported_data()}),
            1,
        )
        points[0].tags["model"] = "mutated"
        self.assertEqual(metric.fetch_reported_data()[0].tags["model"], "test")

    def test_invalid_values_and_empty_histogram(self):
        for bounds in ((), (1, 1), (1, 0), (-1,), (math.inf,), (math.nan,)):
            with self.assertRaises(ValueError):
                HistogramMetric("invalid", {}, bounds)
        metric = HistogramMetric("latency", {}, (1,))
        for value in (-1, math.inf, -math.inf, math.nan):
            with self.assertRaises(ValueError):
                metric.report(value)
        for tag in ("le", "process_instance_id"):
            with self.assertRaises(ValueError):
                metric.report(1, {tag: "override"})
        self.assertEqual(metric.fetch_reported_data(), [])

    def test_concurrent_reports(self):
        metric = HistogramMetric("latency", {}, (0.5, 1))

        def observe(_):
            for _ in range(100):
                metric.report(0.5)
                metric.fetch_reported_data()

        with ThreadPoolExecutor(max_workers=4) as pool:
            list(pool.map(observe, range(4)))
        self.assertEqual(
            [p.value for p in metric.fetch_reported_data()], [400, 400, 400, 200, 400]
        )

    def test_registration_and_existing_flume_wire_format(self):
        with patch.object(ReportWorker, "start"), patch.dict(
            os.environ, {"kmonitorTags": ""}
        ), patch(
            "rtp_llm.aios.kmonitor.python_client.kmonitor.report_worker.HippoHelper.is_hippo_env",
            return_value=False,
        ):
            worker = ReportWorker()
        monitor = KMonitor()
        monitor.report_worker = worker
        monitor.default_tags = {}
        gauge = monitor.register_gauge_metric("legacy_rt", {"unit": "ms"})
        histogram = monitor.register_histogram_metric(
            "latency_seconds", (0.1, 1), {"unit": "s"}
        )
        gauge.report(250)
        histogram.report(0.25)
        events = worker.get_report_events()
        self.assertEqual(events[0].body.split()[0:3:2], [b"legacy_rt", b"250"])
        self.assertTrue(all(e.headers == {"topic": "py_kmonitor"} for e in events))
        names = [e.body.split()[0] for e in events[1:]]
        self.assertEqual(
            names,
            [b"latency_seconds_bucket"] * 3
            + [b"latency_seconds_sum", b"latency_seconds_count"],
        )
        self.assertIn(b"le=+Inf", events[3].body)
        self.assertTrue(all(b"process_instance_id=" in e.body for e in events[1:]))
        # Gauge keeps its drain semantics. Histogram snapshots survive reporting
        # and a failed/missed delivery, so the next cycle can recover counts.
        repeated = worker.get_report_events()
        self.assertEqual(len(repeated), 5)
        self.assertEqual(
            [e.body.split()[2] for e in repeated], [b"0", b"1", b"1", b"0.25", b"1"]
        )

    def test_process_series_do_not_collide(self):
        script = """
from rtp_llm.aios.kmonitor.python_client.kmonitor.metrics.histogram_metric import HistogramMetric
m = HistogramMetric("latency", {}, (1,))
m.report(0.5)
print(m.fetch_reported_data()[-1].tags["process_instance_id"])
"""
        ids = [
            subprocess.check_output([sys.executable, "-c", script], text=True).strip()
            for _ in range(2)
        ]
        self.assertNotEqual(ids[0], ids[1])


if __name__ == "__main__":
    main()
