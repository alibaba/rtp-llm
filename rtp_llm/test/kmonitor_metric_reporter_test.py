import multiprocessing
import os
import tempfile
import threading
import unittest
from unittest.mock import Mock, patch

from rtp_llm.aios.kmonitor.python_client.kmonitor.kmonitor import KMonitor
from rtp_llm.aios.kmonitor.python_client.kmonitor.report_worker import ReportWorker
from rtp_llm.metrics.kmonitor_metric_reporter import (
    SERVICE_STATUS_METRIC,
    SERVICE_STATUS_TAG,
    STARTUP_WARMUP_HEALTH_GATE_FILE_ENV,
    GaugeMetrics,
    MetricReporter,
)


def _observe_draining(event, connection):
    fixture = MetricReporterServingStateTest()
    fixture.setUp()
    try:
        fixture.reporter.bind_service_draining(event)
        fixture.reporter.start_serving_when_ready()
        connection.send(fixture.reporter.is_serving)
        if not event.wait(10):
            raise AssertionError("missing draining notification")
        fixture.worker.do_report()
        connection.send(
            (fixture.reporter.is_serving, fixture.events(SERVICE_STATUS_METRIC)[-1])
        )
        connection.recv()  # Remain alive while the frontend drains requests.
    finally:
        fixture.doCleanups()
        connection.close()


class MetricReporterServingStateTest(unittest.TestCase):
    def setUp(self):
        self.env = patch.dict(os.environ, {}, clear=False)
        self.env.start()
        self.addCleanup(self.env.stop)
        os.environ.pop(STARTUP_WARMUP_HEALTH_GATE_FILE_ENV, None)
        # Use the real metric queues and rendering, but drive reporting cycles
        # explicitly so the tests need neither Flume nor a background thread.
        self.worker = ReportWorker.__new__(ReportWorker)
        self.worker.metrics = {}
        self.worker.metric_lock = threading.Lock()
        self.worker.report_lock = threading.Lock()
        self.worker.before_report = None
        self.worker.flume = Mock()
        kmon = KMonitor.__new__(KMonitor)
        kmon.report_worker = self.worker
        kmon.default_tags = {}
        kmon.metrics = {}
        kmon.lock = threading.Lock()
        self.reporter = MetricReporter(kmon)
        self.reporter.init()

    def events(self, metric_name):
        return [
            event.body.decode()
            for call in self.worker.flume.send_batch.call_args_list
            for event in call.args[0]
            if event.body.decode().split()[0] == metric_name
        ]

    def test_startup_suppresses_even_zero_accumulator_samples(self):
        self.reporter.report(GaugeMetrics.LANTENCY_METRIC, 123)
        self.worker.do_report()
        self.worker.flume.send_batch.assert_not_called()
        self.reporter.start_serving_when_ready()
        self.assertEqual(self.events(GaugeMetrics.LANTENCY_METRIC.value), [])
        self.assertTrue(self.reporter.is_serving)
        for call in self.worker.flume.send_batch.call_args_list:
            for event in call.args[0]:
                self.assertIn(b"is_serving=true", event.body)

    def test_heartbeat_repeats_while_idle_and_draining(self):
        self.reporter.start_serving_when_ready()
        self.worker.do_report()
        self.worker.do_report()
        self.assertEqual(len(self.events(SERVICE_STATUS_METRIC)), 3)
        self.assertTrue(
            all(" 1.0 is_serving=true" in x for x in self.events(SERVICE_STATUS_METRIC))
        )
        self.worker.flume.reset_mock()
        self.reporter.set_serving(False)
        self.worker.do_report()
        self.assertEqual(len(self.events(SERVICE_STATUS_METRIC)), 2)
        self.assertTrue(
            all(
                " 0.0 is_serving=false" in x for x in self.events(SERVICE_STATUS_METRIC)
            )
        )

    def test_shutdown_tags_all_metrics_and_cannot_restart(self):
        self.reporter.start_serving_when_ready()
        self.worker.flume.reset_mock()
        tags = {"source": "test", SERVICE_STATUS_TAG: "stale"}
        self.reporter.report(GaugeMetrics.LANTENCY_METRIC, 7, tags)
        self.reporter.set_serving(False)
        self.reporter.start_serving_when_ready()
        self.assertFalse(self.reporter.is_serving)
        self.assertEqual(tags[SERVICE_STATUS_TAG], "stale")
        self.assertIn("source=test", self.events(GaugeMetrics.LANTENCY_METRIC.value)[0])
        for call in self.worker.flume.send_batch.call_args_list:
            for event in call.args[0]:
                self.assertIn(b"is_serving=false", event.body)

    def test_gate_drops_warmup_samples_before_readiness(self):
        with tempfile.TemporaryDirectory() as tmp:
            gate = os.path.join(tmp, "ready")
            os.environ[STARTUP_WARMUP_HEALTH_GATE_FILE_ENV] = gate
            self.reporter.start_serving_when_ready()
            self.reporter.report(GaugeMetrics.LANTENCY_METRIC, 123)
            self.worker.do_report()
            self.worker.flume.send_batch.assert_not_called()
            open(gate, "w").close()
            self.worker.do_report()
            self.assertTrue(self.reporter.is_serving)
            self.assertEqual(self.events(GaugeMetrics.LANTENCY_METRIC.value), [])
            # Once ready, file I/O leaves the metric and heartbeat hot paths.
            with patch("os.path.exists", side_effect=AssertionError("gate rechecked")):
                self.reporter.report(GaugeMetrics.LANTENCY_METRIC, 5)
                self.worker.do_report()

    def test_shutdown_before_gate_open_cannot_be_overwritten(self):
        with tempfile.TemporaryDirectory() as tmp:
            gate = os.path.join(tmp, "ready")
            os.environ[STARTUP_WARMUP_HEALTH_GATE_FILE_ENV] = gate
            self.reporter.start_serving_when_ready()
            self.reporter.set_serving(False)
            open(gate, "w").close()
            self.worker.do_report()
            self.reporter.set_serving(True)
            self.assertFalse(self.reporter.is_serving)
            self.worker.flume.send_batch.assert_not_called()

    def test_shared_event_updates_all_backends_without_terminating_them(self):
        ctx = multiprocessing.get_context("spawn")
        event = ctx.Event()
        self.reporter.bind_service_draining(event)
        children = []
        try:
            for _ in range(2):
                parent, child = ctx.Pipe()
                process = ctx.Process(target=_observe_draining, args=(event, child))
                process.start()
                child.close()
                children.append((process, parent))
                self.assertTrue(parent.poll(10))
                self.assertTrue(parent.recv())
            self.reporter.start_serving_when_ready()
            self.reporter.set_serving(False)
            self.reporter.set_serving(False)  # Duplicate notifications are harmless.
            for process, parent in children:
                self.assertTrue(parent.poll(10))
                serving, sample = parent.recv()
                self.assertFalse(serving)
                self.assertIn(" 0.0 is_serving=false", sample)
                self.assertTrue(process.is_alive())
                parent.send("exit")
                process.join(10)
                self.assertEqual(process.exitcode, 0)
        finally:
            for process, parent in children:
                if process.is_alive():
                    process.kill()
                process.join(10)
                parent.close()

    def test_notification_before_startup_is_retained(self):
        event = multiprocessing.get_context("spawn").Event()
        event.set()
        self.reporter.bind_service_draining(event)
        self.reporter.start_serving_when_ready()
        self.reporter.set_serving(True)
        self.worker.do_report()
        self.assertFalse(self.reporter.is_serving)
        self.worker.flume.send_batch.assert_not_called()

    def test_shutdown_racing_with_gate_check_remains_terminal(self):
        os.environ[STARTUP_WARMUP_HEALTH_GATE_FILE_ENV] = "/test/warmup_gate"
        with patch("os.path.exists", return_value=False):
            self.reporter.start_serving_when_ready()
        entered = threading.Event()
        proceed = threading.Event()

        def gate_ready(_):
            entered.set()
            if not proceed.wait(5):
                raise AssertionError("test did not release gate check")
            return True

        with patch("os.path.exists", side_effect=gate_ready):
            tick = threading.Thread(target=self.worker.do_report)
            tick.start()
            try:
                self.assertTrue(entered.wait(5))
                shutdown = threading.Thread(
                    target=lambda: self.reporter.set_serving(False)
                )
                shutdown.start()
            finally:
                proceed.set()
                tick.join(5)
            shutdown.join(5)
        self.assertFalse(tick.is_alive())
        self.assertFalse(shutdown.is_alive())
        self.assertFalse(self.reporter.is_serving)
        self.assertIn(" 0.0 is_serving=false", self.events(SERVICE_STATUS_METRIC)[-1])


if __name__ == "__main__":
    unittest.main()
