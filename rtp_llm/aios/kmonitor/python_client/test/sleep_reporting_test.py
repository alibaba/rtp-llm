import multiprocessing
import threading
import unittest
from unittest.mock import MagicMock, patch

from rtp_llm.aios.kmonitor.python_client.kmonitor import reporting
from rtp_llm.aios.kmonitor.python_client.kmonitor.metrics.acc_metric import AccMetric
from rtp_llm.aios.kmonitor.python_client.kmonitor.metrics.gauge_metric import (
    GaugeMetric,
)
from rtp_llm.aios.kmonitor.python_client.kmonitor.report_worker import ReportWorker


def _set_rank(state, rank, enabled):
    reporting.configure(state)
    reporting.set_backend_reporting(enabled, rank)


class SleepReportingTest(unittest.TestCase):
    def setUp(self):
        self.state = reporting.ReportingState(2, multiprocessing.get_context("spawn"))
        self.state_patch = patch.object(reporting, "_state", self.state)
        self.state_patch.start()
        self.addCleanup(self.state_patch.stop)

    def test_all_local_ranks_must_finish_before_frontends_switch(self):
        reporting.set_backend_reporting(False, 0)
        self.assertEqual(reporting.reporting_epoch(), 0)
        reporting.set_backend_reporting(False, 1)
        self.assertEqual(reporting.reporting_epoch(), 1)
        reporting.set_backend_reporting(False, 1)
        self.assertEqual(reporting.reporting_epoch(), 1)
        reporting.set_backend_reporting(True, 0)
        self.assertEqual(reporting.reporting_epoch(), 1)
        reporting.set_backend_reporting(True, 1)
        self.assertEqual(reporting.reporting_epoch(), 2)

    def test_spawned_backend_updates_frontend_state(self):
        ctx = multiprocessing.get_context("spawn")
        for rank in range(2):
            child = ctx.Process(target=_set_rank, args=(self.state, rank, False))
            child.start()
            child.join(15)
            if child.is_alive():
                child.terminate()
                child.join()
                self.fail("child failed to publish sleep state")
            self.assertEqual(child.exitcode, 0)
        self.assertEqual(reporting.reporting_epoch(), 1)

    def test_gauge_discards_sleep_samples_and_old_pending_window(self):
        metric = GaugeMetric("load", {})
        metric.report(37)
        self.state.set_enabled(False)
        for _ in range(100):
            metric.report(0)
        self.assertEqual(metric.report_queue, [])
        self.assertEqual(metric.fetch_reported_data(), [])
        self.state.set_enabled(True)
        metric.report(51)
        self.assertEqual([p.value for p in metric.fetch_reported_data()], [51])

    def test_quick_sleep_wake_drops_old_gauge_without_intermediate_poll(self):
        metric = GaugeMetric("load", {})
        metric.report(37)
        self.state.set_enabled(False)
        self.state.set_enabled(True)
        self.assertEqual(metric.fetch_reported_data(), [])

    def test_qps_resets_denominator_and_does_not_emit_idle_zero_during_sleep(self):
        with patch("time.time", return_value=10):
            metric = AccMetric("qps", {})
            metric.report(100)
        self.state.set_enabled(False)
        with patch("time.time", return_value=20):
            metric.report(500)
            self.assertEqual(metric.fetch_reported_data(), [])
        with patch("time.time", return_value=1000):
            self.state.set_enabled(True)
            metric.report(12)
        with patch("time.time", return_value=1002):
            self.assertEqual([p.value for p in metric.fetch_reported_data()], [6])

    def test_qps_first_window_includes_time_before_first_awake_sample(self):
        with patch("time.time", return_value=10):
            metric = AccMetric("qps", {})
            metric.report(100)
        self.state.set_enabled(False)
        with patch("time.time", return_value=1000):
            self.state.set_enabled(True)
        with patch("time.time", return_value=1009):
            metric.report(10)
        with patch("time.time", return_value=1010):
            self.assertEqual([p.value for p in metric.fetch_reported_data()], [1])

    def make_worker(self):
        with patch.object(ReportWorker, "start"), patch(
            "rtp_llm.aios.kmonitor.python_client.kmonitor.report_worker.HippoHelper.is_hippo_env",
            return_value=False,
        ):
            worker = ReportWorker()
        worker.flume_enabled = True
        worker.flume = MagicMock()
        return worker

    def test_sender_closes_and_reconnects_without_replaying_metrics(self):
        worker = self.make_worker()
        old_client = worker.flume
        metric = GaugeMetric("load", {})
        worker.register_metric(metric)
        metric.report(44)
        self.state.set_enabled(False)
        worker.do_report()
        worker.do_report()
        old_client.close.assert_called_once()
        old_client.send_batch.assert_not_called()
        self.state.set_enabled(True)
        metric.report(52)
        new_client = MagicMock()
        with patch.object(worker, "_connect_flume", return_value=new_client):
            worker.do_report()
        events = new_client.send_batch.call_args.args[0]
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].body.decode().split()[2], "52")

    def test_worker_created_while_paused_connects_only_after_wake(self):
        self.state.set_enabled(False)
        client = MagicMock()
        with patch.object(ReportWorker, "start"), patch(
            "rtp_llm.aios.kmonitor.python_client.kmonitor.report_worker.HippoHelper.is_hippo_env",
            return_value=True,
        ), patch.object(ReportWorker, "_connect_flume", return_value=client) as connect:
            worker = ReportWorker()
            metric = GaugeMetric("load", {})
            worker.register_metric(metric)
            metric.report(100)
            worker.do_report()
            worker.do_report()
            connect.assert_not_called()
            self.assertIsNone(worker.flume)
            client.send_batch.assert_not_called()

            self.state.set_enabled(True)
            metric.report(7)
            worker.do_report()
            connect.assert_called_once()
            events = client.send_batch.call_args.args[0]
            self.assertEqual(len(events), 1)
            self.assertEqual(events[0].body.decode().split()[2], "7")

    def test_sleep_fences_connection_during_worker_construction(self):
        entered, release, pausing, paused, produced = (
            threading.Event() for _ in range(5)
        )
        workers, errors = [], []
        client = MagicMock()
        metric = GaugeMetric("load", {})

        def connect():
            entered.set()
            if not release.wait(5):
                raise RuntimeError("connection was not released")
            return client

        def construct():
            try:
                workers.append(ReportWorker())
            except Exception as error:
                errors.append(error)

        def pause():
            pausing.set()
            self.state.set_enabled(False)
            paused.set()

        def produce():
            metric.report(7)
            produced.set()

        constructor = threading.Thread(target=construct, daemon=True)
        sleeper = threading.Thread(target=pause, daemon=True)
        producer = threading.Thread(target=produce, daemon=True)
        with patch.object(ReportWorker, "start"), patch(
            "rtp_llm.aios.kmonitor.python_client.kmonitor.report_worker.HippoHelper.is_hippo_env",
            return_value=True,
        ), patch.object(ReportWorker, "_connect_flume", side_effect=connect):
            constructor.start()
            try:
                self.assertTrue(entered.wait(5))
                sleeper.start()
                self.assertTrue(pausing.wait(5))
                self.assertFalse(paused.wait(0.05))
                producer.start()
                self.assertTrue(produced.wait(1), "producer waited for connection I/O")
            finally:
                release.set()
                for thread in (constructor, sleeper, producer):
                    if thread.ident is not None:
                        thread.join(5)
        self.assertTrue(all(not t.is_alive() for t in (constructor, sleeper, producer)))
        self.assertEqual(errors, [])
        self.assertTrue(paused.is_set())
        worker = workers[0]
        worker.do_report()
        client.close.assert_called_once()
        client.send_batch.assert_not_called()
        self.assertIsNone(worker.flume)

    def test_sleep_fences_an_inflight_send(self):
        worker = self.make_worker()
        entered, release, paused = (
            threading.Event(),
            threading.Event(),
            threading.Event(),
        )

        def send(_):
            entered.set()
            self.assertTrue(release.wait(5))

        worker.flume.send_batch.side_effect = send
        sender = threading.Thread(target=worker.do_report)
        sender.start()
        self.assertTrue(entered.wait(5))

        def pause():
            self.state.set_enabled(False)
            paused.set()

        sleeper = threading.Thread(target=pause)
        sleeper.start()
        self.assertFalse(paused.wait(0.05))
        release.set()
        sender.join(5)
        sleeper.join(5)
        self.assertFalse(sender.is_alive())
        self.assertFalse(sleeper.is_alive())
        self.assertTrue(paused.is_set())

    def test_slow_send_does_not_block_metric_producers(self):
        worker = self.make_worker()
        entered, release, reported = (threading.Event() for _ in range(3))
        metric = GaugeMetric("load", {})
        worker.register_metric(metric)

        def send(_):
            entered.set()
            release.wait(5)

        def produce():
            metric.report(7)
            reported.set()

        worker.flume.send_batch.side_effect = send
        sender = threading.Thread(target=worker.do_report)
        producer = threading.Thread(target=produce)
        sender.start()
        try:
            self.assertTrue(entered.wait(5))
            producer.start()
            self.assertTrue(reported.wait(1), "metric producer waited for network I/O")
        finally:
            release.set()
            sender.join(5)
            if producer.ident is not None:
                producer.join(5)

    def test_slow_render_does_not_block_metric_producers(self):
        for state in (None, self.state):
            with self.subTest(sleep_enabled=state is not None):
                with patch.object(reporting, "_state", state):
                    worker = self.make_worker()
                    metric = GaugeMetric("load", {})
                    worker.register_metric(metric)
                    metric.report(1)
                    entered, release, produced = (threading.Event() for _ in range(3))
                    original_render = worker.render_event

                    def render(*args):
                        entered.set()
                        if not release.wait(5):
                            raise RuntimeError("renderer was not released")
                        return original_render(*args)

                    def produce():
                        metric.report(2)
                        produced.set()

                    sender = threading.Thread(target=worker.do_report, daemon=True)
                    producer = threading.Thread(target=produce, daemon=True)
                    with patch.object(worker, "render_event", side_effect=render):
                        sender.start()
                        try:
                            self.assertTrue(entered.wait(5))
                            producer.start()
                            self.assertTrue(
                                produced.wait(1), "metric producer waited for rendering"
                            )
                        finally:
                            release.set()
                            sender.join(5)
                            if producer.ident is not None:
                                producer.join(5)
                    self.assertFalse(sender.is_alive())
                    self.assertFalse(producer.is_alive())
                    events = worker.flume.send_batch.call_args.args[0]
                    self.assertEqual(
                        [e.body.decode().split()[2] for e in events], ["1"]
                    )
                    worker.do_report()
                    events = worker.flume.send_batch.call_args.args[0]
                    self.assertEqual(
                        [e.body.decode().split()[2] for e in events], ["2"]
                    )

    def test_coordinator_cannot_overwrite_backend_rank_state(self):
        reporting.set_backend_reporting(False, 0)
        reporting.set_backend_reporting(False, 1)
        reporting.set_instance_reporting(True)
        self.assertEqual(list(self.state.rank_enabled), [0, 0])
        self.assertEqual(reporting.reporting_epoch(), 1)

    def test_installation_waits_for_preexisting_metric_producers(self):
        for installation in ("configure", "first_backend_sleep"):
            for metric_type in (GaugeMetric, AccMetric):
                with self.subTest(
                    installation=installation, metric=metric_type.__name__
                ):
                    with patch.object(reporting, "_state", None):
                        metric = metric_type("load", {})
                        worker = self.make_worker()
                        worker.register_metric(metric)
                        entered, release, installing, installed = (
                            threading.Event() for _ in range(4)
                        )
                        errors = []
                        original_sync = metric._sync_reporting_epoch

                        def sync_after_install_attempt():
                            # report() holds its metric lock but has not read the
                            # epoch yet: changing _state here caused ABBA with
                            # the next collector's state -> metric lock order.
                            entered.set()
                            if not release.wait(5):
                                raise RuntimeError("producer was not released")
                            return original_sync()

                        def produce():
                            try:
                                metric.report(7)
                            except Exception as error:
                                errors.append(error)

                        def install():
                            installing.set()
                            try:
                                if installation == "configure":
                                    reporting.configure(reporting.ReportingState())
                                else:
                                    reporting.set_backend_reporting(False, 0)
                            except Exception as error:
                                errors.append(error)
                            finally:
                                installed.set()

                        producer = threading.Thread(target=produce, daemon=True)
                        installer = threading.Thread(target=install, daemon=True)
                        with patch.object(
                            metric, "_sync_reporting_epoch", sync_after_install_attempt
                        ):
                            producer.start()
                            try:
                                self.assertTrue(entered.wait(5))
                                installer.start()
                                self.assertTrue(installing.wait(5))
                                self.assertFalse(installed.wait(0.05))
                                self.assertIsNone(reporting._state)
                            finally:
                                release.set()
                                producer.join(5)
                                if installer.ident is not None:
                                    installer.join(5)
                        self.assertFalse(producer.is_alive())
                        self.assertFalse(installer.is_alive())
                        self.assertEqual(errors, [])
                        worker.do_report()
                        self.assertEqual(
                            metric.reporting_epoch, reporting.reporting_epoch()
                        )

    def test_disabled_sleep_producers_do_not_wait_for_send_or_installation(self):
        with patch.object(reporting, "_state", None):
            worker = self.make_worker()
            metric = GaugeMetric("load", {})
            entered, release, installing, produced = (
                threading.Event() for _ in range(4)
            )

            def send(_):
                entered.set()
                release.wait(5)

            def install():
                installing.set()
                reporting.configure(reporting.ReportingState())

            def produce():
                metric.report(7)
                produced.set()

            worker.flume.send_batch.side_effect = send
            sender = threading.Thread(target=worker.do_report, daemon=True)
            installer = threading.Thread(target=install, daemon=True)
            producer = threading.Thread(target=produce, daemon=True)
            sender.start()
            try:
                self.assertTrue(entered.wait(5))
                installer.start()
                self.assertTrue(installing.wait(5))
                producer.start()
                self.assertTrue(produced.wait(1), "producer waited for network I/O")
                self.assertIsNone(reporting._state)
            finally:
                release.set()
                for thread in (sender, installer, producer):
                    if thread.ident is not None:
                        thread.join(5)
            self.assertTrue(
                all(not t.is_alive() for t in (sender, installer, producer))
            )

    def test_direct_backend_first_sleep_fences_preexisting_sender(self):
        with patch.object(reporting, "_state", None):
            worker = self.make_worker()
            entered, release, paused = (threading.Event() for _ in range(3))

            def send(_):
                entered.set()
                release.wait(5)

            def pause():
                reporting.set_backend_reporting(False, 0)
                paused.set()

            worker.flume.send_batch.side_effect = send
            sender = threading.Thread(target=worker.do_report)
            sleeper = threading.Thread(target=pause)
            sender.start()
            try:
                self.assertTrue(entered.wait(5))
                sleeper.start()
                self.assertFalse(paused.wait(0.05))
            finally:
                release.set()
                sender.join(5)
                if sleeper.ident is not None:
                    sleeper.join(5)
            self.assertTrue(paused.is_set())
            self.assertEqual(reporting.reporting_epoch(), 1)

    def test_disabled_sleep_keeps_normal_idle_qps(self):
        with patch.object(reporting, "_state", None), patch(
            "time.time", return_value=10
        ):
            metric = AccMetric("qps", {})
        with patch.object(reporting, "_state", None), patch(
            "time.time", return_value=11
        ):
            self.assertEqual([p.value for p in metric.fetch_reported_data()], [0])


if __name__ == "__main__":
    unittest.main()
