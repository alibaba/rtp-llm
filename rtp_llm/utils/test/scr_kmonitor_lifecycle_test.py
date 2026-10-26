"""Verify deferred metric activation at the service-preparation boundary."""

import importlib
import os
import sys
import unittest
from types import SimpleNamespace
from unittest import mock

from rtp_llm.utils import scr_runtime_fixup as runtime
from rtp_llm.utils import scr_template_utils as scr
from rtp_llm.utils.scr_restore_context import RestoreContext
from rtp_llm.utils.scr_template_lifecycle import CallbackHook, TemplateLifecycle

reporters = importlib.import_module(
    "rtp_llm.aios.kmonitor.python_client.kmonitor.report_worker"
)


class ScrKmonitorLifecycleTest(unittest.TestCase):
    def test_non_scr_hippo_transport_still_starts_eagerly(self):
        with mock.patch.object(
            reporters.HippoHelper, "is_hippo_env", return_value=True
        ), mock.patch.object(
            reporters.HippoHelper,
            "get_hippo_tags",
            return_value={"host_ip": "10.0.0.1"},
        ), mock.patch.object(
            reporters.HippoHelper, "refresh_runtime_identity"
        ) as refresh, mock.patch.object(
            reporters.ReportWorker, "start"
        ) as start, mock.patch.object(
            reporters, "FlumeClient"
        ) as flume_cls, mock.patch.dict(
            os.environ,
            {"HIPPO_SLAVE_IP": "10.0.0.1"},
            clear=True,
        ):
            worker = reporters.ReportWorker()

        self.assertFalse(worker._transport_deferred)
        flume_cls.assert_called_once_with("10.0.0.1", 4141, timeout=1000)
        start.assert_called_once_with()
        refresh.assert_not_called()

    def test_normal_startup_preserves_explicit_metric_tags(self):
        for environment in ({}, {"RTPLLM_ENABLE_SCR": "1", "SCR_PHASE": "serving"}):
            with self.subTest(environment=environment), mock.patch.dict(
                os.environ, environment, clear=True
            ), mock.patch.object(
                reporters.HippoHelper, "is_hippo_env", return_value=False
            ), mock.patch.object(
                reporters.HippoHelper,
                "get_hippo_tags",
                return_value={"container_ip": "10.1.0.1", "hippo_role": "default"},
            ), mock.patch.object(
                reporters.ReportWorker, "start"
            ) as start:
                worker = reporters.ReportWorker()
                point = reporters.MetricDataPoint(
                    2, {"container_ip": "10.1.0.99", "hippo_role": "explicit"}
                )
                message = worker.render_event("normal.metric", 1, point).body.decode()
                self.assertIn("container_ip=10.1.0.99", message)
                self.assertIn("hippo_role=explicit", message)
                worker.start_after_restore()
                start.assert_called_once_with()

    def test_scr_defers_transport_and_activates_with_fresh_identity(self):
        kmonitor = importlib.import_module(
            "rtp_llm.aios.kmonitor.python_client.kmonitor.kmonitor"
        )
        # Each iteration represents a fresh clone of the same pre-start template.
        for pod_ip in ("10.1.0.2", "10.1.0.3"):
            with self.subTest(pod_ip=pod_ip), mock.patch.multiple(
                reporters.HippoHelper,
                host_ip="10.0.0.1",
                container_ip="10.1.0.1",
                role="seed_role",
                app="seed_app",
                group="seed_group",
                app_workdir="",
            ), mock.patch.object(
                reporters.HippoHelper,
                "refresh_runtime_identity",
                wraps=reporters.HippoHelper.refresh_runtime_identity,
            ) as refresh, mock.patch.object(
                runtime, "_runtime_identity", None
            ), mock.patch.dict(
                sys.modules, {"libth_transformer": None}
            ), mock.patch.object(
                reporters, "FlumeClient"
            ) as flume_cls, mock.patch.object(
                reporters.ReportWorker, "start"
            ) as start, mock.patch.dict(
                os.environ,
                {
                    "RTPLLM_ENABLE_SCR": "1",
                    "SCR_PHASE": "checkpoint",
                    "kmonitorTags": "configured^kept",
                },
                clear=True,
            ):
                worker = reporters.ReportWorker()
                with mock.patch.object(kmonitor, "report_worker", worker):
                    monitor = kmonitor.KMonitor({"custom_default": "kept"})
                    retained = monitor.register_gauge_metric(
                        "scr.metric", {"custom_metric": "kept"}
                    )
                tags_ref = worker.init_tags
                # A queued sample and its metric both exist before checkpoint.
                retained.report(1)
                self.assertFalse(worker.started)
                self.assertTrue(worker._transport_deferred)
                flume_cls.assert_not_called()
                start.assert_not_called()

                lifecycle = TemplateLifecycle()
                lifecycle.register(
                    "python",
                    CallbackHook(release=lambda _: worker.start_after_restore()),
                )
                lifecycle.prepare_for_template("seed", "checkpoint")
                runtime.fixup_runtime_after_restore(
                    "seed",
                    lifecycle,
                    restore_env={
                        "RequestedIP": pod_ip,
                        "HIPPO_SLAVE_IP": "10.0.0.2",
                        "HIPPO_ROLE": "restored_role",
                        "HIPPO_APP": "restored_app",
                        "HIPPO_SERVICE_NAME": "restored_group",
                    },
                )
                start.assert_not_called()
                lifecycle.release_template("seed")
                worker.start_after_restore()
                refresh.assert_called_once_with(pod_ip=pod_ip)
                flume_cls.assert_called_once_with("10.0.0.2", 4141, timeout=1000)
                start.assert_called_once_with()
                self.assertIs(worker.init_tags, tags_ref)
                self.assertIs(worker.metrics["scr.metric"], retained)
                self.assertEqual(worker.init_tags["container_ip"], pod_ip)
                self.assertEqual(worker.init_tags["custom_default"], "kept")
                retained.report(2)
                monitor.register_gauge_metric("fresh.metric").report(3)
                events = worker.get_report_events()
                self.assertEqual(len(events), 3)
                for event in events:
                    message = event.body.decode()
                    for tag in (
                        "host_ip=10.0.0.2",
                        f"container_ip={pod_ip}",
                        "hippo_role=restored_role",
                        "hippo_app=restored_app",
                        "hippo_group=restored_group",
                        "custom_default=kept",
                        "configured=kept",
                    ):
                        self.assertIn(tag, message)
                    self.assertNotIn("seed_", message)
                    self.assertNotIn("10.0.0.1", message)
                    self.assertNotIn("10.1.0.1", message)
                    if message.startswith("scr.metric "):
                        self.assertIn("custom_metric=kept", message)
                self.assertEqual(retained.tags["container_ip"], "10.1.0.1")

    def test_removed_runtime_tags_do_not_survive_in_retained_metrics(self):
        old_tags = {"hippo_role": "seed_role", "container_ip": "10.1.0.1"}
        with mock.patch.dict(
            os.environ, {"RTPLLM_ENABLE_SCR": "1", "SCR_PHASE": "restore"}, clear=True
        ), mock.patch.object(
            reporters.HippoHelper, "get_hippo_tags", side_effect=[old_tags, {}]
        ), mock.patch.object(
            reporters.HippoHelper, "is_hippo_env", return_value=False
        ), mock.patch.object(
            reporters.ReportWorker, "start"
        ):
            worker = reporters.ReportWorker()
            worker.init_tags["custom"] = "kept"
            worker.start_after_restore()
            self.assertEqual(worker.init_tags, {"custom": "kept"})
            point = reporters.MetricDataPoint(2, {**old_tags, "custom": "kept"})
            message = worker.render_event("scr.metric", 1, point).body.decode()
            self.assertNotIn("hippo_role=", message)
            self.assertNotIn("container_ip=", message)
            self.assertIn("custom=kept", message)
            self.assertEqual(point.tags["hippo_role"], "seed_role")

    def test_hippo_refresh_uses_explicit_ip_without_scr_state_or_dns(self):
        with mock.patch.multiple(
            reporters.HippoHelper,
            host_ip="",
            container_ip="seed_ip",
            role="",
            app="",
            group="",
            app_workdir="",
        ), mock.patch.dict(
            os.environ,
            {"HIPPO_ROLE": "restored_role", "RequestedIP": "seed_ip"},
            clear=True,
        ), mock.patch.object(
            runtime, "get_restore_runtime_identity"
        ) as read_scr, mock.patch(
            "socket.gethostname"
        ) as hostname, mock.patch(
            "socket.gethostbyname"
        ) as resolve:
            tags = reporters.HippoHelper.refresh_runtime_identity(pod_ip="192.0.2.20")
            self.assertEqual(tags["container_ip"], "192.0.2.20")
            self.assertEqual(tags["hippo_role"], "restored_role")
            read_scr.assert_not_called()
            hostname.assert_not_called()
            resolve.assert_not_called()

    def test_non_hippo_template_also_defers_its_reporting_thread(self):
        with mock.patch.object(
            reporters.HippoHelper, "is_hippo_env", return_value=False
        ), mock.patch.dict(
            os.environ, {"RTPLLM_ENABLE_SCR": "1", "SCR_PHASE": "checkpoint"}
        ), mock.patch.object(
            reporters.ReportWorker, "start"
        ) as start:
            worker = reporters.ReportWorker()
            start.assert_not_called()
            self.assertTrue(worker._transport_deferred)
            worker.start_after_restore()
            worker.start_after_restore()
            start.assert_called_once_with()

    def test_reporters_start_only_after_successful_fixup(self):
        order = []
        native = SimpleNamespace(
            resume_kmonitor_after_scr=lambda: order.append("native_start") or True
        )
        worker = SimpleNamespace(
            start_after_restore=lambda: order.append("python_start")
        )
        with mock.patch.dict(
            sys.modules, {"libth_transformer": native}
        ), mock.patch.object(reporters, "report_worker", worker):
            lifecycle = TemplateLifecycle()
            lifecycle.register(
                "native", CallbackHook(release=scr._start_native_kmonitor)
            )
            lifecycle.register(
                "python", CallbackHook(release=scr._start_python_kmonitor)
            )
            lifecycle.prepare_for_template("failed", "checkpoint")
            lifecycle.abort_template("failed")
            self.assertEqual(order, [])
            lifecycle.prepare_for_template("seed", "checkpoint")
            lifecycle.restore_fixup(RestoreContext("seed", "192.0.2.20"))
            self.assertEqual(order, [])
            lifecycle.release_template("seed")
            self.assertEqual(order, ["native_start", "python_start"])

    def test_native_activation_failure_is_not_silently_ignored(self):
        for native in (
            SimpleNamespace(),
            SimpleNamespace(resume_kmonitor_after_scr=lambda: False),
        ):
            with mock.patch.dict(sys.modules, {"libth_transformer": native}):
                with self.assertRaisesRegex(RuntimeError, "did not start"):
                    scr._start_native_kmonitor("seed")


if __name__ == "__main__":
    unittest.main()
