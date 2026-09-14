"""Verify deferred metric activation at the service-preparation boundary."""

import importlib
import os
import sys
import unittest
from types import SimpleNamespace
from unittest import mock

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

    def test_scr_defers_transport_and_activates_with_fresh_identity(self):
        old_tags = {"host_ip": "10.0.0.1", "container_ip": "10.1.0.1"}
        new_tags = {"host_ip": "10.0.0.2", "container_ip": "10.1.0.2"}
        with mock.patch.object(
            reporters.HippoHelper, "is_hippo_env", return_value=True
        ), mock.patch.object(
            reporters.HippoHelper, "get_hippo_tags", return_value=old_tags
        ), mock.patch.object(
            reporters.HippoHelper,
            "refresh_runtime_identity",
            return_value=new_tags,
        ), mock.patch.object(
            reporters, "FlumeClient"
        ) as flume_cls, mock.patch.dict(
            os.environ,
            {
                "RTPLLM_ENABLE_SCR": "1",
                "SCR_PHASE": "checkpoint",
                "HIPPO_SLAVE_IP": "10.0.0.2",
            },
            clear=False,
        ):
            worker = reporters.ReportWorker()
            self.assertTrue(worker._transport_deferred)
            self.assertFalse(worker.started)
            flume_cls.assert_not_called()

            tags_ref = worker.init_tags
            with mock.patch.object(worker, "start") as start:
                worker.start_after_restore()

            flume_cls.assert_called_once_with("10.0.0.2", 4141, timeout=1000)
            self.assertIs(worker.init_tags, tags_ref)
            self.assertEqual(worker.init_tags["container_ip"], "10.1.0.2")
            start.assert_called_once_with()
            event = worker.render_event(
                "scr.metric",
                1,
                reporters.MetricDataPoint(
                    2,
                    {
                        "host_ip": "10.0.0.1",
                        "container_ip": "10.1.0.1",
                        "custom": "kept",
                    },
                ),
            )
            message = event.body.decode("utf-8")
            self.assertIn("host_ip=10.0.0.2", message)
            self.assertIn("container_ip=10.1.0.2", message)
            self.assertIn("custom=kept", message)
            self.assertNotIn("10.0.0.1", message)
            self.assertNotIn("10.1.0.1", message)

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
