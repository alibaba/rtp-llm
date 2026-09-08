"""Exercise reporter sockets and both SCR lifecycle hooks without a GPU."""

import importlib
import os
import socket
import sys
import unittest
from types import SimpleNamespace
from unittest import mock

from rtp_llm.aios.kmonitor.python_client.flume.pyflume import FlumeClient
from rtp_llm.utils import scr_template_utils as scr

reporters = importlib.import_module(
    "rtp_llm.aios.kmonitor.python_client.kmonitor.report_worker"
)


class ScrKmonitorLifecycleTest(unittest.TestCase):
    def test_native_resume_sees_resolved_ip_with_stale_seed_environment(self):
        observed = []
        native = SimpleNamespace(
            resume_kmonitor_after_scr=lambda: observed.append(
                os.getenv("RequestedIP")
            ) or True
        )
        with mock.patch.dict(
            os.environ,
            {"HIPPO_ROLE": "rtp_role", "RequestedIP": "10.1.0.1"},
        ), mock.patch.object(
            socket, "gethostname", return_value="restored-pod"
        ), mock.patch.object(
            socket, "gethostbyname", return_value="10.1.0.2"
        ) as resolve:
            hook = scr._NativeKmonitorTemplateHook()
            hook._paused = True
            with mock.patch.dict(sys.modules, {"libth_transformer": native}):
                hook.release_template("restored-generation")
            resolve.assert_called_once_with("restored-pod")
            self.assertEqual(observed, ["10.1.0.2"])

    def test_native_resume_still_runs_if_hostname_resolution_fails(self):
        native = SimpleNamespace(resume_kmonitor_after_scr=mock.Mock(return_value=True))
        with mock.patch.dict(
            os.environ,
            {"HIPPO_ROLE": "rtp_role", "RequestedIP": "10.1.0.1"},
        ), mock.patch.object(
            socket, "gethostbyname", side_effect=OSError("resolution failed")
        ):
            hook = scr._NativeKmonitorTemplateHook()
            hook._paused = True
            with mock.patch.dict(sys.modules, {"libth_transformer": native}):
                hook.release_template("restored-generation")
            self.assertEqual(os.getenv("RequestedIP"), "10.1.0.1")
        native.resume_kmonitor_after_scr.assert_called_once_with()

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
                worker.resume_after_checkpoint(False)

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

    def test_live_socket_closes_and_reconnects_without_losing_metrics(self):
        with mock.patch.object(
            reporters.HippoHelper, "is_hippo_env", return_value=False
        ):
            worker = reporters.ReportWorker()
        server = socket.socket()
        server.bind(("127.0.0.1", 0))
        server.listen()
        server.settimeout(2)
        peers = []
        try:
            worker.flume = FlumeClient(*server.getsockname(), timeout=100)
            peer, _ = server.accept()
            peers.append(peer)
            peer.settimeout(2)
            metrics = worker.metrics
            was_started = worker.pause_for_checkpoint()
            self.assertTrue(was_started)
            self.assertFalse(worker._report_thread.is_alive())
            while peer.recv(8192):
                pass
            worker.resume_after_checkpoint(was_started)
            peers.append(server.accept()[0])
            self.assertTrue(worker.started)
            self.assertTrue(worker._report_thread.is_alive())
            self.assertIs(worker.metrics, metrics)
        finally:
            worker.pause_for_checkpoint()
            for peer in peers:
                peer.close()
            server.close()

    def test_both_reporters_resume_when_checkpoint_raises(self):
        order = []
        native = SimpleNamespace(
            pause_kmonitor_for_scr=lambda: order.append("native_pause") or True,
            resume_kmonitor_after_scr=lambda: order.append("native_resume") or True,
        )
        worker = SimpleNamespace(
            pause_for_checkpoint=lambda: order.append("python_pause") or True,
            resume_after_checkpoint=lambda state: order.append("python_resume"),
        )

        with mock.patch.dict(
            sys.modules, {"libth_transformer": native}
        ), mock.patch.object(reporters, "report_worker", worker):
            lifecycle = scr.get_template_lifecycle()
            lifecycle.prepare_for_template("g1", "checkpoint")
            self.assertEqual(order, ["native_pause", "python_pause"])
            lifecycle.abort_template("g1")
        self.assertEqual(
            order, ["native_pause", "python_pause", "python_resume", "native_resume"]
        )

    def test_native_resumes_if_python_cannot_quiesce(self):
        native = SimpleNamespace(
            pause_kmonitor_for_scr=mock.Mock(return_value=True),
            resume_kmonitor_after_scr=mock.Mock(return_value=True),
        )
        worker = SimpleNamespace(
            pause_for_checkpoint=mock.Mock(side_effect=RuntimeError("busy"))
        )
        with mock.patch.dict(
            sys.modules, {"libth_transformer": native}
        ), mock.patch.object(reporters, "report_worker", worker):
            with self.assertRaisesRegex(RuntimeError, "busy"):
                scr.get_template_lifecycle().prepare_for_template("g2", "checkpoint")
        native.resume_kmonitor_after_scr.assert_called_once_with()

    def test_older_native_library_cannot_silently_skip_quiescence(self):
        with mock.patch.dict(sys.modules, {"libth_transformer": SimpleNamespace()}):
            with self.assertRaisesRegex(RuntimeError, "lacks SCR Kmonitor hooks"):
                scr.get_template_lifecycle().prepare_for_template("g3", "checkpoint")


if __name__ == "__main__":
    unittest.main()
