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
            resume_kmonitor_after_scr=lambda: order.append("native_resume"),
        )
        worker = SimpleNamespace(
            pause_for_checkpoint=lambda: order.append("python_pause") or True,
            resume_after_checkpoint=lambda state: order.append("python_resume"),
        )

        def checkpoint(**kwargs):
            self.assertEqual(order, ["native_pause", "python_pause"])
            raise RuntimeError("checkpoint failed")

        with mock.patch.dict(
            sys.modules, {"libth_transformer": native}
        ), mock.patch.object(reporters, "report_worker", worker), mock.patch.dict(
            os.environ, {scr.SCR_ENABLE_ENV: "1"}
        ), mock.patch.object(
            scr,
            "_load_epsilon",
            return_value=SimpleNamespace(
                is_snapstart_enable=lambda: True, snapstart_checkpoint=checkpoint
            ),
        ):
            self.assertIsNone(scr.start_scr_checkpoint(worker_id=0, worker_num=1))
        self.assertEqual(
            order, ["native_pause", "python_pause", "python_resume", "native_resume"]
        )

    def test_native_resumes_if_python_cannot_quiesce(self):
        native = SimpleNamespace(
            pause_kmonitor_for_scr=mock.Mock(return_value=True),
            resume_kmonitor_after_scr=mock.Mock(),
        )
        worker = SimpleNamespace(
            pause_for_checkpoint=mock.Mock(side_effect=RuntimeError("busy"))
        )
        with mock.patch.dict(
            sys.modules, {"libth_transformer": native}
        ), mock.patch.object(reporters, "report_worker", worker):
            with self.assertRaisesRegex(RuntimeError, "busy"):
                scr._pause_kmonitor_for_scr()
        native.resume_kmonitor_after_scr.assert_called_once_with()

    def test_older_native_library_cannot_silently_skip_quiescence(self):
        with mock.patch.dict(sys.modules, {"libth_transformer": SimpleNamespace()}):
            with self.assertRaisesRegex(RuntimeError, "lacks SCR Kmonitor hooks"):
                scr._pause_kmonitor_for_scr()


if __name__ == "__main__":
    unittest.main()
