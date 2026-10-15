import os
import signal
import tempfile
import unittest
from unittest.mock import Mock, call, patch

from rtp_llm.test.smoke.remote_kvcm_server import RemoteKVCMServer


class RemoteKVCMServerTest(unittest.TestCase):
    def setUp(self):
        self.server = object.__new__(RemoteKVCMServer)
        self.process = Mock(pid=4242)
        self.server._server_process = self.process
        self.server._fault_trigger = False
        killpg = patch("rtp_llm.test.smoke.remote_kvcm_server.os.killpg")
        self.killpg = killpg.start()
        self.addCleanup(killpg.stop)

    def test_start_creates_a_private_process_group(self):
        with (
            tempfile.TemporaryDirectory() as directory,
            patch("rtp_llm.test.smoke.remote_kvcm_server.PortManager") as ports,
            patch("rtp_llm.test.smoke.remote_kvcm_server.subprocess.Popen") as popen,
            patch.object(RemoteKVCMServer, "wait_sever_done", return_value=False),
            patch.object(RemoteKVCMServer, "stop_server") as stop,
            patch.dict(os.environ),
        ):
            ports.return_value.get_consecutive_ports.return_value = (
                [10001, 10002, 10003, 10004],
                [],
            )
            server = RemoteKVCMServer(directory, {}, directory, directory)
            self.assertFalse(server.start_server())
        self.assertTrue(popen.call_args.kwargs["start_new_session"])
        stop.assert_called_once_with()

    def test_exited_manager_does_not_skip_surviving_children(self):
        self.process.poll.return_value = 0
        self.server._fault_trigger = True
        self.server.clearFaults = Mock()
        with (
            patch(
                "rtp_llm.test.smoke.remote_kvcm_server.time.monotonic",
                side_effect=[0, 0, 5],
            ),
            patch("rtp_llm.test.smoke.remote_kvcm_server.time.sleep"),
        ):
            self.server.stop_server()
        self.assertEqual(
            self.killpg.call_args_list,
            [call(4242, signal.SIGTERM), call(4242, 0), call(4242, signal.SIGKILL)],
        )
        self.server.clearFaults.assert_not_called()
        self.process.wait.assert_called_once_with(timeout=5)
        self.assertIsNone(self.server._server_process)

    def test_graceful_group_exit_does_not_send_sigkill(self):
        self.process.poll.return_value = None
        self.killpg.side_effect = [None, ProcessLookupError()]
        with patch("rtp_llm.test.smoke.remote_kvcm_server.time.monotonic", return_value=0):
            self.server.stop_server()
        self.assertEqual(
            self.killpg.call_args_list,
            [call(4242, signal.SIGTERM), call(4242, 0)],
        )
        self.process.wait.assert_called_once_with(timeout=5)
        self.assertIsNone(self.server._server_process)

    def test_missing_group_and_repeated_stop_are_harmless(self):
        self.killpg.side_effect = ProcessLookupError
        self.server.stop_server()
        self.server.stop_server()
        self.killpg.assert_called_once_with(4242, signal.SIGTERM)
        self.process.wait.assert_called_once_with(timeout=5)
        self.assertIsNone(self.server._server_process)

    def test_fault_cleanup_failure_still_stops_the_group(self):
        self.server._fault_trigger = True
        self.process.poll.return_value = None
        self.server.clearFaults = Mock(side_effect=RuntimeError("manager unavailable"))
        self.killpg.side_effect = ProcessLookupError
        with self.assertLogs(level="ERROR"):
            self.server.stop_server()
        self.killpg.assert_called_once_with(4242, signal.SIGTERM)
        self.process.wait.assert_called_once_with(timeout=5)
        self.assertIsNone(self.server._server_process)

    def test_failed_signal_preserves_the_process_for_cleanup_retry(self):
        self.killpg.side_effect = [PermissionError(), ProcessLookupError()]
        with self.assertLogs(level="ERROR"):
            self.server.stop_server()
        self.assertIs(self.server._server_process, self.process)
        self.process.wait.assert_not_called()
        self.server.stop_server()
        self.assertIsNone(self.server._server_process)
        self.process.wait.assert_called_once_with(timeout=5)


if __name__ == "__main__":
    unittest.main()
