import os
import json
import signal
import tempfile
import unittest
from unittest.mock import MagicMock, Mock, call, patch

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

    def test_pace_requires_an_external_fixture(self):
        with (
            tempfile.TemporaryDirectory() as directory,
            patch("rtp_llm.test.smoke.remote_kvcm_server.PortManager") as ports,
            patch.dict(os.environ, {"KVCM_PACE_FIXTURE": ""}),
            self.assertRaisesRegex(RuntimeError, "KVCM_PACE_FIXTURE"),
        ):
            RemoteKVCMServer(directory, {"PACE_REQUIRED": "true"}, directory, directory + "/output")
        ports.assert_not_called()

    def test_pace_uses_the_supplied_fixture(self):
        with (
            tempfile.TemporaryDirectory() as directory,
            patch("rtp_llm.test.smoke.remote_kvcm_server.PortManager") as ports,
            patch.dict(os.environ),
        ):
            ports.return_value.get_consecutive_ports.return_value = ([10001, 10002, 10003, 10004], [])
            path = os.path.join(directory, "fixture.json")
            with open(path, "w") as output:
                json.dump({"domain": "http://127.0.0.1:12348", "provider_status_urls": ["http://127.0.0.1:12346/api/status"]}, output)
            os.environ["KVCM_PACE_FIXTURE"] = path
            os.environ["TEST_TMPDIR"] = directory
            server = RemoteKVCMServer(directory, {"PACE_REQUIRED": "true"}, directory, directory + "/output")
            self.assertEqual(server.pace_fixture.config["domain"], "http://127.0.0.1:12348")
            server.close()
            server.copy_logs()
            self.assertIsNone(server._work_dir)

    def test_cleanup_failure_does_not_replace_the_original_test_error(self):
        lock = MagicMock()
        self.server._locks = [lock]
        with (
            patch.object(self.server, "stop_server", side_effect=RuntimeError("cleanup failed")),
            self.assertLogs(level="ERROR"),
            self.assertRaisesRegex(ValueError, "model failed"),
        ):
            try:
                raise ValueError("model failed")
            finally:
                self.server.close()
        lock.__exit__.assert_called_once_with(None, None, None)

    def test_non_pace_close_preserves_existing_failed_stop_behavior(self):
        self.server._locks = []
        with patch.object(self.server, "stop_server"):
            self.server.close()

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
