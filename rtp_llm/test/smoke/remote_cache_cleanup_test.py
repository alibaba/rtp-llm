import subprocess
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from smoke.case_runner import CaseRunner
from smoke.remote_kvcm_server import RemoteKVCMServer


class RemoteCacheCleanupTest(unittest.TestCase):
    def setUp(self):
        self.process = subprocess.Popen(
            [sys.executable, "-c", "import time; time.sleep(60)"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        server = object.__new__(RemoteKVCMServer)
        server._server_process = self.process
        server._fault_trigger = False
        logs = tempfile.TemporaryDirectory()
        self.addCleanup(logs.cleanup)
        server._kvcm_src_logs_path = str(Path(logs.name) / "source")
        server._kvcm_dst_logs_path = str(Path(logs.name) / "copied")
        Path(server._kvcm_src_logs_path).mkdir()
        (Path(server._kvcm_src_logs_path) / "server.log").write_text("cache failure")
        self.runner = object.__new__(CaseRunner)
        self.runner.remote_kvcm_server = server

    def tearDown(self):
        if self.process.poll() is None:
            self.process.kill()
        self.process.communicate(timeout=5)

    def assert_server_stopped(self):
        self.assertIsNotNone(self.process.poll(), "remote cache process leaked")
        self.process.communicate(timeout=1)

    @patch("smoke.case_runner.summarize_and_cleanup_coredumps")
    def test_failed_comparison_stops_remote_cache(self, cleanup):
        result = SimpleNamespace(ret=False)
        self.runner._run_impl = Mock(return_value=result)
        self.assertIs(self.runner.run(), result)
        self.assert_server_stopped()
        copied_log = (
            Path(self.runner.remote_kvcm_server._kvcm_dst_logs_path) / "server.log"
        )
        self.assertEqual(copied_log.read_text(), "cache failure")
        cleanup.assert_called_once()

    @patch("smoke.case_runner.summarize_and_cleanup_coredumps")
    def test_query_exception_stops_remote_cache(self, cleanup):
        self.runner._run_impl = Mock(side_effect=RuntimeError("query failed"))
        with self.assertRaisesRegex(RuntimeError, "query failed"):
            self.runner.run()
        self.assert_server_stopped()
        cleanup.assert_called_once()

    def test_repeated_stop_does_not_contact_stopped_server(self):
        server = self.runner.remote_kvcm_server
        server._fault_trigger = True
        server.clearFaults = Mock(return_value=True)
        server.stop_server()
        server.stop_server()
        server.clearFaults.assert_called_once()
        self.assert_server_stopped()

    def test_stop_kills_server_that_ignores_termination(self):
        self.process.kill()
        self.process.communicate(timeout=5)
        self.process = subprocess.Popen(
            [
                sys.executable,
                "-u",
                "-c",
                "import signal, time; signal.signal(signal.SIGTERM, signal.SIG_IGN); "
                "print('ready'); time.sleep(60)",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        self.runner.remote_kvcm_server._server_process = self.process
        self.assertEqual(self.process.stdout.readline(), b"ready\n")
        guard_fired = threading.Event()

        def force_cleanup():
            guard_fired.set()
            self.process.kill()

        guard = threading.Timer(15, force_cleanup)
        guard.start()
        try:
            self.runner.remote_kvcm_server.stop_server()
        finally:
            guard.cancel()
            guard.join()
        self.assertFalse(guard_fired.is_set(), "server shutdown did not time out")
        self.assert_server_stopped()


if __name__ == "__main__":
    unittest.main()
