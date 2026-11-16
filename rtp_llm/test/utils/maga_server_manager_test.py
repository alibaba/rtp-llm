import signal
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from unittest.mock import Mock, mock_open, patch

from rtp_llm.test.utils.maga_server_manager import MagaServerManager


class MagaServerManagerTest(unittest.TestCase):
    def setUp(self):
        self.manager = MagaServerManager(port="18088", health_check_path="/ready")
        self.manager.print_process_log = Mock()

    def tearDown(self):
        with self.manager._state_lock:
            self.manager._server_process = None
        self.manager.stop_server()

    def test_missing_process_does_not_probe_or_report_timeout(self):
        with (
            patch("rtp_llm.utils.util.wait_sever_done") as health_check,
            self.assertLogs(level="WARNING") as logs,
        ):
            self.assertFalse(self.manager.wait_sever_done(timeout=17))

        health_check.assert_not_called()
        self.assertIsNone(self.manager.exit_code)
        self.assertIsNone(self.manager.server_pid)
        self.assertIn("Server process is unavailable", "\n".join(logs.output))
        self.assertNotIn("still alive", "\n".join(logs.output))
        self.manager.print_process_log.assert_called_once_with()

    def test_success_uses_configured_health_check(self):
        process = Mock(pid=4242)
        self.manager._server_process = process
        with patch(
            "rtp_llm.utils.util.wait_sever_done", return_value=True
        ) as health_check:
            self.assertTrue(self.manager.wait_sever_done(timeout=17))

        health_check.assert_called_once_with(process, 18088, 17, "/ready")
        process.poll.assert_not_called()
        self.manager.print_process_log.assert_not_called()
        self.assertEqual(self.manager.server_pid, 4242)

    def test_failed_health_check_reports_exit_status(self):
        cases = [
            (0, "exited with code 0"),
            (3, "exited with code 3"),
            (-signal.SIGKILL, "killed by SIGKILL"),
            (-999, "killed by signal 999"),
            (None, "still alive, health check timed out after 17s"),
        ]
        for exit_code, expected_reason in cases:
            with self.subTest(exit_code=exit_code):
                process = Mock(pid=4242)
                process.poll.return_value = exit_code
                self.manager._server_process = process
                self.manager.print_process_log.reset_mock()
                with (
                    patch("rtp_llm.utils.util.wait_sever_done", return_value=False),
                    self.assertLogs(level="WARNING") as logs,
                ):
                    self.assertFalse(self.manager.wait_sever_done(timeout=17))

                process.poll.assert_called_once_with()
                self.assertEqual(self.manager.exit_code, exit_code)
                self.assertIn(f"pid=4242 {expected_reason}", "\n".join(logs.output))
                self.manager.print_process_log.assert_called_once_with()

    def test_concurrent_stop_preserves_probed_process_diagnostics(self):
        process = Mock(pid=4242)
        process.poll.return_value = -signal.SIGTERM
        self.manager._server_process = process
        health_started = threading.Event()
        release_health = threading.Event()

        def health_check(probed_process, port, timeout, path):
            self.assertIs(probed_process, process)
            health_started.set()
            if not release_health.wait(timeout=10):
                raise TimeoutError("test did not release the health check")
            return False

        parent = Mock()
        parent.children.return_value = []
        with (
            patch("rtp_llm.utils.util.wait_sever_done", side_effect=health_check),
            patch(
                "rtp_llm.test.utils.maga_server_manager.psutil.Process",
                return_value=parent,
            ),
            patch(
                "rtp_llm.test.utils.maga_server_manager.psutil.wait_procs",
                return_value=([], []),
            ),
            self.assertLogs(level="WARNING") as logs,
            ThreadPoolExecutor(max_workers=2) as workers,
        ):
            wait_result = workers.submit(self.manager.wait_sever_done, timeout=17)
            try:
                self.assertTrue(health_started.wait(timeout=5))
                stop_result = workers.submit(self.manager.stop_server)
                # Stop must complete while health is blocked, without holding its lock.
                self.assertTrue(stop_result.result(timeout=5))
                self.assertIsNone(self.manager.server_pid)
                parent.terminate.assert_called_once_with()
            finally:
                release_health.set()
            self.assertFalse(wait_result.result(timeout=5))

        process.poll.assert_called_once_with()
        self.assertEqual(self.manager.exit_code, -signal.SIGTERM)
        self.assertIn("pid=4242 killed by SIGTERM", "\n".join(logs.output))
        self.assertNotIn("still alive", "\n".join(logs.output))
        self.manager.print_process_log.assert_called_once_with()


class MagaServerManagerStartTest(unittest.TestCase):
    def setUp(self):
        mocks = ExitStack()
        self.addCleanup(mocks.close)
        mocks.enter_context(
            patch.dict(
                "os.environ",
                {
                    "HOME": "/mock/home",
                    "TEST_UNDECLARED_OUTPUTS_DIR": "/mock/output",
                    "RTP_PYTHON_BIN": "/mock/python",
                    "CUDA_VISIBLE_DEVICES": "0",
                },
                clear=True,
            )
        )
        mocks.enter_context(
            patch("rtp_llm.test.utils.maga_server_manager.configure_cache_env")
        )
        mocks.enter_context(patch("rtp_llm.test.utils.maga_server_manager.os.makedirs"))
        self.log_open = mocks.enter_context(patch("builtins.open", mock_open()))
        mocks.enter_context(patch("resource.setrlimit"))
        self.process = Mock(pid=4242)
        self.popen = mocks.enter_context(
            patch(
                "rtp_llm.test.utils.maga_server_manager.subprocess.Popen",
                return_value=self.process,
            )
        )
        self.health_check = mocks.enter_context(
            patch("rtp_llm.utils.util.wait_sever_done", return_value=True)
        )
        self.manager = MagaServerManager(
            port="18088", device_ids=[2, 3], health_check_path="/ready"
        )

    def tearDown(self):
        with self.manager._state_lock:
            self.manager._server_process = None
        self.manager.stop_server()

    def test_start_without_readiness_does_not_probe(self):
        self.assertTrue(self.manager.start_server(wait_ready=False))

        self.popen.assert_called_once()
        self.health_check.assert_not_called()
        self.process.poll.assert_not_called()
        self.assertEqual(self.manager.server_pid, 4242)
        self.log_open.assert_called_once_with("/mock/output/main_logs/process.log", "w")
        self.assertEqual(
            self.popen.call_args.kwargs["env"]["CUDA_VISIBLE_DEVICES"], "2,3"
        )
        self.assertIs(self.popen.call_args.kwargs["stdout"], self.log_open.return_value)
        self.assertIs(self.popen.call_args.kwargs["stderr"], self.log_open.return_value)

    def test_start_defaults_to_readiness(self):
        self.assertTrue(self.manager.start_server(timeout=17))

        self.health_check.assert_called_once_with(self.process, 18088, 17, "/ready")
        self.assertEqual(
            self.popen.call_args.args[0],
            ["/mock/python", "-m", "rtp_llm.start_server"],
        )

    def test_start_forwards_custom_module(self):
        self.manager._smoke_args_str = '--checkpoint "/mock/model path"'
        self.assertTrue(
            self.manager.start_server(wait_ready=False, server_module="custom.server")
        )

        self.assertEqual(
            self.popen.call_args.args[0],
            ["/mock/python", "-m", "custom.server", "--checkpoint", "/mock/model path"],
        )
        self.health_check.assert_not_called()

    def test_stop_before_launch_still_returns_false(self):
        self.assertTrue(self.manager.stop_server())
        parent = Mock()
        parent.children.return_value = []
        with (
            patch(
                "rtp_llm.test.utils.maga_server_manager.psutil.Process",
                return_value=parent,
            ) as get_parent,
            patch(
                "rtp_llm.test.utils.maga_server_manager.psutil.wait_procs",
                return_value=([], []),
            ),
        ):
            self.assertFalse(self.manager.start_server(wait_ready=False))

        self.popen.assert_called_once()
        self.health_check.assert_not_called()
        get_parent.assert_called_once_with(4242)
        parent.terminate.assert_called_once_with()
        self.assertIsNone(self.manager.server_pid)
        self.log_open.return_value.close.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
