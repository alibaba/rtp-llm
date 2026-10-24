import select
import signal
import subprocess
import sys
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from textwrap import dedent
from unittest.mock import Mock, call, patch

import psutil

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

    def test_stop_allows_parent_to_reap_children(self):
        self.manager._server_process = Mock(pid=4242)
        child = Mock()
        parent = Mock()
        parent.children.return_value = [child]
        events = []
        parent.terminate.side_effect = lambda: events.append("parent_terminate")
        parent.wait.side_effect = lambda **kwargs: events.append("parent_wait")

        def wait_for_children(processes, timeout):
            events.append("children_wait")
            return processes, []

        with (
            patch(
                "rtp_llm.test.utils.maga_server_manager.psutil.Process",
                return_value=parent,
            ),
            patch(
                "rtp_llm.test.utils.maga_server_manager.psutil.wait_procs",
                side_effect=wait_for_children,
            ),
        ):
            self.assertTrue(self.manager.stop_server())

        self.assertEqual(
            events[:3], ["parent_terminate", "parent_wait", "children_wait"]
        )
        child.terminate.assert_not_called()
        child.kill.assert_not_called()
        parent.kill.assert_not_called()
        self.assertIsNone(self.manager.server_pid)

    def test_stop_preserves_multiprocessing_resource_tracker(self):
        script = dedent(
            """
            import gc
            import multiprocessing
            import signal
            import threading
            import time

            shutdown_requested = threading.Event()

            def request_stop(signum, frame):
                shutdown_requested.set()

            signal.signal(signal.SIGTERM, request_stop)
            semaphore = multiprocessing.get_context("spawn").Semaphore()
            print("ready", flush=True)
            shutdown_requested.wait()
            time.sleep(0.2)
            del semaphore
            gc.collect()
            print("stopped", flush=True)
            """
        )
        process = subprocess.Popen(
            [sys.executable, "-c", script],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        self.manager._server_process = process
        try:
            ready, _, _ = select.select([process.stdout], [], [], 10)
            self.assertTrue(ready, "server fixture did not start")
            self.assertEqual(process.stdout.readline().strip(), "ready")
            self.assertTrue(self.manager.stop_server())
            stdout, stderr = process.communicate(timeout=10)
            self.assertEqual(process.returncode, 0)
            self.assertIn("stopped", stdout)
            self.assertEqual(stderr, "")
        finally:
            if process.poll() is None:
                process.kill()
            process.communicate(timeout=10)

    def test_stop_escalates_after_parent_timeout(self):
        self.manager._server_process = Mock(pid=4242)
        child = Mock(pid=4243)
        late_child = Mock(pid=4244)
        parent = Mock()
        parent.children.side_effect = [[child], [child, late_child]]
        parent.wait.side_effect = [psutil.TimeoutExpired(17), 0]
        parent.kill.side_effect = lambda: late_child.kill.assert_called_once_with()
        with (
            patch(
                "rtp_llm.test.utils.maga_server_manager.psutil.Process",
                return_value=parent,
            ),
            patch(
                "rtp_llm.test.utils.maga_server_manager.psutil.wait_procs",
                side_effect=[
                    ([], [child, late_child]),
                    ([child], [late_child]),
                    ([late_child], []),
                ],
            ),
            self.assertLogs(level="WARNING") as logs,
        ):
            self.assertTrue(self.manager.stop_server(timeout=17))

        self.assertIn("did not exit within 17s", "\n".join(logs.output))
        self.assertEqual(
            parent.wait.call_args_list, [call(timeout=17), call(timeout=5)]
        )
        child.terminate.assert_called_once_with()
        late_child.terminate.assert_called_once_with()
        child.kill.assert_not_called()
        late_child.kill.assert_called_once_with()
        parent.kill.assert_called_once_with()
        self.assertIsNone(self.manager.server_pid)

    def test_stop_continues_when_descendants_disappear(self):
        self.manager._server_process = Mock(pid=4242)
        file_stream = Mock()
        self.manager._file_stream = file_stream
        gone_child = Mock(pid=4243)
        surviving_child = Mock(pid=4244)
        gone_child.terminate.side_effect = psutil.NoSuchProcess(4243)
        surviving_child.kill.side_effect = psutil.NoSuchProcess(4244)
        parent = Mock()
        parent.children.return_value = [gone_child, surviving_child]
        with (
            patch(
                "rtp_llm.test.utils.maga_server_manager.psutil.Process",
                return_value=parent,
            ) as get_process,
            patch(
                "rtp_llm.test.utils.maga_server_manager.psutil.wait_procs",
                side_effect=[
                    ([], [gone_child, surviving_child]),
                    ([gone_child], [surviving_child]),
                    ([], []),
                ],
            ),
        ):
            self.assertTrue(self.manager.stop_server())
            self.assertTrue(self.manager.stop_server())

        get_process.assert_called_once_with(4242)
        surviving_child.terminate.assert_called_once_with()
        surviving_child.kill.assert_called_once_with()
        parent.kill.assert_not_called()
        file_stream.close.assert_called_once_with()
        self.assertIsNone(self.manager.server_pid)

    def test_stop_cleans_children_when_parent_disappears(self):
        self.manager._server_process = Mock(pid=4242)
        child = Mock(pid=4243)
        parent = Mock()
        parent.children.return_value = [child]
        parent.terminate.side_effect = psutil.NoSuchProcess(4242)
        with (
            patch(
                "rtp_llm.test.utils.maga_server_manager.psutil.Process",
                return_value=parent,
            ),
            patch(
                "rtp_llm.test.utils.maga_server_manager.psutil.wait_procs",
                side_effect=[([], [child]), ([child], [])],
            ),
        ):
            self.assertTrue(self.manager.stop_server())

        child.terminate.assert_called_once_with()
        child.kill.assert_not_called()
        parent.wait.assert_not_called()
        parent.kill.assert_not_called()
        self.assertIsNone(self.manager.server_pid)

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


if __name__ == "__main__":
    unittest.main()
