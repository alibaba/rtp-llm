import logging
import multiprocessing
import os
import signal
import time
import unittest
from contextlib import contextmanager
from unittest.mock import Mock, patch

from rtp_llm.utils.process_manager import (
    BACKEND_POST_FRONTEND_DRAIN_SECONDS_ENV,
    DEFERRED_GROUP_SHUTDOWN_HEADROOM_SECONDS_ENV,
    DEFER_FIRST_SIGTERM_ENV,
    DEFER_FIRST_SIGTERM_SECONDS_ENV,
    DEFER_FIRST_SIGTERM_VALUE,
    FRONTEND_PRE_STOP_DRAIN_SECONDS_ENV,
    PRE_STOP_DRAIN_HEADROOM_SECONDS_ENV,
    PRE_STOP_DRAIN_SIGNAL_ENV,
    ProcessManager,
    SHUTDOWN_TIMEOUT_ENV,
    STOP_TIMEOUT_MS_ENV,
)


@contextmanager
def _watchdog(seconds: float, msg: str = "test exceeded watchdog"):
    """Fail-fast guard for tests that would hang on regression.

    Uses SIGALRM (main-thread only); raises AssertionError on timeout so the
    test fails immediately instead of blocking the whole suite.
    """
    def handler(_signum: int, _frame: object) -> None:
        raise AssertionError(f"{msg}: exceeded {seconds}s")

    prev = signal.signal(signal.SIGALRM, handler)
    signal.alarm(int(seconds))
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, prev)


def dummy_worker(duration=1, should_crash=False):
    """Dummy worker function for testing"""
    if should_crash:
        time.sleep(0.5)
        raise RuntimeError("Simulated crash")
    time.sleep(duration)


def forever_worker(queue):
    # Signal that we are ready
    queue.put("ready")

    while True:
        time.sleep(0.1)


def signal_handler_worker(queue):
    """Worker that handles signals and reports back"""
    running = True

    def handler(signum, frame):
        nonlocal running
        queue.put(f"received_{signum}")
        running = False

    signal.signal(signal.SIGTERM, handler)
    while running:
        time.sleep(0.1)


class TestProcessManager(unittest.TestCase):

    def setUp(self):
        """Set up test fixtures"""
        self.manager = ProcessManager(shutdown_timeout=50, monitor_interval=1)
        logging.basicConfig(level=logging.INFO)

    def tearDown(self):
        """Clean up after tests"""
        # Ensure all processes are terminated
        for proc in self.manager.processes:
            # Skip mock processes
            if hasattr(proc, "_mock_name"):
                continue
            if proc.is_alive():
                proc.terminate()
                try:
                    proc.join(timeout=1)
                except Exception:
                    # Ignore join errors in teardown
                    pass
                if proc.is_alive():
                    try:
                        os.kill(proc.pid, signal.SIGKILL)
                    except (OSError, ProcessLookupError):
                        # Process may have already died
                        pass

    def test_init(self):
        """Test ProcessManager initialization"""
        self.assertEqual(self.manager.processes, [])
        self.assertFalse(self.manager.shutdown_requested)
        self.assertFalse(self.manager.failure_detected)

    def test_init_preserves_infinite_shutdown_timeout(self):
        manager = ProcessManager(shutdown_timeout=-1)
        self.assertEqual(manager.shutdown_timeout, -1)
        self.assertIsNone(manager._make_deadline(manager.shutdown_timeout))

    def test_add_single_process(self):
        """Test adding a single process"""
        proc = multiprocessing.Process(target=dummy_worker)
        self.manager.add_process(proc)
        self.assertEqual(len(self.manager.processes), 1)
        self.assertEqual(self.manager.processes[0], proc)

        # Test adding None
        self.manager.add_process(None)
        self.assertEqual(len(self.manager.processes), 1)

    def test_add_multiple_processes(self):
        """Test adding multiple processes"""
        procs = [multiprocessing.Process(target=dummy_worker) for _ in range(3)]
        self.manager.add_processes(procs)
        self.assertEqual(len(self.manager.processes), 3)

        # Test adding None
        self.manager.add_processes(None)
        self.assertEqual(len(self.manager.processes), 3)

        # Test adding empty list
        self.manager.add_processes([])
        self.assertEqual(len(self.manager.processes), 3)

    def test_set_processes(self):
        """Test setting processes (replacing existing)"""
        # Add initial processes
        initial_procs = [multiprocessing.Process(target=dummy_worker)]
        self.manager.add_processes(initial_procs)
        self.assertEqual(len(self.manager.processes), 1)

        # Set new processes
        new_procs = [multiprocessing.Process(target=dummy_worker) for _ in range(2)]
        self.manager.set_processes(new_procs)
        self.assertEqual(len(self.manager.processes), 2)
        self.assertEqual(self.manager.processes, new_procs)

        # Set to None
        self.manager.set_processes(None)
        self.assertEqual(self.manager.processes, [])

    def test_signal_handler(self):
        """Test signal handler"""
        self.assertFalse(self.manager.shutdown_requested)

        # Simulate signal
        self.manager._signal_handler(signal.SIGTERM, None)
        self.assertTrue(self.manager.shutdown_requested)

    def test_is_any_process_alive(self):
        """Test checking if any process is alive"""
        # No processes
        self.assertFalse(self.manager._is_any_process_alive())

        # Add dead processes (not started)
        procs = [multiprocessing.Process(target=dummy_worker) for _ in range(2)]
        self.manager.add_processes(procs)
        self.assertFalse(self.manager._is_any_process_alive())

        # Start one process
        procs[0].start()
        self.assertTrue(self.manager._is_any_process_alive())

        # Terminate it
        procs[0].terminate()
        procs[0].join()
        self.assertFalse(self.manager._is_any_process_alive())

    def test_is_all_processes_alive(self):
        """Test checking if all processes are alive"""
        # No processes
        self.assertTrue(self.manager._is_all_processes_alive())

        # Add and start processes
        procs = [
            multiprocessing.Process(target=dummy_worker, args=(2,)) for _ in range(2)
        ]
        self.manager.add_processes(procs)

        # Not started yet
        self.assertFalse(self.manager._is_all_processes_alive())

        # Start all
        for proc in procs:
            proc.start()
        self.assertTrue(self.manager._is_all_processes_alive())

        # Terminate one
        procs[0].terminate()
        procs[0].join()
        self.assertFalse(self.manager._is_all_processes_alive())

        # Clean up
        procs[1].terminate()
        procs[1].join()

    def test_terminate_processes(self):
        """Test terminating processes"""
        queue = multiprocessing.Queue()
        proc = multiprocessing.Process(target=signal_handler_worker, args=(queue,))
        proc.start()
        self.manager.add_process(proc)

        # Ensure process is running
        time.sleep(0.5)
        self.assertTrue(proc.is_alive())

        # Terminate
        self.manager._terminate_processes(drain_timeout=0)

        # Wait for process to handle SIGTERM and die
        timeout = time.time() + 3
        while proc.is_alive() and time.time() < timeout:
            time.sleep(0.1)

        # If still alive, force join
        if proc.is_alive():
            proc.join(timeout=1)

        self.assertFalse(proc.is_alive())

    def test_terminate_processes_stages_frontend_before_backend(self):
        """Test frontend process group is drained before backend is signaled"""
        events = []

        class FakeProcess:
            def __init__(self, name):
                self.name = name
                self.pid = len(events) + 100
                self._alive = True
                self._popen = None

            def is_alive(self):
                return self._alive

            def terminate(self):
                events.append(self.name)
                self._alive = False

        frontend_proc = FakeProcess("frontend")
        backend_proc = FakeProcess("backend")
        self.manager.add_process(frontend_proc, shutdown_group="frontend")
        self.manager.add_process(backend_proc, shutdown_group="backend")

        self.manager._terminate_processes(drain_timeout=0)

        self.assertEqual(events, ["frontend", "backend"])

    def test_undrained_frontend_force_killed_after_backend_sigterm(self):
        """Undrained frontend is SIGTERM'd, backend SIGTERM'd next, then monitor
        force-kills frontend after POST_KILL_REAP_WINDOW. SIGKILL no longer
        happens inside _terminate_processes — it is the monitor loop's job."""
        events = []

        class FakeProcess:
            def __init__(self, name, pid):
                self.name = name
                self.pid = pid
                self._alive = True
                self._popen = None

            def is_alive(self):
                return self._alive

            def terminate(self):
                events.append(f"term:{self.name}")
                if self.name == "backend":
                    self._alive = False

        frontend_proc = FakeProcess("frontend", 1001)
        backend_proc = FakeProcess("backend", 1002)
        self.manager.shutdown_timeout = 0
        self.manager.monitor_interval = 0.01
        self.manager.POST_KILL_REAP_WINDOW = 0.05
        self.manager.add_process(frontend_proc, shutdown_group="frontend")
        self.manager.add_process(backend_proc, shutdown_group="backend")
        self.manager.shutdown_requested = True  # drive monitor end-to-end

        def fake_kill(pid, sig):
            events.append(f"kill:{pid}:{sig}")
            if pid == frontend_proc.pid:
                frontend_proc._alive = False

        with patch("os.kill", side_effect=fake_kill):
            self.manager._monitor_processes_health()

        # SIGTERM ordering preserved (frontend before backend), and the SIGKILL
        # for the survivor comes AFTER both SIGTERMs — the monitor loop's
        # POST_KILL_REAP_WINDOW expiry is the single source of force-kill.
        self.assertEqual(events[0], "term:frontend")
        self.assertEqual(events[1], "term:backend")
        self.assertIn(f"kill:{frontend_proc.pid}:{signal.SIGKILL}", events[2:])

    def test_force_kill_processes(self):
        """Test force killing processes"""
        proc = multiprocessing.Process(target=dummy_worker, args=(10,))
        proc.start()
        self.manager.add_process(proc)

        # Force kill
        with patch("logging.warning") as mock_warning:
            self.manager._force_kill_processes()
            mock_warning.assert_called()

        # Process should be dead
        proc.join(timeout=1)
        self.assertFalse(proc.is_alive())

    def test_monitor_empty_processes(self):
        """Test monitoring with no processes"""
        with patch("logging.info") as mock_info:
            self.manager.monitor_and_release_processes()
            mock_info.assert_any_call("No processes to monitor")

    def test_monitor_normal_completion(self):
        """Test monitoring processes that complete normally"""
        procs = [
            multiprocessing.Process(target=dummy_worker, args=(0.5,)) for _ in range(2)
        ]
        self.manager.add_processes(procs)

        # Start processes
        for proc in procs:
            proc.start()

        # Monitor until completion
        self.manager.monitor_and_release_processes()

        # All processes should be done
        for proc in procs:
            self.assertFalse(proc.is_alive())

    def test_monitor_with_crash(self):
        """Test monitoring when a process crashes"""
        # One normal, one crashes
        procs = [
            multiprocessing.Process(target=dummy_worker, args=(2,)),
            multiprocessing.Process(target=dummy_worker, args=(0.5, True)),
        ]
        self.manager.add_processes(procs)

        # Start processes
        for proc in procs:
            proc.start()

        # Monitor - should detect crash, terminate all, and fail the parent.
        with patch("logging.error") as mock_error, patch("os._exit") as mock_exit:
            mock_exit.side_effect = SystemExit(1)
            with self.assertRaises(SystemExit):
                self.manager.monitor_and_release_processes()
            mock_error.assert_called()
            mock_exit.assert_called_with(1)

        # All processes should be terminated
        for proc in procs:
            self.assertFalse(proc.is_alive())
        self.assertTrue(self.manager.failure_detected)

    def test_monitor_with_shutdown_signal(self):
        """Test monitoring with shutdown signal"""
        proc = multiprocessing.Process(target=dummy_worker, args=(5,))
        proc.start()
        self.manager.add_process(proc)

        # Start monitoring in thread
        import threading

        monitor_thread = threading.Thread(
            target=self.manager.monitor_and_release_processes
        )
        monitor_thread.start()

        # Send shutdown signal after a short delay (mirrors what the SIGTERM
        # handler does in production — sets shutdown_requested directly,
        # leaves failure_detected=False).
        time.sleep(0.5)
        self.manager.shutdown_requested = True

        # Wait for monitoring to complete
        monitor_thread.join(timeout=5)

        # Wait a bit more for process to fully terminate
        time.sleep(0.5)

        # Process should be terminated
        self.assertFalse(proc.is_alive())
        self.assertTrue(self.manager.shutdown_requested)

    def test_monitor_with_force_kill_timeout(self):
        """Monitor force-kills survivors after the post-SIGTERM reap window."""
        # Mock a process that won't die
        mock_proc = Mock()
        mock_proc.is_alive.return_value = True
        mock_proc.pid = 12345
        # Mark it as a mock to skip in teardown
        mock_proc._mock_name = "mock_proc"

        self.manager.add_process(mock_proc, shutdown_group="frontend")
        self.manager.shutdown_timeout = 0  # immediate drain timeout
        self.manager.POST_KILL_REAP_WINDOW = 0.05
        self.manager.shutdown_requested = True

        # Should force kill after reap window expires
        with patch("os.kill") as mock_kill:
            self.manager._monitor_processes_health()
            mock_kill.assert_called_with(12345, signal.SIGKILL)

    def test_request_failure_shutdown(self):
        """request_failure_shutdown marks both shutdown and failure flags."""
        self.assertFalse(self.manager.shutdown_requested)
        self.assertFalse(self.manager.failure_detected)

        self.manager.request_failure_shutdown()
        self.assertTrue(self.manager.shutdown_requested)
        self.assertTrue(self.manager.failure_detected)

    def test_join_all_processes(self):
        """Test joining all processes"""
        procs = [
            multiprocessing.Process(target=dummy_worker, args=(0.1,)) for _ in range(2)
        ]
        self.manager.add_processes(procs)

        # Start and let them finish
        for proc in procs:
            proc.start()

        time.sleep(0.5)

        # Join should succeed
        with patch("logging.info") as mock_info:
            self.manager._join_all_processes()
            mock_info.assert_called_with("All processes joined")

    def test_join_with_exception(self):
        """Test joining processes with exception"""
        # Mock a process that raises on join
        mock_proc = Mock()
        mock_proc.join.side_effect = RuntimeError("Join failed")
        mock_proc.pid = 123
        mock_proc.is_alive.return_value = False
        # Mark it as a mock to skip in teardown
        mock_proc._mock_name = "mock_proc"

        self.manager.add_process(mock_proc)

        with patch("logging.error") as mock_error:
            self.manager._join_all_processes()
            mock_error.assert_called()

    def test_forever_process_shutdown(self):
        """Test shutting down a process that runs forever"""
        queue = multiprocessing.Queue()
        proc1 = multiprocessing.Process(target=forever_worker, args=(queue,))
        proc1.start()
        self.manager.add_process(proc1)

        proc2 = multiprocessing.Process(target=forever_worker, args=(queue,))
        proc2.start()
        self.manager.add_process(proc2)

        self.manager.shutdown_timeout = 2
        # Wait for process to be ready
        self.assertEqual(queue.get(timeout=5), "ready")

        # Ensure it's running
        self.assertTrue(proc1.is_alive())
        self.assertTrue(proc2.is_alive())

        # Start monitoring in thread
        import threading

        monitor_thread = threading.Thread(
            target=self.manager.monitor_and_release_processes
        )
        monitor_thread.start()

        # Give it a moment
        time.sleep(0.1)

        # Request shutdown (mirrors SIGTERM handler — flag-only, not failure)
        self.manager.shutdown_requested = True

        # Wait for monitoring to complete
        monitor_thread.join()

        # Process should be terminated
        self.assertFalse(proc1.is_alive())
        self.assertFalse(proc2.is_alive())


class TestProcessManagerHealthCheck(unittest.TestCase):
    """Test cases for health check functionality"""

    def setUp(self):
        """Set up test fixtures"""
        self.manager = ProcessManager(shutdown_timeout=50, monitor_interval=1)
        logging.basicConfig(level=logging.INFO)

    def tearDown(self):
        """Clean up after tests"""
        for proc in self.manager.processes:
            if hasattr(proc, "_mock_name"):
                continue
            if proc.is_alive():
                proc.terminate()
                try:
                    proc.join(timeout=1)
                except Exception:
                    pass
                if proc.is_alive():
                    try:
                        os.kill(proc.pid, signal.SIGKILL)
                    except (OSError, ProcessLookupError):
                        pass

    def test_register_health_check(self):
        """Test registering health check for processes"""
        # Create mock processes
        mock_procs = [Mock() for _ in range(2)]
        for mock_proc in mock_procs:
            mock_proc.is_alive.return_value = True
            mock_proc._mock_name = "mock_proc"

        # Create a mock check function
        mock_check_fn = Mock(return_value=True)

        # Register health check
        self.manager.register_health_check(
            processes=mock_procs,
            process_name="test_service",
            check_ready_fn=mock_check_fn,
            retry_interval_seconds=0.1,
        )

        # Verify registration
        self.assertIn("test_service", self.manager.health_check_configs)
        config = self.manager.health_check_configs["test_service"]
        self.assertEqual(config["processes"], mock_procs)
        self.assertEqual(config["retry_interval_seconds"], 0.1)
        self.assertEqual(config["check_ready_fn"], mock_check_fn)

        # Verify status initialization
        self.assertIn("test_service", self.manager.health_check_status)
        status = self.manager.health_check_status["test_service"]
        self.assertFalse(status["ready"])
        self.assertFalse(status["checked"])

    def test_health_check_worker_success(self):
        """Test health check worker with successful check"""
        # Mock processes
        mock_procs = [Mock() for _ in range(2)]
        for mock_proc in mock_procs:
            mock_proc.is_alive.return_value = True
            mock_proc._mock_name = "mock_proc"

        # Create a mock check function that returns True
        mock_check_fn = Mock(return_value=True)

        # Register health check
        self.manager.register_health_check(
            processes=mock_procs,
            process_name="backend_server",
            check_ready_fn=mock_check_fn,
        )

        # Run worker
        self.manager._health_check_worker("backend_server")

        # Verify status updated
        status = self.manager.health_check_status["backend_server"]
        self.assertTrue(status["ready"])
        self.assertTrue(status["checked"])

        # Verify check function was called
        mock_check_fn.assert_called()

    def test_health_check_worker_process_dead(self):
        """Test health check worker when process is dead"""
        # Mock dead processes
        mock_procs = [Mock() for _ in range(2)]
        for mock_proc in mock_procs:
            mock_proc.is_alive.return_value = False
            mock_proc._mock_name = "mock_proc"

        # Create a mock check function
        mock_check_fn = Mock(return_value=True)

        # Register health check
        self.manager.register_health_check(
            processes=mock_procs,
            process_name="backend_server",
            check_ready_fn=mock_check_fn,
        )

        # Run worker
        self.manager._health_check_worker("backend_server")

        # Verify status - should be checked but not ready
        status = self.manager.health_check_status["backend_server"]
        self.assertFalse(status["ready"])
        self.assertTrue(status["checked"])

        # Health check should not be called since process is dead
        mock_check_fn.assert_not_called()

    def test_start_parallel_health_checks_no_registration(self):
        """Test starting parallel health checks with no registered checks"""
        with patch("logging.info") as mock_info:
            self.manager.start_parallel_health_checks()
            mock_info.assert_any_call("No health checks registered")

    def test_start_parallel_health_checks(self):
        """Test starting parallel health checks"""
        # Mock processes for multiple services
        backend_procs = [Mock()]
        backend_procs[0].is_alive.return_value = True
        backend_procs[0]._mock_name = "backend_mock"

        frontend_procs = [Mock()]
        frontend_procs[0].is_alive.return_value = True
        frontend_procs[0]._mock_name = "frontend_mock"

        # Create mock check functions
        backend_check_fn = Mock(return_value=False)
        frontend_check_fn = Mock(return_value=False)

        # Register multiple health checks
        self.manager.register_health_check(
            processes=backend_procs,
            process_name="backend_server",
            check_ready_fn=backend_check_fn,
        )
        self.manager.register_health_check(
            processes=frontend_procs,
            process_name="frontend_server",
            check_ready_fn=frontend_check_fn,
        )

        # Start parallel checks
        self.manager.start_parallel_health_checks()

        backend_check_fn.return_value = True
        frontend_check_fn.return_value = True

        # Verify threads created
        self.assertEqual(len(self.manager.health_check_threads), 2)
        self.assertTrue(all(t.is_alive() for t in self.manager.health_check_threads))

        # Wait for checks to complete
        for thread in self.manager.health_check_threads:
            thread.join(timeout=2)

        # Verify both services are ready
        self.assertTrue(self.manager.health_check_status["backend_server"]["ready"])
        self.assertTrue(self.manager.health_check_status["frontend_server"]["ready"])

    def test_wait_for_health_checks_success(self):
        """Test waiting for health checks to complete successfully"""
        # Mock processes
        mock_procs = [Mock()]
        mock_procs[0].is_alive.return_value = True
        mock_procs[0]._mock_name = "mock_proc"

        # Create a mock check function
        mock_check_fn = Mock(return_value=True)

        # Register and start health check
        self.manager.register_health_check(
            processes=mock_procs,
            process_name="test_service",
            check_ready_fn=mock_check_fn,
        )

        self.manager.start_parallel_health_checks()

        # Wait for completion
        result = self.manager.wait_for_health_checks(timeout=5)

        self.assertTrue(result)
        self.assertTrue(self.manager.health_check_status["test_service"]["ready"])
        self.assertTrue(self.manager.health_check_status["test_service"]["checked"])

    def test_wait_for_health_checks_failure(self):
        """Test waiting for health checks when they fail"""
        # Mock dead processes
        mock_procs = [Mock()]
        mock_procs[0].is_alive.return_value = False
        mock_procs[0]._mock_name = "mock_proc"

        # Create a mock check function
        mock_check_fn = Mock(return_value=True)

        # Register and start health check
        self.manager.register_health_check(
            processes=mock_procs,
            process_name="test_service",
            check_ready_fn=mock_check_fn,
        )

        self.manager.start_parallel_health_checks()

        # Wait for completion
        result = self.manager.wait_for_health_checks(timeout=5)

        self.assertFalse(result)
        self.assertFalse(self.manager.health_check_status["test_service"]["ready"])

    def test_wait_for_health_checks_no_threads(self):
        """Test waiting when no health check threads exist"""
        result = self.manager.wait_for_health_checks()
        self.assertTrue(result)

    def test_wait_for_health_checks_timeout(self):
        """Test waiting for health checks with timeout"""
        # Mock processes
        mock_procs = [Mock()]
        mock_procs[0].is_alive.return_value = True
        mock_procs[0]._mock_name = "mock_proc"

        # Create a mock check function that never succeeds
        mock_check_fn = Mock(return_value=False)

        # Register health check
        self.manager.register_health_check(
            processes=mock_procs,
            process_name="slow_service",
            check_ready_fn=mock_check_fn,
            retry_interval_seconds=10,  # Long interval to simulate slow check
        )

        self.manager.start_parallel_health_checks()

        # Wait with short timeout
        result = self.manager.wait_for_health_checks(timeout=0.5)

        # Should fail because check didn't complete
        self.assertFalse(result)

    def test_parallel_health_checks_multiple_processes(self):
        """Test health checks work correctly with multiple processes per service"""
        # Mock multiple processes for a service (e.g., frontend with multiple workers)
        mock_procs = [Mock() for _ in range(3)]
        for mock_proc in mock_procs:
            mock_proc.is_alive.return_value = True
            mock_proc._mock_name = "mock_proc"

        # Create a mock check function
        mock_check_fn = Mock(return_value=True)

        # Register health check
        self.manager.register_health_check(
            processes=mock_procs,
            process_name="frontend_workers",
            check_ready_fn=mock_check_fn,
        )

        self.manager.start_parallel_health_checks()

        # Wait for completion
        result = self.manager.wait_for_health_checks(timeout=5)

        self.assertTrue(result)
        self.assertTrue(self.manager.health_check_status["frontend_workers"]["ready"])

    def test_health_check_worker_one_process_dies(self):
        """Test health check when one of multiple processes dies"""
        # Mock processes where one dies
        mock_procs = [Mock() for _ in range(2)]
        mock_procs[0].is_alive.return_value = True
        mock_procs[1].is_alive.return_value = False  # One is dead
        for mock_proc in mock_procs:
            mock_proc._mock_name = "mock_proc"

        # Create a mock check function
        mock_check_fn = Mock(return_value=True)

        # Register health check
        self.manager.register_health_check(
            processes=mock_procs,
            process_name="service_with_dead_proc",
            check_ready_fn=mock_check_fn,
        )

        # Run worker
        self.manager._health_check_worker("service_with_dead_proc")

        # Should detect that not all processes are alive
        status = self.manager.health_check_status["service_with_dead_proc"]
        self.assertFalse(status["ready"])
        self.assertTrue(status["checked"])

    def test_run_health_checks(self):
        """Test the convenience method run_health_checks"""
        # Mock processes
        mock_procs = [Mock()]
        mock_procs[0].is_alive.return_value = True
        mock_procs[0]._mock_name = "mock_proc"

        # Create a mock check function
        mock_check_fn = Mock(return_value=True)

        # Register health check
        self.manager.register_health_check(
            processes=mock_procs,
            process_name="test_service",
            check_ready_fn=mock_check_fn,
        )

        # Run health checks (should start and wait)
        result = self.manager.run_health_checks(timeout=5)

        # Verify success
        self.assertTrue(result)
        self.assertTrue(self.manager.health_check_status["test_service"]["ready"])
        self.assertTrue(self.manager.health_check_status["test_service"]["checked"])

    def test_health_check_with_exception(self):
        """Test health check when check_ready_fn raises exception"""
        # Mock processes
        mock_procs = [Mock()]
        mock_procs[0].is_alive.return_value = True
        mock_procs[0]._mock_name = "mock_proc"

        # Create a check function that raises exception first, then succeeds
        call_count = [0]

        def check_fn_with_exception():
            call_count[0] += 1
            if call_count[0] == 1:
                raise RuntimeError("First call fails")
            return True

        # Register health check
        self.manager.register_health_check(
            processes=mock_procs,
            process_name="flaky_service",
            check_ready_fn=check_fn_with_exception,
            retry_interval_seconds=0.1,
        )

        # Run health checks - should eventually succeed after retry
        result = self.manager.run_health_checks(timeout=5)

        # Verify success after retry
        self.assertTrue(result)
        self.assertTrue(self.manager.health_check_status["flaky_service"]["ready"])
        self.assertGreater(call_count[0], 1)  # Should have been called more than once

    def test_custom_check_ready_function(self):
        """Test health check with custom check_ready_fn"""
        # Mock processes
        mock_procs = [Mock()]
        mock_procs[0].is_alive.return_value = True
        mock_procs[0]._mock_name = "mock_proc"

        # Create a custom check function that tracks state
        check_state = {"counter": 0}

        def custom_check():
            check_state["counter"] += 1
            return check_state["counter"] >= 3  # Succeed on third call

        # Register health check
        self.manager.register_health_check(
            processes=mock_procs,
            process_name="custom_service",
            check_ready_fn=custom_check,
            retry_interval_seconds=0.05,
        )

        # Run health checks
        result = self.manager.run_health_checks(timeout=5)

        # Verify custom function was called multiple times
        self.assertTrue(result)
        self.assertEqual(check_state["counter"], 3)
        self.assertTrue(self.manager.health_check_status["custom_service"]["ready"])


class _FakeProc:
    """Minimal fake process for staged-shutdown tests.

    Compatible with the methods ProcessManager invokes (.is_alive, .terminate,
    .join, .pid, .name) without touching the OS.
    """

    _next_pid = 9000

    def __init__(
        self,
        name,
        alive=True,
        dies_on_terminate=False,
        dies_after=None,
        exitcode=0,
    ):
        type(self)._next_pid += 1
        self.pid = type(self)._next_pid
        self.name = name
        self._alive = alive
        self._popen = None
        self._dies_on_terminate = dies_on_terminate
        self._dies_after = dies_after  # absolute time after which is_alive flips False
        self._exitcode = exitcode  # value reported once dead
        self.terminated = False

    def is_alive(self):
        if self._dies_after is not None and time.time() >= self._dies_after:
            self._alive = False
        return self._alive

    @property
    def exitcode(self):
        # Match multiprocessing.Process: None while alive, set after exit.
        return None if self.is_alive() else self._exitcode

    def terminate(self):
        self.terminated = True
        if self._dies_on_terminate:
            self._alive = False

    def join(self, timeout=None):
        # Pretend to wait; flip alive false if the death window already elapsed.
        if self._dies_after is not None and time.time() >= self._dies_after:
            self._alive = False


class TestFailureShutdownPaths(unittest.TestCase):
    """Coverage for the failure-driven shutdown / drain-abort fixes."""

    def setUp(self):
        # Use a short graceful drain so monitor-driven tests finish fast.
        self.manager = ProcessManager(shutdown_timeout=1, monitor_interval=0.01)
        self.manager.POST_KILL_REAP_WINDOW = 0.1

    # --- request_failure_shutdown invariants -------------------------------

    def test_signal_handler_does_not_mark_failure(self):
        """SIGTERM/SIGINT must leave failure_detected=False."""
        self.manager._signal_handler(signal.SIGTERM, None)
        self.assertTrue(self.manager.shutdown_requested)
        self.assertFalse(self.manager.failure_detected)

    def test_backend_process_manager_defers_first_sigterm(self):
        """Backend process managers can survive cgroup-wide SIGTERM until the
        parent sends the staged backend SIGINT after frontend drain."""
        with patch.dict(
            os.environ,
            {
                DEFER_FIRST_SIGTERM_ENV: DEFER_FIRST_SIGTERM_VALUE,
                DEFER_FIRST_SIGTERM_SECONDS_ENV: "30",
            },
        ):
            manager = ProcessManager(
                shutdown_timeout=30,
                monitor_interval=0.01,
                allow_defer_first_sigterm=True,
            )

        manager._signal_handler(signal.SIGTERM, None)
        self.assertFalse(manager.shutdown_requested)
        self.assertTrue(manager.is_deferred_sigterm_pending())

        manager._signal_handler(signal.SIGTERM, None)
        self.assertFalse(manager.shutdown_requested)
        self.assertTrue(manager.is_deferred_sigterm_pending())

        manager._signal_handler(signal.SIGINT, None)
        self.assertTrue(manager.shutdown_requested)
        self.assertFalse(manager.is_deferred_sigterm_pending())
        self.assertFalse(manager.failure_detected)

    def test_infinite_timeout_defers_first_sigterm_without_fallback_timer(self):
        with (
            patch.dict(
                os.environ,
                {
                    DEFER_FIRST_SIGTERM_ENV: DEFER_FIRST_SIGTERM_VALUE,
                    DEFER_FIRST_SIGTERM_SECONDS_ENV: "-1",
                },
            ),
            patch("rtp_llm.utils.process_manager.threading.Timer") as timer_cls,
        ):
            manager = ProcessManager(
                shutdown_timeout=-1,
                monitor_interval=0.01,
                allow_defer_first_sigterm=True,
            )
            manager._signal_handler(signal.SIGTERM, None)

        timer_cls.assert_not_called()
        self.assertFalse(manager.shutdown_requested)
        self.assertTrue(manager.is_deferred_sigterm_pending())

    def test_deferred_backend_group_gets_staged_sigint(self):
        """Parent-staged backend shutdown must not look like duplicate
        cgroup SIGTERM noise to backend children."""

        class FakeProcess:
            name = "backend"
            pid = 123456
            _popen = object()

            def is_alive(self):
                return True

        signals = []

        with patch("os.kill", side_effect=lambda pid, sig: signals.append((pid, sig))):
            self.manager.add_process(FakeProcess(), shutdown_group="backend")
            self.manager._sigterm_deferred_groups()

        self.assertEqual(signals, [(123456, signal.SIGINT)])

    def test_failure_shutdown_sends_sigint_to_deferred_backend_group(self):
        """A frontend/DashSc crash must wake the nested backend manager.

        The backend manager deliberately defers its first SIGTERM, so failure
        shutdown uses the explicit SIGINT handoff for that direct child while
        ordinary frontend children still receive SIGTERM.
        """

        class FakeProcess:
            _popen = object()

            def __init__(self, name, pid):
                self.name = name
                self.pid = pid

            def is_alive(self):
                return True

        frontend = FakeProcess("frontend", 123456)
        backend = FakeProcess("backend", 123457)
        self.manager.add_process(frontend, shutdown_group="frontend")
        self.manager.add_process(backend, shutdown_group="backend")

        signals = []
        with patch(
            "os.kill", side_effect=lambda pid, sig: signals.append((pid, sig))
        ):
            self.manager._terminate_processes(drain_timeout=0, staged=False)

        self.assertEqual(
            signals,
            [
                (frontend.pid, signal.SIGTERM),
                (backend.pid, signal.SIGINT),
            ],
        )

    def test_backend_shutdown_lingers_after_frontend_drain(self):
        events = []

        class FakeProcess:
            _popen = None

            def __init__(self, name):
                self.name = name
                self.pid = len(events) + 100
                self._alive = True

            def is_alive(self):
                return self._alive

            def terminate(self):
                events.append((self.name, time.time()))
                self._alive = False

        frontend_proc = FakeProcess("frontend")
        backend_proc = FakeProcess("backend")
        self.manager.add_process(frontend_proc, shutdown_group="frontend")
        self.manager.add_process(backend_proc, shutdown_group="backend")

        with patch.dict(os.environ, {BACKEND_POST_FRONTEND_DRAIN_SECONDS_ENV: "0.05"}):
            self.manager._terminate_processes(drain_timeout=1)

        self.assertEqual([name for name, _ in events], ["frontend", "backend"])
        self.assertGreaterEqual(events[1][1] - events[0][1], 0.04)

    def test_pre_stop_signal_drains_frontend_before_sigterm(self):
        events = []

        class FakeProcess:
            def __init__(self, name):
                self.name = name
                self.pid = 100 + len(events)
                self._alive = True
                self._popen = object() if name == "frontend" else None

            def is_alive(self):
                return self._alive

            @property
            def exitcode(self):
                return None if self._alive else 0

            def terminate(self):
                events.append((self.name, signal.SIGTERM))
                self._alive = False

        frontend_proc = FakeProcess("frontend")
        backend_proc = FakeProcess("backend")
        self.manager.monitor_interval = 0.005
        self.manager.add_process(frontend_proc, shutdown_group="frontend")
        self.manager.add_process(backend_proc, shutdown_group="backend")

        def fake_kill(pid, sig):
            if pid == frontend_proc.pid:
                events.append(("frontend", sig))

        with patch("os.kill", side_effect=fake_kill), patch.dict(
            os.environ,
            {
                FRONTEND_PRE_STOP_DRAIN_SECONDS_ENV: "0.02",
                BACKEND_POST_FRONTEND_DRAIN_SECONDS_ENV: "0.02",
            },
        ):
            self.manager._terminate_processes(drain_timeout=1)

        self.assertEqual(
            events,
            [
                ("frontend", signal.SIGUSR1),
                ("frontend", signal.SIGTERM),
                ("backend", signal.SIGTERM),
            ],
        )

    def test_pre_stop_signal_sent_to_all_drain_groups_before_sigterm(self):
        events = []

        class FakeProcess:
            _next_pid = 100

            def __init__(self, name):
                type(self)._next_pid += 1
                self.name = name
                self.pid = type(self)._next_pid
                self._alive = True
                self._popen = object() if name in ("frontend", "ingress") else None

            def is_alive(self):
                return self._alive

            @property
            def exitcode(self):
                return None if self._alive else 0

            def terminate(self):
                events.append((self.name, signal.SIGTERM))
                self._alive = False

        frontend_proc = FakeProcess("frontend")
        ingress_proc = FakeProcess("ingress")
        backend_proc = FakeProcess("backend")
        self.manager.monitor_interval = 0.005
        self.manager.add_process(frontend_proc, shutdown_group="frontend")
        self.manager.add_process(ingress_proc, shutdown_group="ingress")
        self.manager.add_process(backend_proc, shutdown_group="backend")

        def fake_kill(pid, sig):
            if pid == frontend_proc.pid:
                events.append(("frontend", sig))
            if pid == ingress_proc.pid:
                events.append(("ingress", sig))

        with patch("os.kill", side_effect=fake_kill), patch.dict(
            os.environ,
            {
                FRONTEND_PRE_STOP_DRAIN_SECONDS_ENV: "0.02",
                BACKEND_POST_FRONTEND_DRAIN_SECONDS_ENV: "0.02",
            },
        ):
            self.manager._terminate_processes(drain_timeout=1)

        self.assertEqual(
            events,
            [
                ("frontend", signal.SIGUSR1),
                ("ingress", signal.SIGUSR1),
                ("frontend", signal.SIGTERM),
                ("ingress", signal.SIGTERM),
                ("backend", signal.SIGTERM),
            ],
        )

    def test_pre_stop_signal_can_be_disabled(self):
        frontend = _FakeProc("frontend", dies_on_terminate=True)
        self.manager.add_process(frontend, shutdown_group="frontend")
        signals = []

        with patch(
            "os.kill", side_effect=lambda pid, sig: signals.append(sig)
        ), patch.dict(
            os.environ,
            {
                FRONTEND_PRE_STOP_DRAIN_SECONDS_ENV: "0.02",
                PRE_STOP_DRAIN_SIGNAL_ENV: "0",
            },
        ):
            self.manager._terminate_processes(drain_timeout=1)

        self.assertNotIn(signal.SIGUSR1, signals)

    def test_pre_stop_signal_window_reserves_shutdown_headroom(self):
        self.manager.shutdown_timeout = 10
        with patch.dict(
            os.environ,
            {
                FRONTEND_PRE_STOP_DRAIN_SECONDS_ENV: "10",
                PRE_STOP_DRAIN_HEADROOM_SECONDS_ENV: "2",
            },
        ):
            window_s = self.manager._pre_stop_drain_signal_window_seconds(
                time.time() + 10
            )

        self.assertGreater(window_s, 7.0)
        self.assertLessEqual(window_s, 8.0)

    def test_backend_linger_is_clamped_to_shutdown_deadline(self):
        frontend = _FakeProc("frontend")
        backend = _FakeProc("backend")
        self.manager.add_process(frontend, shutdown_group="frontend")
        self.manager.add_process(backend, shutdown_group="backend")

        sleeps = []
        with patch.dict(os.environ, {BACKEND_POST_FRONTEND_DRAIN_SECONDS_ENV: "10"}):
            with patch("time.sleep", side_effect=lambda seconds: sleeps.append(seconds)):
                self.manager._linger_before_deferred_group_shutdown(
                    time.time() + 0.05
                )

        self.assertEqual(len(sleeps), 1)
        self.assertGreater(sleeps[0], 0)
        self.assertLessEqual(sleeps[0], 0.05)

    def test_parent_process_manager_does_not_defer_first_sigterm(self):
        """Only backend child managers may defer cgroup-wide SIGTERM.

        The top-level parent must enter staged shutdown immediately so it can
        drain frontend before sending the second SIGTERM to backend.
        """
        with patch.dict(
            os.environ,
            {
                DEFER_FIRST_SIGTERM_ENV: DEFER_FIRST_SIGTERM_VALUE,
                DEFER_FIRST_SIGTERM_SECONDS_ENV: "30",
            },
        ):
            manager = ProcessManager(shutdown_timeout=30, monitor_interval=0.01)

        manager._signal_handler(signal.SIGTERM, None)
        self.assertTrue(manager.shutdown_requested)
        self.assertFalse(manager.failure_detected)

    def test_sync_shutdown_timeout_env_from_parsed_config(self):
        """CLI-only shutdown_timeout must still reach C++ through env."""
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(ProcessManager.sync_shutdown_timeout_env(123), 123)
            self.assertEqual(os.environ[SHUTDOWN_TIMEOUT_ENV], "123")

            self.assertEqual(ProcessManager.sync_shutdown_timeout_env(-1), -1)
            self.assertEqual(os.environ[SHUTDOWN_TIMEOUT_ENV], "-1")

    def test_shutdown_timeout_normalization_only_coerces_invalid_values(self):
        self.assertEqual(ProcessManager.normalize_shutdown_timeout_seconds(-1), -1)
        self.assertEqual(ProcessManager.normalize_shutdown_timeout_seconds(0), 600)
        self.assertEqual(ProcessManager.normalize_shutdown_timeout_seconds(-2), 600)
        self.assertEqual(ProcessManager.normalize_shutdown_timeout_seconds("bad"), 600)
        self.assertIsNone(ProcessManager.deferred_group_shutdown_timeout_seconds(-1))

    def test_deferred_group_budget_includes_stop_timeout_headroom(self):
        """Parent backend wait must outlive the C++ gRPC stop timeout."""
        with patch.dict(
            os.environ,
            {
                STOP_TIMEOUT_MS_ENV: "90000",
                DEFERRED_GROUP_SHUTDOWN_HEADROOM_SECONDS_ENV: "7",
            },
        ):
            self.assertEqual(
                ProcessManager.deferred_group_shutdown_timeout_seconds(10),
                97,
            )

    def test_deferred_group_failure_budget_stays_immediate(self):
        """Failure shutdown must not be expanded by graceful headroom."""
        with patch.dict(
            os.environ,
            {
                STOP_TIMEOUT_MS_ENV: "90000",
                DEFERRED_GROUP_SHUTDOWN_HEADROOM_SECONDS_ENV: "7",
            },
        ):
            self.assertEqual(
                ProcessManager.deferred_group_shutdown_timeout_seconds(0),
                0,
            )

    # --- drain-cap activation -----------------------------------------------

    def test_failure_with_dead_backend_force_kills_frontend(self):
        """failure_detected=True + backend already dead → drain_timeout=0, so
        SIGTERM fires immediately and the monitor loop force-kills the
        non-draining frontend after POST_KILL_REAP_WINDOW."""
        frontend = _FakeProc("frontend")
        backend = _FakeProc("backend", alive=False)
        self.manager.add_process(frontend, shutdown_group="frontend")
        self.manager.add_process(backend, shutdown_group="backend")
        self.manager.request_failure_shutdown()

        kills = []

        def fake_kill(pid, sig):
            kills.append((pid, sig))
            if pid == frontend.pid:
                frontend._alive = False

        with patch("os.kill", side_effect=fake_kill), _watchdog(
            5, "failure path with dead backend regressed"
        ):
            self.manager._monitor_processes_health()

        self.assertTrue(frontend.terminated)
        self.assertIn((frontend.pid, signal.SIGKILL), kills)

    def test_failure_with_alive_backend_force_kills_frontend(self):
        """failure_detected=True with backend alive → SIGTERM frontend+backend,
        then monitor force-kills non-draining frontend within REAP window."""
        frontend = _FakeProc("frontend")  # ignores SIGTERM, never dies
        backend = _FakeProc("backend", dies_on_terminate=True)
        self.manager.add_process(frontend, shutdown_group="frontend")
        self.manager.add_process(backend, shutdown_group="backend")
        self.manager.request_failure_shutdown()

        kills = []

        def fake_kill(pid, sig):
            kills.append((pid, sig))
            if pid == frontend.pid:
                frontend._alive = False

        t0 = time.time()
        with patch("os.kill", side_effect=fake_kill), _watchdog(
            5, "failure path with alive backend regressed"
        ):
            self.manager._monitor_processes_health()
        elapsed = time.time() - t0

        self.assertIn((frontend.pid, signal.SIGKILL), kills)
        self.assertLess(elapsed, 3.0, f"failure shutdown took too long: {elapsed:.2f}s")

    # --- abort-upgrade (Patch A, Scenario 7) --------------------------------

    def test_normal_shutdown_backend_crashes_mid_drain_upgrades_to_failure(self):
        """SIGTERM-driven shutdown, backend alive at start, CRASHES (exitcode!=0)
        during drain wait. Drain-abort branch in _wait_process_list_exit upgrades
        failure_detected so the parent eventually exits non-zero. Monitor loop
        then force-kills the non-draining frontend after POST_KILL_REAP_WINDOW."""
        frontend = _FakeProc("frontend")  # ignores SIGTERM
        # Backend dies 0.1s into drain wait with non-zero exitcode (crash).
        backend = _FakeProc(
            "backend", dies_after=time.time() + 0.1, exitcode=1
        )
        self.manager.add_process(frontend, shutdown_group="frontend")
        self.manager.add_process(backend, shutdown_group="backend")
        self.manager.shutdown_requested = True  # mirror SIGTERM handler

        kills = []

        def fake_kill(pid, sig):
            kills.append((pid, sig))
            if pid == frontend.pid:
                frontend._alive = False

        with patch("os.kill", side_effect=fake_kill), _watchdog(
            15, "crash-mid-drain regressed: force-kill never fired"
        ):
            self.manager._monitor_processes_health()

        # Crash → failure_detected escalated.
        self.assertTrue(
            self.manager.failure_detected,
            "abort branch with crashed backend must upgrade failure_detected",
        )
        self.assertIn((frontend.pid, signal.SIGKILL), kills)

    def test_normal_shutdown_backend_clean_exit_mid_drain_no_failure(self):
        """Cgroup-wide SIGTERM scenario: parent + backend + frontends all get
        SIGTERM. Backend handles it cleanly (exitcode=0) before frontends finish
        draining. Drain abort must NOT flag failure — exit code stays 0 — but
        the monitor still force-kills frontends after POST_KILL_REAP_WINDOW
        because they ignore SIGTERM."""
        frontend = _FakeProc("frontend")  # ignores SIGTERM
        # Backend exits cleanly 0.1s into drain wait.
        backend = _FakeProc(
            "backend", dies_after=time.time() + 0.1, exitcode=0
        )
        self.manager.add_process(frontend, shutdown_group="frontend")
        self.manager.add_process(backend, shutdown_group="backend")
        self.manager.shutdown_requested = True  # mirror SIGTERM handler

        kills = []

        def fake_kill(pid, sig):
            kills.append((pid, sig))
            if pid == frontend.pid:
                frontend._alive = False

        with patch("os.kill", side_effect=fake_kill), _watchdog(
            15, "clean-exit-mid-drain regressed: drain never aborted"
        ):
            self.manager._monitor_processes_health()

        # Clean exit must NOT escalate to failure (k8s sees exit 0).
        self.assertFalse(
            self.manager.failure_detected,
            "clean backend exit must NOT flag failure",
        )
        # Frontend still force-killed by monitor's REAP-window timer.
        self.assertIn((frontend.pid, signal.SIGKILL), kills)

    def test_normal_shutdown_backend_sigterm_exit_mid_drain_no_failure(self):
        """A backend process may exit with -SIGTERM during startup shutdown
        before its Python signal handler is installed; requested shutdown
        should not classify that as a crash."""
        frontend = _FakeProc("frontend")  # ignores SIGTERM
        backend = _FakeProc(
            "backend", dies_after=time.time() + 0.1, exitcode=-signal.SIGTERM
        )
        self.manager.add_process(frontend, shutdown_group="frontend")
        self.manager.add_process(backend, shutdown_group="backend")
        self.manager.shutdown_requested = True

        def fake_kill(pid, sig):
            if pid == frontend.pid:
                frontend._alive = False

        with patch("os.kill", side_effect=fake_kill), _watchdog(
            15, "clean SIGTERM backend exit regressed"
        ):
            self.manager._monitor_processes_health()

        self.assertFalse(self.manager.failure_detected)

    # --- frontend-only scenarios (no backend group registered) -------------

    def test_frontend_only_sigterm_drains_naturally(self):
        """No backend group registered. SIGTERM lets frontend drain (here:
        dies after 0.2s, well under shutdown_timeout). No SIGKILL, no failure."""
        # Frontend exits cleanly on SIGTERM after a short fake drain.
        frontend = _FakeProc(
            "frontend", dies_after=time.time() + 0.2
        )
        self.manager.add_process(frontend, shutdown_group="frontend")
        # No backend at all.
        self.manager.shutdown_requested = True

        kills = []
        with patch(
            "os.kill", side_effect=lambda pid, sig: kills.append((pid, sig))
        ), _watchdog(5, "frontend-only graceful drain regressed"):
            self.manager._monitor_processes_health()

        self.assertFalse(self.manager.failure_detected)
        self.assertNotIn(
            signal.SIGKILL,
            [sig for _, sig in kills],
            "frontend-only graceful path must not SIGKILL",
        )
        self.assertFalse(frontend.is_alive())

    def test_frontend_only_undrained_force_killed_by_monitor(self):
        """No backend group. Frontend ignores SIGTERM → monitor force-kills
        after POST_KILL_REAP_WINDOW. Force kill is NOT a failure."""
        frontend = _FakeProc("frontend")  # never drains
        self.manager.add_process(frontend, shutdown_group="frontend")
        self.manager.shutdown_requested = True

        kills = []

        def fake_kill(pid, sig):
            kills.append((pid, sig))
            if pid == frontend.pid:
                frontend._alive = False

        with patch("os.kill", side_effect=fake_kill), _watchdog(
            5, "frontend-only force kill regressed"
        ):
            self.manager._monitor_processes_health()

        self.assertIn((frontend.pid, signal.SIGKILL), kills)
        # Timeout-driven force kill is NOT a failure.
        self.assertFalse(self.manager.failure_detected)

    # --- timing: graceful waits, failure does not -------------------------

    def test_graceful_drain_actually_waits_for_frontend(self):
        """Graceful path must wait for frontend to drain before SIGTERM'ing
        backend — that's the whole point of staged shutdown. Verifies the wait
        actually happens by checking backend SIGTERM timestamp vs t0."""
        DRAIN_DELAY = 0.3  # frontend drains after this long
        frontend = _FakeProc("frontend", dies_after=time.time() + DRAIN_DELAY)
        backend = _FakeProc("backend", dies_on_terminate=True)
        backend_terminated_at: list = []
        original_terminate = backend.terminate

        def tracked():
            backend_terminated_at.append(time.time())
            original_terminate()

        backend.terminate = tracked  # type: ignore[method-assign]

        self.manager.shutdown_timeout = 5  # plenty of headroom for the drain
        self.manager.add_process(frontend, shutdown_group="frontend")
        self.manager.add_process(backend, shutdown_group="backend")
        self.manager.shutdown_requested = True  # mirror SIGTERM handler

        t0 = time.time()
        with patch("os.kill", side_effect=lambda pid, sig: None), patch.dict(
            os.environ, {BACKEND_POST_FRONTEND_DRAIN_SECONDS_ENV: "0"}
        ), _watchdog(5, "graceful drain timing regressed"):
            self.manager._monitor_processes_health()

        self.assertEqual(len(backend_terminated_at), 1, "backend SIGTERM'd once")
        backend_delay = backend_terminated_at[0] - t0
        self.assertGreaterEqual(
            backend_delay,
            DRAIN_DELAY * 0.8,
            f"backend SIGTERM'd too early ({backend_delay:.3f}s < {DRAIN_DELAY}s); "
            "drain wait did not actually wait for frontend",
        )
        self.assertFalse(self.manager.failure_detected)

    def test_backend_gets_full_shutdown_budget_after_frontend_drain(self):
        """Frontend drain and backend cleanup need separate budgets.

        Regression coverage for graceful restarts where frontend consumes most
        of shutdown_timeout, then backend still needs time to stop C++ RPC
        services without being hit by the monitor's SIGKILL pass.
        """
        t0 = time.time()
        frontend = _FakeProc("frontend", dies_after=t0 + 0.18)
        backend = _FakeProc("backend", dies_after=t0 + 0.32)
        self.manager.shutdown_timeout = 0.2
        self.manager.monitor_interval = 0.01
        self.manager.POST_KILL_REAP_WINDOW = 0.02
        self.manager.add_process(frontend, shutdown_group="frontend")
        self.manager.add_process(backend, shutdown_group="backend")
        self.manager.shutdown_requested = True

        kills = []
        with patch(
            "os.kill", side_effect=lambda pid, sig: kills.append((pid, sig))
        ), _watchdog(3, "backend shutdown budget was squeezed by frontend drain"):
            self.manager._monitor_processes_health()

        self.assertEqual(
            kills,
            [],
            "backend exited within its own shutdown budget and must not be killed",
        )
        self.assertGreaterEqual(
            time.time() - t0,
            0.30,
            "monitor did not wait for backend cleanup after frontend drain",
        )
        self.assertFalse(self.manager.failure_detected)

    def test_backend_only_deferred_group_gets_shutdown_headroom(self):
        """Backend rank managers have no frontend group but still need
        headroom beyond the C++ stop timeout for grpc/http cleanup."""
        t0 = time.time()
        backend = _FakeProc("rank-0", dies_after=t0 + 1.2)
        self.manager.shutdown_timeout = 1
        self.manager.monitor_interval = 0.01
        self.manager.POST_KILL_REAP_WINDOW = 0.02
        self.manager.add_process(backend, shutdown_group="backend")
        self.manager.shutdown_requested = True

        kills = []
        with patch.dict(
            os.environ,
            {DEFERRED_GROUP_SHUTDOWN_HEADROOM_SECONDS_ENV: "1"},
        ), patch(
            "os.kill", side_effect=lambda pid, sig: kills.append((pid, sig))
        ), _watchdog(3, "backend-only deferred headroom regressed"):
            self.manager._monitor_processes_health()

        self.assertEqual(kills, [])
        self.assertGreaterEqual(time.time() - t0, 1.1)
        self.assertFalse(self.manager.failure_detected)

    def test_failure_path_skips_drain_even_with_long_timeout(self):
        """Failure path must use drain_timeout=0 regardless of shutdown_timeout.
        Set shutdown_timeout=60 (would hang for a minute on graceful path);
        verify the failure shutdown completes in well under a second."""
        self.manager.shutdown_timeout = 60
        frontend = _FakeProc("frontend")  # ignores SIGTERM, never drains
        backend = _FakeProc("backend", dies_on_terminate=True)
        self.manager.add_process(frontend, shutdown_group="frontend")
        self.manager.add_process(backend, shutdown_group="backend")
        self.manager.request_failure_shutdown()

        def fake_kill(pid, sig):
            if pid == frontend.pid:
                frontend._alive = False

        t0 = time.time()
        with patch("os.kill", side_effect=fake_kill), _watchdog(
            3, "failure path waited on shutdown_timeout instead of skipping drain"
        ):
            self.manager._monitor_processes_health()
        elapsed = time.time() - t0

        # Should be ~POST_KILL_REAP_WINDOW (0.1s) + a bit, NOT 60s.
        self.assertLess(
            elapsed,
            1.0,
            f"failure shutdown took {elapsed:.2f}s — drain wait was not skipped",
        )

    def test_failure_path_does_not_wait_backend_graceful_headroom(self):
        """A stuck backend on failure path must be SIGKILL'd promptly."""
        self.manager.shutdown_timeout = 60
        self.manager.POST_KILL_REAP_WINDOW = 0.05
        frontend = _FakeProc("frontend", dies_on_terminate=True)
        backend = _FakeProc("backend")  # ignores SIGTERM, never drains
        self.manager.add_process(frontend, shutdown_group="frontend")
        self.manager.add_process(backend, shutdown_group="backend")
        self.manager.request_failure_shutdown()

        kills = []

        def fake_kill(pid, sig):
            kills.append((pid, sig))
            if pid == backend.pid:
                backend._alive = False

        t0 = time.time()
        with patch("os.kill", side_effect=fake_kill), _watchdog(
            3, "failure path waited on backend graceful headroom"
        ):
            self.manager._monitor_processes_health()
        elapsed = time.time() - t0

        self.assertLess(
            elapsed,
            1.0,
            f"failure shutdown took {elapsed:.2f}s — backend headroom was used",
        )
        self.assertIn((backend.pid, signal.SIGKILL), kills)

    def test_graceful_shutdown_waits_for_default_backend_group(self):
        """Backend-rank ProcessManager registers ranks in the default group, not
        frontend/backend groups. A normal SIGTERM must still wait the graceful
        budget for those ranks to exit before the monitor's SIGKILL pass."""
        rank = _FakeProc("rank-0", dies_after=time.time() + 0.3)
        self.manager.shutdown_timeout = 2
        self.manager.POST_KILL_REAP_WINDOW = 0.05
        self.manager.add_process(rank, shutdown_group="default")
        self.manager.shutdown_requested = True

        kills = []
        t0 = time.time()
        with patch(
            "os.kill", side_effect=lambda pid, sig: kills.append((pid, sig))
        ), _watchdog(5, "default backend group graceful wait regressed"):
            self.manager._monitor_processes_health()
        elapsed = time.time() - t0

        self.assertGreaterEqual(
            elapsed,
            0.25,
            "default backend rank was not given the graceful shutdown budget",
        )
        self.assertEqual(kills, [], "rank exited within budget and must not be killed")
        self.assertFalse(self.manager.failure_detected)

    # --- _join_all_processes always uses POST_KILL_REAP_WINDOW --------------

    def test_join_uses_post_kill_reap_window_regardless_of_state(self):
        """After refactor, _join_all_processes is just bounded wait4(); the
        graceful drain happens upstream in _wait_process_list_exit. The reap
        window is identical whether shutdown_timeout is small/large or
        failure_detected — join's job is only to reap, not to honor drain
        budgets."""
        for shutdown_timeout, failure in [(1, False), (1, True), (60, True)]:
            with self.subTest(shutdown_timeout=shutdown_timeout, failure=failure):
                manager = ProcessManager(
                    shutdown_timeout=shutdown_timeout, monitor_interval=0.01
                )
                proc = _FakeProc("any", alive=True)
                proc._mock_name = "fake"
                manager.add_process(proc)
                if failure:
                    manager.failure_detected = True

                captured = []

                def capture_join(self_proc, timeout=None):
                    captured.append(timeout)
                    proc._alive = False

                with patch.object(_FakeProc, "join", capture_join):
                    manager._join_all_processes()

                self.assertEqual(len(captured), 1)
                self.assertIsNotNone(captured[0])
                self.assertGreater(captured[0], 0)
                self.assertLessEqual(captured[0], manager.POST_KILL_REAP_WINDOW)

    # --- monitor_and_release_processes empty-list --------------------------

    def test_empty_processes_with_failure_flag_still_exits_nonzero(self):
        """Empty processes + failure_detected must still os._exit(1).

        Production path: backend Process construction (pickle/spawn) or
        config validation raises before add_process() runs. The except branch
        calls request_failure_shutdown() + monitor_and_release_processes()
        in finally; processes is still []. Without this, the parent silently
        exits 0 and k8s/systemd never restarts.
        """
        self.manager.request_failure_shutdown()
        with patch("os._exit") as mock_exit:
            self.manager.monitor_and_release_processes()
            mock_exit.assert_called_once_with(1)

    def test_empty_processes_without_failure_returns_cleanly(self):
        """No processes + no failure → quiet return, no os._exit."""
        with patch("os._exit") as mock_exit:
            self.manager.monitor_and_release_processes()
            mock_exit.assert_not_called()

    def test_unrequested_clean_child_exit_still_exits_parent_nonzero(self):
        """Exit code 0 is only graceful after the parent requested shutdown."""
        self.manager.add_process(_FakeProc("frontend", alive=False, exitcode=0))

        with patch("os._exit") as mock_exit:
            self.manager.monitor_and_release_processes()

        self.assertTrue(self.manager.failure_detected)
        mock_exit.assert_called_once_with(1)

    # --- normal SIGTERM with alive backend keeps -1 semantics ---------------

    def test_signal_shutdown_alive_backend_no_force_kill(self):
        """SIGTERM with backend healthy and frontend that dies on SIGTERM →
        graceful drain succeeds, no SIGKILL, no failure flag."""
        frontend = _FakeProc("frontend", dies_on_terminate=True)
        backend = _FakeProc("backend", dies_on_terminate=True)
        self.manager.add_process(frontend, shutdown_group="frontend")
        self.manager.add_process(backend, shutdown_group="backend")
        self.manager.shutdown_requested = True  # like SIGTERM handler

        kills = []
        with patch(
            "os.kill", side_effect=lambda pid, sig: kills.append((pid, sig))
        ), _watchdog(5, "graceful path regressed"):
            self.manager._monitor_processes_health()

        self.assertNotIn(signal.SIGKILL, [sig for _, sig in kills])
        self.assertFalse(
            self.manager.failure_detected,
            "graceful path should not flip failure_detected",
        )


if __name__ == "__main__":
    unittest.main()
