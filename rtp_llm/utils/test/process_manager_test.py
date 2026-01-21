import logging
import multiprocessing
import os
import signal
import time
import unittest
from unittest.mock import Mock, patch

from rtp_llm.utils.process_manager import ProcessManager


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
        self.assertFalse(self.manager.terminated)
        self.assertEqual(self.manager.first_dead_time, 0)

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
        self.manager._terminate_processes()
        self.assertTrue(self.manager.terminated)
        self.assertGreater(self.manager.first_dead_time, 0)

        # Wait for process to handle SIGTERM and die
        timeout = time.time() + 3
        while proc.is_alive() and time.time() < timeout:
            time.sleep(0.1)

        # If still alive, force join
        if proc.is_alive():
            proc.join(timeout=1)

        self.assertFalse(proc.is_alive())

        # Test idempotency
        old_time = self.manager.first_dead_time
        self.manager._terminate_processes()
        self.assertEqual(self.manager.first_dead_time, old_time)

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

        # Monitor - should detect crash and terminate all
        with patch("logging.error") as mock_error:
            self.manager.monitor_and_release_processes()
            mock_error.assert_called()

        # All processes should be terminated
        for proc in procs:
            self.assertFalse(proc.is_alive())
        self.assertTrue(self.manager.terminated)

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

        # Send shutdown signal after a short delay
        time.sleep(0.5)
        self.manager.graceful_shutdown()

        # Wait for monitoring to complete
        monitor_thread.join(timeout=5)

        # Wait a bit more for process to fully terminate
        time.sleep(0.5)

        # Process should be terminated
        self.assertFalse(proc.is_alive())
        self.assertTrue(self.manager.shutdown_requested)
        self.assertTrue(self.manager.terminated)

    def test_monitor_with_force_kill_timeout(self):
        """Test force kill after timeout"""
        # Mock a process that won't die
        mock_proc = Mock()
        mock_proc.is_alive.return_value = True
        mock_proc.pid = 12345
        # Mark it as a mock to skip in teardown
        mock_proc._mock_name = "mock_proc"

        self.manager.add_process(mock_proc)
        self.manager.terminated = True
        self.manager.first_dead_time = time.time() - self.manager.shutdown_timeout - 1

        # Should force kill
        with patch("os.kill") as mock_kill:
            self.manager._monitor_processes_health()
            mock_kill.assert_called_with(12345, signal.SIGKILL)

    def test_graceful_shutdown(self):
        """Test graceful shutdown method"""
        self.assertFalse(self.manager.shutdown_requested)
        self.manager.graceful_shutdown()
        self.assertTrue(self.manager.shutdown_requested)

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

        # Request shutdown
        self.manager.graceful_shutdown()

        # Wait for monitoring to complete
        monitor_thread.join()

        # Process should be terminated
        self.assertFalse(proc1.is_alive())
        self.assertFalse(proc2.is_alive())
        self.assertTrue(self.manager.terminated)


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


if __name__ == "__main__":
    unittest.main()
