import logging
import os
import signal
import threading
import time
from multiprocessing import Process
from typing import Callable, Dict, List, Optional


class ProcessManager:
    """Process manager for managing and monitoring processes"""

    def __init__(self, shutdown_timeout: int = 50, monitor_interval: int = 1):
        self.processes: List[Process] = []
        self.shutdown_requested = False
        self.terminated = False
        self.first_dead_time = 0
        self.shutdown_timeout = shutdown_timeout
        self.monitor_interval = monitor_interval

        # Health check related attributes
        self.health_check_processes: List[Process] = []
        self.health_check_configs: Dict[str, dict] = {}  # process_name -> config
        self.health_check_threads: List[threading.Thread] = []
        self.health_check_status: Dict[str, dict] = {}  # process_name -> status
        self.health_check_lock = threading.Lock()

        self._setup_signal_handlers()

    def _setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown"""
        signal.signal(signal.SIGTERM, self._signal_handler)
        signal.signal(signal.SIGINT, self._signal_handler)

    def _signal_handler(self, signum, frame):
        """Handle termination signals gracefully"""
        logging.info(
            f"Process manager received signal {signum}, initiating shutdown..."
        )
        self.shutdown_requested = True

    def set_processes(self, processes: List[Process]):
        """Set the processes to manage (replaces existing list)"""
        self.processes = processes if processes else []

    def add_process(self, process: Process):
        """Add a single process to manage"""
        if process:
            self.processes.append(process)

    def add_processes(self, processes: List[Process]):
        """Add multiple processes to manage"""
        if processes:
            self.processes.extend(processes)

    def _terminate_processes(self):
        """Terminate all managed processes"""
        if self.terminated:
            return

        logging.info("Shutdown requested, terminating processes...")
        for proc in self.processes:
            if proc.is_alive():
                logging.info(f"Sending SIGTERM to process {proc.pid}")
                proc.terminate()
            else:
                logging.info(f"yemu_debug proc.name [{proc.name}] pid[{proc.pid}]")
        self.terminated = True
        self.first_dead_time = time.time()

    def _force_kill_processes(self):
        """Force kill processes after timeout"""
        logging.warning(
            f"Graceful shutdown timeout ({self.shutdown_timeout}s), force killing..."
        )
        for proc in self.processes:
            if proc.is_alive():
                logging.warning(f"Force killing process {proc.pid}")
                try:
                    os.kill(proc.pid, signal.SIGKILL)
                except (OSError, ProcessLookupError):
                    # Process may have already died
                    pass

    def _is_any_process_alive(self) -> bool:
        """Check if any process is still alive"""
        return any(proc.is_alive() for proc in self.processes)

    def _is_all_processes_alive(self) -> bool:
        """Check if all processes are still alive"""
        return all(proc.is_alive() for proc in self.processes)

    def is_available(self) -> bool:
        """
        Check if ProcessManager is available.
        Returns False if:
        - Shutdown has been requested
        - Any managed process has died
        """
        if self.shutdown_requested:
            return False
        if self.processes and not self._is_all_processes_alive():
            return False
        return True

    def _join_all_processes(self):
        """Join all processes"""
        deadline = None
        if self.shutdown_timeout != -1:
            if self.first_dead_time > 0:
                deadline = self.first_dead_time + self.shutdown_timeout
            else:
                deadline = time.time() + self.shutdown_timeout

        for proc in self.processes:
            try:
                timeout = None
                if deadline is not None:
                    timeout = max(0.1, deadline - time.time())
                proc.join(timeout)
            except Exception as e:
                logging.error(f"Error joining process {proc.pid}: {e}")
        logging.info("All processes joined")

    def _monitor_processes_health(self):
        """Monitor process health and handle failures"""
        while self._is_any_process_alive():
            # Check shutdown signal
            if self.shutdown_requested and not self.terminated:
                self._terminate_processes()

            # Check sub-process status
            elif not self._is_all_processes_alive() and not self.terminated:
                for proc in self.processes:
                    if not proc.is_alive():
                        logging.error(f"Process {proc.pid} died unexpectedly")
                if self.first_dead_time == 0:
                    self.first_dead_time = time.time()
                logging.error("Some processes died unexpectedly, terminating all...")
                self._terminate_processes()

            # Force kill after timeout (only if shutdown_timeout != -1)
            if (
                self.terminated
                and self.shutdown_timeout != -1
                and (time.time() - self.first_dead_time) > self.shutdown_timeout
            ):
                self._force_kill_processes()
                break

            time.sleep(self.monitor_interval)

    def monitor_and_release_processes(self):
        """Monitor all processes until completion or failure"""
        if not self.processes:
            logging.info("No processes to monitor")
            return

        logging.info(f"Monitoring {len(self.processes)} processes")
        self._monitor_processes_health()
        self._join_all_processes()
        logging.info("Process monitoring completed")

    def graceful_shutdown(self):
        """Trigger graceful shutdown"""
        self.shutdown_requested = True

    def register_health_check(
        self,
        processes: list[Process],
        process_name: str,
        check_ready_fn: Callable[[], bool],
        retry_interval_seconds: float = 0.1,
    ):
        """
        Register a health check for a process

        Args:
            processes: The processes to monitor
            process_name: Name identifier for the process
            check_ready_fn: Custom function to check if service is ready.
                          Should return True when ready, False otherwise.
            retry_interval_seconds: Interval between health checks
        """
        self.health_check_processes.extend(processes)
        self.health_check_configs[process_name] = {
            "processes": processes,
            "retry_interval_seconds": retry_interval_seconds,
            "check_ready_fn": check_ready_fn,
        }

        # Initialize status
        with self.health_check_lock:
            self.health_check_status[process_name] = {
                "ready": False,
                "checked": False,
            }

    def _health_check_worker(self, process_name: str):
        """
        Worker thread for health checking a specific process

        Args:
            process_name: Name of the process to check
        """
        config = self.health_check_configs[process_name]
        # fail fast, if backend fail, frontend should exit as soon as possible
        processes = self.health_check_processes
        retry_interval = config["retry_interval_seconds"]
        check_ready_fn = config["check_ready_fn"]

        while True:
            if not self.is_available():
                with self.health_check_lock:
                    self.health_check_status[process_name]["ready"] = False
                    self.health_check_status[process_name]["checked"] = True
                logging.error(f"{process_name} process manager is not available")
                return
            # Check if process is still alive
            if not all(proc.is_alive() for proc in processes):
                with self.health_check_lock:
                    self.health_check_status[process_name]["ready"] = False
                    self.health_check_status[process_name]["checked"] = True
                logging.error(f"{process_name} process is not alive")
                return

            try:
                if check_ready_fn():
                    with self.health_check_lock:
                        self.health_check_status[process_name]["ready"] = True
                        self.health_check_status[process_name]["checked"] = True
                    logging.info(f"{process_name} is ready")
                    return
            except Exception as e:
                logging.debug(f"{process_name} health check exception: {str(e)}")
            time.sleep(retry_interval)

    def start_parallel_health_checks(self):
        """
        Start parallel health checks for all registered processes
        Creates a thread for each registered health check
        """
        if not self.health_check_configs:
            logging.info("No health checks registered")
            return

        logging.info(
            f"Starting parallel health checks for {len(self.health_check_configs)} processes"
        )

        for process_name in self.health_check_configs.keys():
            thread = threading.Thread(
                target=self._health_check_worker,
                args=(process_name,),
                daemon=True,
                name=f"health_check_{process_name}",
            )
            self.health_check_threads.append(thread)
            thread.start()

    def wait_for_health_checks(self, timeout: Optional[float] = None) -> bool:
        """
        Wait for all health check threads to complete

        Args:
            timeout: Maximum time to wait in seconds. None means wait indefinitely

        Returns:
            True if all health checks passed, False otherwise
        """
        if not self.health_check_threads:
            logging.info("No health check threads to wait for")
            return True

        logging.info(
            f"Waiting for {len(self.health_check_threads)} health checks to complete..."
        )

        for thread in self.health_check_threads:
            thread.join(timeout=timeout)

        # Check results
        all_ready = True
        with self.health_check_lock:
            for process_name, status in self.health_check_status.items():
                if not status["checked"]:
                    logging.warning(f"{process_name} health check did not complete")
                    all_ready = False
                elif not status["ready"]:
                    logging.error(f"{process_name} health check failed")
                    all_ready = False
                else:
                    logging.info(f"{process_name} health check passed")

        return all_ready

    def run_health_checks(self, timeout: Optional[float] = None) -> bool:
        """
        Start parallel health checks and wait for completion
        This is a convenience method that combines start_parallel_health_checks()
        and wait_for_health_checks()

        Args:
            timeout: Maximum time to wait in seconds. None means wait indefinitely

        Returns:
            True if all health checks passed, False otherwise
        """
        self.start_parallel_health_checks()
        return self.wait_for_health_checks(timeout=timeout)
