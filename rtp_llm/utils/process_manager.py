import logging
import os
import signal
import threading
import time
from multiprocessing import Process
from typing import Callable, Dict, List, Optional, Tuple

import psutil


class ProcessManager:
    """Process manager for managing and monitoring processes"""

    def __init__(self, shutdown_timeout: int = 50, monitor_interval: int = 1):
        self.processes: List[Process] = []
        self.shutdown_requested = False
        self.terminated = False
        self.first_dead_time = 0
        self.shutdown_timeout = shutdown_timeout
        self.monitor_interval = monitor_interval
        self._shutdown_processes: Dict[int, Tuple[psutil.Process, float]] = {}
        self._shutdown_process_order: List[int] = []

        # Health check related attributes
        self.health_check_processes: List[Process] = []
        self.health_check_configs: Dict[str, dict] = {}  # process_name -> config
        self.health_check_threads: List[threading.Thread] = []
        self.health_check_status: Dict[str, dict] = {}  # process_name -> status
        self.health_check_lock = threading.Lock()
        self._shutdown_event = threading.Event()

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
        self._shutdown_event.set()

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

        self._snapshot_process_trees()
        logging.info("Shutdown requested, terminating processes...")
        for proc in self.processes:
            if proc.is_alive():
                logging.info(f"Sending SIGTERM to process {proc.pid}")
                proc.terminate()
            else:
                logging.info(f"proc.name [{proc.name}] pid[{proc.pid}] is not alived")
        self.terminated = True
        self.first_dead_time = time.monotonic()

    def _remember_shutdown_process(self, process: psutil.Process) -> None:
        try:
            create_time = process.create_time()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            return

        if process.pid not in self._shutdown_processes:
            self._shutdown_process_order.append(process.pid)
        self._shutdown_processes[process.pid] = (process, create_time)

    def _snapshot_process_trees(self) -> None:
        """Remember process identities before parents can exit and orphan children."""
        for managed_process in self.processes:
            try:
                if managed_process.pid is None or not managed_process.is_alive():
                    continue
                root = psutil.Process(managed_process.pid)
                descendants = root.children(recursive=True)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue

            self._remember_shutdown_process(root)
            for descendant in descendants:
                self._remember_shutdown_process(descendant)

    @staticmethod
    def _tracked_process_is_alive(
        tracked_process: Tuple[psutil.Process, float]
    ) -> bool:
        process, create_time = tracked_process
        try:
            if process.create_time() != create_time:
                return False
            return process.is_running() and process.status() != psutil.STATUS_ZOMBIE
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            return False

    def _is_any_shutdown_process_alive(self) -> bool:
        return any(
            self._tracked_process_is_alive(process)
            for process in self._shutdown_processes.values()
        )

    def _force_kill_processes(self):
        """Force kill processes after timeout"""
        logging.warning(
            f"Graceful shutdown timeout ({self.shutdown_timeout}s), force killing..."
        )
        self._snapshot_process_trees()
        root_pids = {proc.pid for proc in self.processes if proc.pid is not None}
        killed_processes = []
        killed_pids = set()

        for pid in reversed(self._shutdown_process_order):
            tracked_process = self._shutdown_processes.get(pid)
            if tracked_process is None or not self._tracked_process_is_alive(
                tracked_process
            ):
                continue
            process, _ = tracked_process
            process_type = "process" if pid in root_pids else "descendant process"
            logging.warning(f"Force killing {process_type} {pid}")
            try:
                process.kill()
                if pid not in root_pids:
                    killed_processes.append(process)
                killed_pids.add(pid)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass

        if killed_processes:
            psutil.wait_procs(killed_processes, timeout=1)

        for proc in self.processes:
            if proc.is_alive() and proc.pid not in killed_pids:
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
                deadline = time.monotonic() + self.shutdown_timeout

        for proc in self.processes:
            try:
                timeout = None
                if deadline is not None:
                    timeout = max(0.1, deadline - time.monotonic())
                proc.join(timeout)
            except Exception as e:
                logging.error(f"Error joining process {proc.pid}: {e}")
        logging.info("All processes joined")

    def _monitor_processes_health(self):
        """Monitor process health and handle failures"""
        while True:
            if not self.terminated:
                self._snapshot_process_trees()

            direct_process_alive = self._is_any_process_alive()
            shutdown_process_alive = (
                self.terminated and self._is_any_shutdown_process_alive()
            )
            if not direct_process_alive and not shutdown_process_alive:
                break

            # Check shutdown signal
            if self.shutdown_requested and not self.terminated:
                self._terminate_processes()

            # Check sub-process status
            elif not self._is_all_processes_alive() and not self.terminated:
                for proc in self.processes:
                    if not proc.is_alive():
                        logging.error(f"Process {proc.pid} died unexpectedly")
                if self.first_dead_time == 0:
                    self.first_dead_time = time.monotonic()
                logging.error("Some processes died unexpectedly, terminating all...")
                self._terminate_processes()

            # Force kill after timeout (only if shutdown_timeout != -1)
            if (
                self.terminated
                and self.shutdown_timeout != -1
                and (time.monotonic() - self.first_dead_time) > self.shutdown_timeout
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
        self._shutdown_event.set()

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
            if self.shutdown_requested:
                with self.health_check_lock:
                    self.health_check_status[process_name]["ready"] = False
                    self.health_check_status[process_name]["checked"] = True
                logging.info(
                    f"{process_name} health check cancelled by shutdown request"
                )
                return
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
                        ready = not self.shutdown_requested
                        self.health_check_status[process_name]["ready"] = ready
                        self.health_check_status[process_name]["checked"] = True
                    if not ready:
                        logging.info(
                            f"{process_name} health check cancelled by shutdown request"
                        )
                        return
                    logging.info(f"{process_name} is ready")
                    return
            except Exception as e:
                logging.debug(f"{process_name} health check exception: {str(e)}")
            self._shutdown_event.wait(retry_interval)

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

        deadline = None if timeout is None else time.monotonic() + timeout
        while True:
            if self.shutdown_requested:
                logging.info("Health checks cancelled by shutdown request")
                return False

            active_thread = next(
                (thread for thread in self.health_check_threads if thread.is_alive()),
                None,
            )
            if active_thread is None:
                break

            wait_timeout = 0.1
            if deadline is not None:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                wait_timeout = min(wait_timeout, remaining)
            active_thread.join(timeout=wait_timeout)

        if self.shutdown_requested:
            logging.info("Health checks cancelled by shutdown request")
            return False

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
