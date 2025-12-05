import logging
import os
import signal
import sys
import time
from abc import ABC, abstractmethod
from multiprocessing import Process
from typing import List


class BaseProcessManager(ABC):
    """Base class for process management with common functionality"""

    DEFAULT_SHUTDOWN_TIMEOUT = 50
    MONITOR_INTERVAL = 1

    def __init__(self):
        self.processes: List[Process] = []
        self.shutdown_requested = False
        self.terminated = False
        self.first_dead_time = 0
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
        """Set the processes to manage"""
        self.processes = processes if processes else []

    def set_backend_process(self, process: Process):
        """Set a single backend process"""
        if process:
            self.processes = [process]

    def set_frontend_processes(self, processes: List[Process]):
        """Set frontend processes"""
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
        self.terminated = True
        self.first_dead_time = time.time()

    def _force_kill_processes(self):
        """Force kill processes after timeout"""
        logging.warning(
            f"Graceful shutdown timeout ({self.DEFAULT_SHUTDOWN_TIMEOUT}s), force killing..."
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

    def _join_all_processes(self):
        """Join all processes"""
        for proc in self.processes:
            try:
                proc.join()
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
                if self.first_dead_time == 0:
                    self.first_dead_time = time.time()
                logging.error("Some processes died unexpectedly, terminating all...")
                self._terminate_processes()

            # Force kill after timeout
            if (
                self.terminated
                and (time.time() - self.first_dead_time) > self.DEFAULT_SHUTDOWN_TIMEOUT
            ):
                self._force_kill_processes()
                break

            time.sleep(self.MONITOR_INTERVAL)

    @abstractmethod
    def monitor_and_release_processes(self):
        """Monitor and release processes - subclasses must implement"""
        pass

    def graceful_shutdown(self):
        """Trigger graceful shutdown"""
        self.shutdown_requested = True


class ServerProcessManager(BaseProcessManager):
    """Process manager for server processes (backend + frontend)"""

    def __init__(self):
        super().__init__()
        self.backend_process = None
        self.frontend_processes = []

    def set_backend_process(self, process):
        """Set the backend process (for backward compatibility)"""
        if process:
            self.backend_process = process
            self.processes = [process]

    def set_frontend_processes(self, processes):
        """Set the frontend processes (for backward compatibility)"""
        if processes:
            self.frontend_processes = processes
            # Combine with backend process
            all_processes = []
            if self.backend_process:
                all_processes.append(self.backend_process)
            all_processes.extend(processes)
            self.processes = all_processes

    def monitor_and_release_processes(self):
        """Monitor all managed processes and handle failures"""
        if not self.processes:
            logging.info("No processes to monitor")
            return

        logging.info(f"Monitoring processes: {[proc.name for proc in self.processes]}")
        self._monitor_processes_health()
        self._join_all_processes()


class BackendRankProcessManager(BaseProcessManager):
    """Process manager specifically for backend rank processes in multi-TP scenarios"""

    def monitor_and_join(self):
        """Monitor process health and join processes when complete"""
        if not self.processes:
            logging.info("No rank processes to monitor")
            return

        logging.info(
            f"Monitoring rank processes: {[proc.name for proc in self.processes]}"
        )
        self._monitor_processes_health()
        self._join_all_processes()

    def monitor_and_release_processes(self):
        """Alias for backward compatibility"""
        self.monitor_and_join()
