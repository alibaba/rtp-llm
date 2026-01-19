import logging
import os
import signal
import sys
import time
from multiprocessing import Process
from typing import List


class ProcessManager:
    """Generic process manager for managing and monitoring processes"""

    def __init__(self, shutdown_timeout: int = 50, monitor_interval: int = 1):
        self.processes: List[Process] = []
        self.shutdown_requested = False
        self.terminated = False
        self.first_dead_time = 0
        self.shutdown_timeout = shutdown_timeout
        self.monitor_interval = monitor_interval
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
                logging.info(f"Sending SIGKILL to process {proc.pid}")
                # Use proc.kill() which sends SIGKILL, or os.kill() directly
                try:
                    proc.kill()  # This sends SIGKILL
                except Exception as e:
                    logging.warning(
                        f"Failed to kill process {proc.pid} via proc.kill(): {e}"
                    )
                    try:
                        os.kill(proc.pid, signal.SIGKILL)
                    except (OSError, ProcessLookupError) as e2:
                        logging.warning(
                            f"Failed to kill process {proc.pid} via os.kill(): {e2}"
                        )
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
