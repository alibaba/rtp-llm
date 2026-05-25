import logging
import os
import signal
import threading
import time
from multiprocessing import Process
from typing import Callable, Dict, List, Optional, Set


class ProcessManager:
    """Process manager for managing and monitoring processes"""

    POST_KILL_REAP_WINDOW = 5  # seconds: post-SIGTERM wait before SIGKILL/reap
    # Groups whose in-flight work we let drain after SIGTERM.
    DRAIN_GROUPS: Set[str] = {"frontend", "ingress"}
    # Groups SIGTERM'd only after drain groups stop sending RPCs.
    DEFERRED_GROUPS: Set[str] = {"backend"}

    def __init__(self, shutdown_timeout: int = 600, monitor_interval: int = 1):
        if shutdown_timeout <= 0:
            logging.warning(
                f"shutdown_timeout={shutdown_timeout} is non-positive; "
                "coercing to 600s so the parent cannot hang on a "
                "non-draining child."
            )
            shutdown_timeout = 600
        self.processes: List[Process] = []
        self.shutdown_requested = False
        self.failure_detected = False
        self.shutdown_timeout = shutdown_timeout
        self.monitor_interval = monitor_interval
        self.process_groups: Dict[str, List[Process]] = {}
        self.shutdown_group_order: List[str] = []

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

    def _register_group(self, shutdown_group: str):
        if shutdown_group not in self.process_groups:
            self.process_groups[shutdown_group] = []
        if shutdown_group not in self.shutdown_group_order:
            self.shutdown_group_order.append(shutdown_group)

    def set_processes(
        self, processes: List[Process], shutdown_group: str = "default"
    ):
        """Set the processes to manage (replaces existing list)"""
        self.processes = processes if processes else []
        self.process_groups = {}
        self.shutdown_group_order = []
        self._register_group(shutdown_group)
        self.process_groups[shutdown_group] = self.processes.copy()

    def add_process(self, process: Process, shutdown_group: str = "default"):
        """Add a single process to manage"""
        if process:
            self.processes.append(process)
            self._register_group(shutdown_group)
            self.process_groups[shutdown_group].append(process)

    def add_processes(
        self, processes: List[Process], shutdown_group: str = "default"
    ):
        """Add multiple processes to manage"""
        if processes:
            self.processes.extend(processes)
            self._register_group(shutdown_group)
            self.process_groups[shutdown_group].extend(processes)

    def _terminate_process_list(self, processes: List[Process], group_name: str):
        for proc in processes:
            if proc.is_alive():
                logging.info(
                    f"Sending SIGTERM to {group_name} process {proc.name} pid={proc.pid}"
                )
                proc.terminate()
            else:
                logging.info(f"proc.name [{proc.name}] pid[{proc.pid}] is not alived")

    def _wait_process_list_exit(
        self,
        processes: List[Process],
        timeout: Optional[int],
        group_name: str,
        abort_if_dead_groups: Optional[Set[str]] = None,
    ):
        """Block until SIGTERM'd `processes` exit, or a terminal condition fires.

        Per-tick state machine (priority order):
          1. ALL_EXITED        → every target proc dead; graceful drain succeeded.
          2. DEADLINE_REACHED  → timeout elapsed; caller will SIGKILL survivors.
          3. DEPENDENCY_LOST   → a dep group has dead members; further drain is
                                 futile (e.g. frontend's backend RPC is gone).
                                 Crashes flip `failure_detected`; clean exits
                                 do not (cgroup-wide SIGTERM stays exit 0).
          4. STILL_DRAINING    → log, sleep one monitor_interval, retry.
        """
        if not processes:
            return
        deadline = None if (timeout is None or timeout < 0) else time.time() + timeout
        abort_groups = abort_if_dead_groups or set()

        while True:
            alive_pids = [p.pid for p in processes if p.is_alive()]
            if not alive_pids:
                return  # 1. ALL_EXITED
            if self._drain_deadline_passed(deadline, group_name, alive_pids):
                return  # 2. DEADLINE_REACHED
            if self._dependency_group_lost(abort_groups, group_name, alive_pids):
                return  # 3. DEPENDENCY_LOST
            logging.info(
                f"Waiting for {group_name} process group to exit, alive={alive_pids}"
            )
            time.sleep(self.monitor_interval)

    def _drain_deadline_passed(
        self,
        deadline: Optional[float],
        group_name: str,
        alive_pids: List[Optional[int]],
    ) -> bool:
        if deadline is None or time.time() < deadline:
            return False
        logging.warning(
            f"Timed out waiting for {group_name} process group to exit, "
            f"alive={alive_pids}"
        )
        return True

    def _dependency_group_lost(
        self,
        dep_groups: Set[str],
        waiting_for: str,
        alive_pids: List[Optional[int]],
    ) -> bool:
        """Inspect dep groups; abort drain if any has dead members.

        A multi-process backend losing one member already breaks frontend RPC,
        so we trip on ANY dead proc (mirrors the `all(p.is_alive())` check at
        `_terminate_processes` start). Distinguishes crash from cgroup-wide
        clean exit by inspecting exitcodes — only crashes flag failure.
        """
        for grp in dep_groups:
            grp_procs = self.process_groups.get(grp, [])
            if not grp_procs or all(p.is_alive() for p in grp_procs):
                continue

            dead_exitcodes = [p.exitcode for p in grp_procs if not p.is_alive()]
            crashed = any(c is None or c != 0 for c in dead_exitcodes)
            if crashed:
                logging.warning(
                    f"{grp} group crashed (exitcodes={dead_exitcodes}); "
                    f"aborting drain for {waiting_for}, alive={alive_pids}"
                )
                self.failure_detected = True
            else:
                logging.info(
                    f"{grp} group exited cleanly (exitcodes={dead_exitcodes}); "
                    f"aborting drain for {waiting_for} since drain depends on it, "
                    f"alive={alive_pids}"
                )
            return True
        return False

    def _terminate_processes(self, drain_timeout: int, staged: bool = True):
        """SIGTERM all managed processes. Caller picks `drain_timeout`:
          - graceful exit → user's shutdown_timeout (e.g., 5min).
          - failure exit  → 0 (skip drain wait, SIGTERM backend immediately).

        Force-kill of survivors is the outer monitor loop's responsibility;
        this function only sequences SIGTERMs.

        Staged mode (production path):
          Phase 1 — SIGTERM drain groups (frontend/ingress), wait for them
                    to drain in-flight work (or hit drain_timeout / dep dies).
          Phase 2 — SIGTERM deferred groups (backend); outer loop polls.

        Non-staged mode (post-crash all-stop): SIGTERM everyone at once.
        """
        logging.info(f"Sending SIGTERM (drain_timeout={drain_timeout}s)")

        if staged and self.process_groups:
            self._sigterm_and_drain_groups(drain_timeout)
            self._sigterm_deferred_groups()
        else:
            self._terminate_process_list(self.processes, "managed")

    def _sigterm_and_drain_groups(self, drain_timeout: int):
        """SIGTERM non-deferred groups; wait for drainable ones to drain."""
        for group_name in self.shutdown_group_order:
            if group_name in self.DEFERRED_GROUPS:
                continue
            group_processes = self.process_groups.get(group_name, [])
            self._terminate_process_list(group_processes, group_name)
            if group_name in self.DRAIN_GROUPS:
                self._wait_process_list_exit(
                    group_processes,
                    drain_timeout,
                    group_name,
                    abort_if_dead_groups=set(self.DEFERRED_GROUPS),
                )

    def _sigterm_deferred_groups(self):
        """SIGTERM deferred groups (backend); outer monitor loop polls."""
        for group_name in self.shutdown_group_order:
            if group_name not in self.DEFERRED_GROUPS:
                continue
            self._terminate_process_list(
                self.process_groups.get(group_name, []), group_name
            )

    def _force_kill_processes(self):
        """SIGKILL all surviving children (outer loop's last-resort cleanup)."""
        logging.warning("Force-killing surviving children")
        self._force_kill_process_list(self.processes)

    def _force_kill_process_list(self, processes: List[Process]):
        for proc in processes:
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
        """Reap children with a bounded wait4() window.

        By the time this runs, _monitor_processes_health has already either
        watched all children exit (graceful path) or SIGKILL'd survivors
        (timeout/force path). All that's left is wait4(): already-dead procs
        return instantly; SIGKILL stragglers usually finalize within ~1s;
        D-state procs (CUDA/NFS/NCCL stuck syscall) get a warning and are
        left for kernel reap once the syscall unblocks.
        """
        deadline = time.time() + self.POST_KILL_REAP_WINDOW
        for proc in self.processes:
            try:
                proc.join(max(0.1, deadline - time.time()))
                if proc.is_alive():
                    logging.warning(
                        f"Process {proc.name} (pid={proc.pid}) still alive "
                        f"after {self.POST_KILL_REAP_WINDOW}s reap window — "
                        "likely D-state; leaving for kernel reap"
                    )
            except Exception as e:
                logging.error(f"Error joining process {proc.pid}: {e}")
        logging.info("All processes joined")

    def _monitor_processes_health(self):
        """Watch children; on shutdown or unexpected death, SIGTERM then SIGKILL.

        Steady state: poll every monitor_interval while everything is healthy.
        Otherwise: send SIGTERM (with caller-picked drain budget), wait the
        POST_KILL_REAP_WINDOW, SIGKILL any survivor, exit the loop.
        """
        while self._is_any_process_alive():
            if not self.shutdown_requested and self._is_all_processes_alive():
                time.sleep(self.monitor_interval)
                continue

            if self.shutdown_requested:
                drain_timeout = 0 if self.failure_detected else self.shutdown_timeout
                self._terminate_processes(drain_timeout, staged=True)
            else:
                # Unexpected death → escalate to failure shutdown.
                for proc in self.processes:
                    if not proc.is_alive():
                        logging.error(f"Process {proc.pid} died unexpectedly")
                self.failure_detected = True
                logging.error("Some processes died unexpectedly, terminating all...")
                self._terminate_processes(drain_timeout=0, staged=False)

            time.sleep(self.POST_KILL_REAP_WINDOW)
            self._force_kill_processes()
            break

    def monitor_and_release_processes(self):
        """Monitor all processes until completion or failure.

        The failure_detected → os._exit(1) check runs even when self.processes
        is empty: callers may construct the manager, hit an exception before
        any child is registered (e.g. backend Process pickle/spawn fails,
        import error, config validation), then call request_failure_shutdown()
        + monitor_and_release_processes() in a finally. The parent must still
        exit non-zero so the supervisor (k8s/systemd) restarts.
        """
        if self.processes:
            logging.info(f"Monitoring {len(self.processes)} processes")
            self._monitor_processes_health()
            self._join_all_processes()

            # All children may have exited before the monitor loop first ran
            # (race at startup) or during it without tripping the in-loop
            # dead-detection branch (e.g. simultaneous death between
            # iterations). Inspect final exitcodes to surface silent crashes
            # the monitor missed.
            if not self.shutdown_requested and not self.failure_detected:
                crashed = [
                    (p.name, p.exitcode)
                    for p in self.processes
                    if p.exitcode is not None and p.exitcode != 0
                ]
                if crashed:
                    logging.error(
                        f"Children exited non-zero without shutdown request: {crashed}"
                    )
                    self.failure_detected = True
        else:
            logging.info("No processes to monitor")

        if self.failure_detected:
            logging.error("Child process failure cleanup completed, exiting parent")
            os._exit(1)
        logging.info("Process monitoring completed")

    def request_failure_shutdown(self):
        """Mark failure-driven shutdown.

        After this returns, the monitor skips the graceful drain wait
        (drain_timeout=0) and the parent exits non-zero via
        monitor_and_release_processes.

        Normal SIGTERM/SIGINT shutdown does NOT call this method; the signal
        handler sets shutdown_requested directly, leaving failure_detected=False.
        """
        self.shutdown_requested = True
        self.failure_detected = True

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
