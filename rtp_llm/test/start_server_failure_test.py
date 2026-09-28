import multiprocessing
import socket
import unittest
from unittest.mock import patch

from rtp_llm.config.py_config_modules import PyEnvConfigs
from rtp_llm.ops import RoleType
from rtp_llm.start_server import start_server, start_vit_server_impl
from rtp_llm.utils.process_manager import ProcessManager


def _hold_loopback_port(port, ready, stop):
    """Spawn target used only to prove partial-start cleanup releases a port."""
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind(("127.0.0.1", port))
        listener.listen()
        ready.set()
        stop.wait()
    finally:
        listener.close()


class _ReadyListenerProcess:
    """Process-shaped adapter that waits until its test listener owns the port."""

    def __init__(self, port):
        context = multiprocessing.get_context("spawn")
        self._ready = context.Event()
        self._stop = context.Event()
        self._process = context.Process(
            target=_hold_loopback_port, args=(port, self._ready, self._stop)
        )
        self.name = "vit_test_listener"

    @property
    def pid(self):
        return self._process.pid

    def start(self):
        self._process.start()
        if not self._ready.wait(timeout=5):
            self.terminate()
            self.join(timeout=5)
            raise RuntimeError("test listener did not bind its loopback port")

    def is_alive(self):
        return self._process.is_alive()

    def terminate(self):
        self._stop.set()
        if self._process.is_alive():
            self._process.terminate()

    def kill(self):
        if self._process.is_alive():
            self._process.kill()

    def join(self, timeout=None):
        self._process.join(timeout=timeout)


class StartServerFailureTest(unittest.TestCase):
    def test_health_check_failure_requests_failure_shutdown_and_exits_nonzero(self):
        py_env_configs = PyEnvConfigs()
        py_env_configs.role_config.role_type = RoleType.VIT

        original_request_failure_shutdown = ProcessManager.request_failure_shutdown

        def request_failure_shutdown(manager):
            return original_request_failure_shutdown(manager)

        with (
            patch("rtp_llm.start_server.start_vit_server_impl", return_value=[]),
            patch.object(ProcessManager, "run_health_checks", return_value=False),
            patch.object(
                ProcessManager,
                "request_failure_shutdown",
                autospec=True,
                side_effect=request_failure_shutdown,
            ) as request_shutdown,
            patch(
                "rtp_llm.utils.process_manager.os._exit",
                side_effect=SystemExit(1),
            ) as exit_parent,
        ):
            with self.assertRaises(SystemExit) as exit_context:
                start_server(py_env_configs)

        request_shutdown.assert_called_once()
        exit_parent.assert_called_once_with(1)
        self.assertEqual(exit_context.exception.code, 1)

    def test_health_check_failure_after_shutdown_preserves_graceful_exit(self):
        py_env_configs = PyEnvConfigs()
        py_env_configs.role_config.role_type = RoleType.VIT

        def health_check_after_shutdown(manager):
            manager.shutdown_requested = True
            return False

        with (
            patch("rtp_llm.start_server.start_vit_server_impl", return_value=[]),
            patch.object(
                ProcessManager,
                "run_health_checks",
                autospec=True,
                side_effect=health_check_after_shutdown,
            ),
            patch.object(
                ProcessManager,
                "request_failure_shutdown",
                autospec=True,
            ) as request_shutdown,
            patch.object(
                ProcessManager,
                "monitor_and_release_processes",
                autospec=True,
            ) as monitor_and_release,
        ):
            start_server(py_env_configs)

        request_shutdown.assert_not_called()
        monitor_and_release.assert_called_once()

    def test_partial_vit_start_failure_enters_manager_shutdown_with_owned_child(self):
        py_env_configs = PyEnvConfigs()
        py_env_configs.role_config.role_type = RoleType.VIT
        child = _VitFakeProcess("vit_worker_0")
        captured = {}

        def fail_after_registering_child(_, manager):
            manager.add_process(child)
            captured["manager"] = manager
            raise RuntimeError("vit proxy start failed")

        with (
            patch(
                "rtp_llm.start_server.start_vit_server_impl",
                side_effect=fail_after_registering_child,
            ),
            patch.object(
                ProcessManager, "request_failure_shutdown", autospec=True
            ) as request_shutdown,
            patch.object(
                ProcessManager, "monitor_and_release_processes", autospec=True
            ) as monitor_and_release,
        ):
            start_server(py_env_configs)

        self.assertEqual(captured["manager"].processes, [child])
        request_shutdown.assert_called_once_with(captured["manager"])
        monitor_and_release.assert_called_once_with(captured["manager"])


class _VitFakeProcess:
    def __init__(
        self,
        name,
        start_error=None,
        terminate_error=None,
        join_error=None,
    ):
        self.name = name
        self.pid = None
        self._alive = False
        self._start_error = start_error
        self._terminate_error = terminate_error
        self._join_error = join_error
        self.terminate_calls = 0
        self.kill_calls = 0
        self.join_timeouts = []

    def start(self):
        if self._start_error:
            raise self._start_error
        self.pid = 1
        self._alive = True

    def is_alive(self):
        return self._alive

    def terminate(self):
        self.terminate_calls += 1
        if self._terminate_error:
            raise self._terminate_error
        self._alive = False

    def kill(self):
        self.kill_calls += 1
        self._alive = False

    def join(self, timeout=None):
        self.join_timeouts.append(timeout)
        if self._join_error:
            raise self._join_error


class _VitFakeManager:
    def __init__(self):
        self.processes = []
        self.process_groups = {"default": []}
        self.health_check_processes = []
        self.health_check_configs = {}

    def add_process(self, process, shutdown_group="default"):
        self.processes.append(process)
        self.process_groups.setdefault(shutdown_group, []).append(process)

    def register_health_check(
        self, processes, process_name, check_ready_fn, retry_interval_seconds
    ):
        self.health_check_processes.extend(processes)
        self.health_check_configs[process_name] = {
            "processes": processes,
            "check_ready_fn": check_ready_fn,
            "retry_interval_seconds": retry_interval_seconds,
        }


class StartVitServerLifecycleTest(unittest.TestCase):
    def _configs(self, count=2):
        configs = PyEnvConfigs()
        configs.server_config.vit_server_count = count
        configs.server_config.start_port = 31000
        configs.vit_config.output_transport.rdma.port = 0
        return configs

    def _start(self, configs, processes, manager=None):
        with (
            patch("rtp_llm.start_server.load_gpu_nic_affinity"),
            patch(
                "rtp_llm.start_server.torch.multiprocessing.Process",
                side_effect=processes,
            ),
        ):
            return start_vit_server_impl(configs, manager)

    def test_proxy_start_failure_leaves_prior_children_managed(self):
        worker0 = _VitFakeProcess("vit_worker_0")
        worker1 = _VitFakeProcess("vit_worker_1")
        proxy = _VitFakeProcess("vit_proxy", RuntimeError("proxy start failed"))
        manager = _VitFakeManager()

        with self.assertRaisesRegex(RuntimeError, "proxy start failed"):
            self._start(self._configs(), [worker0, worker1, proxy], manager)

        self.assertEqual(manager.processes, [worker0, worker1])
        self.assertEqual(manager.process_groups["default"], [worker0, worker1])

    def test_invalid_threshold_or_worker_count_fails_before_spawning(self):
        for count, minimum in ((0, 0), (-1, 0), (2, 3)):
            with self.subTest(count=count, minimum=minimum):
                configs = self._configs(count)
                configs.vit_config.vit_proxy_min_healthy_workers = minimum
                with patch(
                    "rtp_llm.start_server.torch.multiprocessing.Process"
                ) as make_process:
                    with self.assertRaises(ValueError):
                        start_vit_server_impl(configs, _VitFakeManager())
                make_process.assert_not_called()

    def test_second_worker_start_failure_leaves_first_child_managed(self):
        worker0 = _VitFakeProcess("vit_worker_0")
        worker1 = _VitFakeProcess("vit_worker_1", RuntimeError("worker start failed"))
        manager = _VitFakeManager()
        with self.assertRaisesRegex(RuntimeError, "worker start failed"):
            self._start(self._configs(), [worker0, worker1], manager)
        self.assertEqual(manager.processes, [worker0])

    def test_single_worker_is_registered_once(self):
        worker = _VitFakeProcess("vit_server")
        manager = _VitFakeManager()
        self.assertEqual(self._start(self._configs(1), [worker], manager), [worker])
        self.assertEqual(manager.processes, [worker])

    def test_success_registers_each_child_once_and_returns_proxy_first(self):
        worker0 = _VitFakeProcess("vit_worker_0")
        worker1 = _VitFakeProcess("vit_worker_1")
        proxy = _VitFakeProcess("vit_proxy")
        manager = _VitFakeManager()

        processes = self._start(self._configs(), [worker0, worker1, proxy], manager)

        self.assertEqual(processes, [proxy, worker0, worker1])
        self.assertEqual(manager.processes, [worker0, worker1, proxy])
        self.assertEqual(len(manager.processes), len(set(map(id, manager.processes))))
        self.assertEqual(manager.health_check_processes, [proxy, worker0, worker1])

    def test_unmanaged_cleanup_continues_after_one_child_cleanup_error(self):
        worker0 = _VitFakeProcess(
            "vit_worker_0",
            terminate_error=RuntimeError("terminate failed"),
            join_error=RuntimeError("join failed"),
        )
        worker1 = _VitFakeProcess("vit_worker_1")
        failed_proxy = _VitFakeProcess("vit_proxy", RuntimeError("proxy start failed"))

        with self.assertRaisesRegex(RuntimeError, "proxy start failed"):
            self._start(self._configs(), [worker0, worker1, failed_proxy])

        self.assertEqual(worker0.terminate_calls, 1)
        self.assertEqual(worker0.kill_calls, 1)
        self.assertEqual(worker1.terminate_calls, 1)
        self.assertFalse(worker1.is_alive())
        self.assertEqual(len(worker1.join_timeouts), 2)

    def test_unmanaged_partial_start_reaps_real_listener_and_releases_port(self):
        while True:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as reservation:
                reservation.bind(("127.0.0.1", 0))
                listener_port = reservation.getsockname()[1]
            if listener_port <= 65532:
                break

        listener = _ReadyListenerProcess(listener_port)
        failed_worker = _VitFakeProcess(
            "vit_worker_1", RuntimeError("second worker start failed")
        )
        configs = self._configs()
        configs.server_config.start_port = listener_port - 2

        try:
            with self.assertRaisesRegex(RuntimeError, "second worker start failed"):
                self._start(configs, [listener, failed_worker])
            self.assertFalse(listener.is_alive())
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as rebound:
                rebound.bind(("127.0.0.1", listener_port))
        finally:
            if listener.is_alive():
                listener.terminate()
                listener.join(timeout=5)
            if listener.is_alive():
                listener.kill()
                listener.join(timeout=5)


if __name__ == "__main__":
    unittest.main()
