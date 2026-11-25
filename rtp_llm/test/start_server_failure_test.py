import multiprocessing
import os
import socket
import unittest
from unittest.mock import patch

from rtp_llm.config.py_config_modules import PyEnvConfigs
from rtp_llm.ops import RoleType
from rtp_llm.start_server import (
    _backend_service_draining,
    _create_backend_restarter,
    start_server,
    start_vit_server_impl,
)
from rtp_llm.utils.backend_restart import ShutdownRequested
from rtp_llm.utils.process_manager import ProcessManager


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

        def health_check_after_shutdown(manager, timeout=None):
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


def run_server(mode="none", limit=0, exits_cleanly=True, env=None):
    """Run start_server around a stubbed backend and frontends; report what the monitor saw."""
    py_env_configs = PyEnvConfigs()
    py_env_configs.role_config.role_type = RoleType.PREFILL
    py_env_configs.server_config.engine_restart_limit = limit
    py_env_configs.server_config.start_port = 23456
    py_env_configs.load_config.resident_weights_config.mode = mode
    seen = {}

    def monitor(manager):
        seen["restarter"] = manager.backend_restarter
        seen["store_env"] = os.environ.get("RTP_LLM_RESIDENT_STORE")
        seen["gate_env"] = os.environ.get("RTP_LLM_STARTUP_WARMUP_HEALTH_GATE_FILE")
        if not exits_cleanly:
            raise SystemExit(1)  # stands for os._exit(1) inside the monitor

    with (
        patch.dict(os.environ, env or {}, clear=False),
        patch("rtp_llm.start_server.start_backend_server_impl"),
        patch("rtp_llm.start_server.start_frontend_server_impl"),
        patch("rtp_llm.start_server.start_dash_sc_server_impl"),
        patch("rtp_llm.start_server._maybe_run_startup_real_warmup"),
        patch.object(ProcessManager, "run_health_checks", return_value=True),
        patch.object(
            ProcessManager,
            "monitor_and_release_processes",
            autospec=True,
            side_effect=monitor,
        ),
        patch("rtp_llm.start_server._stop_instance_resident_store") as stop_store,
    ):
        if "RTP_LLM_RESIDENT_STORE" not in (env or {}):
            os.environ.pop("RTP_LLM_RESIDENT_STORE", None)
        try:
            start_server(py_env_configs)
        except SystemExit:
            pass
    return seen, stop_store


class StartServerResidentStoreTest(unittest.TestCase):
    """Which resident store start_server owns, by --resident_weights_mode."""

    def test_none_and_dev_leave_the_store_alone(self):
        for mode in ("none", "dev"):
            with self.subTest(mode=mode):
                seen, stop_store = run_server(mode=mode)
                self.assertIsNone(seen["store_env"])
                stop_store.assert_not_called()

    def test_prod_stops_the_instance_store_on_clean_exit(self):
        seen, stop_store = run_server(mode="prod")
        name = f"rtp_llm_resident_weights.{os.getuid()}.23456"
        self.assertEqual(seen["store_env"], name)
        stop_store.assert_called_once_with(name)

    def test_operator_store_name_is_kept(self):
        seen, stop_store = run_server(
            mode="prod", env={"RTP_LLM_RESIDENT_STORE": "resident-my-task"}
        )
        self.assertEqual(seen["store_env"], "resident-my-task")
        stop_store.assert_called_once_with("resident-my-task")

    def test_failure_exit_keeps_the_store(self):
        _, stop_store = run_server(mode="prod", exits_cleanly=False)
        stop_store.assert_not_called()


class StartServerEngineRestartTest(unittest.TestCase):
    """In-place backend restart is independent of the resident store lifecycle."""

    def test_disabled_changes_nothing(self):
        seen, _ = run_server(limit=0)
        self.assertIsNone(seen["restarter"])
        self.assertIsNone(seen["gate_env"])

    def test_enabled_arms_restart_and_leaves_the_store_alone(self):
        seen, stop_store = run_server(limit=2)
        self.assertTrue(seen["gate_env"])
        self.assertTrue(seen["restarter"]._armed)
        self.assertEqual(seen["restarter"].restart_limit, 2)
        self.assertIsNone(seen["store_env"])
        stop_store.assert_not_called()

    def test_prod_store_lifecycle_is_unchanged_by_restart(self):
        name = f"rtp_llm_resident_weights.{os.getuid()}.23456"
        seen, stop_store = run_server(mode="prod", limit=2)
        self.assertTrue(seen["restarter"]._armed)
        self.assertEqual(seen["store_env"], name)
        stop_store.assert_called_once_with(name)
        _, stop_store = run_server(mode="prod", limit=2, exits_cleanly=False)
        stop_store.assert_not_called()

    def test_default_restart_limit_enables_three_restarts_only_in_prod(self):
        seen_prod, _ = run_server(mode="prod", limit=-1)
        self.assertIsNotNone(seen_prod["restarter"])
        self.assertEqual(seen_prod["restarter"].restart_limit, 3)

        seen_prod_off, _ = run_server(mode="prod", limit=0)
        self.assertIsNone(seen_prod_off["restarter"])

        for mode in ("none", "dev"):
            with self.subTest(mode=mode):
                seen, _ = run_server(mode=mode, limit=-1)
                self.assertIsNone(seen["restarter"])

    def test_stop_while_waiting_for_part0_starts_nothing_and_exits_cleanly(self):
        py_env_configs = PyEnvConfigs()
        py_env_configs.role_config.role_type = RoleType.PREFILL
        py_env_configs.server_config.engine_restart_limit = 2
        py_env_configs.server_config.start_port = 23456
        py_env_configs.load_config.resident_weights_config.mode = "prod"
        with (
            patch.dict(os.environ, {}, clear=False),
            patch(
                "rtp_llm.start_server._create_backend_restarter",
                side_effect=ShutdownRequested(),
            ),
            patch("rtp_llm.start_server.start_backend_server_impl") as backend,
            patch("rtp_llm.start_server.start_frontend_server_impl") as frontend,
            patch("rtp_llm.start_server.start_dash_sc_server_impl"),
            patch.object(ProcessManager, "request_failure_shutdown") as failure,
            patch.object(ProcessManager, "monitor_and_release_processes") as monitor,
            patch("rtp_llm.start_server._stop_instance_resident_store") as stop_store,
        ):
            os.environ.pop("RTP_LLM_RESIDENT_STORE", None)
            start_server(py_env_configs)
        backend.assert_not_called()
        frontend.assert_not_called()
        failure.assert_not_called()
        monitor.assert_called_once()
        stop_store.assert_called_once()


class HeartbeatTimeoutTest(unittest.TestCase):
    def test_timeouts_that_never_fire_or_fire_between_beats_are_refused(self):
        for value in (float("nan"), float("inf"), 0.0, -1.0, 0.5, 1.0):
            with self.subTest(value=value):
                configs = PyEnvConfigs()
                configs.server_config.engine_restart_limit = 1
                configs.server_config.engine_restart_heartbeat_timeout_s = value
                with self.assertRaisesRegex(ValueError, "heartbeat"):
                    _create_backend_restarter(
                        None, configs, None, None, multiprocessing.Event()
                    )


class BackendServiceDrainingTest(unittest.TestCase):
    """A restarted backend reports serving again; only the instance's stop drains everything."""

    def test_without_restart_the_instance_shares_one_event(self):
        instance = multiprocessing.Event()
        manager = ProcessManager(service_draining=instance)
        self.assertIs(_backend_service_draining(instance, manager, False), instance)
        self.assertIsNone(manager.backend_service_draining)

    def test_a_failed_generation_does_not_drain_the_instance_or_the_next_one(self):
        instance = multiprocessing.Event()
        manager = ProcessManager(service_draining=instance)
        first = _backend_service_draining(instance, manager, True)
        first.set()  # what a failing backend does
        second = _backend_service_draining(instance, manager, True)
        self.assertIsNot(second, first)
        self.assertFalse(instance.is_set())
        self.assertFalse(second.is_set())

    def test_the_instance_stop_reaches_the_current_generation(self):
        instance = multiprocessing.Event()
        manager = ProcessManager(service_draining=instance)
        _backend_service_draining(instance, manager, True)
        current = _backend_service_draining(instance, manager, True)
        manager._mark_service_draining()
        self.assertTrue(instance.is_set())
        self.assertTrue(current.is_set())
        # A generation started while stopping begins drained.
        self.assertTrue(_backend_service_draining(instance, manager, True).is_set())


if __name__ == "__main__":
    unittest.main()
