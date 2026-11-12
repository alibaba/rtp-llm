import os
import unittest
from unittest.mock import patch

from rtp_llm.config.py_config_modules import PyEnvConfigs
from rtp_llm.ops import RoleType
from rtp_llm.start_server import start_server
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

    def test_warmup_failure_drains_all_children_without_opening_gate(self):
        configs = PyEnvConfigs()
        configs.role_config.role_type = RoleType.FRONTEND
        with (
            patch(
                "rtp_llm.start_server.start_frontend_server_impl", return_value=[]
            ) as frontend,
            patch("rtp_llm.start_server.start_dash_sc_server_impl", return_value=[]),
            patch(
                "rtp_llm.start_server._setup_startup_warmup_health_gate",
                return_value="/test/gate",
            ),
            patch(
                "rtp_llm.start_server._maybe_run_startup_real_warmup",
                side_effect=RuntimeError("warmup failed"),
            ),
            patch(
                "rtp_llm.start_server._mark_startup_warmup_health_gate_ready"
            ) as mark_ready,
            patch.object(ProcessManager, "run_health_checks", return_value=True),
            patch.object(ProcessManager, "monitor_and_release_processes"),
        ):
            start_server(configs)
        mark_ready.assert_not_called()
        self.assertTrue(frontend.call_args.args[-1].is_set())

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


def run_server(mode="none", exits_cleanly=True, env=None):
    """Run start_server around a stubbed backend and frontends; report what the monitor saw."""
    py_env_configs = PyEnvConfigs()
    py_env_configs.role_config.role_type = RoleType.PREFILL
    py_env_configs.server_config.start_port = 23456
    py_env_configs.load_config.resident_weights_config.mode = mode
    seen = {}

    def monitor(manager):
        seen["store_env"] = os.environ.get("RTP_LLM_RESIDENT_STORE")
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


if __name__ == "__main__":
    unittest.main()
