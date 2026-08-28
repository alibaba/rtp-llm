import unittest
from unittest.mock import patch

from rtp_llm.config.py_config_modules import PyEnvConfigs
from rtp_llm.ops import RoleType
from rtp_llm.start_server import start_server
from rtp_llm.utils.process_manager import ProcessManager


class StartServerFailureTest(unittest.TestCase):
    def test_health_check_failure_requests_failure_shutdown_and_raises(self):
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
        ):
            with self.assertRaisesRegex(
                RuntimeError, "managed server processes exited abnormally"
            ):
                start_server(py_env_configs)

        request_shutdown.assert_called_once()

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
            patch("rtp_llm.start_server._maybe_run_startup_real_warmup") as warmup,
            patch(
                "rtp_llm.start_server._mark_startup_warmup_health_gate_ready"
            ) as mark_ready,
        ):
            start_server(py_env_configs)

        request_shutdown.assert_not_called()
        monitor_and_release.assert_called_once()
        warmup.assert_not_called()
        mark_ready.assert_not_called()


if __name__ == "__main__":
    unittest.main()
