import unittest
from unittest.mock import MagicMock, patch

from rtp_llm.config.py_config_modules import PyEnvConfigs
from rtp_llm.ops import RoleType
from rtp_llm.start_server import start_server
from rtp_llm.utils.process_manager import ProcessManager


class StartServerFailureTest(unittest.TestCase):
    def test_prompt_generator_starts_as_separate_frontend_processes(self):
        for role in (RoleType.PDFUSION, RoleType.FRONTEND):
            with self.subTest(role=role):
                config = PyEnvConfigs()
                config.role_config.role_type = role
                config.server_config.enable_prompt_generator = True
                config.server_config.enable_prompt_generator_mps = False
                backend, pg = MagicMock(), MagicMock()
                with (
                    patch("rtp_llm.start_server.normalize_prompt_generator_config"),
                    patch("rtp_llm.start_server.ProcessManager") as manager_cls,
                    patch(
                        "rtp_llm.start_server.start_backend_server_impl",
                        return_value=backend,
                    ) as spawn_backend,
                    patch(
                        "rtp_llm.start_server.start_prompt_generator_impl",
                        return_value=[pg],
                    ) as spawn_pg,
                    patch(
                        "rtp_llm.start_server.start_frontend_server_impl"
                    ) as spawn_frontend,
                ):
                    start_server(config)
                manager = manager_cls.return_value
                spawn_pg.assert_called_once()
                manager.add_processes.assert_called_once_with(
                    [pg], shutdown_group="frontend"
                )
                spawn_frontend.assert_not_called()
                if role == RoleType.PDFUSION:
                    spawn_backend.assert_called_once()
                    manager.add_process.assert_called_once_with(
                        backend, shutdown_group="backend"
                    )
                else:
                    spawn_backend.assert_not_called()
                manager.monitor_and_release_processes.assert_called_once()

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


if __name__ == "__main__":
    unittest.main()
