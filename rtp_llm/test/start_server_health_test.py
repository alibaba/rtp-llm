import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from rtp_llm.ops import RoleType
from rtp_llm.start_server import check_server_health, start_server


class StartServerHealthTest(unittest.TestCase):
    @patch("rtp_llm.start_server.requests.get")
    def test_check_server_health_accepts_frontend_string_response(self, mock_get):
        response = Mock(status_code=200, text='"ok"')
        response.json.return_value = "ok"
        mock_get.return_value = response

        self.assertTrue(check_server_health(30000))

    @patch("rtp_llm.start_server.requests.get")
    def test_check_server_health_accepts_status_object_response(self, mock_get):
        response = Mock(status_code=200, text='{"status":"ok"}')
        response.json.return_value = {"status": "ok"}
        mock_get.return_value = response

        self.assertTrue(check_server_health(30000))

    @patch("rtp_llm.start_server.start_frontend_server_impl")
    @patch("rtp_llm.start_server.start_backend_server_impl")
    @patch("rtp_llm.start_server.init_controller")
    @patch("rtp_llm.start_server.ProcessManager")
    @patch("rtp_llm.start_server.logging.error")
    def test_shutdown_during_health_check_is_not_startup_failure(
        self,
        mock_error,
        mock_process_manager_cls,
        mock_init_controller,
        mock_start_backend,
        mock_start_frontend,
    ):
        process_manager = mock_process_manager_cls.return_value
        process_manager.run_health_checks.return_value = False
        process_manager.shutdown_requested = True
        mock_start_backend.return_value = Mock()
        mock_start_frontend.return_value = [Mock()]
        config = SimpleNamespace(
            parallelism_config=SimpleNamespace(dp_size=1),
            concurrency_config=Mock(),
            server_config=SimpleNamespace(shutdown_timeout=20, monitor_interval=1),
            vit_config=SimpleNamespace(vit_separation=object()),
            role_config=SimpleNamespace(role_type=RoleType.PDFUSION),
        )

        start_server(config)

        mock_error.assert_not_called()
        process_manager.graceful_shutdown.assert_not_called()
        process_manager.monitor_and_release_processes.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
