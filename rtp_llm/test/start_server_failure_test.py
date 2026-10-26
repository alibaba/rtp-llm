import os
import unittest
from unittest.mock import Mock, patch

from rtp_llm.config.py_config_modules import PyEnvConfigs
from rtp_llm.ops import RoleType
from rtp_llm.start_backend_server import (
    _create_rank_processes,
    local_rank_start,
    start_backend_server,
)
from rtp_llm.start_server import start_backend_server_impl, start_server
from rtp_llm.utils.parent_death_signal import run_with_parent_death_signal
from rtp_llm.utils.process_manager import ProcessManager


class StartServerFailureTest(unittest.TestCase):
    def test_backend_manager_is_bound_to_server_parent(self):
        config = PyEnvConfigs()
        reader, writer = Mock(), Mock()
        with (
            patch.dict(os.environ),
            patch(
                "rtp_llm.start_server.torch.multiprocessing.Pipe",
                return_value=(reader, writer),
            ),
            patch("rtp_llm.start_server.torch.multiprocessing.Process") as process,
        ):
            result = start_backend_server_impl(None, config)
        process.assert_called_once_with(
            target=run_with_parent_death_signal,
            args=(os.getpid(), start_backend_server, None, config, writer),
            name="backend_manager",
        )
        self.assertIs(result, process.return_value)
        result.start.assert_called_once_with()
        writer.close.assert_called_once_with()

    def test_every_rank_is_bound_to_backend_manager_parent(self):
        config = PyEnvConfigs()
        ctx = Mock()
        pipes = [(Mock(), Mock()), (Mock(), Mock())]
        ctx.Pipe.side_effect = pipes
        processes, readers = [], []
        with (
            patch.dict(os.environ),
            patch("rtp_llm.start_backend_server._get_local_world_size", return_value=2),
            patch(
                "rtp_llm.start_backend_server._get_cuda_device_list",
                return_value=["0", "1"],
            ),
        ):
            _create_rank_processes(None, config, ctx, processes, readers)
        self.assertEqual(len(processes), 2)
        self.assertEqual(readers, [reader for reader, _ in pipes])
        for rank, call in enumerate(ctx.Process.call_args_list):
            self.assertIs(call.kwargs["target"], run_with_parent_death_signal)
            self.assertEqual(
                call.kwargs["args"],
                (os.getpid(), local_rank_start, None, config, rank, pipes[rank][1]),
            )
            pipes[rank][1].close.assert_called_once_with()

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
