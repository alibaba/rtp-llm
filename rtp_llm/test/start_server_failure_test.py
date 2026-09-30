import os
import unittest
from unittest.mock import MagicMock, patch

from rtp_llm import start_backend_server, start_dash_sc_server, start_frontend_server
from rtp_llm import start_server as launcher
from rtp_llm.aios.kmonitor.python_client.kmonitor import reporting
from rtp_llm.config.py_config_modules import PyEnvConfigs
from rtp_llm.ops import RoleType
from rtp_llm.start_server import start_server
from rtp_llm.utils.process_manager import ProcessManager


class StartServerFailureTest(unittest.TestCase):
    def test_reporting_state_is_forwarded_through_spawn_arguments(self):
        configs = PyEnvConfigs()
        configs.parallelism_config.world_size = 2
        configs.parallelism_config.tp_size = 1
        configs.server_config.frontend_server_count = 2
        state = reporting.ReportingState(2)
        with (
            patch.dict("os.environ", {"LOCAL_WORLD_SIZE": "2"}),
            patch("multiprocessing.Process") as process,
            patch("torch.multiprocessing.Process", process),
            patch.object(launcher, "start_memory_saver_configured_process"),
        ):
            for launch in (
                launcher.start_backend_server_impl,
                launcher.start_frontend_server_impl,
                launcher.start_dash_sc_server_impl,
            ):
                process.reset_mock()
                launch(None, configs, MagicMock(), reporting_state=state)
                self.assertGreater(process.call_count, 0)
                for call in process.call_args_list:
                    self.assertIs(call.kwargs["args"][-1], state)

        ctx = MagicMock()
        ctx.Pipe.side_effect = lambda **_: (MagicMock(), MagicMock())
        with (
            patch.dict("os.environ", {"LOCAL_WORLD_SIZE": "2"}),
            patch.object(
                start_backend_server, "_get_cuda_device_list", return_value=["0", "1"]
            ),
            patch("torch.cuda.device_count", return_value=2),
            patch.object(start_backend_server, "start_memory_saver_configured_process"),
        ):
            start_backend_server._create_rank_processes(
                None, configs, ctx, [], [], state
            )
        self.assertEqual(ctx.Process.call_count, 2)
        for call in ctx.Process.call_args_list:
            self.assertIs(call.kwargs["args"][-1], state)

    def test_children_install_state_before_starting_services(self):
        state = reporting.ReportingState(2)
        configs = PyEnvConfigs()
        for module, entry, args in (
            (
                start_backend_server,
                start_backend_server.local_rank_start,
                (None, configs),
            ),
            (
                start_frontend_server,
                start_frontend_server.start_frontend_server,
                (0, 0, None, configs),
            ),
            (
                start_dash_sc_server,
                start_dash_sc_server.start_dash_sc_server,
                (0, 0, None, configs),
            ),
        ):
            with (
                self.subTest(entry=entry.__name__),
                patch.object(reporting, "_state", None),
                patch.object(
                    module,
                    "_install_hot_hook_runtime",
                    side_effect=RuntimeError("stop before service startup"),
                ),
            ):
                with self.assertRaisesRegex(
                    RuntimeError, "stop before service startup"
                ):
                    entry(*args, reporting_state=state)
                self.assertIs(reporting._state, state)

    def test_sleep_reporting_state_reaches_backend_and_all_ingress_groups(self):
        for enabled, role in (
            (False, RoleType.PDFUSION),
            (True, RoleType.PDFUSION),
            (True, RoleType.FRONTEND),
        ):
            with self.subTest(enabled=enabled, role=role):
                configs = PyEnvConfigs()
                configs.role_config.role_type = role
                configs.runtime_config.enable_sleep_mode = enabled
                configs.parallelism_config.world_size = 4
                configs.runtime_config.warm_up = False
                with (
                    patch("rtp_llm.start_server.ProcessManager") as manager,
                    patch("rtp_llm.start_server.start_backend_server_impl") as backend,
                    patch(
                        "rtp_llm.start_server.start_frontend_server_impl"
                    ) as frontend,
                    patch("rtp_llm.start_server.start_dash_sc_server_impl") as dash,
                    patch.dict("os.environ", {"LOCAL_WORLD_SIZE": "2"}),
                ):
                    start_server(configs)
                manager.return_value.request_failure_shutdown.assert_not_called()
                state = frontend.call_args.kwargs.get("reporting_state")
                self.assertIs(dash.call_args.kwargs.get("reporting_state"), state)
                if not enabled:
                    self.assertIsNone(state)
                    continue
                self.assertIsNotNone(state)
                if role == RoleType.FRONTEND:
                    backend.assert_not_called()
                    self.assertTrue(state.frontend_only)
                    state.set_enabled(False)
                    self.assertEqual(state.epoch.value, 1)
                else:
                    self.assertIs(backend.call_args.kwargs["reporting_state"], state)
                    self.assertFalse(state.frontend_only)
                    self.assertEqual(len(state.rank_enabled), 2)
                    state.set_rank_enabled(0, False)
                    self.assertEqual(state.epoch.value, 0)
                    state.set_rank_enabled(1, False)
                    self.assertEqual(state.epoch.value, 1)

    def test_reporting_rank_count_does_not_probe_cuda_or_write_environment(self):
        configs = PyEnvConfigs()
        configs.parallelism_config.world_size = 4
        configs.parallelism_config.local_world_size = 2
        configs.server_config.frontend_server_count = 1
        with (
            patch.dict("os.environ", {}, clear=True),
            patch(
                "rtp_llm.start_backend_server._get_local_world_size",
                side_effect=AssertionError("backend sizing has side effects"),
            ),
            patch(
                "torch.cuda.device_count",
                side_effect=AssertionError("main process must not probe CUDA"),
            ),
        ):
            self.assertEqual(launcher._local_world_size_for_serving(configs), 2)
            self.assertNotIn("LOCAL_WORLD_SIZE", os.environ)
            self.assertEqual(list(launcher._iter_serving_ranks(configs)), [0, 1])
            with (
                patch("multiprocessing.Process") as process,
                patch.object(launcher, "start_memory_saver_configured_process"),
            ):
                launcher.start_frontend_server_impl(None, configs, MagicMock())
            self.assertEqual(process.call_count, 2)

        with patch.dict("os.environ", {"LOCAL_WORLD_SIZE": "3"}):
            self.assertEqual(launcher._local_world_size_for_serving(configs), 3)
        with patch.dict("os.environ", {"LOCAL_WORLD_SIZE": "0"}):
            with self.assertRaisesRegex(ValueError, "must be positive"):
                launcher._local_world_size_for_serving(configs)

    def test_backend_rank_count_matches_resolved_cli_local_world_size(self):
        configs = PyEnvConfigs()
        configs.parallelism_config.world_size = 4
        configs.parallelism_config.local_world_size = 2
        with (
            patch.dict("os.environ", {}, clear=True),
            patch("torch.cuda.device_count", return_value=4),
        ):
            self.assertEqual(launcher._local_world_size_for_serving(configs), 2)
            self.assertEqual(start_backend_server._get_local_world_size(configs), 2)
            self.assertEqual(os.environ["LOCAL_WORLD_SIZE"], "2")

    def test_backend_rejects_more_local_ranks_than_visible_devices(self):
        configs = PyEnvConfigs()
        configs.parallelism_config.world_size = 8
        configs.parallelism_config.local_world_size = 8
        ctx = MagicMock()
        ctx.Pipe.side_effect = lambda **_: (MagicMock(), MagicMock())
        with (
            patch.dict("os.environ", {}, clear=True),
            patch("torch.cuda.device_count", return_value=4),
            patch.object(
                start_backend_server,
                "_get_cuda_device_list",
                return_value=["0", "1", "2", "3"],
            ),
            patch.object(start_backend_server, "start_memory_saver_configured_process"),
        ):
            with self.assertRaisesRegex(
                ValueError, "local_world_size=8 exceeds 4 visible CUDA devices"
            ):
                start_backend_server._create_rank_processes(
                    None, configs, ctx, [], []
                )
            self.assertNotIn("LOCAL_WORLD_SIZE", os.environ)
        ctx.Process.assert_not_called()

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
