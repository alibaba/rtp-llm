import os
import sys
import types
import unittest
from contextlib import ExitStack
from unittest.mock import MagicMock, call, patch

from rtp_llm import start_server
from rtp_llm.config.py_config_modules import PyEnvConfigs, VitSeparation
from rtp_llm.ops import RoleType


class StartPromptGeneratorTest(unittest.TestCase):
    def setUp(self):
        self.cfg = PyEnvConfigs()
        self.cfg.server_config.enable_prompt_generator = True
        self.cfg.server_config.enable_prompt_generator_mps = False
        self.cfg.server_config.prompt_generator_server_count = 2
        self.cfg.parallelism_config.world_size = 4
        self.cfg.parallelism_config.world_rank = 0
        self.cfg.parallelism_config.tp_size = 2
        self.controller = object()
        self.manager = MagicMock()
        self.manager.shutdown_requested = False
        self.manager.run_health_checks.return_value = True

        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        modules = {}
        for name, entry in (
            ("start_server", "start_prompt_generator"),
            ("start_mps", "start_mps"),
        ):
            path = f"internal_source.rtp_llm.prompt_generator.service.{name}"
            module = types.ModuleType(path)
            setattr(module, entry, MagicMock(name=entry))
            modules[path] = module
        self.pg_entry = modules[
            "internal_source.rtp_llm.prompt_generator.service.start_server"
        ].start_prompt_generator
        self.mps_entry = modules[
            "internal_source.rtp_llm.prompt_generator.service.start_mps"
        ].start_mps
        self.stack.enter_context(patch.dict(sys.modules, modules))
        self.stack.enter_context(
            patch.object(start_server, "has_internal_source", return_value=True)
        )

    def _mock_startup_boundaries(self):
        mocks = {}
        for name in (
            "configure_warmup",
            "init_controller",
            "_sync_server_shutdown_timeout",
            "ProcessManager",
            "_setup_startup_warmup_health_gate",
            "_maybe_run_startup_real_warmup",
            "_mark_startup_warmup_health_gate_ready",
            "start_backend_server_impl",
            "start_frontend_server_impl",
            "start_dash_sc_server_impl",
            "start_vit_server_impl",
        ):
            mocks[name] = self.stack.enter_context(patch.object(start_server, name))
        mocks["init_controller"].return_value = self.controller
        mocks["ProcessManager"].return_value = self.manager
        self.stack.enter_context(
            patch.object(start_server.multiprocessing, "set_start_method")
        )
        self.stack.enter_context(
            patch.object(start_server.torch.multiprocessing, "set_start_method")
        )
        self.stack.enter_context(patch("rtp_llm.telemetry.resolve_region_env"))
        return mocks

    def test_rank_topology_process_arguments_and_health_registration(self):
        for world_rank, expected_ranks in ((0, (0, 2)), (1, (0, 1, 3))):
            with self.subTest(world_rank=world_rank):
                self.cfg.parallelism_config.world_rank = world_rank
                manager = MagicMock()
                with (
                    patch.dict(os.environ, {"LOCAL_WORLD_SIZE": "4"}),
                    patch.object(
                        start_server.multiprocessing,
                        "Process",
                        side_effect=lambda **kwargs: MagicMock(),
                    ) as process_factory,
                ):
                    processes = start_server.start_prompt_generator_impl(
                        self.controller, self.cfg, manager
                    )
                expected_calls = [
                    call(
                        target=self.pg_entry,
                        args=(self.cfg, rank, server_index, self.controller),
                        name=f"prompt_generator_rank{rank}_server{server_index}",
                    )
                    for rank in expected_ranks
                    for server_index in range(2)
                ]
                self.assertEqual(process_factory.call_args_list, expected_calls)
                self.assertEqual(len(processes), len(expected_calls))
                for process in processes:
                    process.start.assert_called_once_with()
                manager.register_health_check.assert_called_once()
                health = manager.register_health_check.call_args.kwargs
                self.assertEqual(health["processes"], processes)
                self.assertEqual(health["process_name"], "prompt_generator_server")
                self.assertEqual(health["retry_interval_seconds"], 1)
                with patch.object(
                    start_server, "check_server_health", return_value=True
                ) as check_health:
                    self.assertTrue(health["check_ready_fn"]())
                check_health.assert_called_once_with(self.cfg.server_config.start_port)

    def test_local_world_size_override_and_world_size_fallback(self):
        for local_world_size, expected_ranks in (("2", (0,)), (None, (0, 2))):
            with self.subTest(local_world_size=local_world_size):
                with (
                    patch.dict(os.environ),
                    patch.object(
                        start_server.multiprocessing,
                        "Process",
                        side_effect=lambda **kwargs: MagicMock(),
                    ) as process_factory,
                ):
                    if local_world_size is None:
                        os.environ.pop("LOCAL_WORLD_SIZE", None)
                    else:
                        os.environ["LOCAL_WORLD_SIZE"] = local_world_size
                    processes = start_server.start_prompt_generator_impl(
                        self.controller, self.cfg
                    )
                self.assertEqual(len(processes), len(expected_ranks) * 2)
                self.assertEqual(
                    [args.kwargs["args"][1] for args in process_factory.call_args_list],
                    [rank for rank in expected_ranks for _ in range(2)],
                )

    def test_final_vit_role_rejected_before_startup_side_effects(self):
        mocks = self._mock_startup_boundaries()
        self.cfg.server_config.enable_prompt_generator_mps = True
        for role in (RoleType.VIT, RoleType.PDFUSION):
            with self.subTest(role=role):
                self.cfg.role_config.role_type = role
                self.cfg.vit_config.vit_separation = VitSeparation.VIT_SEPARATION_ROLE
                with self.assertRaisesRegex(ValueError, "unsupported for the VIT role"):
                    start_server.start_server(self.cfg)
        for mocked in mocks.values():
            mocked.assert_not_called()
        self.mps_entry.assert_not_called()

    def test_invalid_count_rejected_before_startup_side_effects(self):
        mocks = self._mock_startup_boundaries()
        self.cfg.server_config.enable_prompt_generator_mps = True
        for count in (0, -1):
            with self.subTest(count=count):
                self.cfg.server_config.prompt_generator_server_count = count
                with self.assertRaisesRegex(ValueError, "server count must be greater"):
                    start_server.start_server(self.cfg)
        for mocked in mocks.values():
            mocked.assert_not_called()
        self.mps_entry.assert_not_called()
        self.pg_entry.assert_not_called()

    def test_direct_pg_start_rejects_invalid_count_without_processes(self):
        self.cfg.server_config.prompt_generator_server_count = 0
        with patch.object(start_server.multiprocessing, "Process") as process_factory:
            with self.assertRaisesRegex(ValueError, "server count must be greater"):
                start_server.start_prompt_generator_impl(
                    self.controller, self.cfg, self.manager
                )
        process_factory.assert_not_called()
        self.manager.register_health_check.assert_not_called()

    def test_invalid_tp_size_rejected_before_any_pg_process_starts(self):
        # rank 0 short-circuits the modulo branch; multiple local ranks expose
        # the previous partial-start failure on rank 1.
        for tp_size in (0, -1):
            with self.subTest(tp_size=tp_size):
                self.cfg.parallelism_config.tp_size = tp_size
                with (
                    patch.dict(os.environ, {"LOCAL_WORLD_SIZE": "2"}),
                    patch.object(start_server.multiprocessing, "Process") as factory,
                ):
                    with self.assertRaisesRegex(ValueError, "tp_size must be greater"):
                        start_server.start_prompt_generator_impl(
                            self.controller, self.cfg, self.manager
                        )
                factory.assert_not_called()
        self.manager.register_health_check.assert_not_called()

    def test_invalid_tp_size_rejected_before_server_startup_side_effects(self):
        mocks = self._mock_startup_boundaries()
        self.cfg.parallelism_config.tp_size = 0
        self.cfg.server_config.enable_prompt_generator_mps = True
        with self.assertRaisesRegex(ValueError, "tp_size must be greater"):
            start_server.start_server(self.cfg)
        for mocked in mocks.values():
            mocked.assert_not_called()
        self.mps_entry.assert_not_called()

    def test_unavailable_pg_preserves_frontend_fallback_with_invalid_count(self):
        self.cfg.server_config.prompt_generator_server_count = 0
        self.cfg.server_config.enable_prompt_generator_mps = True
        with patch.object(start_server, "has_internal_source", return_value=False):
            start_server.normalize_prompt_generator_config(self.cfg)
        self.assertFalse(self.cfg.server_config.enable_prompt_generator)
        self.assertFalse(self.cfg.server_config.enable_prompt_generator_mps)
        self.mps_entry.assert_not_called()

    def test_pg_replaces_frontend_and_dash_and_registers_shutdown_group(self):
        mocks = self._mock_startup_boundaries()
        self.cfg.role_config.role_type = RoleType.PDFUSION
        with (
            patch.dict(os.environ, {"LOCAL_WORLD_SIZE": "2"}),
            patch.object(
                start_server.multiprocessing,
                "Process",
                side_effect=lambda **kwargs: MagicMock(),
            ),
        ):
            start_server.start_server(self.cfg)
        mocks["start_backend_server_impl"].assert_called_once_with(
            self.controller, self.cfg, self.manager
        )
        self.manager.add_process.assert_called_once_with(
            mocks["start_backend_server_impl"].return_value, shutdown_group="backend"
        )
        mocks["start_frontend_server_impl"].assert_not_called()
        mocks["start_dash_sc_server_impl"].assert_not_called()
        processes = self.manager.register_health_check.call_args.kwargs["processes"]
        self.assertEqual(len(processes), 2)
        self.manager.add_processes.assert_called_once_with(
            processes, shutdown_group="frontend"
        )
        self.manager.run_health_checks.assert_called_once_with()
        self.manager.request_failure_shutdown.assert_not_called()
        self.manager.monitor_and_release_processes.assert_called_once_with()
        self.mps_entry.assert_not_called()

    def test_mps_failure_uses_process_manager_cleanup(self):
        mocks = self._mock_startup_boundaries()
        self.cfg.server_config.enable_prompt_generator_mps = True
        self.mps_entry.side_effect = RuntimeError("MPS startup failed")
        with patch.object(start_server.multiprocessing, "Process") as process_factory:
            start_server.start_server(self.cfg)
        self.mps_entry.assert_called_once_with()
        mocks["start_backend_server_impl"].assert_not_called()
        mocks["start_frontend_server_impl"].assert_not_called()
        process_factory.assert_not_called()
        self.manager.request_failure_shutdown.assert_called_once_with()
        self.manager.monitor_and_release_processes.assert_called_once_with()
        self.manager.run_health_checks.assert_not_called()


if __name__ == "__main__":
    unittest.main()
