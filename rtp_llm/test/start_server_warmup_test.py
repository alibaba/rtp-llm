import unittest
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import Mock, patch

from rtp_llm import start_server
from rtp_llm.config.py_config_modules import PyEnvConfigs
from rtp_llm.ops import RoleType, SpeculativeType
from rtp_llm.utils.drain_flag import DrainFlag


class ServiceDrainingPropagationTest(unittest.TestCase):
    def test_each_launch_passes_a_fresh_event_to_all_process_groups(self):
        configs = Mock()
        configs.parallelism_config.dp_size = 1
        # Without in-place restart every process group shares the instance's event.
        configs.server_config.engine_restart_limit = 0
        with ExitStack() as stack:
            for name in (
                "init_controller",
                "_sync_server_shutdown_timeout",
                "_setup_startup_warmup_health_gate",
                "_maybe_run_startup_real_warmup",
                "_mark_startup_warmup_health_gate_ready",
            ):
                stack.enter_context(patch.object(start_server, name))
            manager_class = stack.enter_context(
                patch.object(start_server, "ProcessManager")
            )
            starters = [
                stack.enter_context(patch.object(start_server, name))
                for name in (
                    "start_backend_server_impl",
                    "start_frontend_server_impl",
                    "start_dash_sc_server_impl",
                )
            ]
            vit = stack.enter_context(
                patch.object(start_server, "start_vit_server_impl")
            )
            previous = None
            for role in (RoleType.PDFUSION, RoleType.PDFUSION, RoleType.VIT):
                configs.role_config.role_type = role
                start_server.start_server(configs)
                event = manager_class.call_args.kwargs["service_draining"]
                # multiprocessing.Event.set() hangs once a waiter was killed.
                self.assertIsInstance(event, DrainFlag)
                self.assertIsNot(event, previous)
                self.assertFalse(event.is_set())
                for starter in [vit] if role == RoleType.VIT else starters:
                    self.assertIs(starter.call_args.args[-1], event)
                event.set()
                previous = event


class StartupRealWarmupTest(unittest.TestCase):
    def test_single_stage_dp_ranks_pass_startup_warmup_gate(self):
        for world_size in (2, 3, 4):
            for world_rank in range(world_size):
                with self.subTest(world_size=world_size, world_rank=world_rank):
                    config = SimpleNamespace(
                        parallelism_config=SimpleNamespace(
                            pp_size=1,
                            dp_size=world_size,
                            tp_size=1,
                            world_size=world_size,
                            world_rank=world_rank,
                        ),
                        runtime_config=SimpleNamespace(
                            warm_up=True, model_warm_up=True
                        ),
                        role_config=SimpleNamespace(role_type=RoleType.PREFILL),
                        model_args=SimpleNamespace(model_type="qwen_3"),
                    )
                    self.assertTrue(
                        start_server._is_startup_real_warmup_entry_rank(config)
                    )
                    self.assertFalse(
                        start_server._should_run_startup_real_warmup(config)
                    )

    def test_warmup_entry_rank_preserves_pp_and_tp_boundaries(self):
        for pp_size, world_rank, expected in (
            (1, 0, True),
            (1, 1, False),
            (1, 2, True),
            (2, 0, True),
            (2, 2, True),
            (2, 4, False),
            (2, 5, False),
            (2, 6, False),
        ):
            with self.subTest(pp_size=pp_size, world_rank=world_rank):
                config = SimpleNamespace(
                    parallelism_config=SimpleNamespace(
                        pp_size=pp_size,
                        dp_size=2,
                        tp_size=2,
                        world_size=pp_size * 4,
                        world_rank=world_rank,
                    )
                )
                self.assertEqual(
                    start_server._is_startup_real_warmup_entry_rank(config),
                    expected,
                )

    @staticmethod
    def _configs(sp_type, gamma):
        return SimpleNamespace(
            sp_config=SimpleNamespace(
                type=sp_type,
                gen_num_per_cycle=gamma,
            )
        )

    def test_speculative_reserve_matches_engine(self):
        mtp = self._configs(SpeculativeType.MTP, 3)
        dspark = self._configs(SpeculativeType.DSPARK, 3)
        dflash = self._configs(SpeculativeType.DFLASH, 3)

        with patch.dict("os.environ", {"RTP_LLM_STREAM_ASYNC": "0"}):
            self.assertEqual(
                start_server._get_startup_real_warmup_speculative_reserve_step(mtp),
                4,
            )
            self.assertEqual(
                start_server._get_startup_real_warmup_speculative_reserve_step(dspark),
                9,
            )
            self.assertEqual(
                start_server._get_startup_real_warmup_speculative_reserve_step(dflash),
                9,
            )
        with patch.dict("os.environ", {"RTP_LLM_STREAM_ASYNC": "1"}):
            self.assertEqual(
                start_server._get_startup_real_warmup_speculative_reserve_step(mtp),
                7,
            )
            self.assertEqual(
                start_server._get_startup_real_warmup_speculative_reserve_step(dspark),
                9,
            )
            self.assertEqual(
                start_server._get_startup_real_warmup_speculative_reserve_step(dflash),
                9,
            )
        self.assertEqual(
            start_server._get_startup_real_warmup_request_token_len(
                token_len=1048576,
                max_len=1048576,
                reserve_step=8,
            ),
            1048568,
        )

    def test_glm_startup_warmup_gate(self):
        config = PyEnvConfigs()
        config.model_args.model_type = "glm_5"
        config.parallelism_config.tp_size = 8
        config.parallelism_config.world_size = 8
        for warm_up, model_warm_up, role, rank, expected in (
            (True, True, RoleType.PREFILL, 0, True),
            (False, True, RoleType.PREFILL, 0, False),
            (True, False, RoleType.PREFILL, 0, False),
            (True, True, RoleType.DECODE, 0, False),
            (True, True, RoleType.PREFILL, 1, False),
        ):
            with self.subTest(case=(warm_up, model_warm_up, role, rank)):
                config.runtime_config.warm_up = warm_up
                config.runtime_config.model_warm_up = model_warm_up
                config.role_config.role_type = role
                config.parallelism_config.world_rank = rank
                self.assertEqual(
                    start_server._should_run_startup_real_warmup(config), expected
                )


if __name__ == "__main__":
    unittest.main()
