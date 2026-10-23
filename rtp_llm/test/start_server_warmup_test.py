import unittest
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import Mock, patch

from rtp_llm import start_server
from rtp_llm.ops import RoleType, SpeculativeType


class ServiceDrainingPropagationTest(unittest.TestCase):
    def test_each_launch_passes_a_fresh_event_to_all_process_groups(self):
        configs = Mock()
        configs.parallelism_config.dp_size = 1
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
                self.assertIsNot(event, previous)
                self.assertFalse(event.is_set())
                for starter in [vit] if role == RoleType.VIT else starters:
                    self.assertIs(starter.call_args.args[-1], event)
                event.set()
                previous = event


class StartupRealWarmupTest(unittest.TestCase):
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

        with patch.dict("os.environ", {"RTP_LLM_STREAM_ASYNC": "0"}):
            self.assertEqual(
                start_server._get_startup_real_warmup_speculative_reserve_step(mtp),
                4,
            )
            self.assertEqual(
                start_server._get_startup_real_warmup_speculative_reserve_step(dspark),
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
            start_server._get_startup_real_warmup_request_token_len(
                token_len=1048576,
                max_len=1048576,
                reserve_step=8,
            ),
            1048568,
        )


if __name__ == "__main__":
    unittest.main()
