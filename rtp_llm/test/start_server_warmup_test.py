import os
import unittest
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import Mock, patch

from rtp_llm import start_server


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
            previous = None
            for _ in range(2):
                start_server.start_server(configs)
                event = manager_class.call_args.kwargs["service_draining"]
                self.assertIsNot(event, previous)
                self.assertFalse(event.is_set())
                for starter in starters:
                    self.assertIs(starter.call_args.args[-1], event)
                event.set()
                previous = event


class StartupRealWarmupTokenLensTest(unittest.TestCase):
    @staticmethod
    def _configs(max_seq_len=262144):
        return SimpleNamespace(
            model_args=SimpleNamespace(max_seq_len=max_seq_len),
        )

    def test_default_uses_pow2_token_lens(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("RTP_LLM_STARTUP_REAL_WARMUP_TOKEN_LENS", None)
            self.assertEqual(
                start_server._get_startup_real_warmup_token_lens(
                    self._configs(max_seq_len=16)
                ),
                [2, 4, 8, 16],
            )

    def test_configured_token_lens(self):
        with patch.dict(
            os.environ,
            {"RTP_LLM_STARTUP_REAL_WARMUP_TOKEN_LENS": ("4096,131072,262144")},
        ):
            self.assertEqual(
                start_server._get_startup_real_warmup_token_lens(self._configs()),
                [4096, 131072, 262144],
            )

    def test_configured_token_lens_must_fit_model_max(self):
        with patch.dict(
            os.environ,
            {"RTP_LLM_STARTUP_REAL_WARMUP_TOKEN_LENS": "4096,262144"},
        ):
            with self.assertRaises(ValueError):
                start_server._get_startup_real_warmup_token_lens(
                    self._configs(max_seq_len=131072)
                )


class StartupRealWarmupAddressTest(unittest.TestCase):
    def test_warms_all_local_tp_entries_on_each_node(self):
        for world_rank in (0, 4):
            config = SimpleNamespace(
                server_config=SimpleNamespace(
                    ip="127.0.0.1",
                    start_port=8088,
                    rpc_server_port=8089,
                    worker_info_port_num=10,
                ),
                distribute_config=SimpleNamespace(
                    remote_server_port=8088, zone_name="test"
                ),
                parallelism_config=SimpleNamespace(
                    world_rank=world_rank,
                    world_size=8,
                    local_world_size=4,
                    local_rank=0,
                    tp_size=2,
                    ffn_disaggregate_config=SimpleNamespace(
                        enable_ffn_disaggregate=False, to_string=lambda: "test"
                    ),
                ),
            )
            with patch("socket.gethostbyname", return_value="127.0.0.1"):
                self.assertEqual(
                    start_server._get_startup_real_warmup_grpc_addresses(config),
                    ["127.0.0.1:8089", "127.0.0.1:8109"],
                )

    def test_multi_rank_resolution_failure_does_not_silently_skip_peers(self):
        config = SimpleNamespace(
            server_config=SimpleNamespace(rpc_server_port=8089),
            distribute_config=SimpleNamespace(),
            parallelism_config=SimpleNamespace(world_size=8),
        )
        with patch(
            "rtp_llm.distribute.distributed_server.get_world_info",
            side_effect=RuntimeError("world info unavailable"),
        ):
            with self.assertRaises(
                start_server.StartupRealWarmupAddressResolutionError
            ):
                start_server._get_startup_real_warmup_grpc_addresses(config)


if __name__ == "__main__":
    unittest.main()
