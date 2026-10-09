import asyncio
import threading
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from rtp_llm import start_server
from rtp_llm.ops import RoleType, SpeculativeType


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

    def test_k3_prefill_entry_only_and_both_switches_required(self):
        config = SimpleNamespace(
            runtime_config=SimpleNamespace(warm_up=True, model_warm_up=True),
            role_config=SimpleNamespace(role_type=RoleType.PREFILL),
            parallelism_config=SimpleNamespace(world_rank=0, world_size=8, tp_size=4),
            model_args=SimpleNamespace(model_type="kimi_k3"),
        )
        self.assertTrue(start_server._should_run_startup_real_warmup(config))
        config.parallelism_config.world_rank = 1
        self.assertFalse(start_server._should_run_startup_real_warmup(config))
        config.parallelism_config.world_rank = 0
        config.role_config.role_type = RoleType.DECODE
        self.assertFalse(start_server._should_run_startup_real_warmup(config))
        config.role_config.role_type = RoleType.PREFILL
        config.runtime_config.model_warm_up = False
        self.assertFalse(start_server._should_run_startup_real_warmup(config))

    def test_real_warmup_uses_model_capability_instead_of_model_name(self):
        config = SimpleNamespace(
            runtime_config=SimpleNamespace(warm_up=True, model_warm_up=True),
            role_config=SimpleNamespace(role_type=RoleType.PREFILL),
            parallelism_config=SimpleNamespace(world_rank=0, world_size=1, tp_size=1),
            model_args=SimpleNamespace(model_type="custom_model"),
        )
        policy = SimpleNamespace(
            supports_startup_real_warmup=True,
            startup_real_warmup_timeout_env="CUSTOM_WARMUP_TIMEOUT",
        )
        with patch(
            "rtp_llm.model_factory.ModelFactory.get_model_cls", return_value=policy
        ):
            self.assertTrue(start_server._should_run_startup_real_warmup(config))
            policy.supports_startup_real_warmup = False
            self.assertFalse(start_server._should_run_startup_real_warmup(config))
            with patch.dict("os.environ", {"CUSTOM_WARMUP_TIMEOUT": "19"}, clear=True):
                self.assertEqual(
                    start_server._get_startup_real_warmup_timeout_s("custom_model"), 19
                )
                self.assertEqual(
                    start_server._get_startup_real_warmup_timeout_s("custom_model", 7),
                    7,
                )

    def test_timeout_and_length_configuration_keep_legacy_dsv4_fallback(self):
        config = SimpleNamespace(
            model_args=SimpleNamespace(max_seq_len=4096),
            jit_config=SimpleNamespace(startup_real_warmup_max_len=512),
        )
        self.assertEqual(start_server._get_startup_real_warmup_max_len(config), 512)
        config.jit_config.startup_real_warmup_max_len = 8192
        with self.assertRaisesRegex(ValueError, "between"):
            start_server._get_startup_real_warmup_max_len(config)
        with patch.dict("os.environ", {"DSV4_STARTUP_REAL_WARMUP_TIMEOUT_S": "13"}, clear=True):
            self.assertEqual(start_server._get_startup_real_warmup_timeout_s("deepseek_v4"), 13)
            self.assertEqual(start_server._get_startup_real_warmup_timeout_s("kimi_k3"), 600)
            self.assertEqual(start_server._get_startup_real_warmup_timeout_s("kimi_k3", 7), 7)

    def test_publish_notification_requires_successful_request_warmup(self):
        event = threading.Event()

        async def succeed(_configs):
            self.assertFalse(event.is_set())
            await asyncio.sleep(0)
            self.assertFalse(event.is_set())

        async def fail(_configs):
            self.assertFalse(event.is_set())
            raise RuntimeError("warmup request failed")

        with patch.object(
            start_server, "_should_run_startup_real_warmup", return_value=True
        ), patch.object(start_server, "_run_startup_real_warmup_grpc", succeed):
            self.assertTrue(start_server._maybe_run_startup_real_warmup(None, event))
            self.assertTrue(event.is_set())

        event.clear()
        with patch.object(
            start_server, "_should_run_startup_real_warmup", return_value=True
        ), patch.object(start_server, "_run_startup_real_warmup_grpc", fail):
            with self.assertRaisesRegex(RuntimeError, "warmup request failed"):
                start_server._maybe_run_startup_real_warmup(None, event)
            self.assertFalse(event.is_set())

        with patch.object(
            start_server, "_should_run_startup_real_warmup", return_value=False
        ), patch.object(start_server, "_run_startup_real_warmup_grpc") as request:
            self.assertFalse(start_server._maybe_run_startup_real_warmup(None, event))
            self.assertFalse(event.is_set())
            request.assert_not_called()


if __name__ == "__main__":
    unittest.main()
