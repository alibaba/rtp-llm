import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from rtp_llm import start_server
from rtp_llm.ops import RoleType, SpeculativeType


class StartupRealWarmupTest(unittest.TestCase):
    def test_backend_status_carries_resolved_model_limit(self):
        from rtp_llm.start_backend_server import _send_pipe_status

        writer = Mock()
        _send_pipe_status(writer, "success", "ready", model_max_seq_len=4096)
        writer.send.assert_called_once_with(
            {
                "status": "success",
                "message": "ready",
                "traceback": "",
                "model_max_seq_len": 4096,
            }
        )
        writer.close.assert_called_once()

    def test_resolved_limit_reaches_rocm_warmup_without_changing_cli_config(self):
        config = SimpleNamespace(model_args=SimpleNamespace(max_seq_len=None))
        with patch.object(
            start_server, "_should_run_startup_real_warmup", return_value=True
        ), patch.object(
            start_server, "_rocm_startup_warmup_requested", return_value=True
        ), patch.object(
            start_server, "_run_rocm_startup_real_warmup_grpc", new=Mock()
        ) as warmup, patch.object(
            start_server, "_run_startup_real_warmup_async"
        ) as run:
            self.assertTrue(
                start_server._maybe_run_startup_real_warmup(
                    config, model_max_seq_len=4096
                )
            )
        warmup.assert_called_once_with(config, model_max_seq_len=4096)
        run.assert_called_once_with(warmup.return_value)
        self.assertIsNone(config.model_args.max_seq_len)

    def test_rocm_warmup_auto_default_and_override_use_the_readiness_gate(self):
        config = SimpleNamespace(
            parallelism_config=SimpleNamespace(world_size=1, world_rank=0, tp_size=1),
            runtime_config=SimpleNamespace(warm_up=True, model_warm_up=True),
            role_config=SimpleNamespace(role_type=RoleType.PDFUSION),
            model_args=SimpleNamespace(model_type="qwen_3"),
            kv_cache_config=SimpleNamespace(reuse_cache=True),
        )
        with patch.dict("os.environ", {"RTP_LLM_ROCM_STARTUP_WARMUP": "0"}):
            self.assertFalse(start_server._should_run_startup_real_warmup(config))
        with patch.dict("os.environ", {}, clear=True), patch.object(
            start_server.torch.version, "hip", "test-rocm"
        ):
            self.assertTrue(start_server._should_run_startup_real_warmup(config))
        with patch.dict(
            "os.environ", {"RTP_LLM_ROCM_STARTUP_WARMUP": "1"}
        ), patch.object(start_server.torch.version, "hip", "test-rocm"):
            self.assertTrue(start_server._should_run_startup_real_warmup(config))
            config.runtime_config.model_warm_up = False
            self.assertFalse(start_server._should_run_startup_real_warmup(config))
            config.runtime_config.model_warm_up = True
            config.kv_cache_config.reuse_cache = False
            with self.assertRaisesRegex(ValueError, "REUSE_CACHE"):
                start_server._should_run_startup_real_warmup(config)
            config.kv_cache_config.reuse_cache = True
            config.parallelism_config.world_size = 2
            with self.assertRaisesRegex(ValueError, "single-rank"):
                start_server._should_run_startup_real_warmup(config)
            config.parallelism_config.world_size = 1
            config.role_config.role_type = RoleType.DECODE
            with self.assertRaisesRegex(ValueError, "PDFUSION"):
                start_server._should_run_startup_real_warmup(config)
            config.role_config.role_type = RoleType.PDFUSION
            with patch.object(start_server.torch.version, "hip", None):
                with self.assertRaisesRegex(ValueError, "ROCm build"):
                    start_server._should_run_startup_real_warmup(config)

    def test_rocm_warmup_failure_propagates_before_marking_ready(self):
        config = SimpleNamespace()
        with patch.object(
            start_server, "_should_run_startup_real_warmup", return_value=True
        ), patch.object(
            start_server, "_rocm_startup_warmup_requested", return_value=True
        ), patch.object(
            start_server, "_run_rocm_startup_real_warmup_grpc", return_value=None
        ), patch.object(
            start_server,
            "_run_startup_real_warmup_async",
            side_effect=RuntimeError("JIT failed"),
        ):
            with self.assertRaisesRegex(RuntimeError, "JIT failed"):
                start_server._maybe_run_startup_real_warmup(config)

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
