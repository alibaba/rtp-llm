import unittest
from types import SimpleNamespace
from unittest.mock import patch

from rtp_llm.frontend.frontend_worker import FrontendWorker
from rtp_llm.pipeline.pipeline import Pipeline
from rtp_llm.server.cache_key_routing import get_v41_cache_key_seed


class FrontendCacheKeySeedTest(unittest.TestCase):
    def test_frontend_preserves_model_cache_namespace_through_pipeline(self):
        for seed in (0x4453563431494231, 0x4453563431494631):
            with self.subTest(seed=seed):
                model = SimpleNamespace(
                    cache_key_hash_seed=seed,
                    ckpt_path="unused",
                    tokenizer_path="unused",
                    model_type="deepseek_v41",
                    max_seq_len=32768,
                    attn_config=SimpleNamespace(tokens_per_block=128),
                )
                env = SimpleNamespace(
                    server_config=None,
                    distribute_config=None,
                    parallelism_config=None,
                    prefill_cp_config=None,
                    vit_config=None,
                    sp_config=None,
                    grpc_config=None,
                    master_config=None,
                    generate_env_config=None,
                )
                with (
                    patch("rtp_llm.frontend.frontend_worker.TokenizerFactory.create"),
                    patch("rtp_llm.frontend.frontend_worker.EngineConfig.create"),
                    patch("rtp_llm.frontend.frontend_worker.get_world_info"),
                    patch("rtp_llm.frontend.frontend_worker.get_dp_addrs_from_world_info"),
                    patch("rtp_llm.pipeline.pipeline.BackendRPCServerVisitor") as visitor,
                ):
                    FrontendWorker(env, model, SimpleNamespace())
                received = visitor.call_args.kwargs.get("model_config")
                self.assertIs(received, model)
                self.assertEqual(get_v41_cache_key_seed(received), seed)

    def test_pipeline_keeps_optional_model_config_for_existing_callers(self):
        with patch("rtp_llm.pipeline.pipeline.BackendRPCServerVisitor") as visitor:
            Pipeline(SimpleNamespace(), None, [], 32768, 128, None)
        self.assertIsNone(visitor.call_args.kwargs["model_config"])


if __name__ == "__main__":
    unittest.main()
