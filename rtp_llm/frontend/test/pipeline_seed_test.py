import unittest
from types import SimpleNamespace

from rtp_llm.config.generate_config import RoleType
from rtp_llm.ops import PDSepConfig
from rtp_llm.pipeline.pipeline import Pipeline
from rtp_llm.server.backend_rpc_server_visitor import BackendRPCServerVisitor

V41_BOUNDED_REPLAY_SEED = 0x4453563431494231


def _pd_config():
    config = PDSepConfig()
    config.role_type = RoleType.FRONTEND
    return config


class PipelineCacheKeySeedTest(unittest.TestCase):
    """Frontend routing keys must share the engine's cache-key namespace.

    The V4.1 engine indexes its prefix cache with the model's
    cache_key_hash_seed (0x4453563431494231 in bounded-replay mode). The
    frontend Pipeline must forward model_config to BackendRPCServerVisitor so
    block cache keys sent to the load balancer are computed in the same
    namespace; otherwise every worker lookup reports zero prefix hits and
    requests alternate between prefill replicas.
    """

    def _pipeline(self, model_config):
        return Pipeline(
            special_tokens=SimpleNamespace(),
            pd_sep_config=_pd_config(),
            addresses=[],
            max_seq_len=100,
            seq_size_per_block=2,
            tokenizer=None,
            model_config=model_config,
        )

    def test_pipeline_forwards_model_config_seed(self):
        model_config = SimpleNamespace(cache_key_hash_seed=V41_BOUNDED_REPLAY_SEED)
        pipeline = self._pipeline(model_config)
        self.assertEqual(
            pipeline.backend_rpc_server_visitor.cache_key_seed,
            V41_BOUNDED_REPLAY_SEED,
        )

    def test_pipeline_without_model_config_keeps_seed_zero(self):
        pipeline = self._pipeline(None)
        self.assertEqual(pipeline.backend_rpc_server_visitor.cache_key_seed, 0)

    def test_visitor_uses_model_seed(self):
        visitor = BackendRPCServerVisitor(
            max_seq_len=100,
            seq_size_per_block=2,
            pd_sep_config=_pd_config(),
            addresses=[],
            model_config=SimpleNamespace(cache_key_hash_seed=V41_BOUNDED_REPLAY_SEED),
        )
        self.assertEqual(visitor.cache_key_seed, V41_BOUNDED_REPLAY_SEED)

    def test_visitor_without_model_config_defaults_to_zero(self):
        visitor = BackendRPCServerVisitor(
            max_seq_len=100,
            seq_size_per_block=2,
            pd_sep_config=_pd_config(),
            addresses=[],
        )
        self.assertEqual(visitor.cache_key_seed, 0)


if __name__ == "__main__":
    unittest.main()
