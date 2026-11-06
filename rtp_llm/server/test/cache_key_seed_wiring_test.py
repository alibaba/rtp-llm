"""Cache-key seed wiring regression tests.

Lock the frontend routing path (FrontendWorker -> Pipeline ->
BackendRPCServerVisitor) so the model's cache_key_hash_seed namespace reaches
the schedule-request block cache keys instead of silently falling back to 0.
"""

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from rtp_llm.server.backend_rpc_server_visitor import BackendRPCServerVisitor
from rtp_llm.server.cache_key_routing import get_v41_cache_key_seed

V41_BOUNDED_SEED = 0x4453563431494231


def _model_config(seed=V41_BOUNDED_SEED):
    return SimpleNamespace(
        cache_key_hash_seed=seed,
        mm_model_config=None,
        max_seq_len=32768,
        ckpt_path="",
        tokenizer_path="",
        model_type="dsv41",
        attn_config=SimpleNamespace(tokens_per_block=128),
    )


class CacheKeySeedContractTest(unittest.TestCase):
    def test_seed_passes_through_from_model_config(self):
        self.assertEqual(get_v41_cache_key_seed(_model_config()), V41_BOUNDED_SEED)

    def test_seed_defaults_to_zero_without_config(self):
        self.assertEqual(get_v41_cache_key_seed(SimpleNamespace()), 0)
        self.assertEqual(get_v41_cache_key_seed(None), 0)


class VisitorSeedTest(unittest.TestCase):
    def _build(self, **kwargs):
        with patch("rtp_llm.server.backend_rpc_server_visitor.ModelRpcClient"), patch(
            "rtp_llm.server.backend_rpc_server_visitor.HostService"
        ), patch("rtp_llm.server.backend_rpc_server_visitor.HostServiceArgs"), patch(
            "rtp_llm.server.backend_rpc_server_visitor.MasterClient"
        ):
            return BackendRPCServerVisitor(
                max_seq_len=32768,
                seq_size_per_block=128,
                pd_sep_config=MagicMock(),
                addresses=[],
                **kwargs,
            )

    def test_model_config_seed_reaches_visitor(self):
        visitor = self._build(model_config=_model_config())
        self.assertEqual(visitor.cache_key_seed, V41_BOUNDED_SEED)

    def test_missing_model_config_keeps_zero_seed(self):
        visitor = self._build()
        self.assertEqual(visitor.cache_key_seed, 0)


class PipelineForwardingTest(unittest.TestCase):
    @staticmethod
    def _kwargs():
        return dict(
            special_tokens=MagicMock(),
            pd_sep_config=MagicMock(),
            addresses=[],
            max_seq_len=32768,
            seq_size_per_block=128,
            tokenizer=None,
        )

    def test_model_config_forwarded_to_visitor(self):
        from rtp_llm.pipeline import pipeline as pipeline_module

        mc = _model_config()
        with patch.object(pipeline_module, "BackendRPCServerVisitor") as visitor_cls:
            pipeline_module.Pipeline(model_config=mc, **self._kwargs())
        self.assertIs(visitor_cls.call_args.kwargs["model_config"], mc)

    def test_model_config_defaults_to_none(self):
        from rtp_llm.pipeline import pipeline as pipeline_module

        with patch.object(pipeline_module, "BackendRPCServerVisitor") as visitor_cls:
            pipeline_module.Pipeline(**self._kwargs())
        self.assertIsNone(visitor_cls.call_args.kwargs["model_config"])


class FrontendWorkerWiringTest(unittest.TestCase):
    def test_frontend_worker_passes_model_config_to_pipeline(self):
        import rtp_llm.frontend.frontend_worker as fw

        mc = _model_config()
        py_env = SimpleNamespace(
            sp_config=None,
            grpc_config=None,
            vit_config=None,
            distribute_config=None,
            parallelism_config=None,
            server_config=SimpleNamespace(ip=""),
            master_config=None,
            prefill_cp_config=None,
            generate_env_config=None,
        )
        with patch.object(fw, "TokenizerFactory"), patch.object(
            fw, "EngineConfig"
        ), patch.object(fw, "get_world_info"), patch.object(
            fw, "get_dp_addrs_from_world_info", return_value=[]
        ), patch.object(
            fw, "Pipeline"
        ) as pipeline_cls:
            fw.FrontendWorker(py_env, mc, MagicMock())
        self.assertIs(pipeline_cls.call_args.kwargs["model_config"], mc)


if __name__ == "__main__":
    unittest.main()
