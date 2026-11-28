import pickle
import unittest
from dataclasses import dataclass, replace
from types import SimpleNamespace
from unittest.mock import patch

from rtp_llm.async_decoder_engine.embedding.embedding_engine import (
    EmbeddingCppEngine,
    _cacheless_prefill_graph_backend_available,
)
from rtp_llm.models_py.modules.factory.attention import attn_factory
from rtp_llm.ops import FMHAConfig, HWKernelConfig


@dataclass
class _EngineConfigStub:
    hw_kernel_config: HWKernelConfig


class _ModelStub:
    def __init__(self, declares: bool):
        self._declares = declares
        self.model_config = SimpleNamespace(getAttentionConfigs=lambda tp: object())
        self.parallelism_config = SimpleNamespace(get_attn_tp_size=lambda: 1)
        self.fmha_config = FMHAConfig()

    def support_embedding_prefill_graph(self):
        return self._declares


def _engine(declares: bool, enable_cuda_graph: bool):
    engine = EmbeddingCppEngine.__new__(EmbeddingCppEngine)
    engine.model = _ModelStub(declares)
    engine.engine_config = _EngineConfigStub(hw_kernel_config=HWKernelConfig())
    engine.engine_config.hw_kernel_config.enable_cuda_graph = enable_cuda_graph
    return engine


class EmbeddingEngineGraphGateTest(unittest.TestCase):
    def test_graph_disabled_config_passes_through(self):
        engine = _engine(declares=True, enable_cuda_graph=False)
        gated = engine._graph_gated_engine_config()
        self.assertIs(gated, engine.engine_config)
        self.assertFalse(gated.hw_kernel_config.enable_cuda_graph)

    def test_declaring_model_on_capable_device_keeps_graph(self):
        engine = _engine(declares=True, enable_cuda_graph=True)
        with patch.object(
            attn_factory,
            "has_cacheless_prefill_cuda_graph_backend",
            return_value=True,
        ):
            gated = engine._graph_gated_engine_config()
        self.assertIs(gated, engine.engine_config)
        self.assertTrue(gated.hw_kernel_config.enable_cuda_graph)

    def test_declaring_model_on_incapable_device_degrades_to_eager(self):
        engine = _engine(declares=True, enable_cuda_graph=True)
        with patch.object(
            attn_factory,
            "has_cacheless_prefill_cuda_graph_backend",
            return_value=False,
        ):
            gated = engine._graph_gated_engine_config()
        self.assertIsNot(gated, engine.engine_config)
        self.assertFalse(gated.hw_kernel_config.enable_cuda_graph)
        # The original config is left untouched for other consumers.
        self.assertTrue(engine.engine_config.hw_kernel_config.enable_cuda_graph)
        # The eager copy stays picklable (the gate deep-copies hw_kernel_config).
        roundtrip = pickle.loads(pickle.dumps(gated.hw_kernel_config))
        self.assertFalse(roundtrip.enable_cuda_graph)

    def test_undeclaring_model_still_degrades_to_eager(self):
        engine = _engine(declares=False, enable_cuda_graph=True)
        with patch.object(
            attn_factory,
            "has_cacheless_prefill_cuda_graph_backend",
            return_value=True,
        ):
            gated = engine._graph_gated_engine_config()
        self.assertIsNot(gated, engine.engine_config)
        self.assertFalse(gated.hw_kernel_config.enable_cuda_graph)

    def test_capability_probe_uses_model_attention_config(self):
        model = _ModelStub(declares=True)
        model.fmha_config = None
        seen = {}

        def fake_get_attn_configs(tp_size):
            seen["tp_size"] = tp_size
            return "attn-configs"

        model.model_config = SimpleNamespace(
            getAttentionConfigs=fake_get_attn_configs
        )

        def record(attn_configs, fmha_config):
            seen["attn_configs"] = attn_configs
            seen["fmha_config"] = fmha_config
            return True

        with patch.object(
            attn_factory,
            "has_cacheless_prefill_cuda_graph_backend",
            side_effect=record,
        ):
            self.assertTrue(_cacheless_prefill_graph_backend_available(model))
        self.assertEqual(seen["tp_size"], 1)
        self.assertEqual(seen["attn_configs"], "attn-configs")
        self.assertIsNone(seen["fmha_config"])


if __name__ == "__main__":
    unittest.main()
