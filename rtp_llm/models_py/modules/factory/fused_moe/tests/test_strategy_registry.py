"""Strategy-registry diagnostics for public MOE_STRATEGY values."""

import subprocess
import sys
import unittest
from unittest.mock import MagicMock

from rtp_llm.config.model_config import ModelConfig
from rtp_llm.models_py.modules.factory.fused_moe.defs.config_adapter import (
    MoEConfigAdapter,
)
from rtp_llm.models_py.modules.factory.fused_moe.strategy_registry import (
    StrategyRegistry,
)
from rtp_llm.ops import MoeConfig, ParallelismConfig


def _config(strategy: str) -> MoEConfigAdapter:
    model_config = ModelConfig()
    model_config.model_type = "test_model"
    model_config.quant_config = None

    parallelism_config = ParallelismConfig()
    parallelism_config.ep_size = 1
    parallelism_config.tp_size = 1
    parallelism_config.dp_size = 1
    parallelism_config.world_size = 1

    moe_config = MoeConfig()
    moe_config.moe_strategy = strategy
    moe_config.use_deepep_low_latency = False
    return MoEConfigAdapter(
        model_config=model_config,
        parallelism_config=parallelism_config,
        moe_config=moe_config,
    )


class StrategyRegistryDiagnosticsTest(unittest.TestCase):
    def test_request_names_value_and_current_model_scope(self):
        with self.assertRaises(ValueError) as cm:
            StrategyRegistry().get_strategy(_config("grouped_fp4"))

        message = str(cm.exception)
        self.assertIn("MOE_STRATEGY='grouped_fp4'", message)
        self.assertIn("model scope 'test_model'", message)
        self.assertIn("generic fused-MoE factory", message)

    def test_explicit_strategy_does_not_fall_back_to_another_backend(self):
        registry = StrategyRegistry()
        fallback = MagicMock()
        fallback.strategy_name = "fallback"
        fallback.supported_moe_quant_method = None
        fallback.can_handle.return_value = True
        registry.register(fallback)

        with self.assertRaises(ValueError) as cm:
            registry.get_strategy(_config("requested"))

        fallback.can_handle.assert_not_called()
        self.assertIn("MOE_STRATEGY='requested'", str(cm.exception))

    def test_explicit_strategy_selects_only_its_registered_backend(self):
        registry = StrategyRegistry()
        requested = MagicMock()
        requested.strategy_name = "requested"
        requested.supported_moe_quant_method = None
        requested.can_handle.return_value = True
        requested.get_attributes.return_value.calculate_priority.return_value = 1
        fallback = MagicMock()
        fallback.strategy_name = "fallback"
        fallback.supported_moe_quant_method = None
        fallback.can_handle.return_value = True
        registry.register(fallback)
        registry.register(requested)

        self.assertIs(registry.get_strategy(_config("requested")), requested)
        fallback.can_handle.assert_not_called()

    def test_package_import_does_not_eagerly_import_deep_gemm(self):
        code = """
import sys

class DeepGemmImportBlocker:
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "deep_gemm" or fullname.startswith("deep_gemm."):
            raise ImportError("deep_gemm deliberately unavailable")
        return None

sys.meta_path.insert(0, DeepGemmImportBlocker())
import rtp_llm.models_py.modules.factory.fused_moe
assert not any(
    name == "deep_gemm" or name.startswith("deep_gemm.") for name in sys.modules
)
"""
        subprocess.run([sys.executable, "-c", code], check=True)


if __name__ == "__main__":
    unittest.main()
