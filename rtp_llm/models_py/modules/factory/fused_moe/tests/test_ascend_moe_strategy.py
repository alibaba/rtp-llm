"""Ascend MoE strategy tests (CPU-only).

AscendBf16FallbackStrategy is a placeholder that rejects every MoE
configuration on Ascend: no NPU-capable MoE executor exists (Triton-based
executors are excluded from Ascend deps) and Ascend TP (tp_size > 1) is not
implemented either. These tests pin the fail-fast contract.
"""

import unittest

from rtp_llm.config.model_config import ModelConfig
from rtp_llm.models_py.modules.factory.fused_moe.defs.config_adapter import (
    MoEConfigAdapter,
)
from rtp_llm.models_py.modules.factory.fused_moe.impl.ascend.strategy.pytorch_fallback import (
    AscendBf16FallbackStrategy,
)
from rtp_llm.models_py.modules.factory.fused_moe.strategy_registry import (
    StrategyRegistry,
)
from rtp_llm.models_py.modules.factory.fused_moe.utils.condition_checker import (
    ConditionChecker,
)
from rtp_llm.ops import MoeConfig, ParallelismConfig


def create_config_adapter(
    ep_size: int = 1,
    tp_size: int = 1,
    quant_config=None,
) -> MoEConfigAdapter:
    """Helper function to create MoEConfigAdapter for testing"""
    model_config = ModelConfig()
    model_config.hidden_size = 1024
    model_config.expert_num = 8
    model_config.moe_k = 2
    model_config.data_type = "fp16"
    model_config.quant_config = quant_config

    parallelism_config = ParallelismConfig()
    parallelism_config.ep_size = ep_size
    parallelism_config.tp_size = tp_size
    parallelism_config.dp_size = 1
    parallelism_config.ep_rank = 0
    parallelism_config.tp_rank = 0
    parallelism_config.dp_rank = 0
    parallelism_config.world_size = ep_size * tp_size
    parallelism_config.world_rank = 0
    parallelism_config.local_rank = 0
    parallelism_config.local_world_size = 1

    moe_config = MoeConfig()
    moe_config.ll_num_max_token = 128
    return MoEConfigAdapter(
        model_config=model_config,
        parallelism_config=parallelism_config,
        moe_config=moe_config,
    )


class TestAscendMoeStrategyRejected(unittest.TestCase):
    """Ascend MoE must fail fast with a clear error."""

    def _assert_rejected(self, config: MoEConfigAdapter) -> None:
        checker = ConditionChecker("AscendBf16FallbackStrategy.check_conditions()")
        with self.assertRaises(ValueError) as ctx:
            AscendBf16FallbackStrategy.check_conditions(checker, config)
        self.assertIn("Ascend MoE is not supported", str(ctx.exception))
        self.assertIn("tp_size > 1", str(ctx.exception))

    def test_rejects_single_gpu_bf16(self):
        """Non-quantized, tp_size=1 (previously 'supported') must be rejected."""
        self._assert_rejected(create_config_adapter())

    def test_rejects_tp_gt_1(self):
        """tp_size > 1 (pure TP topology, expert-dropping risk) must be rejected."""
        self._assert_rejected(create_config_adapter(tp_size=2))

    def test_rejects_ep_enabled(self):
        """EP topologies must be rejected."""
        self._assert_rejected(create_config_adapter(ep_size=2))

    def test_registry_surfaces_error(self):
        """The rejection must propagate through strategy selection.

        can_handle() consults get_attributes() before check_conditions(), so
        the registry path raises RuntimeError from get_attributes(); direct
        check_conditions() calls raise ValueError. Both carry the same
        message.
        """
        registry = StrategyRegistry()
        registry.register(AscendBf16FallbackStrategy())
        with self.assertRaises((ValueError, RuntimeError)) as ctx:
            registry.get_strategy(create_config_adapter())
        self.assertIn("Ascend MoE is not supported", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
