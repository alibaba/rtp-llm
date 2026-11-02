"""Ascend MoE strategy tests (CPU-only).

Pins the fused-MoE strategy contract on Ascend:

- AscendCannStrategy (priority 52) serves bf16 / non-quantized / pure-TP /
  auto / non-fake-balance configs;
- AscendBf16FallbackStrategy is a candidate for non-quantized configs CANN
  rejects (e.g. fp16 act_type or non-auto MOE_STRATEGY), routing through
  the shared BatchedDataRouter + BatchedTritonExperts chain;
- quantized configs fail strategy selection with the registry's
  "No suitable MOE strategy found" error.
"""

import unittest

from rtp_llm.config.model_config import ModelConfig
from rtp_llm.config.quant_config import WeightOnlyInt8PerChannelQuantConfig
from rtp_llm.models_py.modules.factory.fused_moe.defs.config_adapter import (
    MoEConfigAdapter,
)
from rtp_llm.models_py.modules.factory.fused_moe.impl.ascend.strategy import (
    AscendBf16FallbackStrategy,
    AscendCannStrategy,
)
from rtp_llm.models_py.modules.factory.fused_moe.strategy_registry import (
    StrategyRegistry,
)
from rtp_llm.models_py.modules.factory.fused_moe.utils.condition_checker import (
    ConditionChecker,
)
from rtp_llm.ops import MoeConfig, ParallelismConfig

try:
    import torch_npu  # noqa: F401

    _HAS_TORCH_NPU = True
except ImportError:
    _HAS_TORCH_NPU = False


def create_config_adapter(
    ep_size: int = 1,
    tp_size: int = 1,
    quant_config=None,
    data_type: str = "fp16",
    use_all_gather: bool = False,
) -> MoEConfigAdapter:
    """Helper function to create MoEConfigAdapter for testing"""
    model_config = ModelConfig()
    model_config.hidden_size = 1024
    model_config.expert_num = 8
    model_config.moe_k = 2
    model_config.data_type = data_type
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
    moe_config.use_all_gather = use_all_gather
    return MoEConfigAdapter(
        model_config=model_config,
        parallelism_config=parallelism_config,
        moe_config=moe_config,
    )


class TestAscendMoeStrategyContract(unittest.TestCase):
    """CANN serves the default; the fallback covers CANN opt-out configs."""

    def test_fallback_accepts_non_quantized(self):
        """check_conditions only records conditions; it never raises."""
        checker = ConditionChecker("AscendBf16FallbackStrategy.check_conditions()")
        AscendBf16FallbackStrategy.check_conditions(checker, create_config_adapter())
        self.assertTrue(checker.all_passed())

    def test_fallback_attributes_are_real(self):
        """get_attributes wires the shared batched router/executor chain."""
        attrs = AscendBf16FallbackStrategy().get_attributes()

        from rtp_llm.models_py.modules.factory.fused_moe.impl.common.executor.batched_triton_executor import (
            BatchedTritonExperts,
        )
        from rtp_llm.models_py.modules.factory.fused_moe.impl.common.router.batched_data_router import (
            BatchedDataRouter,
        )

        self.assertIs(attrs.router_class, BatchedDataRouter)
        self.assertIs(attrs.executor_class, BatchedTritonExperts)

    @unittest.skipUnless(
        _HAS_TORCH_NPU, "torch_npu is required for CANN strategy wiring"
    )
    def test_cann_selected_for_default_bf16_config(self):
        """bf16 + non-quant + pure-TP + auto picks AscendCannStrategy."""
        registry = StrategyRegistry()
        registry.register(AscendCannStrategy())
        registry.register(AscendBf16FallbackStrategy())
        strategy = registry.get_strategy(
            create_config_adapter(data_type="bf16", use_all_gather=True)
        )
        self.assertIsInstance(strategy, AscendCannStrategy)

    def test_fallback_serves_cann_rejected_config(self):
        """fp16 (CANN is bf16-only) falls through to the fallback strategy."""
        registry = StrategyRegistry()
        registry.register(AscendCannStrategy())
        registry.register(AscendBf16FallbackStrategy())
        strategy = registry.get_strategy(create_config_adapter())
        self.assertIsInstance(strategy, AscendBf16FallbackStrategy)

    def test_quantized_config_fails_selection(self):
        """Quantized configs have no Ascend candidate; registry raises."""
        registry = StrategyRegistry()
        registry.register(AscendBf16FallbackStrategy())
        with self.assertRaises(ValueError) as ctx:
            registry.get_strategy(
                create_config_adapter(
                    quant_config=WeightOnlyInt8PerChannelQuantConfig()
                )
            )
        self.assertIn("No suitable MOE strategy found", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
