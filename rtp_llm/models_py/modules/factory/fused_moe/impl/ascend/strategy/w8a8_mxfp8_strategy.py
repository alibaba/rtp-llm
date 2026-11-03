"""Ascend NPU W8A8_MXFP8 MoE strategy (W8A8_MXFP8: Weight & Activation FP8 Quantization).

Reuses the pure-torch ``BatchedDataRouter`` (its ``prepare`` asserts
``a1_scale/a2_scale is None``, which holds here because the activation MX
quantization happens inside the executor) with relaxed quantization
conditions, and pairs it with ``AscendW8A8MXFP8Executor``.

The existing ``AscendBf16FallbackStrategy`` requires "no quantization", so
without this strategy a quantized config would find no candidate and
``StrategyRegistry.get_strategy`` would raise.
"""

from typing import Any

from rtp_llm.models_py.modules.factory.fused_moe.defs.priority_attributes import (
    StrategyAttributes,
)
from rtp_llm.models_py.modules.factory.fused_moe.defs.quant_config import (
    FusedMoEQuantConfig,
)
from rtp_llm.models_py.modules.factory.fused_moe.defs.strategy_base import MoeStrategy
from rtp_llm.models_py.modules.factory.fused_moe.impl.common.router.batched_data_router import (
    BatchedDataRouter,
)


class AscendW8A8MXFP8BatchedDataRouter(BatchedDataRouter):
    """BatchedDataRouter for the MXFP8 quantized path (single GPU / tp==ep)."""

    @classmethod
    def check_conditions(cls, checker: Any, config: Any) -> None:
        from rtp_llm.models_py.modules.factory.fused_moe.utils.config_resolver import (
            MoeConfigResolver,
        )

        resolver = MoeConfigResolver()
        # Unlike the base router, quantization IS expected here (ModelSlim /
        # W8A8_MXFP8); activation quantization happens inside the executor so
        # prepare()'s a1_scale/a2_scale=None assertion still holds.
        checker.check(
            resolver.get_quant_method(config) in ("ASCEND_W8A8_MXFP8",)
        )
        checker.check(resolver.is_single_gpu(config) or resolver.is_tp_equal_ep(config))


class AscendW8A8MXFP8MoeStrategy(MoeStrategy):
    """Ascend W8A8_MXFP8 MoE strategy."""

    def get_attributes(self) -> StrategyAttributes:
        from rtp_llm.models_py.modules.factory.fused_moe.impl.ascend.executors.w8a8_mxfp8_executor import (
            AscendW8A8MXFP8Executor,
        )

        return StrategyAttributes(
            router_class=AscendW8A8MXFP8BatchedDataRouter,
            executor_class=AscendW8A8MXFP8Executor,
            quant_config=FusedMoEQuantConfig(quant_dtype=None),
        )

    @classmethod
    def check_conditions(cls, checker: Any, config: Any) -> None:
        from rtp_llm.models_py.modules.factory.fused_moe.utils.config_resolver import (
            MoeConfigResolver,
        )

        resolver = MoeConfigResolver()
        checker.check(resolver.is_bf16(config))
        # Static (ModelSlim) path only in this stage; the dynamic load-quant
        # path (FP8_PER_BLOCK) is enabled by a follow-up TODO.
        checker.check(resolver.get_quant_method(config) in ("ASCEND_W8A8_MXFP8",))
