"""Batched Triton strategy for common platforms"""

from rtp_llm.models_py.modules.factory.fused_moe.defs.priority_attributes import (
    StrategyAttributes,
)
from rtp_llm.models_py.modules.factory.fused_moe.defs.quant_config import (
    FusedMoEQuantConfig,
)
from rtp_llm.models_py.modules.factory.fused_moe.defs.strategy_base import MoeStrategy


class BatchedTritonStrategy(MoeStrategy):
    """CUDA single GPU without quantization strategy"""

    def get_attributes(self) -> StrategyAttributes:
        from rtp_llm.models_py.modules.factory.fused_moe.impl.common.executor.batched_triton_executor import (
            BatchedTritonExperts,
        )
        from rtp_llm.models_py.modules.factory.fused_moe.impl.common.router.batched_data_router import (
            BatchedDataRouter,
        )

        quant_config = FusedMoEQuantConfig(quant_dtype=None)
        return StrategyAttributes(
            router_class=BatchedDataRouter,
            executor_class=BatchedTritonExperts,
            quant_config=quant_config,
        )
