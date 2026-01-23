"""CUDA strategies without quantization"""

from typing import Any, Dict

import torch

from rtp_llm.models_py.modules.factory.fused_moe.defs.priority_attributes import (
    StrategyAttributes,
)
from rtp_llm.models_py.modules.factory.fused_moe.defs.quant_config import (
    FusedMoEQuantConfig,
)
from rtp_llm.models_py.modules.factory.fused_moe.defs.strategy_base import MoeStrategy


class CudaAfDisaggregateStrategy(MoeStrategy):
    """CUDA AF disaggregate strategy"""

    # def create_router(self, config: GptInitModelParameters) -> Any:
    #     from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.routers.afd_data_router import (
    #         AfdDataRouterFfn,
    #     )

    #     return AfdDataRouterFfn(
    #         config,
    #         use_fp8_dispatch=False,
    #         zero_copy=False,
    #         async_finish=True,
    #         return_recv_hook=False,
    #     )

    # def create_executor(
    #     self, config: GptInitModelParameters, weights: Dict[str, torch.Tensor]
    # ) -> Any:
    #     from rtp_llm.models_py.modules.factory.fused_moe.impl.common.executor.batched_triton_executor import (
    #         BatchedTritonExperts,
    #     )
    #     from rtp_llm.utils.model_weight import W

    #     num_max_dispatch_tokens_per_rank = (
    #         config.max_generate_batch_size + config.tp_size - 1
    #     ) // config.tp_size

    #     return BatchedTritonExperts(
    #         max_num_tokens=num_max_dispatch_tokens_per_rank * config.world_size,
    #         num_dispatchers=1,
    #         w1=weights[W.moe_w1],
    #         w2=weights[W.moe_w2],
    #     )

    def get_attributes(self) -> StrategyAttributes:
        from rtp_llm.models_py.modules.factory.fused_moe.impl.common.executor.batched_triton_executor import (
            BatchedTritonExperts,
        )
        from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.routers.afd_data_router import (
            AfdDataRouterFfn,
        )

        quant_config = FusedMoEQuantConfig(
            quant_dtype=torch.float8_e4m3fn,
            block_shape=[128, 128],
        )
        return StrategyAttributes(
            router_class=AfdDataRouterFfn,
            executor_class=BatchedTritonExperts,
            quant_config=quant_config,
        )
