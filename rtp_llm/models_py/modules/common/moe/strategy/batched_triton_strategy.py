from typing import Any, Dict

import torch

from rtp_llm.config.gpt_init_model_parameters import GptInitModelParameters
from rtp_llm.models_py.modules.factory.fused_moe.strategies.base import MoeStrategy
from rtp_llm.models_py.modules.factory.fused_moe.strategies.priority_attributes import (
    StrategyAttributes,
)


class BatchedTritonStrategy(MoeStrategy):
    """CUDA single GPU without quantization strategy"""

    def create_router(self, config: GptInitModelParameters) -> Any:
        from rtp_llm.models_py.modules.common.moe.router.batched_data_router import (
            BatchedDataRouter,
        )

        max_num_tokens = (
            config.max_generate_batch_size + config.tp_size - 1
        ) // config.tp_size

        return BatchedDataRouter(
            max_num_tokens=max_num_tokens,
            num_local_experts=config.expert_num,
            num_dispatchers=1,
            rank=0,
            num_experts=config.expert_num,
        )

    def create_executor(
        self, config: GptInitModelParameters, weights: Dict[str, torch.Tensor]
    ) -> Any:
        from rtp_llm.models_py.modules.common.moe.executor.batched_triton_executor import (
            BatchedTritonExperts,
        )
        from rtp_llm.utils.model_weight import W

        max_num_tokens = (
            config.max_generate_batch_size + config.tp_size - 1
        ) // config.tp_size

        return BatchedTritonExperts(
            max_num_tokens=max_num_tokens,
            num_dispatchers=1,
            w1=weights[W.moe_w1],
            w2=weights[W.moe_w2],
        )

    def get_attributes(self) -> StrategyAttributes:
        from rtp_llm.models_py.modules.common.moe.executor.batched_triton_executor import (
            BatchedTritonExperts,
        )
        from rtp_llm.models_py.modules.common.moe.router.batched_data_router import (
            BatchedDataRouter,
        )

        return StrategyAttributes(
            router_class=BatchedDataRouter,
            executor_class=BatchedTritonExperts,
        )
