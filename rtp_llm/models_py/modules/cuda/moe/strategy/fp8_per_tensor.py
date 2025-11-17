"""CUDA FP8 PerTensor quantization strategies"""

from typing import Any, Dict

import torch

from rtp_llm.config.gpt_init_model_parameters import GptInitModelParameters
from rtp_llm.models_py.modules.factory.fused_moe.strategies.base import MoeStrategy
from rtp_llm.models_py.modules.factory.fused_moe.strategies.priority_attributes import (
    StrategyAttributes,
)
from rtp_llm.utils.model_weight import W


class CudaFp8PerTensorEpLowLatencyStrategy(MoeStrategy):
    """CUDA FP8 PerTensor EP low latency strategy"""

    def create_router(self, config: GptInitModelParameters) -> Any:
        from rtp_llm.models_py.modules.cuda.moe.routers.deepep_low_latency_router import (
            DeepEpLowLatencyRouter,
        )

        return DeepEpLowLatencyRouter(config, use_fp8_dispatch=True)

    def create_executor(
        self, config: GptInitModelParameters, weights: Dict[str, torch.Tensor]
    ) -> Any:
        from rtp_llm.models_py.modules.cuda.moe.executors.cutlass_moe import (
            CutlassBatchedExpertsFp8,
        )

        max_num_tokens = (
            config.max_generate_batch_size + config.tp_size - 1
        ) // config.tp_size
        num_dispatchers = config.world_size // config.tp_size

        return CutlassBatchedExpertsFp8(
            max_num_tokens=max_num_tokens,
            num_dispatchers=num_dispatchers,
            w1=weights[W.moe_w1],
            w2=weights[W.moe_w2],
            w1_scale=weights[W.moe_s1],
            w2_scale=weights[W.moe_s2],
            a1q_scale=weights.get(W.moe_w1_input_sr, None),
            a2_scale=weights.get(W.moe_w2_input_sr, None),
            num_experts=config.expert_num,
            per_act_token_quant=True,
        )

    def get_attributes(self) -> StrategyAttributes:
        from rtp_llm.models_py.modules.cuda.moe.executors.cutlass_moe import (
            CutlassBatchedExpertsFp8,
        )
        from rtp_llm.models_py.modules.cuda.moe.routers.deepep_low_latency_router import (
            DeepEpLowLatencyRouter,
        )

        return StrategyAttributes(
            router_class=DeepEpLowLatencyRouter,
            executor_class=CutlassBatchedExpertsFp8,
        )


class CudaFp8PerTensorEpNormalStrategy(MoeStrategy):
    """CUDA FP8 PerTensor EP normal mode strategy"""

    def create_router(self, config: GptInitModelParameters) -> Any:
        from rtp_llm.models_py.modules.cuda.moe.routers.deepep_normal_router import (
            DeepepNormalRouter,
        )

        return DeepepNormalRouter(config, use_fp8=True, expert_alignment=1)

    def create_executor(
        self, config: GptInitModelParameters, weights: Dict[str, torch.Tensor]
    ) -> Any:
        from rtp_llm.models_py.modules.cuda.moe.executors.cutlass_moe import (
            CutlassExpertsFp8,
        )

        return CutlassExpertsFp8(
            w1=weights[W.moe_w1],
            w2=weights[W.moe_w2],
            w1_scale=weights[W.moe_s1],
            w2_scale=weights[W.moe_s2],
            a1q_scale=weights.get(W.moe_w1_input_sr, None),
            a2_scale=weights.get(W.moe_w2_input_sr, None),
            num_experts=config.expert_num,
            per_act_token_quant=True,
        )

    def get_attributes(self) -> StrategyAttributes:
        from rtp_llm.models_py.modules.cuda.moe.executors.cutlass_moe import (
            CutlassExpertsFp8,
        )
        from rtp_llm.models_py.modules.cuda.moe.routers.deepep_normal_router import (
            DeepepNormalRouter,
        )

        return StrategyAttributes(
            router_class=DeepepNormalRouter,
            executor_class=CutlassExpertsFp8,
        )


# TODO expand to TP=EP case
class CudaFp8PerTensorSingleGpuStrategy(MoeStrategy):
    """CUDA FP8 PerTensor single GPU strategy"""

    def create_router(self, config: GptInitModelParameters) -> Any:
        from rtp_llm.models_py.modules.cuda.moe.routers.no_ep_standard_router import (
            DataRouterNoEPStandard,
        )

        return DataRouterNoEPStandard(num_dispatchers=1)

    def create_executor(
        self, config: GptInitModelParameters, weights: Dict[str, torch.Tensor]
    ) -> Any:
        from rtp_llm.models_py.modules.cuda.moe.executors.cutlass_moe import (
            CutlassExpertsFp8,
        )

        return CutlassExpertsFp8(
            w1=weights[W.moe_w1],
            w2=weights[W.moe_w2],
            w1_scale=weights[W.moe_s1],
            w2_scale=weights[W.moe_s2],
            a1q_scale=weights.get(W.moe_w1_input_sr, None),
            a2_scale=weights.get(W.moe_w2_input_sr, None),
            num_experts=config.expert_num,
            per_act_token_quant=True,
        )

    def get_attributes(self) -> StrategyAttributes:
        from rtp_llm.models_py.modules.cuda.moe.executors.cutlass_moe import (
            CutlassExpertsFp8,
        )
        from rtp_llm.models_py.modules.cuda.moe.routers.no_ep_standard_router import (
            DataRouterNoEPStandard,
        )

        return StrategyAttributes(
            router_class=DataRouterNoEPStandard,
            executor_class=CutlassExpertsFp8,
        )
