"""Single-rank SM90 FP8 experts with per-output-channel checkpoint scales."""

import torch

from rtp_llm.models_py.modules.factory.fused_moe.defs.priority_attributes import (
    StrategyAttributes,
)
from rtp_llm.models_py.modules.factory.fused_moe.defs.quant_config import (
    FusedMoEQuantConfig,
)
from rtp_llm.models_py.modules.factory.fused_moe.defs.strategy_base import MoeStrategy
from rtp_llm.models_py.modules.factory.fused_moe.utils.config_resolver import (
    MoeConfigResolver,
)


class CudaFp8PerChannelNoDPStrategy(MoeStrategy):
    strategy_name = "fp8_per_channel_no_dp"
    supported_moe_quant_method = "FP8_PER_CHANNEL_COMPRESSED"

    @classmethod
    def check_conditions(cls, checker, config):
        from rtp_llm.models_py.kernels.cuda.fp8_kernel.fp8_kernel import (
            fp8_grouped_gemm_ptpc,
        )
        from rtp_llm.ops import ActivationType

        checker.check(callable(fp8_grouped_gemm_ptpc))
        checker.check(config.moe_strategy in ("auto", cls.strategy_name))
        checker.check(
            config.model_config.activation_type in ("SiGLU", ActivationType.Swiglu)
        )
        checker.check(not config.enable_cuda_graph)
        checker.check(config.world_size == 1 and MoeConfigResolver.is_bf16(config))
        checker.check(
            torch.cuda.is_available() and torch.cuda.get_device_capability()[0] == 9
        )

    def get_attributes(self):
        from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.executors.cutlass_moe import (
            CutlassExpertsFp8PerChannel,
        )
        from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.routers.pure_tp_router import (
            PureTpRouterFp8PerChannel,
        )

        return StrategyAttributes(
            router_class=PureTpRouterFp8PerChannel,
            executor_class=CutlassExpertsFp8PerChannel,
            quant_config=FusedMoEQuantConfig(
                quant_dtype=torch.float8_e4m3fn,
                per_act_token_quant=True,
                per_out_ch_quant=True,
            ),
        )
