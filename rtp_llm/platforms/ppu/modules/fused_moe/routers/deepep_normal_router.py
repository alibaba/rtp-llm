"""PPU DeepEP Normal router implementations."""

from typing import Any, Optional

import torch
from rtp_llm.models_py.distributed.deepep_wrapper import DeepEPWrapper
from rtp_llm.models_py.modules.factory.fused_moe.defs.config_adapter import (
    MoEConfigAdapter,
)
from rtp_llm.models_py.modules.factory.fused_moe.defs.quant_config import (
    FusedMoEQuantConfig,
)
from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.routers.deepep_normal_router import (
    DeepepNormalRouterNoQuant,
)
from rtp_llm.models_py.modules.factory.fused_moe.utils.config_resolver import (
    MoeConfigResolver,
)
from rtp_llm.platforms.ppu.kernels.int8.int8_kernel.int8_quant import (
    per_token_quant_int8,
)


class PpuDeepepNormalRouterNoQuant(DeepepNormalRouterNoQuant):
    """DeepEP Normal BF16 router without CUDA SM capability checks."""

    @classmethod
    def check_conditions(cls, checker: Any, config: MoEConfigAdapter) -> None:
        resolver = MoeConfigResolver()
        checker.check(resolver.is_ep_enabled(config))
        checker.check(not resolver.use_low_latency(config))
        checker.check(resolver.get_quant_method(config) is None)
        checker.check(DeepEPWrapper.supported())

    def __init__(
        self,
        config: MoEConfigAdapter,
        quant_config: FusedMoEQuantConfig,
    ) -> None:
        super().__init__(config, quant_config)

    def prepare(
        self,
        a1: torch.Tensor,
        a1_scale: Optional[torch.Tensor],
        a2_scale: Optional[torch.Tensor],
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
    ):
        token_num = a1.size(0)
        tp_token_size = (token_num + self.config.tp_size - 1) // self.config.tp_size
        slice_begin = min(tp_token_size * self.config.tp_rank, token_num)
        slice_size = min(token_num - slice_begin, tp_token_size)
        if slice_size == 0 and token_num > 0:
            a1 = a1.new_empty((0, a1.shape[1]))
            topk_ids = topk_ids.new_empty((0, topk_ids.shape[1]))
            topk_weights = topk_weights.new_empty((0, topk_weights.shape[1]))
        return super().prepare(a1, a1_scale, a2_scale, topk_weights, topk_ids)


class PpuDeepepNormalRouterW8A8Int8(PpuDeepepNormalRouterNoQuant):
    """Quantize each local token once and dispatch INT8 data with its scale."""

    QUANT_METHOD = "W8A8_INT8_PER_CHANNEL_COMPRESSED"

    @classmethod
    def check_conditions(cls, checker: Any, config: MoEConfigAdapter) -> None:
        resolver = MoeConfigResolver()
        checker.check(resolver.is_ep_enabled(config))
        checker.check(not resolver.use_low_latency(config))
        checker.check(resolver.get_quant_method(config) == cls.QUANT_METHOD)
        checker.check(DeepEPWrapper.supported())

    def _prepare_dispatch_input(
        self,
        a1: torch.Tensor,
        slice_begin: int,
        slice_size: int,
        use_fp8: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if use_fp8:
            raise ValueError("W8A8 INT8 router cannot use FP8 dispatch")
        if a1.dtype != torch.bfloat16:
            raise ValueError(f"W8A8 DeepEP Normal expects BF16 input, got {a1.dtype}")
        tp_expert_a1 = torch.narrow(a1, 0, slice_begin, slice_size)
        return per_token_quant_int8(tp_expert_a1)
