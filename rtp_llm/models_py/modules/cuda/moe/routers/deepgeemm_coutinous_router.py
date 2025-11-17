from typing import Any, Optional

import torch

from rtp_llm.config.gpt_init_model_parameters import GptInitModelParameters
from rtp_llm.distribute.collective import Group, all_reduce
from rtp_llm.models_py.modules.common.moe.fused_moe import (
    ExpertForwardPayload,
    ExpertTokensMetadata,
    FusedMoeDataRouter,
)
from rtp_llm.models_py.modules.factory.fused_moe.quant_config import FusedMoEQuantConfig
from rtp_llm.models_py.modules.factory.fused_moe.type import RouterType
from rtp_llm.models_py.triton_kernels.moe.ep_kernels import (
    recompute_topk_ids_sum_expert_count,
)
from rtp_llm.ops.compute_ops import trt_fp8_quantize_128


class PureTpRouter(FusedMoeDataRouter):
    @classmethod
    def router_type(cls):
        return RouterType.PURE_TP

    @classmethod
    def check_conditions(cls, checker: Any, config: GptInitModelParameters) -> None:
        """Check if PureTpRouter can handle the configuration"""
        from rtp_llm.models_py.modules.factory.fused_moe.config_resolver import (
            MoeConfigResolver,
        )

        resolver = MoeConfigResolver()
        checker.check(resolver.is_single_gpu(config) or resolver.is_tp_equal_ep(config))

    def __init__(
        self,
        config: GptInitModelParameters,
        use_fp8: bool = True,
        async_mode: bool = False,
        expert_alignment: int = 128,
    ):
        self.config = config
        self.tp_size = config.tp_size
        self.tp_rank = config.tp_rank
        self.dp_size = config.dp_size
        self.dp_rank = config.dp_rank
        self.ep_size = config.ep_size
        self.ep_rank = config.ep_rank
        self.expert_num = config.expert_num
        self.expert_num_per_rank = self.expert_num // self.ep_size
        self.expert_start_id = self.ep_rank * self.expert_num_per_rank
        self.top_k = config.moe_topk_group
        self.use_fp8 = use_fp8
        self.async_mode = async_mode
        self.expert_alignment = expert_alignment
        if self.async_mode:
            raise ValueError("DeepEPNormal not supports async mode now")

    def prepare(
        self,
        a1: torch.Tensor,
        a1_scale: Optional[torch.Tensor],
        a2_scale: Optional[torch.Tensor],
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        quant_config: FusedMoEQuantConfig,
    ) -> ExpertForwardPayload:
        # recompute top_k ids to current expert, mask out of range expert to -1
        if self.use_fp8:
            expert_x, expert_x_scale = trt_fp8_quantize_128(a1, False)
        else:
            expert_x = a1
            expert_x_scale = None
        adjusted_topk_ids, num_recv_tokens_per_expert = (
            recompute_topk_ids_sum_expert_count(
                topk_ids, self.expert_start_id, self.expert_num_per_rank
            )
        )
        return ExpertForwardPayload(
            expert_x,
            None,
            expert_x_scale,
            ExpertTokensMetadata(num_recv_tokens_per_expert, None),
            adjusted_topk_ids,
            topk_weights,
        )

    def finalize(
        self,
        fused_expert_output: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        apply_router_weight_on_input: bool,
        extra_finalize_args: Optional[dict[str, Any]],
    ) -> torch.Tensor:
        if self.tp_size > 1:
            fused_expert_output = all_reduce(fused_expert_output, group=Group.TP)
        return fused_expert_output
