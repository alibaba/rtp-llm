# Adapt from https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/layers/moe/ep_moe/kernels.py
# but make some modifications for RTP-LLM
# Licensed under the Apache License, Version 2.0
from typing import Any, Dict, Optional

import torch

from rtp_llm.models_py.modules.factory.fused_moe.defs.config_adapter import (
    MoEConfigAdapter,
)
from rtp_llm.models_py.modules.factory.fused_moe.defs.fused_moe import (
    CombineForwardPayload,
    ExpertForwardPayload,
    FusedMoeExpertExecutor,
)
from rtp_llm.models_py.modules.factory.fused_moe.defs.quant_config import (
    FusedMoEQuantConfig,
)
from rtp_llm.models_py.modules.factory.fused_moe.defs.type import ExecutorType
from rtp_llm.models_py.modules.factory.fused_moe.utils.config_resolver import (
    MoeConfigResolver,
)
from rtp_llm.ops.compute_ops import FusedMoEOp
from rtp_llm.utils.model_weight import W


class CppMoeExecutor(FusedMoeExpertExecutor):
    @classmethod
    def executor_type(cls):
        return ExecutorType.FUSED_MOE

    @classmethod
    def check_conditions(cls, checker: Any, config: MoEConfigAdapter) -> None:
        resolver = MoeConfigResolver()
        checker.check(not resolver.has_quantization(config))

    def __init__(
        self,
        config: MoEConfigAdapter,
        quant_config: FusedMoEQuantConfig,
        weights: Dict[str, torch.Tensor],
    ):
        super().__init__(config, quant_config, weights)

        self.ep_size = config.ep_size
        self.ep_rank = config.ep_rank
        self.num_experts = config.expert_num
        assert self.num_experts % self.ep_size == 0
        self.num_experts_per_partition = self.num_experts // self.ep_size
        self.start_expert_id = self.ep_rank * self.num_experts_per_partition
        self.end_expert_id = self.start_expert_id + self.num_experts_per_partition - 1
        self.top_k = config.moe_k
        self.intermediate_size = config.model_config.moe_inter_size
        self.activation = config.activation_type
        self.renormalize = True
        self.use_fp8_w8a8 = True
        self.use_block_quant = True

        # 权重初始化
        self.w13_weight = weights[W.moe_w1]
        self.w2_weight = weights[W.moe_w2]
        self.moe_op = FusedMoEOp(config.model_config, config.parallelism_config)

    @property
    def topk_ids_dtype(self) -> torch.dtype:
        return torch.int32

    def execute(
        self,
        payload: ExpertForwardPayload,
        activation: str,
        expert_map: Optional[torch.Tensor],
        a2_scale: Optional[torch.Tensor],
        apply_router_weight_on_input: bool,
        extra_expert_args: Optional[dict[str, Any]],
    ) -> CombineForwardPayload:
        output = torch.zeros_like(payload.expert_x)
        assert payload.expert_topk_weights is not None, "expert_topk_weights is None"
        assert payload.expert_topk_ids is not None, "expert_topk_ids is None"
        self.moe_op.forward(
            payload.expert_x,
            self.w13_weight,
            self.w2_weight,
            payload.expert_topk_weights,
            payload.expert_topk_ids.to(torch.int32),
            output,
        )
        return CombineForwardPayload(fused_expert_output=output)
