"""DCU fused MoE executor using vllm-dcu fused_experts."""

from typing import Any, Dict, Optional

import torch
import aiter
from aiter.moe import get_aiter_moe_config, aiter_moe, MoeQuantType
from aiter.ops.triton.fused_moe import fused_experts_impl

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
from rtp_llm.models_py.modules.factory.fused_moe.impl.dcu.marlin_w16a16_pack import shapes_match_marlin_packed
from rtp_llm.utils.model_weight import W


def _moe_activation_type(activation: str) -> aiter.ActivationType:
    if activation in ("silu", "SiGLU"):
        return "silu"
    return "gelu"


class DcuExpertsBf16(FusedMoeExpertExecutor):
    """DCU BF16 (no quantization) MoE expert executor using vllm-dcu fused_experts."""

    @classmethod
    def executor_type(cls):
        return ExecutorType.FUSED_MOE

    @classmethod
    def check_conditions(cls, checker: Any, config: Any) -> None:
        resolver = MoeConfigResolver()
        quant_method = resolver.get_quant_method(config)
        checker.check(quant_method is None)

    @property
    def topk_ids_dtype(self) -> torch.dtype:
        # vllm-dcu moe_align_block_size uses int32 internally
        return torch.int32

    def __init__(
        self,
        config: MoEConfigAdapter,
        quant_config: FusedMoEQuantConfig,
        weights: Dict[str, torch.Tensor],
    ):
        super().__init__(config, quant_config, weights)
        self.num_experts = config.expert_num
        # print(f"======={config.__dict__=}")
        # print(f"======={config.model_config.activation_type=}, {config.model_config.inter_size=}, {config.model_config.moe_inter_size=}, {config.hidden_size=}")
        self.ep_size = config.ep_size
        self.ep_rank = config.ep_rank
        self.w1 = weights[W.moe_w1]
        self.w2 = weights[W.moe_w2]
        self._marlin_packed = shapes_match_marlin_packed(self.w1, self.w2)
        # Hidden size K: packed layout stores w1 as [E, K/16, 2N*16].
        self._hidden_size = (
            self.w1.size(1) * 16 if self._marlin_packed else self.w1.size(2)
        )
        
        # ==================== 使用aiter moe ======================
        # print(f"w1={self.w1[-1,:10]},w2={self.w2[-1,:10]}")
        #_, self.aiter_moe_config = get_aiter_moe_config(
        #    M=64,
        #    E=config.expert_num,
        #    N1=2 * config.model_config.moe_inter_size,
        #    N2=config.hidden_size,
        #    K=config.hidden_size,
        #    top_k=config.moe_k,
        #    block_size=0,
        #    dtype=torch.bfloat16,
        #    quant_type=MoeQuantType.W16A16,
        #    activation=_moe_activation_type(config.model_config.activation_type),
        #)

    @property
    def local_num_experts(self) -> int:
        return self.w1.size(0)

    def execute(
        self,
        payload: ExpertForwardPayload,
        activation: str,
        expert_map: Optional[torch.Tensor],
        a2_scale: Optional[torch.Tensor],
        apply_router_weight_on_input: bool,
        extra_expert_args: Optional[dict[str, Any]],
    ) -> CombineForwardPayload:
        assert payload.expert_x is not None, "expert_x is None"
        assert payload.expert_x.size(-1) == self._hidden_size, (
            f"Hidden size mismatch {payload.expert_x.size(-1)} != {self._hidden_size}"
        )
        assert payload.expert_x.is_contiguous(), "Hidden_states must be contiguous"
        assert self.w1.stride(-1) == 1, "Stride of last dimension must be 1"
        assert self.w2.stride(-1) == 1, "Stride of last dimension must be 1"
        assert payload.expert_tokens_meta is not None

        topk_ids = payload.expert_topk_ids
        topk_weights = payload.expert_topk_weights
        assert topk_ids is not None
        assert topk_weights is not None

        assert self.w1.size(0) == self.local_num_experts
        assert self.w2.size(0) == self.local_num_experts

        hidden_states = payload.expert_x

        if apply_router_weight_on_input:
            assert (
                topk_weights.dim() == 2
            ), "`topk_weights` should be in shape (num_tokens, topk)"
            _, topk = topk_weights.shape
            assert (
                topk == 1
            ), "Only support topk=1 when `apply_router_weight_on_input` is True"
            hidden_states = hidden_states * topk_weights.to(hidden_states.dtype)
            topk_weights = torch.ones_like(topk_weights, dtype=torch.float32)
        
        # ===================== 使用aiter moe ============================
        #print(f"input hidden={hidden_states[-1,:10]}\n{topk_ids[-1]=}")
        #output = aiter_moe(
        #    hidden_states=hidden_states,
        #    w1=self.w1,
        #    w2=self.w2,
        #    topk_weights=topk_weights,
        #    topk_ids=topk_ids,
        #    moe_config=self.aiter_moe_config,
        #    inplace=False,
        #    activation=_moe_activation_type(activation),
        #    global_num_experts=self.num_experts,
        #    expert_map=expert_map,
        #)
        #print(f"output_hidden={output[-1,:10]}")
        
        # ==================== 使用aiter triton ==========================
        # print(f"{hidden_states[-1:10]=}, {self.w1[-1,:32,-1]=}, {self.w1[-1,-32:,-1]=}, {self.w2[-1,-1,:64]=}")
        output = fused_experts_impl(
            hidden_states=hidden_states,
            w1=self.w1,
            w2=self.w2,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            global_num_experts=self.num_experts
        )
        return CombineForwardPayload(fused_expert_output=output)
