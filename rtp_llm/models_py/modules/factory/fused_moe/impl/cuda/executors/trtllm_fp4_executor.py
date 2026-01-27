from typing import Any, Dict, Optional

import torch

from rtp_llm.models_py.modules.factory.fused_moe.defs.config_adapter import MoEConfigAdapter
from rtp_llm.models_py.modules.factory.fused_moe.defs.fused_moe import (
    ExpertForwardPayload,
    FusedMoeExpertExecutor,
)
from rtp_llm.models_py.modules.factory.fused_moe.defs.quant_config import (
    FusedMoEQuantConfig,
)
from rtp_llm.utils.model_weight import W

from flashinfer import (
    fp4_quantize,
)
from flashinfer.fused_moe import (
    GatedActType,
    trtllm_fp4_block_scale_routed_moe,
)
from flashinfer.utils import (
    device_support_pdl,
)

from rtp_llm.models_py.modules.factory.fused_moe.defs.type import ExecutorType


class TrtllmFp4Executor(FusedMoeExpertExecutor):
    @classmethod
    def executor_type(cls):
        return ExecutorType.TRTLLM_FP4

    @classmethod
    def check_conditions(cls, checker: Any, config: MoEConfigAdapter) -> None:
        from rtp_llm.models_py.modules.factory.fused_moe.utils.config_resolver import (
            MoeConfigResolver,
        )

        resolver = MoeConfigResolver()
        checker.check(resolver.is_bf16(config))
        # Check if quantization is enabled and uses FP4 (uint8 dtype)
        # FP4 weights are packed as uint8, so we check for quant_config with uint8 dtype
        checker.check(resolver.has_quantization(config) and resolver.get_quant_method(config) == "modelopt_fp4")

    def __init__(
        self,
        config: MoEConfigAdapter,
        weights: Dict[str, torch.Tensor],
        quant_config: FusedMoEQuantConfig,
    ):
        super().__init__(config, quant_config, weights)

        self.w1 = weights.get(W.moe_w1, None)
        self.w2 = weights.get(W.moe_w2, None)
        self.w1_scale = weights.get(W.moe_s1, None)
        self.w2_scale = weights.get(W.moe_s2, None)

        w13_input_scale = weights.get(W.moe_w1_i_s, None)
        w13_weight_scale_2 = weights.get(W.moe_w1_s2, None)
        w2_input_scale = weights.get(W.moe_w2_i_s, None)
        w2_weight_scale_2 = weights.get(W.moe_w2_s2, None)

        assert self.w1 is not None
        assert self.w2 is not None
        assert self.w1_scale is not None
        assert self.w2_scale is not None
        assert w13_input_scale is not None
        assert w13_weight_scale_2 is not None
        assert w2_input_scale is not None
        assert w2_weight_scale_2 is not None

        self.expert_x_scale = 1 / w13_input_scale
        self.g1_alphas = w13_input_scale * w13_weight_scale_2
        self.g2_alphas = w2_input_scale * w2_weight_scale_2
        self.g1_scale_c = self.g1_alphas / w2_input_scale

        self.global_num_experts = config.expert_num
        self._enable_pdl = device_support_pdl(self.w1.device)

    @property
    def local_num_experts(self) -> int:
        assert self.w1 is not None
        return self.w1.size(0)
    @property
    def intermediate_size(self) -> int:
        assert self.w1 is not None
        return int(self.w1.size(1) / 2)
    @property
    def hidden_size(self) -> int:
        assert self.w2 is not None
        return self.w2.size(-2)

    def execute(
        self,
        payload: ExpertForwardPayload,
        activation: str,
        expert_map: Optional[torch.Tensor],
        a2_scale: Optional[torch.Tensor],
        apply_router_weight_on_input: bool,
        extra_expert_args: Optional[dict[str, Any]],
    ) -> torch.Tensor:
        topk_ids = payload.expert_topk_ids
        topk_weights = payload.expert_topk_weights

        topk = topk_ids.size(-1)

        act_type_map = {
            "silu": GatedActType.SwiGlu.value,
            "swiglu": GatedActType.SwiGlu.value,
            "geglu": GatedActType.GeGlu.value,
            "siglu": GatedActType.SwiGlu.value,
        }
        gated_act_type = act_type_map[activation.lower()]

        packed_tensor = (topk_ids.to(torch.int32) << 16) | topk_weights.to(
            torch.bfloat16
        ).view(torch.int16)

        if payload.expert_x.dtype is torch.bfloat16:
            hidden_states, hidden_states_scale = fp4_quantize(
                payload.expert_x, self.expert_x_scale, is_sf_swizzled_layout=False)
        else:
            hidden_states, hidden_states_scale = payload.expert_x, payload.expert_x_scale
            assert hidden_states.dtype is torch.uint8, f"hidden_states: {hidden_states.dtype}"
            assert hidden_states_scale is not None, f"hidden_states_scale: {hidden_states_scale}"
            assert hidden_states_scale.dtype is torch.uint8, f"hidden_states_scale: {hidden_states_scale.dtype}"
            assert hidden_states.shape[-1] == hidden_states_scale.shape[-1] * 8, (
                f"hidden_states: {hidden_states.shape}"
                f"hidden_states_scale: {hidden_states_scale.shape}"
            )

        output = trtllm_fp4_block_scale_routed_moe(
            topk_ids=packed_tensor, # topk_ids
            routing_bias=None,  # routing_bias
            hidden_states=hidden_states, # hidden_states
            hidden_states_scale=hidden_states_scale.view(torch.float8_e4m3fn), # hidden_states_scale
            gemm1_weights=self.w1, # gemm1_weights
            gemm1_weights_scale=self.w1_scale.view(torch.float8_e4m3fn), # gemm1_weights_scale
            gemm1_bias=None,  # gemm1_bias
            gemm1_alpha=None,  # gemm1_alpha
            gemm1_beta=None,  # gemm1_beta
            gemm1_clamp_limit=None,  # gemm1_clamp_limit
            gemm2_weights=self.w2, # gemm2_weights
            gemm2_weights_scale=self.w2_scale.view(torch.float8_e4m3fn), # gemm2_weights_scale
            gemm2_bias=None,  # gemm2_bias
            output1_scale_scalar=self.g1_scale_c, # output1_scale_scalar
            output1_scale_gate_scalar=self.g1_alphas, # output1_scale_gate_scalar
            output2_scale_scalar=self.g2_alphas, # output2_scale_scalar
            num_experts=self.global_num_experts, # num_experts
            top_k=topk, # top_k
            n_group=None,  # n_group
            topk_group=None,  # topk_group
            intermediate_size=self.intermediate_size, # intermediate_size
            local_expert_offset=0,  # local_expert_offset
            local_num_experts=self.local_num_experts,  # local_num_experts
            routed_scaling_factor=None,  # routed_scaling_factor
            routing_method_type=1,  # routing_method_type: Renormalize
            do_finalize=True,  # do_finalize
            enable_pdl=self._enable_pdl, # enable_pdl
            gated_act_type=gated_act_type, # gated_act_type
            output=None,  # output (optional inplace)
            # tune_max_num_tokens: int = 8192
        )[0]  # Returns list, get first element

        return output
