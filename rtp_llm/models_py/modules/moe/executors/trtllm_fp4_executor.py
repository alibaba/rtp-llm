from typing import Any, Dict, Optional

import torch

from rtp_llm.config.gpt_init_model_parameters import GptInitModelParameters
from rtp_llm.models_py.modules.moe.fused_moe import (
    ExpertForwardPayload,
    FusedMoeExpertExecutor,
)
# from rtp_llm.models_py.modules.factory.fused_moe.quant_config import FusedMoEQuantConfig
from rtp_llm.models_py.modules.moe.utils import FusedMoEQuantConfig
# from rtp_llm.models_py.modules.factory.fused_moe.type import ExecutorType
from rtp_llm.async_decoder_engine.engine_creator import ExecutorType
from rtp_llm.utils.model_weight import W

# Try to import trtllm_fp4_block_scale_routed_moe from flashinfer
from flashinfer import (
    fp4_quantize,
)
from flashinfer.fused_moe import (
    GatedActType,
    trtllm_fp4_block_scale_routed_moe,
)
from flashinfer.utils import device_support_pdl

# NVFP4 constants
FLOAT8_E4M3_MAX = 448.0
FLOAT4_E2M1_MAX = 6.0
NVFP4_BLOCK_SIZE = 16


class TrtllmFp4Executor(FusedMoeExpertExecutor):
    def __init__(
        self,
        config: GptInitModelParameters,
        weights: Dict[str, torch.Tensor],
        quant_config: FusedMoEQuantConfig,
    ):
        super().__init__(quant_config=quant_config)

        self.w1 = weights.get(W.moe_w1, None)
        self.w2 = weights.get(W.moe_w2, None)
        self.w1_scale = weights.get(W.moe_s1, None)
        self.w2_scale = weights.get(W.moe_s2, None)

        self.output1_scale_scalar = weights.get("output1_scale_scalar", None)
        self.output1_scale_gate_scalar = weights.get("output1_scale_gate_scalar", None)
        self.output2_scale_scalar = weights.get("output2_scale_scalar", None)

        assert self.output1_scale_scalar is not None
        assert self.output1_scale_gate_scalar is not None
        assert self.output2_scale_scalar is not None
        assert self.w1 is not None and self.w2 is not None
        assert self.w1_scale is not None and self.w2_scale is not None
        
        self._enable_pdl = device_support_pdl(self.w1.device)

    @property
    def local_num_experts(self) -> int:
        assert self.w1 is not None
        return self.w1.size(0)
    @property
    def intermediate_size(self) -> int:
        assert self.w1 is not None
        return int(self.w1.size(1) / 2)

    def execute(
        self,
        payload: ExpertForwardPayload,
        activation: str,
        global_num_experts: int,
        expert_map: Optional[torch.Tensor],
        a2_scale: Optional[torch.Tensor],
        apply_router_weight_on_input: bool,
        extra_expert_args: Optional[dict[str, Any]],
    ) -> torch.Tensor:
        topk_ids = payload.expert_topk_ids
        topk_weights = payload.expert_topk_weights

        topk = topk_ids.size(-1)

        if activation.lower() == "silu" or activation.lower() == "swiglu":
            gated_act_type = GatedActType.SwiGlu.value
        elif activation.lower() == "geglu":
            gated_act_type = GatedActType.GeGlu.value
        else:
            raise NotImplementedError(f"Activation {activation} not supported")

        packed_tensor = (topk_ids.to(torch.int32) << 16) | topk_weights.to(
            torch.bfloat16
        ).view(torch.int16)

        # Quantize input to FP4
        (hidden_states, hidden_states_scale) = fp4_quantize(
            payload.expert_x,
            payload.expert_x_scale,
            is_sf_swizzled_layout=False,
        )

        # Print all parameters
        def print_param(name, value):
            if isinstance(value, torch.Tensor):
                print(f"{name}: shape={value.shape}, dtype={value.dtype}")
            else:
                print(f"{name}: {value}")

        print("=" * 80)
        print("trtllm_fp4_block_scale_routed_moe parameters:")
        print("=" * 80)
        print_param("packed_tensor", packed_tensor)
        print_param("routing_bias", None)
        print_param("hidden_states", hidden_states)
        print_param("hidden_states_scale", hidden_states_scale)
        print_param("self.w1", self.w1)
        print_param("self.w1_scale", self.w1_scale)
        print_param("w13_bias", None)
        print_param("gemm1_alpha", None)
        print_param("gemm1_beta", None)
        print_param("gemm1_clamp_limit", None)
        print_param("self.w2", self.w2)
        print_param("self.w2_scale", self.w2_scale)
        print_param("w2_bias", None)
        print_param("self.output1_scale_scalar", self.output1_scale_scalar)
        print_param("self.output1_scale_gate_scalar", self.output1_scale_gate_scalar)
        print_param("self.output2_scale_scalar", self.output2_scale_scalar)
        print_param("global_num_experts", global_num_experts)
        print_param("topk", topk)
        print_param("n_group", None)
        print_param("topk_group", None)
        print_param("self.intermediate_size", self.intermediate_size)
        print_param("local_expert_offset", 0)
        print_param("self.local_num_experts", self.local_num_experts)
        print_param("routed_scaling_factor", None)
        print_param("tile_tokens_dim", None)
        print_param("routing_method_type", 1)
        print_param("do_finalize", True)
        print_param("self._enable_pdl", self._enable_pdl)
        print_param("gated_act_type", gated_act_type)
        print_param("output (optional inplace)", None)
        print("=" * 80)

        output = trtllm_fp4_block_scale_routed_moe(
            packed_tensor,
            None,  # routing_bias
            hidden_states,
            hidden_states_scale.view(torch.float8_e4m3fn),
            self.w1,
            self.w1_scale,
            None,  # w13_bias
            None,  # gemm1_alpha
            None,  # gemm1_beta
            None,  # gemm1_clamp_limit
            self.w2,
            self.w2_scale,
            None,  # w2_bias
            self.output1_scale_scalar,
            self.output1_scale_gate_scalar,
            self.output2_scale_scalar,
            global_num_experts,
            topk,
            None,  # n_group
            None,  # topk_group
            self.intermediate_size,
            0,  # local_expert_offset
            self.local_num_experts,  # local_num_experts
            None,  # routed_scaling_factor
            None,  # tile_tokens_dim
            1,  # routing_method_type: Renormalize
            True,  # do_finalize
            self._enable_pdl,
            gated_act_type,
            None,  # output (optional inplace)
        )[0]  # Returns list, get first element

        return output
