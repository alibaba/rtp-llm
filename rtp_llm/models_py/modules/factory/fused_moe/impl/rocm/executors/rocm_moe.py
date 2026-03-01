from math import prod
from typing import Any, Callable, Dict, Optional
import aiter 

import torch

import rtp_llm.ops.compute_ops as compute_ops
from rtp_llm.models_py.kernels.cuda.fp8_kernel import (
    cutlass_moe_mm_fp8_scaled,
    get_best_config_swap_ab,
)
from rtp_llm.models_py.modules.factory import (
    LinearFactory,
)

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
from rtp_llm.models_py.triton_kernels.common.activation import (
    silu_and_mul,
    silu_mul_fp8_per_token_quant_batched,
)

from rtp_llm.utils.model_weight import W

BLOCK_SIZE_M = 32

class RocmExpertsFp8PerChannel(FusedMoeExpertExecutor):
    @classmethod
    def executor_type(cls):
        return ExecutorType.FUSED_MOE

    @classmethod
    def check_conditions(cls, checker: Any, config: Any) -> None:
        """Check if RocmExpertsFp8PerChannel can handle the configuration"""
        from rtp_llm.models_py.modules.factory.fused_moe.utils.config_resolver import (
            MoeConfigResolver,
        )

        resolver = MoeConfigResolver()
        quant_method = resolver.get_quant_method(config)
        checker.check(
            quant_method is "FP8_PER_CHANNEL_COMPRESSED"
        )

    @property
    def topk_ids_dtype(self) -> torch.dtype:
        return torch.int32

    def __init__(
        self,
        config: MoEConfigAdapter,
        quant_config: FusedMoEQuantConfig,
        weights: Dict[str, torch.Tensor],
    ):
        super().__init__(config, quant_config, weights)

        # Update quant_config with FP8-specific settings
        self.quant_config.quant_dtype = torch.float8_e4m3fnuz
        self.quant_config.per_act_token_quant = True
        self.quant_config.per_out_ch_quant = True
        self.quant_config.block_shape = None

        self.num_experts = config.expert_num
        self.ep_rank = config.ep_rank
        self.ep_size = config.ep_size
        # Extract weights from dictionary
        self.w1 = weights[W.moe_w1]
        self.w2 = weights[W.moe_w2]
        self.w1_scale = weights[W.moe_s1]
        self.w2_scale = weights[W.moe_s2]
        self.a1q_scale = weights.get(W.moe_w1_input_sr, None)
        self.a2_scale = weights.get(W.moe_w2_input_sr, None)

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
        assert payload.expert_x.size(-1) == self.w1.size(
            2
        ), f"Hidden size mismatch {payload.expert_x.size(-1)} != {self.w1.size(2)}"

        assert payload.expert_x.is_contiguous(), "Hidden_states must be contiguous"
        assert self.w1.stride(-1) == 1, "Stride of last dimension must be 1"
        assert self.w2.stride(-1) == 1, "Stride of last dimension must be 1"
        assert payload.expert_tokens_meta is not None

        E = self.local_num_experts
        global_E = self.num_experts
        #print("self.local_num_experts: ", self.local_num_experts)
        #print("self.num_experts: ", self.num_experts)
        # temp fix to reshape experts
        E = global_E
        N = self.w1.size(1)
        assert payload.expert_topk_ids is not None

        assert self.w1.size(0) == E
        assert self.w2.size(0) == E

        topk_ids = payload.expert_topk_ids
        topk_weights = payload.expert_topk_weights
    
        device = topk_ids.device
        M, topk = topk_ids.shape
        model_dim = self.w1.size(2)
        num_token = payload.expert_x.size(0)
        inter_dim = self.w2.size(2)
    
        max_num_tokens_padded = M * topk + global_E * BLOCK_SIZE_M - topk

        max_num_m_blocks = int((max_num_tokens_padded + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M)
        sorted_ids = torch.empty((max_num_tokens_padded,), dtype=torch.int32, device=device)
        sorted_weights = torch.empty(
            (max_num_tokens_padded,), dtype=torch.float32, device=device
        )
        sorted_expert_ids = torch.empty(
            (max_num_m_blocks,), dtype=torch.int32, device=device
        )
        num_valid_ids = torch.empty((2,), dtype=torch.int32, device=device)
        moe_out = torch.empty((num_token, model_dim), dtype=torch.bfloat16, device=device)
        
        aiter.moe_sorting_fwd(
            topk_ids,
            topk_weights,
            sorted_ids,
            sorted_weights,
            sorted_expert_ids,
            num_valid_ids,
            moe_out,
            global_E,
            BLOCK_SIZE_M,
            local_expert_mask=None,
            num_local_tokens=None,
            dispatch_policy=0,
        )
        
        tmp_out = torch.zeros((num_token, topk, inter_dim), dtype=torch.bfloat16, device=device)
        # inp.scalar_type() == torch_fp8) &&
        #      (w1.scalar_type() == torch_fp8) &&
        #      out.scalar_type() == at::ScalarType::BFloat16 &&
        #      quant_type == QuantType::per_Token && 
        #      !do_weight)
        aiter.moe_stage1_g1u1(
            payload.expert_x,
            self.w1,
            self.w2,
            sorted_ids,
            sorted_expert_ids,
            num_valid_ids,
            tmp_out,
            inter_dim,
            "",
            BLOCK_SIZE_M,
            ksplit=0,
            activation=aiter.ActivationType.Silu,
            quant_type=aiter.QuantType.per_Token,
            a1_scale=payload.expert_x_scale,
            w1_scale=self.w1_scale,
            sorted_weights=None,
        )
        a2_scale = torch.empty((num_token, topk, 1), dtype=torch.float32, device=device)
        a2 = torch.empty((num_token, topk, inter_dim), dtype=self.quant_config.quant_dtype, device=device)
        aiter.dynamic_per_token_scaled_quant(a2, tmp_out, a2_scale)
        aiter.ck_moe_stage2(
            a2,
            self.w1,
            self.w2,
            sorted_ids,
            sorted_expert_ids,
            num_valid_ids,
            moe_out,
            topk,
            "",
            self.w2_scale,
            a2_scale,
            BLOCK_SIZE_M,
            sorted_weights,
            aiter.QuantType.per_Token,
            aiter.ActivationType.Silu
        )
        
        return CombineForwardPayload(fused_expert_output=moe_out)
