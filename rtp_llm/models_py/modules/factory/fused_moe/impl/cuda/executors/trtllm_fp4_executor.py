from typing import Any, Dict, Optional
from functools import partial

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

def prepare_static_weights_for_trtllm_fp4_moe(
    # args_dequant,
    # args,
    gemm1_weights,
    gemm2_weights,
    gemm1_scales_linear_fp4_bytes,
    gemm2_scales_linear_fp4_bytes,
    hidden_size,
    intermediate_size,
    num_experts,
):
    from flashinfer import nvfp4_block_scale_interleave
    from flashinfer.fused_moe.core import (
        _maybe_get_cached_w3_w1_permute_indices,
        get_w2_permute_indices_with_cache,
    )

    _cache_permute_indices: dict[torch.Size, torch.Tensor] = {}
    """Prepare quantized weights for kernel (done offline with weights)."""
    epilogue_tile_m = 128  # FIXME: this depends on the kernel internals

    # Convert quantized weights to proper formats
    gemm1_weights_fp4 = gemm1_weights.view(torch.float8_e4m3fn).reshape(
        num_experts, 2 * intermediate_size, hidden_size // 2
    )  # packed fp4
    gemm1_scales_linear_fp4 = gemm1_scales_linear_fp4_bytes.view(
        torch.float8_e4m3fn
    ).reshape(
        num_experts, 2 * intermediate_size, hidden_size // 16
    )  # fp8 scaling factors

    gemm2_weights_fp4 = gemm2_weights.view(torch.float8_e4m3fn).reshape(
        num_experts, hidden_size, intermediate_size // 2
    )  # packed fp4
    gemm2_scales_linear_fp4 = gemm2_scales_linear_fp4_bytes.view(
        torch.float8_e4m3fn
    ).reshape(num_experts, hidden_size, intermediate_size // 16)  # fp8 scaling factors

    gemm1_weights_fp4_shuffled = []
    gemm1_scales_fp4_shuffled = []
    gemm2_weights_fp4_shuffled = []
    gemm2_scales_fp4_shuffled = []
    for i in range(num_experts):
        # Calculate the permute indices for the following:
        # 1. Reorder rows of W1 and scales for fused gated activation
        # 2. Shuffle weights and scaling factors for transposed mma output
        # for both w3_w1 and w2 weights and scale factors
        permute_indices = _maybe_get_cached_w3_w1_permute_indices(
            _cache_permute_indices,
            gemm1_weights_fp4[i].view(torch.uint8),
            epilogue_tile_m,
        )
        gemm1_weights_fp4_shuffled.append(
            gemm1_weights_fp4[i]
            .view(torch.uint8)[permute_indices.to(gemm1_weights_fp4.device)]
            .contiguous()
        )

        permute_sf_indices = _maybe_get_cached_w3_w1_permute_indices(
            _cache_permute_indices,
            gemm1_scales_linear_fp4[i].view(torch.uint8),
            epilogue_tile_m,
            num_elts_per_sf=16,
        )
        gemm1_scales_fp4_shuffled.append(
            nvfp4_block_scale_interleave(
                gemm1_scales_linear_fp4[i]
                .view(torch.uint8)[
                    permute_sf_indices.to(gemm1_scales_linear_fp4.device)
                ]
                .contiguous()
            )
        )

        permute_indices = get_w2_permute_indices_with_cache(
            _cache_permute_indices,
            gemm2_weights_fp4[i].view(torch.uint8),
            epilogue_tile_m,
        )
        gemm2_weights_fp4_shuffled.append(
            gemm2_weights_fp4[i]
            .view(torch.uint8)[permute_indices.to(gemm2_weights_fp4.device)]
            .contiguous()
        )

        permute_sf_indices = get_w2_permute_indices_with_cache(
            _cache_permute_indices,
            gemm2_scales_linear_fp4[i].view(torch.uint8),
            epilogue_tile_m,
            num_elts_per_sf=16,
        )
        gemm2_scales_fp4_shuffled.append(
            nvfp4_block_scale_interleave(
                gemm2_scales_linear_fp4[i]
                .view(torch.uint8)[
                    permute_sf_indices.to(gemm2_scales_linear_fp4.device)
                ]
                .contiguous()
            )
        )

    # Stack weights for all experts
    gemm1_weights_fp4_shuffled = torch.stack(gemm1_weights_fp4_shuffled)
    gemm1_scales_fp4_shuffled = (
        torch.stack(gemm1_scales_fp4_shuffled)
        .view(torch.float8_e4m3fn)
        .reshape(num_experts, 2 * intermediate_size, hidden_size // 16)
    )

    gemm2_weights_fp4_shuffled = torch.stack(gemm2_weights_fp4_shuffled)
    gemm2_scales_fp4_shuffled = (
        torch.stack(gemm2_scales_fp4_shuffled)
        .view(torch.float8_e4m3fn)
        .reshape(num_experts, hidden_size, intermediate_size // 16)
    )
    return (
        gemm1_weights_fp4_shuffled,
        gemm1_scales_fp4_shuffled,
        gemm2_weights_fp4_shuffled,
        gemm2_scales_fp4_shuffled,
    )

class TrtllmFp4Executor(FusedMoeExpertExecutor):
    @classmethod
    def executor_type(cls):
        return ExecutorType.TRTLLM_FP4

    @classmethod
    def check_conditions(cls, checker: Any, config: MoEConfigAdapter) -> None:
        pass

    def __init__(
        self,
        config: MoEConfigAdapter,
        weights: Dict[str, torch.Tensor],
        quant_config: FusedMoEQuantConfig,
    ):
        super().__init__(quant_config=quant_config)

        self.w1 = weights.get(W.moe_w1, None)
        self.w2 = weights.get(W.moe_w2, None)
        self.w1_scale = weights.get(W.moe_s1, None)
        self.w2_scale = weights.get(W.moe_s2, None)
        (
            self.w1,
            self.w1_scale,
            self.w2,
            self.w2_scale,
        ) = prepare_static_weights_for_trtllm_fp4_moe(
            self.w1,
            self.w2,
            self.w1_scale,
            self.w2_scale,
            self.hidden_size,
            self.intermediate_size,
            self.local_num_experts,
        )
        w13_input_scale = weights.get("w13_input_scale", None)
        w13_weight_scale_2 = weights.get("w13_weight_scale_2", None)
        w2_input_scale = weights.get("w2_input_scale", None)
        w2_weight_scale_2 = weights.get("w2_weight_scale_2", None)

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
        }
        gated_act_type = act_type_map[activation.lower()]

        packed_tensor = (topk_ids.to(torch.int32) << 16) | topk_weights.to(
            torch.bfloat16
        ).view(torch.int16)

        if payload.expert_x.dtype is torch.bfloat16:
            hidden_states, hidden_states_scale = fp4_quantize(
                payload.expert_x, self.expert_x_scale, is_sf_swizzled_layout=False)
        else:
            assert payload.expert_x.dtype is torch.uint8
            assert payload.expert_x_scale is not None
            hidden_states, hidden_states_scale = payload.expert_x, payload.expert_x_scale

        output = trtllm_fp4_block_scale_routed_moe(
            packed_tensor,
            None,  # routing_bias
            hidden_states,
            hidden_states_scale.view(torch.float8_e4m3fn),
            self.w1,
            self.w1_scale.view(torch.float8_e4m3fn),
            None,  # w13_bias
            None,  # gemm1_alpha
            None,  # gemm1_beta
            None,  # gemm1_clamp_limit
            self.w2,
            self.w2_scale.view(torch.float8_e4m3fn),
            None,  # w2_bias
            self.g1_scale_c,
            self.g1_alphas,
            self.g2_alphas,
            self.global_num_experts,
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
