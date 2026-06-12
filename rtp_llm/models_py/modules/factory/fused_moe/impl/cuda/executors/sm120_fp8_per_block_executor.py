"""FP8 PER_BLOCK MoE executor for sm_120 (consumer Blackwell).

Uses the vLLM-ported cutlass_scaled_mm_blockwise_sm120_fp8 kernel (PR-5)
in a per-expert loop. Functional fallback for sm_120 where DeepGEMM and
rtp_kernel grouped GEMMs are unavailable.
"""

from typing import Any, Dict, Optional

import torch

from rtp_llm.models_py.kernels.cuda.fp8_kernel import sgl_per_token_group_quant_fp8
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
from rtp_llm.models_py.triton_kernels.common.activation import silu_and_mul
from rtp_llm.models_py.utils.arch import get_sm, is_sm12x
from rtp_llm.ops.compute_ops import cutlass_scaled_mm_blockwise_sm120_fp8
from rtp_llm.utils.model_weight import W


class Sm120Fp8PerBlockMoeExecutor(FusedMoeExpertExecutor):
    """Per-expert loop FP8 MoE executor for sm_120.

    Processes each active expert sequentially using the sm_120 CUTLASS
    blockwise FP8 GEMM kernel. Less efficient than grouped GEMM but
    functionally correct on consumer Blackwell GPUs.
    """

    BLOCK_SIZE = 128

    @classmethod
    def executor_type(cls) -> ExecutorType:
        return ExecutorType.DEEPGEMM_CONTINUOUS

    @classmethod
    def check_conditions(cls, checker: Any, config: MoEConfigAdapter) -> None:
        from rtp_llm.models_py.modules.factory.fused_moe.utils.config_resolver import (
            MoeConfigResolver,
        )

        resolver = MoeConfigResolver()
        quant_method = resolver.get_quant_method(config)
        checker.check(quant_method == "FP8_PER_BLOCK")
        checker.check(resolver.is_bf16(config))
        checker.check(is_sm12x())

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
        self.top_k = config.moe_k

        self.w13_weight = weights[W.moe_w1]
        self.w2_weight = weights[W.moe_w2]
        self.w13_scale = weights[W.moe_s1]
        self.w2_scale = weights[W.moe_s2]

        self.E, self.N, self.K = self.w13_weight.size()
        assert self.N % 2 == 0
        assert self.w2_weight.size(0) == self.E
        assert self.w2_weight.size(1) == self.K
        assert self.w2_weight.size(2) == self.N // 2

        self.inter_size = self.N // 2
        self.K_out = self.K

    def execute(
        self,
        payload: ExpertForwardPayload,
        activation: str,
        expert_map: Optional[torch.Tensor],
        a2_scale: Optional[torch.Tensor],
        apply_router_weight_on_input: bool,
        extra_expert_args: Optional[dict[str, Any]],
    ) -> CombineForwardPayload:
        assert payload.expert_x is not None
        assert payload.expert_x_scale is not None
        assert payload.expert_topk_ids is not None
        assert payload.expert_topk_weights is not None

        expert_x = payload.expert_x
        expert_x_scale = payload.expert_x_scale
        topk_ids = payload.expert_topk_ids
        topk_weights = payload.expert_topk_weights

        M = expert_x.shape[0]
        top_k = topk_ids.size(1)
        device = expert_x.device

        flat_topk_ids = topk_ids.view(-1)
        flat_topk_weights = topk_weights.view(-1)

        expanded_x = expert_x.repeat_interleave(top_k, dim=0)
        expanded_scales = expert_x_scale.repeat_interleave(top_k, dim=0)

        per_slot_output = torch.zeros(
            M * top_k, self.K_out, dtype=torch.bfloat16, device=device
        )

        for expert_id in range(self.num_experts_per_partition):
            indices = (flat_topk_ids == expert_id).nonzero(as_tuple=True)[0]
            if indices.numel() == 0:
                continue

            n_tokens = indices.numel()
            tokens_fp8 = expanded_x[indices]
            scales_row = expanded_scales[indices]
            scales_cm = scales_row.t().contiguous()

            inter = torch.empty(n_tokens, self.N, dtype=torch.bfloat16, device=device)
            cutlass_scaled_mm_blockwise_sm120_fp8(
                inter,
                tokens_fp8,
                self.w13_weight[expert_id],
                scales_cm,
                self.w13_scale[expert_id],
            )

            inter_act = torch.empty(
                n_tokens, self.inter_size, dtype=torch.bfloat16, device=device
            )
            silu_and_mul(inter_act, inter)

            inter_fp8, inter_scales = sgl_per_token_group_quant_fp8(
                inter_act,
                group_size=self.BLOCK_SIZE,
                column_major_scales=True,
                scale_tma_aligned=False,
                scale_ue8m0=False,
            )

            expert_out = torch.empty(
                n_tokens, self.K_out, dtype=torch.bfloat16, device=device
            )
            cutlass_scaled_mm_blockwise_sm120_fp8(
                expert_out,
                inter_fp8,
                self.w2_weight[expert_id],
                inter_scales,
                self.w2_scale[expert_id],
            )

            per_slot_output[indices] = expert_out

        per_slot_output = per_slot_output * flat_topk_weights.to(
            torch.bfloat16
        ).unsqueeze(1)
        output = per_slot_output.view(M, top_k, self.K_out).sum(dim=1)

        return CombineForwardPayload(fused_expert_output=output)
