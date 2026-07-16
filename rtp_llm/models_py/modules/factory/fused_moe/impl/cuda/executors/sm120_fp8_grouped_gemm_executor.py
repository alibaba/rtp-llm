"""FP8 PER_BLOCK grouped GEMM MoE executor for sm_120 (consumer Blackwell).

Mirrors the SM9x DeepGemmHybridExecutor masked path but replaces DeepGEMM
with a Triton FP8 blockwise batched grouped GEMM kernel.

Execution sequence (6 kernel launches):
  1a/1b. ep_scatter_v2   - scatter [M, K] → [E, alignment, K] fp8 + scale
  2.     fp8_grouped_gemm - gate+up GEMM [E, alignment, K] × [E, N, K]^T → [E, alignment, N] bf16
  3.     silu_mul_masked_fp8_post_quant_fwd - SiLU + requant → [E, alignment, N//2] fp8
  4.     fp8_grouped_gemm - down GEMM [E, alignment, N//2] × [E, K, N//2]^T → [E, alignment, K] bf16
  5.     ep_gather         - weighted-sum reduce → [M, K] bf16
"""

import math
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
from rtp_llm.models_py.triton_kernels.common.activation import (
    silu_mul_masked_fp8_post_quant_fwd,
)
from rtp_llm.models_py.triton_kernels.moe.ep_kernels import ep_gather, ep_scatter_v2
from rtp_llm.models_py.triton_kernels.moe.fp8_grouped_gemm import (
    invoke_sm120_fp8_grouped_gemm,
)
from rtp_llm.models_py.utils.arch import is_sm12x
from rtp_llm.models_py.utils.math import align, ceil_div
from rtp_llm.utils.model_weight import W


def _align_up(n: int, alignment: int) -> int:
    return int(math.ceil(n / alignment)) * alignment


class Sm120Fp8GroupedGemmExecutor(FusedMoeExpertExecutor):
    """Triton FP8 blockwise batched grouped GEMM executor for sm_120.

    Replaces the per-expert CUTLASS loop with a single batched kernel launch
    per GEMM, matching the SM9x DeepGemmHybridExecutor masked path structure.
    """

    BLOCK_SIZE = 128
    EXPERT_ALIGNMENT = 128

    @classmethod
    def executor_type(cls) -> ExecutorType:
        return ExecutorType.SM120_FP8_GROUPED

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

        self.w13_weight = weights[W.moe_w1]  # [E, N, K]
        self.w2_weight = weights[W.moe_w2]  # [E, K, N//2]
        self.w13_scale = weights[W.moe_s1]  # [E, N//128, K//128]
        self.w2_scale = weights[W.moe_s2]  # [E, K//128, (N//2)//128]

        self.E, self.N, self.K = self.w13_weight.size()
        assert self.N % 2 == 0
        assert self.w2_weight.size(0) == self.E
        assert self.w2_weight.size(1) == self.K
        assert self.w2_weight.size(2) == self.N // 2

        self.inter_size = self.N // 2

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
        assert payload.expert_tokens_meta is not None
        assert payload.expert_tokens_meta.expert_num_tokens is not None

        hidden_states_fp8 = payload.expert_x  # [M, K] fp8
        hidden_states_scale = payload.expert_x_scale  # [M, K//128] float32
        topk_ids = payload.expert_topk_ids  # [M, top_k]
        topk_weights = payload.expert_topk_weights  # [M, top_k]
        num_recv_tokens_per_expert = (
            payload.expert_tokens_meta.expert_num_tokens
        )  # [E] int32

        token_num = hidden_states_fp8.shape[0]
        device = hidden_states_fp8.device
        hidden_states_fp8_shape = hidden_states_fp8.shape

        max_token_num = token_num * self.top_k
        token_num_mean_per_expert = ceil_div(
            max_token_num, self.num_experts_per_partition
        )
        alignment = align(token_num, self.EXPERT_ALIGNMENT)
        expected_m = min(alignment, token_num_mean_per_expert)

        E = self.num_experts_per_partition

        # Step 1: Scatter [M, K] fp8 → [E, alignment, K] fp8 + [E, alignment, K//128] float32 scale
        input_packed = torch.empty(
            (E * alignment, self.K),
            device=device,
            dtype=hidden_states_fp8.dtype,
        )
        input_sf_packed = torch.empty(
            (E, alignment, self.K // self.BLOCK_SIZE),
            device=device,
            dtype=torch.float32,
        )
        output_index = torch.empty_like(topk_ids)
        expert_start_loc = torch.empty_like(num_recv_tokens_per_expert)

        ep_scatter_v2(
            hidden_states_fp8,
            hidden_states_scale,
            topk_ids,
            alignment,
            expert_start_loc,
            input_packed,
            input_sf_packed,
            output_index,
            scale_ue8m0=False,
        )
        input_packed_3d = input_packed.view(E, alignment, self.K)

        # Step 2: Gate+Up GEMM: [E, alignment, K] × [E, N, K]^T → [E, alignment, N] bf16
        # w13_weight: [E, N, K], stored as [E, N, K], transposed B in GEMM notation
        upgate_output = torch.empty(
            (E, alignment, self.N),
            device=device,
            dtype=torch.bfloat16,
        )
        invoke_sm120_fp8_grouped_gemm(
            A=input_packed_3d,
            A_sf=input_sf_packed,
            B=self.w13_weight,
            B_sf=self.w13_scale,
            expert_num_tokens=num_recv_tokens_per_expert,
            C=upgate_output,
        )
        del input_packed, input_packed_3d, input_sf_packed

        # Step 3: SiLU + FP8 requant: [E, alignment, N] → [E, alignment, N//2] fp8 + scale
        down_input = torch.empty(
            (E, alignment, self.inter_size),
            device=device,
            dtype=torch.float8_e4m3fn,
        )
        down_input_scale = torch.empty(
            (E, alignment, self.inter_size // self.BLOCK_SIZE),
            device=device,
            dtype=torch.float32,
        )
        silu_mul_masked_fp8_post_quant_fwd(
            input=upgate_output,
            output=down_input,
            output_scale=down_input_scale,
            quant_group_size=self.BLOCK_SIZE,
            masked_m=num_recv_tokens_per_expert,
            expected_m=expected_m,
            scale_ue8m0=False,
        )
        del upgate_output

        # Step 4: Down GEMM: [E, alignment, N//2] × [E, K, N//2]^T → [E, alignment, K] bf16
        # w2_weight: [E, K, N//2], transposed B gives [E, N//2, K], so result is [E, alignment, K]
        down_output = torch.empty(
            (E, alignment, self.K),
            device=device,
            dtype=torch.bfloat16,
        )
        invoke_sm120_fp8_grouped_gemm(
            A=down_input,
            A_sf=down_input_scale,
            B=self.w2_weight,
            B_sf=self.w2_scale,
            expert_num_tokens=num_recv_tokens_per_expert,
            C=down_output,
        )
        del down_input, down_input_scale

        # Step 5: Gather + weighted sum: [E*alignment, K] → [M, K] bf16
        gather_out = torch.empty(
            hidden_states_fp8_shape,
            device=device,
            dtype=torch.bfloat16,
        )
        ep_gather(
            down_output.view(E * alignment, self.K),
            topk_ids,
            topk_weights,
            output_index,
            gather_out,
        )

        return CombineForwardPayload(fused_expert_output=gather_out)
