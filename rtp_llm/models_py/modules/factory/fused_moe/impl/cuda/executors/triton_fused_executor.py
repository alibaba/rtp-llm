from typing import Any, Dict, Optional

import torch
import triton.language as tl

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
from rtp_llm.models_py.triton_kernels.common.activation import silu_and_mul
from rtp_llm.models_py.triton_kernels.moe.fused_moe_kernel import (
    get_default_config,
    invoke_fused_moe_kernel,
    moe_align_block_size_torch,
)
from rtp_llm.utils.model_weight import W


class TritonFusedMoeExecutor(FusedMoeExpertExecutor):
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

        # w1: [E, 2*inter_size, hidden_size], w2: [E, hidden_size, inter_size]
        self.w1 = weights[W.moe_w1]
        self.w2 = weights[W.moe_w2]
        self.E = self.w1.size(0)
        self.N = self.w1.size(1)  # 2 * inter_size
        self.K = self.w1.size(2)  # hidden_size
        self.inter_size = self.N // 2

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
        hidden_states = payload.expert_x  # [M, K]
        topk_ids = payload.expert_topk_ids  # [M, top_k]
        topk_weights = payload.expert_topk_weights  # [M, top_k]

        assert topk_ids is not None
        assert topk_weights is not None

        if self.ep_size > 1:
            local_ids = topk_ids - self.start_expert_id
            local_mask = (local_ids >= 0) & (local_ids < self.num_experts_per_partition)
            topk_ids = local_ids.clamp(min=0, max=self.num_experts_per_partition - 1)
            topk_weights = topk_weights * local_mask

        M, K = hidden_states.shape
        top_k = topk_ids.size(1)
        device = hidden_states.device
        dtype = hidden_states.dtype
        compute_type = tl.bfloat16 if dtype == torch.bfloat16 else tl.float16

        # Config selection for both GEMMs
        config1 = get_default_config(M, self.E, self.N, K, top_k)
        config2 = get_default_config(M, self.E, K, self.inter_size, top_k)

        # Flat routing tensors for the kernel
        flat_topk_weights = topk_weights.view(-1)  # [M * top_k]
        flat_topk_ids = topk_ids.view(-1)  # [M * top_k]

        # Token-expert alignment: compute once if block sizes match, else twice
        block_m1 = config1["BLOCK_SIZE_M"]
        block_m2 = config2["BLOCK_SIZE_M"]
        sorted_token_ids, expert_ids, num_tokens_post_padded = (
            moe_align_block_size_torch(topk_ids, block_m1, self.E)
        )
        if block_m1 == block_m2:
            sorted_token_ids2, expert_ids2, num_tokens_post_padded2 = (
                sorted_token_ids,
                expert_ids,
                num_tokens_post_padded,
            )
        else:
            sorted_token_ids2, expert_ids2, num_tokens_post_padded2 = (
                moe_align_block_size_torch(topk_ids, block_m2, self.E)
            )

        # GEMM1: hidden_states @ w1.T → intermediate1 [M*top_k, 2*inter]
        intermediate1 = torch.empty(M * top_k, self.N, device=device, dtype=dtype)
        invoke_fused_moe_kernel(
            hidden_states,
            self.w1,
            intermediate1,
            flat_topk_weights,
            flat_topk_ids,
            sorted_token_ids,
            expert_ids,
            num_tokens_post_padded,
            mul_routed_weight=False,
            top_k=top_k,
            config=config1,
            compute_type=compute_type,
        )

        # Activation: silu_and_mul(intermediate1) → intermediate2 [M*top_k, inter]
        intermediate2 = torch.empty(
            M * top_k, self.inter_size, device=device, dtype=dtype
        )
        silu_and_mul(intermediate2, intermediate1)

        # GEMM2: intermediate2 @ w2.T → out [M*top_k, K]
        out = torch.empty(M * top_k, K, device=device, dtype=dtype)
        invoke_fused_moe_kernel(
            intermediate2,
            self.w2,
            out,
            flat_topk_weights,
            flat_topk_ids,
            sorted_token_ids2,
            expert_ids2,
            num_tokens_post_padded2,
            mul_routed_weight=True,
            top_k=1,
            config=config2,
            compute_type=compute_type,
        )

        # Sum reduce over top_k: out is [M*top_k, K], reshape and sum
        output = out.view(M, top_k, K).sum(dim=1)

        return CombineForwardPayload(fused_expert_output=output)
