"""DeepGemm BF16 Hybrid Executor for DeepEP Normal mode.

Supports two compute paths selected at runtime by token count:

- Masked  (token_num <= masked_max_token_num, typical for decode):
    ep_scatter_v2_bf16 → deepgemm bf16 masked (gate+up) → silu_mul_masked_bf16
    → deepgemm bf16 masked (down) → ep_gather

- Contiguous (token_num > masked_max_token_num, typical for prefill):
    ep_scatter_bf16 → deepgemm bf16 contiguous (gate+up) → silu_and_mul
    → deepgemm bf16 contiguous (down) → ep_gather

The masked path uses a 3D [E, alignment, K] layout where alignment = align(token_num, 128),
which is memory-efficient only when token_num is small (decode). For large token counts
(prefill), the alignment blows up to token_num, wasting E × token_num × K memory and
launching GEMM tiles over mostly-empty rows.

The contiguous path uses a flat [Σ align(ei, 128), K] layout where each expert contributes
only its actual (padded) tokens, giving ~E× better memory utilization for prefill.
"""

import logging
from typing import Any, Dict, Optional

import torch

from rtp_llm.models_py.kernels.cuda.deepgemm_wrapper import (
    configure_deep_gemm_num_sms,
    m_grouped_bf16_gemm_nt_contiguous,
    m_grouped_bf16_gemm_nt_masked,
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
    silu_mul_masked_bf16_no_post_quant_fwd,
)
from rtp_llm.models_py.triton_kernels.moe.ep_kernels import (
    ep_gather,
    ep_scatter_bf16,
    ep_scatter_v2_bf16,
)
from rtp_llm.models_py.utils.arch import get_num_device_sms, get_sm
from rtp_llm.models_py.utils.math import align, ceil_div
from rtp_llm.models_py.utils.memory import dispose_tensor
from rtp_llm.utils.model_weight import W


logger = logging.getLogger(__name__)


class DeepGemmBf16HybridExecutor(FusedMoeExpertExecutor):
    """Executor for DeepEP Normal bf16 mode using deepgemm grouped GEMM.

    Dispatches between two paths at runtime based on token count:
    - Masked  (token_num <= masked_max_token_num): 3D layout, efficient for decode.
    - Contiguous (token_num > masked_max_token_num): flat layout, efficient for prefill.
    """

    EXPERT_ALIGNMENT = 128
    DEEPGEMM_BLOCK_SHAPE: list[int] = [128, 128]

    @classmethod
    def executor_type(cls) -> ExecutorType:
        # Returns DEEPGEMM_MASKED as the nominal type; consumers use this for
        # logging/registration only — actual dispatch (masked vs contiguous) is
        # done at runtime inside execute() based on token count.
        return ExecutorType.DEEPGEMM_MASKED

    @classmethod
    def check_conditions(cls, checker: Any, config: MoEConfigAdapter) -> None:
        from rtp_llm.models_py.kernels.cuda.deepgemm_wrapper import has_deep_gemm
        from rtp_llm.models_py.modules.factory.fused_moe.utils.config_resolver import (
            MoeConfigResolver,
        )

        resolver = MoeConfigResolver()
        quant_method = resolver.get_quant_method(config)
        checker.check(quant_method is None)
        checker.check(resolver.is_bf16(config))
        checker.check(has_deep_gemm())
        checker.check(get_sm()[0] >= 9)
        checker.check(not config.enable_cuda_graph)

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
        self.activation = config.activation_type
        self.masked_max_token_num = config.masked_max_token_num

        # Weight initialization (bf16, no quantization)
        self.w1 = weights[W.moe_w1]
        self.w2 = weights[W.moe_w2]

        self.num_local_experts, self.intermediate_size, self.hidden_size = self.w1.size()
        assert self.intermediate_size % 2 == 0
        assert self.w2.size(0) == self.num_local_experts
        assert self.w2.size(1) == self.hidden_size
        assert self.w2.size(2) == self.intermediate_size // 2

        self.num_gemm_sms = get_num_device_sms()

    def _to_local_expert_ids(self, topk_idx: torch.Tensor) -> torch.Tensor:
        """Convert global expert IDs to partition-local IDs (0-based), -1 for out-of-partition."""
        local = topk_idx - self.start_expert_id
        return torch.where(
            (local >= 0) & (local < self.num_experts_per_partition),
            local,
            torch.tensor(-1, device=local.device, dtype=local.dtype),
        )

    def execute(
        self,
        payload: ExpertForwardPayload,
        activation: str,
        expert_map: Optional[torch.Tensor],
        a2_scale: Optional[torch.Tensor],
        apply_router_weight_on_input: bool,
        extra_expert_args: Optional[dict[str, Any]],
    ) -> CombineForwardPayload:
        assert payload.expert_x is not None, "hidden_states is not initialized"
        assert payload.expert_topk_ids is not None, "expert_topk_ids is not initialized"
        assert payload.expert_topk_weights is not None, "expert_topk_weights is not initialized"
        assert payload.expert_tokens_meta is not None, "expert_tokens_meta is not initialized"
        assert payload.expert_tokens_meta.expert_num_tokens is not None
        # Router weight is always applied at the gather stage (ep_gather). DeepEP Normal
        # callers must pass apply_router_weight_on_input=False.
        assert not apply_router_weight_on_input, (
            "DeepGemmBf16HybridExecutor applies router weight at gather; "
            "apply_router_weight_on_input=True is not supported."
        )

        token_num = payload.expert_x.shape[0]
        if token_num <= self.masked_max_token_num:
            return self.execute_masked(
                payload, activation, expert_map, a2_scale, apply_router_weight_on_input, extra_expert_args
            )
        else:
            return self.execute_contiguous(
                payload, activation, expert_map, a2_scale, apply_router_weight_on_input, extra_expert_args
            )

    def execute_masked(
        self,
        payload: ExpertForwardPayload,
        activation: str,
        expert_map: Optional[torch.Tensor],
        a2_scale: Optional[torch.Tensor],
        apply_router_weight_on_input: bool,
        extra_expert_args: Optional[dict[str, Any]],
    ) -> CombineForwardPayload:
        with configure_deep_gemm_num_sms(self.num_gemm_sms):
            hidden_states = payload.expert_x
            topk_idx = self._to_local_expert_ids(payload.expert_topk_ids)
            topk_weights = payload.expert_topk_weights
            num_recv_tokens_per_expert = payload.expert_tokens_meta.expert_num_tokens

            token_num = hidden_states.shape[0]
            num_experts = num_recv_tokens_per_expert.shape[0]
            max_token_num = token_num * self.top_k
            token_num_mean_per_expert = ceil_div(max_token_num, num_experts)
            alignment = align(token_num, self.EXPERT_ALIGNMENT)
            expected_m = min(alignment, token_num_mean_per_expert)

            device = hidden_states.device
            hidden_states_shape = hidden_states.shape

            # Step 1: Scatter flat [M, K] → 3D [E, alignment, K]
            input_tensor = torch.empty(
                (self.num_experts_per_partition, alignment, self.hidden_size),
                device=device,
                dtype=torch.bfloat16,
            )
            output_index = torch.empty_like(topk_idx)
            expert_start_loc = torch.empty_like(num_recv_tokens_per_expert)

            ep_scatter_v2_bf16(
                hidden_states,
                topk_idx,
                alignment,
                expert_start_loc,
                input_tensor.view(self.num_experts_per_partition * alignment, self.hidden_size),
                output_index,
            )
            dispose_tensor(hidden_states)

            # Step 2: Gate and Up GEMM (deepgemm bf16 masked)
            upgate_output = torch.empty(
                (self.num_experts_per_partition, alignment, self.intermediate_size),
                device=device,
                dtype=torch.bfloat16,
            )
            m_grouped_bf16_gemm_nt_masked(
                input_tensor,
                self.w1,
                upgate_output,
                num_recv_tokens_per_expert,
                expected_m,
            )
            dispose_tensor(input_tensor)

            # Step 3: SiLU Activation (masked bf16)
            down_input = torch.empty(
                (self.num_experts_per_partition, alignment, self.intermediate_size // 2),
                device=device,
                dtype=torch.bfloat16,
            )
            silu_mul_masked_bf16_no_post_quant_fwd(
                input=upgate_output,
                output=down_input,
                masked_m=num_recv_tokens_per_expert,
                expected_m=expected_m,
                group_size=self.DEEPGEMM_BLOCK_SHAPE[0],
            )
            dispose_tensor(upgate_output)

            # Step 4: Down GEMM (deepgemm bf16 masked)
            down_output = torch.empty(
                (self.num_experts_per_partition, alignment, self.hidden_size),
                device=device,
                dtype=torch.bfloat16,
            )
            m_grouped_bf16_gemm_nt_masked(
                down_input,
                self.w2,
                down_output,
                num_recv_tokens_per_expert,
                expected_m,
            )
            dispose_tensor(down_input)

            # Step 5: Gather 3D → flat, with router weight multiplication
            gather_out = torch.empty(hidden_states_shape, device=device, dtype=torch.bfloat16)
            ep_gather(
                down_output.view(self.num_experts_per_partition * alignment, self.hidden_size),
                topk_idx,
                topk_weights,
                output_index,
                gather_out,
            )
            dispose_tensor(down_output)

            return CombineForwardPayload(fused_expert_output=gather_out)

    def execute_contiguous(
        self,
        payload: ExpertForwardPayload,
        activation: str,
        expert_map: Optional[torch.Tensor],
        a2_scale: Optional[torch.Tensor],
        apply_router_weight_on_input: bool,
        extra_expert_args: Optional[dict[str, Any]],
    ) -> CombineForwardPayload:
        """Large-token-count path: flat [all_tokens, K] layout, efficient for prefill.

        Each expert's tokens are packed contiguously (padded to EXPERT_ALIGNMENT),
        so GEMM operates on dense data without wasting tiles over empty rows.
        """
        hidden_states = payload.expert_x
        topk_idx = self._to_local_expert_ids(payload.expert_topk_ids)
        topk_weights = payload.expert_topk_weights

        # Get per-expert token counts as a Python list (needed for alignment arithmetic)
        if payload.expert_tokens_meta.expert_num_tokens_cpu is not None:
            tokens_per_expert_list = payload.expert_tokens_meta.expert_num_tokens_cpu
        else:
            tokens_per_expert_list = payload.expert_tokens_meta.expert_num_tokens.cpu().tolist()
        if isinstance(tokens_per_expert_list, torch.Tensor):
            tokens_per_expert_list = tokens_per_expert_list.tolist()

        # Align each expert's token count to EXPERT_ALIGNMENT (128)
        aligned_tokens = [align(x, self.EXPERT_ALIGNMENT) for x in tokens_per_expert_list]
        all_tokens = sum(aligned_tokens)

        device = hidden_states.device
        hidden_states_shape = hidden_states.shape

        if all_tokens <= 0:
            return CombineForwardPayload(
                fused_expert_output=torch.zeros(hidden_states_shape, device=device, dtype=torch.bfloat16)
            )

        num_recv_tokens_per_expert_gpu = torch.tensor(
            aligned_tokens, dtype=torch.int32, pin_memory=True, device="cpu"
        ).cuda(non_blocking=True)
        expert_start_loc = torch.empty_like(num_recv_tokens_per_expert_gpu)
        m_indices = torch.empty(all_tokens, device=device, dtype=torch.int32)
        output_index = torch.empty_like(topk_idx)

        # Step 1: Scatter flat [M, K] → expert-sorted flat [all_tokens, K]
        input_tensor = torch.empty((all_tokens, self.hidden_size), device=device, dtype=torch.bfloat16)
        ep_scatter_bf16(
            hidden_states,
            topk_idx,
            num_recv_tokens_per_expert_gpu,
            expert_start_loc,
            input_tensor,
            m_indices,
            output_index,
        )
        # ep_scatter_bf16 fills m_indices for occupied slots and leaves padding slots at 0
        # (from torch.empty initialization). clamp_ guards against any stale values in
        # unoccupied trailing slots that deepgemm uses as the expert-index array.
        m_indices.clamp_(min=0, max=self.num_experts_per_partition - 1)
        dispose_tensor(hidden_states)

        # Step 2: Gate and Up GEMM (deepgemm bf16 contiguous)
        gateup_output = torch.empty((all_tokens, self.intermediate_size), device=device, dtype=torch.bfloat16)
        with configure_deep_gemm_num_sms(self.num_gemm_sms):
            m_grouped_bf16_gemm_nt_contiguous(input_tensor, self.w1, gateup_output, m_indices)
        dispose_tensor(input_tensor)

        # Step 3: SiLU activation (flat, no mask needed)
        down_input = torch.empty((all_tokens, self.intermediate_size // 2), device=device, dtype=torch.bfloat16)
        silu_and_mul(down_input, gateup_output)
        dispose_tensor(gateup_output)

        # Step 4: Down GEMM (deepgemm bf16 contiguous)
        down_output = torch.empty((all_tokens, self.hidden_size), device=device, dtype=torch.bfloat16)
        with configure_deep_gemm_num_sms(self.num_gemm_sms):
            m_grouped_bf16_gemm_nt_contiguous(down_input, self.w2, down_output, m_indices)
        dispose_tensor(down_input)

        # Step 5: Gather flat → [M, K], apply router weights
        gather_out = torch.empty(hidden_states_shape, device=device, dtype=torch.bfloat16)
        ep_gather(down_output, topk_idx, topk_weights, output_index, gather_out)
        dispose_tensor(down_output)

        return CombineForwardPayload(fused_expert_output=gather_out)
