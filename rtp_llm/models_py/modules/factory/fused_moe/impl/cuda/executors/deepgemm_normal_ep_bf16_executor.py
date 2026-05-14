import logging
from typing import Any, Dict, Optional

import torch

from rtp_llm.models_py.kernels.cuda.deepgemm_wrapper import (
    configure_deep_gemm_num_sms,
    is_deep_gemm_e8m0_used,
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
    silu_mul_masked_bf16_no_post_quant_fwd,
)
from rtp_llm.models_py.triton_kernels.moe.ep_kernels import ep_gather, ep_scatter_bf16
from rtp_llm.models_py.utils.arch import get_num_device_sms, get_sm
from rtp_llm.models_py.utils.math import align, ceil_div
from rtp_llm.models_py.utils.memory import dispose_tensor
from rtp_llm.utils.model_weight import W

logger = logging.getLogger(__name__)


class DeepGemmNormalEpBf16Executor(FusedMoeExpertExecutor):
    """
    BF16 MoE executor for normal DeepEP routing using deep_gemm masked grouped GEMM.

    Receives contiguous (M, K) BF16 tokens from the normal DeepEP dispatch,
    scatters them into a masked (E, alignment, K) layout, runs two grouped
    masked GEMMs (gate+up and down), and gathers the results back.

    This path is selected over TritonFusedMoeExecutor when deep_gemm is
    available (SM9+) and the model uses no weight quantization.
    """

    EXPERT_ALIGNMENT = 128
    DEEPGEMM_BLOCK_SHAPE: list[int] = [128, 128]

    @classmethod
    def executor_type(cls) -> ExecutorType:
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
        checker.check(not is_deep_gemm_e8m0_used())

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
        self.rank_expert_offset = self.ep_rank * self.num_experts_per_partition

        self.top_k = config.moe_k

        self._w1 = weights[W.moe_w1]  # (E_local, N, K)
        self._w2 = weights[W.moe_w2]  # (E_local, K, N//2)

        self._E, self._N, self._K = self._w1.size()
        assert self._N % 2 == 0
        assert self._w2.size(0) == self._E
        assert self._w2.size(1) == self._K
        assert self._w2.size(2) == self._N // 2

        self._num_gemm_sms = get_num_device_sms()
        logger.info(
            "DeepGemmNormalEpBf16Executor initialized: E_local=%d K=%d N=%d ep_rank=%d/%d",
            self._E,
            self._K,
            self._N,
            self.ep_rank,
            self.ep_size,
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
        assert payload.expert_x is not None
        assert payload.expert_topk_ids is not None
        assert payload.expert_topk_weights is not None
        assert payload.expert_tokens_meta is not None
        assert payload.expert_tokens_meta.expert_num_tokens is not None

        expert_x = payload.expert_x  # (M, K) BF16 contiguous
        global_topk_ids = payload.expert_topk_ids  # (M, topk) int64, global IDs
        topk_weights = payload.expert_topk_weights
        num_recv_tokens_per_expert = (
            payload.expert_tokens_meta.expert_num_tokens
        )  # (E_local,) int32

        M, K = expert_x.shape
        device = expert_x.device

        if M == 0:
            return CombineForwardPayload(
                fused_expert_output=torch.zeros(
                    expert_x.shape, device=device, dtype=torch.bfloat16
                )
            )

        # Convert global expert IDs → local IDs.
        # The normal router substituted original -1 slots with (expert_num-1) or 0.
        # After subtraction, those become out-of-range (>= E_local or < 0) and
        # are safely skipped by ep_scatter_bf16's bounds check.
        local_topk_ids = (global_topk_ids - self.rank_expert_offset).to(torch.int32)

        # alignment must be >= max tokens any single expert can receive
        alignment = align(M, self.EXPERT_ALIGNMENT)
        token_num_mean_per_expert = ceil_div(M * self.top_k, self._E)
        expected_m = min(alignment, token_num_mean_per_expert)

        # Masked layout: (E_local * alignment, K), zeroed for empty slots
        output_tensor = torch.zeros(
            (self._E * alignment, K), device=device, dtype=torch.bfloat16
        )
        # Initialize to -1 so ep_gather skips unvisited slots via bounds check
        output_index = torch.full(
            local_topk_ids.shape, -1, device=device, dtype=torch.int32
        )
        expert_start_loc = torch.empty(self._E, device=device, dtype=torch.int32)

        logger.debug(
            "DeepGemmNormalEpBf16Executor: M=%d E=%d alignment=%d expected_m=%d",
            M,
            self._E,
            alignment,
            expected_m,
        )

        ep_scatter_bf16(
            recv_x=expert_x,
            recv_topk=local_topk_ids,
            alignment=alignment,
            expert_start_loc=expert_start_loc,
            output_tensor=output_tensor,
            output_index=output_index,
        )

        masked_input = output_tensor.view(self._E, alignment, K)

        with configure_deep_gemm_num_sms(self._num_gemm_sms):
            upgate_output = torch.empty(
                (self._E, alignment, self._N), device=device, dtype=torch.bfloat16
            )
            m_grouped_bf16_gemm_nt_masked(
                masked_input,
                self._w1,
                upgate_output,
                num_recv_tokens_per_expert,
                expected_m,
            )
            dispose_tensor(output_tensor)

            down_input = torch.empty(
                (self._E, alignment, self._N // 2), device=device, dtype=torch.bfloat16
            )
            silu_mul_masked_bf16_no_post_quant_fwd(
                input=upgate_output,
                output=down_input,
                masked_m=num_recv_tokens_per_expert,
                expected_m=expected_m,
                group_size=self.DEEPGEMM_BLOCK_SHAPE[0],
            )
            dispose_tensor(upgate_output)

            down_output = torch.empty(
                (self._E, alignment, K), device=device, dtype=torch.bfloat16
            )
            m_grouped_bf16_gemm_nt_masked(
                down_input,
                self._w2,
                down_output,
                num_recv_tokens_per_expert,
                expected_m,
            )
            dispose_tensor(down_input)

        gather_out = torch.empty(expert_x.shape, device=device, dtype=torch.bfloat16)
        ep_gather(
            down_output.view(self._E * alignment, K),
            local_topk_ids,
            topk_weights,
            output_index,
            gather_out,
        )

        return CombineForwardPayload(fused_expert_output=gather_out)
