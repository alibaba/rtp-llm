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
from rtp_llm.models_py.triton_kernels.common.activation import silu_and_mul
from rtp_llm.models_py.triton_kernels.moe.grouped_gemm import (
    invoke_moe_batched_triton_kernel,
)
from rtp_llm.utils.model_weight import W


class BatchedTritonExperts(FusedMoeExpertExecutor):
    """Triton experts for BatchedDataRouter's (E, rows, hidden) layout."""

    @classmethod
    def executor_type(cls):
        return ExecutorType.BATCHED_TRITON

    @property
    def topk_ids_dtype(self) -> torch.dtype:
        return torch.int32

    @classmethod
    def check_conditions(cls, checker: Any, config: MoEConfigAdapter) -> None:
        from rtp_llm.models_py.modules.factory.fused_moe.utils.config_resolver import (
            MoeConfigResolver,
        )

        resolver = MoeConfigResolver()
        checker.check(not resolver.has_quantization(config))

    def __init__(
        self,
        config: MoEConfigAdapter,
        quant_config: FusedMoEQuantConfig,
        weights: Dict[str, torch.Tensor],
    ):
        super().__init__(config, quant_config, weights)

        self.w1 = weights[W.moe_w1]
        self.w2 = weights[W.moe_w2]
        assert self.w1.stride(-1) == 1 and self.w2.stride(-1) == 1
        assert self.w2.size(0) == self.w1.size(0)

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
        if expert_map is not None:
            raise ValueError("BatchedTritonExperts does not support expert_map")
        if apply_router_weight_on_input:
            raise ValueError(
                "BatchedTritonExperts requires output-side router weighting"
            )

        expert_x = payload.expert_x
        meta = payload.expert_tokens_meta
        if meta is None or meta.expert_num_tokens is None:
            raise ValueError("expert_num_tokens is required")
        expert_num_tokens = meta.expert_num_tokens

        E = self.local_num_experts
        N = self.w1.size(1)
        if (
            expert_x.dim() != 3
            or expert_x.size(0) != E
            or expert_x.size(2) != self.w1.size(2)
            or expert_num_tokens.shape != (E,)
        ):
            raise ValueError(
                f"Invalid expert shapes: {tuple(expert_x.shape)}, "
                f"{tuple(expert_num_tokens.shape)}"
            )

        if expert_x.dtype == torch.bfloat16:
            compute_type = tl.bfloat16
        elif expert_x.dtype == torch.float16:
            compute_type = tl.float16
        else:
            raise ValueError(f"Unsupported compute_type: {expert_x.dtype}")

        num_rows = expert_x.size(1)
        # GEMMs skip rows past expert_num_tokens; activation padding is discarded.
        intermediate_cache1 = torch.empty(
            (E, num_rows, N),
            device=expert_x.device,
            dtype=expert_x.dtype,
        )
        intermediate_cache2 = torch.empty(
            (E, num_rows, N // 2),
            device=expert_x.device,
            dtype=expert_x.dtype,
        )
        output = torch.empty(
            (E, num_rows, self.w2.size(1)),
            device=expert_x.device,
            dtype=self.w2.dtype,
        )

        invoke_moe_batched_triton_kernel(
            A=expert_x,
            B=self.w1,
            C=intermediate_cache1,
            expert_num_tokens=expert_num_tokens,
            compute_type=compute_type,
        )

        silu_and_mul(
            intermediate_cache2.view(-1, N // 2),
            intermediate_cache1.view(-1, N),
        )

        invoke_moe_batched_triton_kernel(
            A=intermediate_cache2,
            B=self.w2,
            C=output,
            expert_num_tokens=expert_num_tokens,
            compute_type=compute_type,
        )

        return CombineForwardPayload(fused_expert_output=output)
