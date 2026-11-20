from typing import Any, Dict, Optional

import torch

from rtp_llm.models_py.modules.factory.fused_moe.defs.config_adapter import (
    MoEConfigAdapter,
)
from rtp_llm.models_py.modules.factory.fused_moe.defs.fused_moe import (
    CombineForwardPayload,
    ExpertForwardPayload,
    GlobalExpertIdsContext,
)
from rtp_llm.models_py.modules.factory.fused_moe.defs.quant_config import (
    FusedMoEQuantConfig,
)
from rtp_llm.models_py.modules.factory.fused_moe.defs.type import ExecutorType
from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.executors.deepgemm_hybrid_executor import (
    DeepGemmHybridExecutor,
    _is_cuda_graph_warmup_or_capture,
)
from rtp_llm.models_py.utils.arch import get_sm


class DeepGemmMaskedExecutorV2(DeepGemmHybridExecutor):

    @classmethod
    def executor_type(cls) -> ExecutorType:
        return ExecutorType.DEEPGEMM_MASKED

    @classmethod
    def check_conditions(cls, checker: Any, config: MoEConfigAdapter) -> None:
        """Check if DeepGemmMaskedExecutorV2 can handle the configuration"""
        from rtp_llm.models_py.kernels.cuda.deepgemm_wrapper import has_deep_gemm
        from rtp_llm.models_py.modules.factory.fused_moe.utils.config_resolver import (
            MoeConfigResolver,
        )

        resolver = MoeConfigResolver()
        quant_method = resolver.get_quant_method(config)
        checker.check(quant_method == "FP8_PER_BLOCK")
        checker.check(resolver.is_bf16(config))
        checker.check(has_deep_gemm())
        checker.check(get_sm()[0] >= 9)

    def __init__(
        self,
        config: MoEConfigAdapter,
        quant_config: FusedMoEQuantConfig,
        weights: Dict[str, torch.Tensor],
    ):
        super().__init__(config, quant_config, weights)
        # Qwen4 auto-selects this executor for decode graph capture. Large
        # prefills still need the compact grouped layout: the masked layout
        # reserves a full token-capacity slab for every local expert.
        self._qwen4_sm90_mixed_graph = (
            config.moe_strategy == "auto"
            and config.enable_cuda_graph
            and torch.cuda.get_device_capability(self.w13_weight.device) == (9, 0)
            and getattr(config.model_config, "_qwen4_full_fp8", False)
            and not getattr(config.model_config, "_qwen4_full_fp8_nvfp4", False)
        )

    def can_consume_global_expert_ids(
        self,
        hidden_states,
        topk_ids,
        activation,
        expert_map,
        a1_scale,
        a2_scale,
        apply_router_weight_on_input,
        extra_expert_args,
    ) -> bool:
        return self._can_consume_h20_compact_global_ids(
            hidden_states,
            topk_ids,
            activation,
            expert_map,
            a1_scale,
            a2_scale,
            apply_router_weight_on_input,
            extra_expert_args,
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
        if (
            self._qwen4_sm90_mixed_graph
            and payload.expert_x is not None
            and payload.expert_x.shape[0] > self.masked_max_token_num
            and not _is_cuda_graph_warmup_or_capture()
        ):
            if isinstance(
                getattr(payload, "router_context", None), GlobalExpertIdsContext
            ):
                raise RuntimeError(
                    "Global expert ID payload cannot fall through to a legacy consumer"
                )
            return self.execute_contiguous(
                payload,
                activation,
                expert_map,
                a2_scale,
                apply_router_weight_on_input,
                extra_expert_args,
                use_eager_layout=True,
            )
        return self.execute_masked(
            payload,
            activation,
            expert_map,
            a2_scale,
            apply_router_weight_on_input,
            extra_expert_args,
        )
