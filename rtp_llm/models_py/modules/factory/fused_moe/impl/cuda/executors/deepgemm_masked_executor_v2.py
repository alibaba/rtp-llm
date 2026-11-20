import os
from typing import Any, Dict, Optional

import torch

from rtp_llm.models_py.modules.factory.fused_moe.defs.config_adapter import (
    MoEConfigAdapter,
)
from rtp_llm.models_py.modules.factory.fused_moe.defs.fused_moe import (
    CombineForwardPayload,
    ExpertForwardPayload,
)
from rtp_llm.models_py.modules.factory.fused_moe.defs.quant_config import (
    FusedMoEQuantConfig,
)
from rtp_llm.models_py.modules.factory.fused_moe.defs.type import ExecutorType
from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.executors.deepgemm_hybrid_executor import (
    DeepGemmHybridExecutor,
    _is_cuda_graph_warmup_or_capture,
)
from rtp_llm.models_py.modules.qwen4_exp.moe.compact_fp8_decode import (
    compact_fp8_decode,
    h20_compact_moe_decode,
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

        self._h20_compact_fp8_decode = (
            os.environ.get("RTP_LLM_H20_COMPACT_FP8_MOE", "1") == "1"
            and self._qwen4_sm90_mixed_graph
            and self._sm90_mn_major_scales
            and config.tp_size == config.ep_size == 2
            and getattr(config, "dp_size", 0) == 1
            and "H20" in torch.cuda.get_device_name(self.w13_weight.device)
            and self.w13_weight.dtype == self.w2_weight.dtype == torch.float8_e4m3fn
            and self.w13_weight_scale_inv.dtype
            == self.w2_weight_scale_inv.dtype
            == torch.float32
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
            return self.execute_contiguous(
                payload,
                activation,
                expert_map,
                a2_scale,
                apply_router_weight_on_input,
                extra_expert_args,
                use_eager_layout=True,
            )
        if self._h20_compact_fp8_decode and h20_compact_moe_decode.get():
            compact = self._execute_compact_decode(
                payload,
                activation,
                expert_map,
                a2_scale,
                apply_router_weight_on_input,
                extra_expert_args,
            )
            if compact is not None:
                return compact
        return self.execute_masked(
            payload,
            activation,
            expert_map,
            a2_scale,
            apply_router_weight_on_input,
            extra_expert_args,
        )

    def _execute_compact_decode(
        self,
        payload: ExpertForwardPayload,
        activation: str,
        expert_map: Optional[torch.Tensor],
        a2_scale: Optional[torch.Tensor],
        apply_router_weight_on_input: bool,
        extra_expert_args: Optional[dict[str, Any]],
    ) -> Optional[CombineForwardPayload]:
        x = payload.expert_x
        scales = getattr(payload, "expert_x_scale", None)
        ids = getattr(payload, "expert_topk_ids", None)
        route_weights = getattr(payload, "expert_topk_weights", None)
        meta = getattr(payload, "expert_tokens_meta", None)
        counts = meta.expert_num_tokens if meta is not None else None
        if (
            x is not None
            and x.ndim == 2
            and 0 < x.shape[0] <= 2048
            and x.shape[1] == self.K
            and x.dtype == torch.float8_e4m3fn
            and scales is not None
            and scales.dtype == torch.float32
            and scales.shape == (x.shape[0], self.K // self.BLOCK_SIZE)
            and ids is not None
            and ids.dtype in (torch.int32, torch.int64)
            and ids.shape == (x.shape[0], self.top_k)
            and route_weights is not None
            and route_weights.dtype == torch.float32
            and route_weights.shape == ids.shape
            and counts is not None
            and counts.shape == (self.E,)
            and counts.device == x.device
            and counts.dtype == torch.int32
            and counts.is_contiguous()
            and activation == "SiGLU"
            and expert_map is None
            and a2_scale is None
            and not apply_router_weight_on_input
            and not extra_expert_args
            and _is_cuda_graph_warmup_or_capture()
        ):
            # The PureTP router has already converted remote routes to -1.
            # Rebuild block metadata on each replay from the current route IDs.
            return CombineForwardPayload(
                fused_expert_output=compact_fp8_decode(
                    x,
                    scales,
                    ids,
                    route_weights,
                    counts,
                    self.w13_weight_fp8,
                    self.w2_weight_fp8,
                )
            )
        return None
