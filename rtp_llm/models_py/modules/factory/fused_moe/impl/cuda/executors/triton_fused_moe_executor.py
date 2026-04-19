# Adapt from https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/layers/moe/moe_runner/triton.py
# Adapted for RTP-LLM's FusedMoeExpertExecutor interface. Wraps the Triton
# fused_moe path (see rtp_llm.models_py.triton_kernels.moe.fused_moe_triton)
# so it can be selected by the MoE strategy registry.
# Licensed under the Apache License, Version 2.0
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
from rtp_llm.models_py.modules.factory.fused_moe.utils.config_resolver import (
    MoeConfigResolver,
)
from rtp_llm.models_py.triton_kernels.moe.fused_moe_triton import fused_experts_impl
from rtp_llm.utils.model_weight import W


class TritonFusedMoeExecutor(FusedMoeExpertExecutor):
    """Triton fused_moe_kernel + custom moe_sum_reduce executor (no DeepEP).

    This is the RTP-LLM port of sglang's ``TritonRunnerCore``. It expects the
    pure-TP routing layout produced by ``PureTpRouter*``: ``expert_x`` is a
    plain ``(num_tokens, hidden_size)`` tensor and ``expert_topk_ids`` /
    ``expert_topk_weights`` are ``(num_tokens, top_k)``.
    """

    @classmethod
    def executor_type(cls) -> ExecutorType:
        return ExecutorType.BATCHED_TRITON

    @classmethod
    def check_conditions(cls, checker: Any, config: MoEConfigAdapter) -> None:
        resolver = MoeConfigResolver()
        quant_method = resolver.get_quant_method(config)
        # Currently supports the no-quant and FP8 W8A8 paths (per-tensor /
        # per-token / per-block). Other quant schemes should fall through to
        # their dedicated executors.
        checker.check(
            quant_method is None
            or quant_method
            in (
                "FP8_PER_BLOCK",
                "FP8_PER_TENSOR_COMPRESSED",
                "FP8_DYNAMIC_PER_TENSOR",
            )
        )

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
        self.num_local_experts = self.num_experts // self.ep_size
        self.top_k = config.moe_k

        self.use_fp8_w8a8 = (
            quant_config.is_quantized
            and quant_config.quant_dtype == torch.float8_e4m3fn
        )
        self.block_shape = quant_config.block_shape
        self.per_channel_quant = quant_config.is_per_act_token

        self.w13_weight = weights[W.moe_w1]
        self.w2_weight = weights[W.moe_w2]
        self.w13_weight_scale = weights.get(W.moe_s1, None)
        self.w2_weight_scale = weights.get(W.moe_s2, None)

        # Filter sentinel-marked rows when EP is in use (some topk_ids may be -1
        # after PureTpRouter recompute).
        self.filter_expert = self.num_local_experts != self.num_experts

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
        extra_expert_args: Optional[Dict[str, Any]],
    ) -> CombineForwardPayload:
        assert payload.expert_topk_ids is not None
        assert payload.expert_topk_weights is not None
        # PureTpRouter produces 2D ``expert_x`` with shape (num_tokens, K).
        assert (
            payload.expert_x.dim() == 2
        ), f"TritonFusedMoeExecutor expects 2D expert_x, got {payload.expert_x.shape}"

        # Activation arrives in upstream's stylized form (e.g. "SiGLU"); the
        # Triton kernel only knows the lower-cased gate part.
        act = activation.lower()
        if "silu" in act or "swiglu" in act or "siglu" in act:
            act = "silu"
        elif "gelu" in act:
            act = "gelu"
        else:
            raise ValueError(f"Unsupported activation: {activation}")

        # PureTpRouter passes either the raw fp8 quantized tensor or a bf16
        # tensor that we must quantize ourselves. fused_experts_impl handles
        # both branches via use_fp8_w8a8.
        hidden_states = payload.expert_x
        a1_scale = payload.expert_x_scale
        if self.use_fp8_w8a8 and hidden_states.dtype != torch.float8_e4m3fn:
            # Router did not pre-quantize: let the orchestrator do it.
            a1_scale = None

        topk_ids = payload.expert_topk_ids.to(torch.int32)
        # Output dtype must match the un-quantized model dtype, not the (FP8)
        # storage dtype of ``hidden_states`` after a router pre-quant.
        out_dtype = payload.expert_x_origin_dtype or hidden_states.dtype

        out = fused_experts_impl(
            hidden_states=hidden_states.contiguous(),
            w1=self.w13_weight,
            w2=self.w2_weight,
            topk_weights=payload.expert_topk_weights.contiguous(),
            topk_ids=topk_ids.contiguous(),
            inplace=False,
            activation=act,
            apply_router_weight_on_input=apply_router_weight_on_input,
            use_fp8_w8a8=self.use_fp8_w8a8,
            per_channel_quant=self.per_channel_quant,
            w1_scale=self.w13_weight_scale,
            w2_scale=self.w2_weight_scale,
            a1_scale=a1_scale,
            a2_scale=a2_scale,
            block_shape=self.block_shape,
            filter_expert=self.filter_expert,
            out_dtype=out_dtype,
        )
        return CombineForwardPayload(fused_expert_output=out)
