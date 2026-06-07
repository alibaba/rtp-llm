"""MXFP8 contiguous grouped MoE executor for MiniMax-M3.

Consumes the BF16 payload produced by ``PureTpRouterMxfp8`` (no pre-quant),
routes tokens into a contiguous per-expert layout and runs the full FFN
(gate/up grouped fp8_fp4 GEMM -> SwiGLU-OAI -> down grouped fp8_fp4 GEMM) with
``recipe=(1, 32)`` via :func:`mxfp8_moe_forward`. Weights are e4m3 with
prepacked int32 UE8M0 ``[1, 32]`` block scales loaded by ``Mxfp8Weight``.
"""

from typing import Any, Dict, Optional

import torch

from rtp_llm.models_py.kernels.cuda.mxfp8_moe import mxfp8_moe_forward
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
from rtp_llm.models_py.triton_kernels.common.swiglu_oai import (
    is_swiglu_oai,
    swiglu_oai_alpha_limit,
)
from rtp_llm.models_py.utils.arch import get_sm
from rtp_llm.utils.model_weight import W


class Mxfp8ContiguousExecutor(FusedMoeExpertExecutor):
    @classmethod
    def executor_type(cls) -> ExecutorType:
        return ExecutorType.DEEPGEMM_CONTINUOUS

    @classmethod
    def check_conditions(cls, checker: Any, config: MoEConfigAdapter) -> None:
        from rtp_llm.models_py.kernels.cuda.deepgemm_wrapper import has_deep_gemm

        resolver = MoeConfigResolver()
        checker.check(resolver.get_quant_method(config) == "MXFP8")
        checker.check(has_deep_gemm())
        checker.check(get_sm()[0] >= 10)

    def __init__(
        self,
        config: MoEConfigAdapter,
        quant_config: FusedMoEQuantConfig,
        weights: Dict[str, torch.Tensor],
    ):
        super().__init__(config, quant_config, weights)
        self.w1 = weights[W.moe_w1]          # [E, 2*inter, hidden] (up|gate) e4m3
        self.w2 = weights[W.moe_w2]          # [E, hidden, inter] (down) e4m3
        self.w1_scale = weights[W.moe_s1]    # fp32 (1,32) power-of-two scale
        self.w2_scale = weights[W.moe_s2]
        self.E = self.w1.size(0)
        self.top_k = config.moe_k
        self._w1_sp = None
        self._w2_sp = None

    @property
    def topk_ids_dtype(self) -> torch.dtype:
        return torch.int32

    def _packed_scales(self):
        # Pack the (1,32) fp32 expert scales into DeepGEMM int32 layout on first
        # forward (deferred from load; see Mxfp8Weight._postprocess) and cache.
        if self._w1_sp is None:
            from rtp_llm.models_py.kernels.cuda.mxfp8_ops import pack_mxfp8_scale
            e, ngu, k1 = self.w1.shape
            self._w1_sp = pack_mxfp8_scale(self.w1_scale, mn=ngu, k=k1, num_groups=e)
            e2, hid, k2 = self.w2.shape
            self._w2_sp = pack_mxfp8_scale(self.w2_scale, mn=hid, k=k2, num_groups=e2)
        return self._w1_sp, self._w2_sp

    def execute(
        self,
        payload: ExpertForwardPayload,
        activation: str,
        expert_map: Optional[torch.Tensor],
        a2_scale: Optional[torch.Tensor],
        apply_router_weight_on_input: bool,
        extra_expert_args: Optional[dict[str, Any]],
    ) -> CombineForwardPayload:
        hidden = payload.expert_x
        topk_ids = payload.expert_topk_ids
        topk_weights = payload.expert_topk_weights
        assert topk_ids is not None and topk_weights is not None

        if is_swiglu_oai(activation):
            alpha, limit = swiglu_oai_alpha_limit(extra_expert_args)
        else:
            alpha, limit = None, None

        w1_sp, w2_sp = self._packed_scales()
        output = mxfp8_moe_forward(
            hidden,
            topk_ids,
            topk_weights,
            self.w1,
            w1_sp,
            self.w2,
            w2_sp,
            num_experts=self.E,
            alpha=alpha,
            limit=limit,
            gate_first=False,
        )
        return CombineForwardPayload(fused_expert_output=output)
