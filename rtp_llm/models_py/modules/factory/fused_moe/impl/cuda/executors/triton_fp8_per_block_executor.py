"""Block-scaled FP8 MoE on CUDA devices without DeepGEMM support."""

from typing import Any, Dict, Optional

import torch
import triton.language as tl

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
from rtp_llm.models_py.modules.factory.fused_moe.utils.config_resolver import (
    MoeConfigResolver,
)
from rtp_llm.models_py.triton_kernels.common.activation import silu_and_mul
from rtp_llm.models_py.triton_kernels.moe.fused_moe_kernel import (
    get_default_config,
    invoke_fused_moe_kernel,
    moe_align_block_size_torch,
)
from rtp_llm.models_py.utils.arch import get_sm
from rtp_llm.utils.model_weight import W


class TritonFp8PerBlockExecutor(FusedMoeExpertExecutor):
    """Local routed FP8 MoE with FP32 per-128x128 weight scales.

    Each FP8 GEMM applies the activation's per-token 128-wide scale and the
    checkpoint weight's 128x128 scale. This path does not call DeepGEMM and
    keeps the same BF16 intermediate rounding as the DeepGEMM executor.
    """

    BLOCK_SIZE = 128

    @classmethod
    def executor_type(cls) -> ExecutorType:
        # Lower priority than DeepGEMM when both backends are available.
        return ExecutorType.BATCHED_TRITON

    @classmethod
    def check_conditions(cls, checker: Any, config: MoEConfigAdapter) -> None:
        resolver = MoeConfigResolver()
        checker.check(resolver.get_quant_method(config) == "FP8_PER_BLOCK")
        checker.check(resolver.is_bf16(config))
        # SM100/SM120 loading rewrites scales to packed UE8M0, which this
        # executor deliberately does not consume.
        checker.check((8, 9) <= get_sm() < (10, 0))
        checker.check(not config.enable_cuda_graph)

    def __init__(
        self,
        config: MoEConfigAdapter,
        quant_config: FusedMoEQuantConfig,
        weights: Dict[str, torch.Tensor],
    ):
        super().__init__(config, quant_config, weights)
        self.ep_size = config.ep_size
        self.num_experts = config.expert_num // self.ep_size
        self.w1 = weights[W.moe_w1]
        self.w2 = weights[W.moe_w2]
        self.s1 = weights[W.moe_s1]
        self.s2 = weights[W.moe_s2]

        if self.w1.ndim != 3 or self.w2.ndim != 3:
            raise ValueError("FP8 MoE weights must have shape [experts, N, K]")
        expert_count, gate_up_size, hidden_size = self.w1.shape
        if expert_count != self.num_experts or self.w2.shape != (
            expert_count,
            hidden_size,
            gate_up_size // 2,
        ):
            raise ValueError("FP8 MoE expert weight shapes do not match the model")
        if (
            gate_up_size % (2 * self.BLOCK_SIZE) != 0
            or hidden_size % self.BLOCK_SIZE != 0
        ):
            raise ValueError("FP8 MoE dimensions must be multiples of 128")
        if self.w1.dtype != torch.float8_e4m3fn or self.w2.dtype != torch.float8_e4m3fn:
            raise ValueError("FP8 MoE weights must use float8_e4m3fn")
        if self.s1.dtype != torch.float32 or self.s2.dtype != torch.float32:
            raise ValueError("FP8 MoE block scales must use float32")
        if self.s1.shape != (
            expert_count,
            gate_up_size // self.BLOCK_SIZE,
            hidden_size // self.BLOCK_SIZE,
        ) or self.s2.shape != (
            expert_count,
            hidden_size // self.BLOCK_SIZE,
            (gate_up_size // 2) // self.BLOCK_SIZE,
        ):
            raise ValueError("FP8 MoE block scale shapes do not match the weights")

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
        if activation != "SiGLU":
            raise ValueError("Triton FP8 MoE supports SiGLU activation only")
        if (
            expert_map is not None
            or a2_scale is not None
            or apply_router_weight_on_input
        ):
            raise ValueError(
                "Triton FP8 MoE does not support expert remapping or input weights"
            )
        x = payload.expert_x
        x_scale = payload.expert_x_scale
        topk_ids = payload.expert_topk_ids
        topk_weights = payload.expert_topk_weights
        if x_scale is None or topk_ids is None or topk_weights is None:
            raise ValueError(
                "Triton FP8 MoE requires FP8 input scales and top-k routes"
            )
        if x.dtype != torch.float8_e4m3fn or x_scale.dtype != torch.float32:
            raise ValueError("Triton FP8 MoE requires FP8 input and float32 scales")
        tokens, hidden_size = x.shape
        topk = topk_ids.shape[1]
        if hidden_size != self.w1.shape[2] or topk_weights.shape != topk_ids.shape:
            raise ValueError("Triton FP8 MoE input and top-k shapes do not match")
        if tokens == 0:
            return CombineForwardPayload(
                fused_expert_output=torch.empty(
                    (0, hidden_size), device=x.device, dtype=torch.bfloat16
                )
            )

        # Pure-TP routing uses -1 for experts owned by another rank. Route
        # those slots through a valid local expert with zero combine weight.
        valid_routes = (topk_ids >= 0) & (topk_ids < self.num_experts)
        local_ids = topk_ids.clamp(0, self.num_experts - 1)
        local_weights = topk_weights * valid_routes
        flat_ids = local_ids.reshape(-1)
        flat_weights = local_weights.reshape(-1)
        gate_up_size = self.w1.shape[1]
        intermediate_size = gate_up_size // 2
        config1 = get_default_config(
            tokens, self.num_experts, gate_up_size, hidden_size, topk
        )
        config2 = get_default_config(
            tokens, self.num_experts, hidden_size, intermediate_size, topk
        )

        def aligned_routes(block_m: int):
            return moe_align_block_size_torch(local_ids, block_m, self.num_experts)

        sorted_ids1, expert_ids1, padded1 = aligned_routes(config1["BLOCK_SIZE_M"])
        if config1["BLOCK_SIZE_M"] == config2["BLOCK_SIZE_M"]:
            sorted_ids2, expert_ids2, padded2 = sorted_ids1, expert_ids1, padded1
        else:
            sorted_ids2, expert_ids2, padded2 = aligned_routes(config2["BLOCK_SIZE_M"])

        route_count = tokens * topk
        gate_up = torch.empty(
            (route_count, gate_up_size), device=x.device, dtype=torch.bfloat16
        )
        invoke_fused_moe_kernel(
            x,
            self.w1,
            gate_up,
            flat_weights,
            flat_ids,
            sorted_ids1,
            expert_ids1,
            padded1,
            False,
            topk,
            config1,
            tl.bfloat16,
            A_scale=x_scale,
            B_scale=self.s1,
            block_shape=[self.BLOCK_SIZE, self.BLOCK_SIZE],
        )
        down_input = torch.empty(
            (route_count, intermediate_size), device=x.device, dtype=torch.bfloat16
        )
        silu_and_mul(down_input, gate_up)
        down_fp8, down_scale = sgl_per_token_group_quant_fp8(
            down_input, group_size=self.BLOCK_SIZE
        )
        down_output = torch.empty(
            (route_count, hidden_size), device=x.device, dtype=torch.bfloat16
        )
        invoke_fused_moe_kernel(
            down_fp8,
            self.w2,
            down_output,
            flat_weights,
            flat_ids,
            sorted_ids2,
            expert_ids2,
            padded2,
            True,
            1,
            config2,
            tl.bfloat16,
            A_scale=down_scale,
            B_scale=self.s2,
            block_shape=[self.BLOCK_SIZE, self.BLOCK_SIZE],
        )
        return CombineForwardPayload(
            fused_expert_output=down_output.view(tokens, topk, hidden_size).sum(dim=1)
        )
