"""BF16 CANN implementation of the Qwen MoE expert pipeline."""

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
from rtp_llm.utils.model_weight import W


class NpuFusedExpertsExecutor(FusedMoeExpertExecutor):
    """Run init-routing, two grouped GEMMs, SwiGLU and unpermute on NPU.

    Qwen stores W1 as ``[expert, up+gate, hidden]``.  CANN SwiGLU consumes
    ``[gate, up]``, so the initialization transform both swaps the halves and
    changes the GEMM layout to ``[expert, hidden, 2 * intermediate]``.  This is
    intentionally done once per layer construction, never in ``execute``.
    """

    @classmethod
    def executor_type(cls) -> ExecutorType:
        return ExecutorType.FUSED_MOE

    @classmethod
    def check_conditions(cls, checker: Any, config: MoEConfigAdapter) -> None:
        resolver = MoeConfigResolver()
        checker.check(resolver.is_bf16(config))
        checker.check(not resolver.has_quantization(config))
        # cuda-graph gate removed: aclnn MoE ops are expected to be capturable
        # by aclgraph; graph-mode validation for Qwen3.5 in progress.

    @property
    def topk_ids_dtype(self) -> torch.dtype:
        return torch.int32

    def __init__(
        self,
        config: MoEConfigAdapter,
        quant_config: FusedMoEQuantConfig,
        weights: Dict[str, torch.Tensor],
    ) -> None:
        super().__init__(config, quant_config, weights)
        self._torch_npu = self._require_cann_ops()
        self.num_experts = config.expert_num
        w1 = weights[W.moe_w1]
        w2 = weights[W.moe_w2]
        self._validate_weights(w1, w2)

        # This is an Ascend-only weight dictionary.  Replace the source entries
        # rather than retaining a second full expert-weight copy for the life of
        # the model.  GenericMoeLayer only reads their leading expert dimension.
        self.w1_cann = self._prepare_w1(w1)
        self.w2_cann = w2.transpose(1, 2).contiguous()
        weights[W.moe_w1] = self.w1_cann
        weights[W.moe_w2] = self.w2_cann

    @staticmethod
    def _require_cann_ops() -> Any:
        try:
            import torch_npu
        except ImportError as error:
            raise RuntimeError(
                "Ascend CANN MoE requires the pinned torch_npu package"
            ) from error

        required = (
            "npu_moe_init_routing_v2",
            "npu_grouped_matmul",
            "npu_swiglu",
            "npu_moe_token_unpermute",
        )
        missing = [name for name in required if not hasattr(torch_npu, name)]
        if missing:
            raise RuntimeError(
                "torch_npu is missing required CANN MoE operators: "
                + ", ".join(missing)
            )
        return torch_npu

    def _validate_weights(self, w1: torch.Tensor, w2: torch.Tensor) -> None:
        if w1.dtype != torch.bfloat16 or w2.dtype != torch.bfloat16:
            raise TypeError("Ascend CANN MoE currently supports BF16 expert weights only")
        if w1.dim() != 3 or w2.dim() != 3:
            raise ValueError("Ascend CANN MoE expert weights must be rank-3")
        if w1.size(0) != self.num_experts or w2.size(0) != self.num_experts:
            raise ValueError("Ascend CANN MoE requires all experts on every TP rank")
        if w1.size(1) % 2 != 0:
            raise ValueError("W.moe_w1 intermediate dimension must contain gate/up halves")
        if w1.size(2) != w2.size(1):
            raise ValueError("expert W1/W2 hidden dimensions are inconsistent")
        if w1.size(1) // 2 != w2.size(2):
            raise ValueError("expert W1/W2 intermediate dimensions are inconsistent")

    @staticmethod
    def _prepare_w1(w1: torch.Tensor) -> torch.Tensor:
        """Convert Qwen's [up, gate] channels to CANN SwiGLU [gate, up]."""
        up, gate = w1.chunk(2, dim=1)
        return torch.cat((gate, up), dim=1).transpose(1, 2).contiguous()

    def execute(
        self,
        payload: ExpertForwardPayload,
        activation: str,
        expert_map: Optional[torch.Tensor],
        a2_scale: Optional[torch.Tensor],
        apply_router_weight_on_input: bool,
        extra_expert_args: Optional[dict[str, Any]],
    ) -> CombineForwardPayload:
        if activation.lower() not in ("siglu", "swiglu", "silu"):
            raise NotImplementedError(
                f"Ascend CANN MoE supports SwiGLU only, got activation={activation}"
            )
        if expert_map is not None:
            raise NotImplementedError("Ascend CANN MoE does not support expert maps")
        if a2_scale is not None or apply_router_weight_on_input:
            raise NotImplementedError("Ascend CANN MoE does not support quantized/router-input scaling")
        if payload.expert_topk_ids is None or payload.expert_topk_weights is None:
            raise ValueError("Ascend CANN MoE requires top-k ids and weights")
        if payload.expert_x.dtype != torch.bfloat16:
            raise TypeError("Ascend CANN MoE currently supports BF16 activations only")

        topk_ids = payload.expert_topk_ids.contiguous()
        topk_weights = payload.expert_topk_weights.contiguous()
        if topk_ids.dtype != torch.int32 or topk_weights.dtype != torch.float32:
            raise TypeError("Ascend CANN MoE requires int32 ids and float32 weights")

        sorted_x, row_idx, group_list, _ = self._torch_npu.npu_moe_init_routing_v2(
            payload.expert_x,
            expert_idx=topk_ids,
            scale=None,
            active_num=topk_ids.numel(),
            expert_num=self.num_experts,
            active_expert_range=[0, self.num_experts],
            drop_pad_mode=0,
            expert_tokens_num_type=0,
            expert_tokens_num_flag=True,
            quant_mode=-1,
            row_idx_type=1,
        )
        group_list = group_list.to(torch.int64)
        gate_up = self._torch_npu.npu_grouped_matmul(
            x=[sorted_x],
            weight=[self.w1_cann],
            bias=None,
            split_item=3,
            group_type=0,
            group_list_type=0,
            group_list=group_list,
        )[0]
        activated = self._torch_npu.npu_swiglu(gate_up.contiguous(), dim=-1)
        expert_out = self._torch_npu.npu_grouped_matmul(
            x=[activated],
            weight=[self.w2_cann],
            bias=None,
            split_item=3,
            group_type=0,
            group_list_type=0,
            group_list=group_list,
        )[0]
        # npu_moe_init_routing_v2(row_idx_type=1) returns a gather index:
        # row_idx[permuted_row] == original (token, topk) flat row.  But
        # npu_moe_token_unpermute expects a scatter index, i.e.
        # sorted_indices[original_row] == permuted_row.  Feeding the gather
        # index straight through silently reshuffles tokens between rows, so
        # invert the permutation first.
        gather_idx = torch.abs(row_idx)
        scatter_idx = torch.empty_like(gather_idx)
        scatter_idx.scatter_(
            0,
            gather_idx.long(),
            torch.arange(
                gather_idx.numel(),
                device=gather_idx.device,
                dtype=gather_idx.dtype,
            ),
        )
        output = self._torch_npu.npu_moe_token_unpermute(
            permuted_tokens=expert_out,
            sorted_indices=scatter_idx,
            probs=topk_weights,
        )
        return CombineForwardPayload(fused_expert_output=output)
