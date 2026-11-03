"""Ascend NPU W8A8_MXFP8 MoE executor (W8A8_MXFP8: Weight & Activation FP8 Quantization).

This module is only imported when ``get_device_type() == DeviceType.Ascend``
(see ``factory/fused_moe/__init__.py`` device branch), so a top-level
``import torch_npu`` is allowed here.

Weight layouts after ``PerBlockFp8Weight._postprocess`` (NPU branch):
  * ``moe_w1``: fp8_e4m3fn ``[E, K_hidden, 2N]`` (up|gate order along the last dim)
  * ``moe_w2``: fp8_e4m3fn ``[E, K_inter, N_hidden]``
  * ``moe_s1``: uint8 (E8M0) ``[E, kp1, 2N, 2]`` swizzled pair-split layout
  * ``moe_s2``: uint8 (E8M0) ``[E, kp2, N_hidden, 2]``

Execution path (fully controlled semantics): npu_moe_init_routing_v2
(fused dispatch) -> npu_dynamic_mx_quant -> grouped GEMM1 -> silu*mul ->
npu_dynamic_mx_quant -> grouped GEMM2 -> npu_moe_token_unpermute (fused
combine incl. topk weights). Numerically verified against a per-expert
bf16 reference on Ascend950PR (cos_sim ~0.997).

NOTE: the fused ``npu_grouped_matmul_swiglu_quant_v2`` path was evaluated
(semantics mismatch, cos_sim 0.88/0.73 vs both half-order references) and
removed; re-introduce only after validation against the CANN op spec.
"""

from typing import Any, Dict, Optional

import torch
import torch_npu

from rtp_llm.device.device_type import is_ascend
from rtp_llm.models_py.kernels.ascend.w8a8_mx_layout import get_e8m0_dtype
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
from rtp_llm.utils.model_weight import W

# Logical dtype of E8M0 scales (prefer torch_npu, fall back to torch).
FLOAT8_E8M0FNU_DTYPE = get_e8m0_dtype()


def _first(output: Any) -> torch.Tensor:
    """npu grouped matmul ops may return a list or a single tensor."""
    if isinstance(output, (list, tuple)):
        return output[0]
    return output


class AscendW8A8MXFP8Executor(FusedMoeExpertExecutor):
    """NPU W8A8_MXFP8 MoE executor."""

    @classmethod
    def executor_type(cls) -> ExecutorType:
        return ExecutorType.ASCEND_W8A8_MXFP8

    @classmethod
    def check_conditions(cls, checker: Any, config: MoEConfigAdapter) -> None:
        checker.check(is_ascend())
        checker.check(torch.npu.is_available())

    @property
    def topk_ids_dtype(self) -> torch.dtype:
        # The MoE layer builds topk_ids in this dtype (generic_moe);
        # npu_moe_init_routing_v2 requires int32 expert_idx.
        return torch.int32

    def __init__(
        self,
        config: MoEConfigAdapter,
        quant_config: FusedMoEQuantConfig,
        weights: Dict[str, torch.Tensor],
    ):
        super().__init__(config, quant_config, weights)
        self._w1 = weights[W.moe_w1]  # fp8 [E, K_hidden, 2N] (up|gate)
        self._w2 = weights[W.moe_w2]  # fp8 [E, K_inter, N_hidden]
        self._w1_scale = weights.get(W.moe_s1, None)  # uint8 [E, kp1, 2N, 2]
        self._w2_scale = weights.get(W.moe_s2, None)  # uint8 [E, kp2, N, 2]
        if self._w1_scale is None or self._w2_scale is None:
            raise ValueError(
                "AscendW8A8MXFP8Executor requires moe_s1/moe_s2 scales (W8A8_MXFP8)"
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
        if expert_map is not None:
            raise NotImplementedError("Ascend W8A8_MXFP8 MoE does not support expert maps")
        if a2_scale is not None or apply_router_weight_on_input:
            raise NotImplementedError(
                "Ascend W8A8_MXFP8 MoE applies router weights in token unpermute"
            )
        if payload.expert_topk_ids is None or payload.expert_topk_weights is None:
            raise ValueError("Ascend W8A8_MXFP8 MoE requires top-k ids and weights")
        expert_x = payload.expert_x  # flat bf16 [tokens, hidden]
        topk_ids = payload.expert_topk_ids.contiguous()
        topk_weights = payload.expert_topk_weights.contiguous()
        if topk_ids.dtype != torch.int32:
            raise TypeError(
                f"Ascend W8A8_MXFP8 MoE requires int32 topk_ids, got {topk_ids.dtype}"
            )

        # Fused dispatch: expand to [tokens*topk, hidden] expert-grouped rows +
        # the group_list / row_idx metadata in one CANN call (replaces the
        # legacy torch argsort/bincount/cumsum bucketing + boolean-mask
        # compaction, ~66 kernels/MoE-layer/step and a .item() D2H sync).
        sorted_x, row_idx, group_list, _ = torch_npu.npu_moe_init_routing_v2(
            expert_x,
            expert_idx=topk_ids,
            scale=None,
            active_num=topk_ids.numel(),
            expert_num=self._w1.shape[0],
            active_expert_range=[0, self._w1.shape[0]],
            drop_pad_mode=0,
            expert_tokens_num_type=0,
            expert_tokens_num_flag=True,
            quant_mode=-1,  # int8 dynamic quant only; MXFP8 stays below
            row_idx_type=1,
        )
        # npu_grouped_matmul requires int64 groupList. Zero-length groups
        # (empty experts) are legal; no special-casing needed.
        group_list = group_list.to(torch.int64)

        # Activation MX quantization stays between dispatch and GEMM1
        # (same rows as the legacy x_grouped, unchanged numerics).
        x_fp8, x_scale = torch_npu.npu_dynamic_mx_quant(
            sorted_x, dst_type=torch.float8_e4m3fn
        )
        down_output = self._stepwise_grouped_ffn(x_fp8, x_scale, group_list)

        # npu_moe_init_routing_v2(row_idx_type=1) returns a gather index:
        # row_idx[permuted_row] == original (token, topk) flat row.  But
        # npu_moe_token_unpermute expects a scatter index, i.e.
        # sorted_indices[original_row] == permuted_row.  Feeding the gather
        # index straight through silently reshuffles tokens between rows, so
        # invert the permutation first (same as the bf16 CANN pipeline).
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
        # Fused combine: restore token order, apply topk weights, and sum
        # the topk expert outputs in one CANN call.
        output = torch_npu.npu_moe_token_unpermute(
            permuted_tokens=down_output,
            sorted_indices=scatter_idx,
            probs=topk_weights,
        )
        return CombineForwardPayload(fused_expert_output=output)

    def _stepwise_grouped_ffn(
        self, x_fp8: torch.Tensor, x_scale: torch.Tensor, group_list: torch.Tensor
    ) -> torch.Tensor:
        """GEMM1 -> silu*mul -> MX quant -> GEMM2."""
        upgate = _first(
            torch_npu.npu_grouped_matmul(
                x=[x_fp8],
                weight=[self._w1],
                scale=[self._w1_scale],
                per_token_scale=[x_scale],
                split_item=2,
                group_type=0,
                group_list=group_list,
                group_list_type=0,
                output_dtype=torch.bfloat16,
                scale_dtype=FLOAT8_E8M0FNU_DTYPE,
                per_token_scale_dtype=FLOAT8_E8M0FNU_DTYPE,
            )
        )
        # w1 is up|gate along the last dim.
        up, gate = upgate.chunk(2, dim=-1)
        act = torch.nn.functional.silu(gate) * up
        act_fp8, act_scale = torch_npu.npu_dynamic_mx_quant(
            act, dst_type=torch.float8_e4m3fn
        )
        return _first(
            torch_npu.npu_grouped_matmul(
                x=[act_fp8],
                weight=[self._w2],
                scale=[self._w2_scale],
                per_token_scale=[act_scale],
                split_item=2,
                group_type=0,
                group_list=group_list,
                group_list_type=0,
                output_dtype=torch.bfloat16,
                scale_dtype=FLOAT8_E8M0FNU_DTYPE,
                per_token_scale_dtype=FLOAT8_E8M0FNU_DTYPE,
            )
        )
