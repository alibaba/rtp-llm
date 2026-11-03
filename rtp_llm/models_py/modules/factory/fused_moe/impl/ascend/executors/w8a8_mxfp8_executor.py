"""Ascend NPU W8A8_MXFP8 MoE executor (W8A8_MXFP8: Weight & Activation FP8 Quantization).

This module is only imported when ``get_device_type() == DeviceType.Ascend``
(see ``factory/fused_moe/__init__.py`` device branch), so a top-level
``import torch_npu`` is allowed here.

Weight layouts after ``PerBlockFp8Weight._postprocess`` (NPU branch):
  * ``moe_w1``: fp8_e4m3fn ``[E, K_hidden, 2N]`` (up|gate order along the last dim)
  * ``moe_w2``: fp8_e4m3fn ``[E, K_inter, N_hidden]``
  * ``moe_s1``: uint8 (E8M0) ``[E, kp1, 2N, 2]`` swizzled pair-split layout
  * ``moe_s2``: uint8 (E8M0) ``[E, kp2, N_hidden, 2]``

Execution path (fully controlled semantics): grouped GEMM1 -> silu*mul ->
npu_dynamic_mx_quant -> grouped GEMM2. Numerically verified against a
per-expert bf16 reference on Ascend950PR (cos_sim ~0.997).

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
        expert_x = payload.expert_x  # bf16 [E, T, K]
        masked_m = payload.expert_tokens_meta.expert_num_tokens  # [E]

        e, t, k = expert_x.shape
        # Vectorized compaction to the grouped-contiguous layout (expert-major).
        mask = (
            torch.arange(t, device=expert_x.device).unsqueeze(0)
            < masked_m.unsqueeze(1)
        )  # [E, T]
        x_grouped = expert_x[mask]  # [sum_tokens, K]
        # aclnnGroupedMatmulV4 requires groupList to be int64.
        group_list = torch.cumsum(masked_m, dim=0).to(torch.int64)

        if x_grouped.shape[0] == 0:
            out = torch.zeros(
                (e, t, self._w2.shape[-1]),
                dtype=expert_x.dtype,
                device=expert_x.device,
            )
            return CombineForwardPayload(fused_expert_output=out)

        # One MX dynamic quantization for the whole grouped batch.
        x_fp8, x_scale = torch_npu.npu_dynamic_mx_quant(
            x_grouped, dst_type=torch.float8_e4m3fn
        )
        down_output = self._stepwise_grouped_ffn(x_fp8, x_scale, group_list)

        # Scatter back to the batched [E, T, hidden] layout.
        out = torch.zeros(
            (e, t, down_output.shape[-1]),
            dtype=down_output.dtype,
            device=down_output.device,
        )
        out[mask] = down_output
        return CombineForwardPayload(fused_expert_output=out)

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
