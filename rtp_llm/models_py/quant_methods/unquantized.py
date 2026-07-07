from typing import Any, Dict, Optional

import torch

from rtp_llm.models_py.quant_methods.base import (
    FusedMoEMethodBase,
    QuantizeMethodBase,
    register_moe_quant_method,
    register_quant_method,
)


@register_quant_method("none", "")
class UnquantizedLinearMethod(QuantizeMethodBase):

    def create_weights(
        self,
        layer,
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **kwargs,
    ):
        weight = torch.nn.Parameter(
            torch.empty(output_size, input_size, dtype=params_dtype),
            requires_grad=False,
        )
        layer.register_parameter("weight", weight)

    def apply(
        self, layer, x: torch.Tensor, bias: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        return torch.nn.functional.linear(x, layer.weight, bias)

    def process_weights_after_loading(self, layer):
        pass


@register_moe_quant_method("none", "")
class UnquantizedFusedMoEMethod(FusedMoEMethodBase):
    """MoE 未量化（BF16/FP16）方法。

    对齐 vLLM:连「不量化」也是一个 method，使 BaseMoEExperts 像 vLLM 的 FusedMoE
    一样成为「不认识量化」的纯壳。行为与 BaseMoEExperts._init_buffers 的 else 分支
    逐字一致:只建 bf16 的 w13 [E, 2*M_tp, H] / w2 [E, H, M_tp]，无 scale、无后处理、
    不往权重字典加 scale（dispatch_scale / add_weight_tensors 用基类默认 no-op）。
    """

    def __init__(self, quant_config: Any = None):
        self.quant_config = quant_config

    def create_weights(
        self,
        layer,
        num_experts: int,
        hidden_size: int,
        intermediate_size: int,
        params_dtype: torch.dtype,
        **kwargs,
    ):
        layer.w13 = torch.nn.Parameter(
            torch.empty(
                num_experts, 2 * intermediate_size, hidden_size, dtype=params_dtype
            ),
            requires_grad=False,
        )
        layer.w2 = torch.nn.Parameter(
            torch.empty(
                num_experts, hidden_size, intermediate_size, dtype=params_dtype
            ),
            requires_grad=False,
        )

    def process_weights_after_loading(self, layer):
        # BF16/FP16 无需后处理（无 scale 融合 / 在线量化）。
        return None
