from typing import Optional

import torch

from rtp_llm.models_py.quant_methods.base import (
    QuantizeMethodBase,
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
        **kwargs
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
