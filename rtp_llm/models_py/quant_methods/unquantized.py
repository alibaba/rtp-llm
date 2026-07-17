from typing import Optional

import torch

from rtp_llm.models_py.quant_methods.base import (
    QuantizeMethodBase,
    register_quant_method,
)


@register_quant_method("none")
class UnquantizedLinearMethod(QuantizeMethodBase):
    def create_weights(
        self,
        layer,
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
    ) -> None:
        backing = torch.empty(input_size, output_size, dtype=params_dtype)
        layer.register_parameter(
            "weight",
            torch.nn.Parameter(
                backing.T,
                requires_grad=False,
            ),
        )

    def apply(
        self, layer, x: torch.Tensor, bias: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        return torch.nn.functional.linear(x, layer.weight, bias)

    def process_weights_after_loading(self, layer) -> None:
        expected_stride = (1, layer.weight.shape[0])
        if layer.weight.ndim != 2 or layer.weight.stride() != expected_stride:
            raise RuntimeError(
                f"Unquantized weight has unexpected stride {layer.weight.stride()}; "
                f"expected {expected_stride}"
            )
