from typing import Optional

import torch
from rtp_llm.models_py.quant_methods.base import (
    FusedMoEMethodBase,
    QuantizeMethodBase,
    register_moe_quant_method,
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


@register_moe_quant_method("none")
class UnquantizedFusedMoEMethod(FusedMoEMethodBase):
    def create_weights(
        self,
        layer,
        num_experts: int,
        hidden_size: int,
        intermediate_size: int,
        params_dtype: torch.dtype,
        **kwargs,
    ) -> None:
        layer.register_parameter(
            "w13",
            torch.nn.Parameter(
                torch.empty(
                    num_experts,
                    2 * intermediate_size,
                    hidden_size,
                    dtype=params_dtype,
                ),
                requires_grad=False,
            ),
        )
        layer.register_parameter(
            "w2",
            torch.nn.Parameter(
                torch.empty(
                    num_experts,
                    hidden_size,
                    intermediate_size,
                    dtype=params_dtype,
                ),
                requires_grad=False,
            ),
        )

    def process_weights_after_loading(self, layer) -> None:
        return None

    def dispatch_weight(self, layer, local_id, projection, parameter_name, tensor):
        if parameter_name == "weight" and not tensor.is_floating_point():
            raise TypeError(
                f"Unquantized MoE {projection}.weight must be floating point, "
                f"got {tensor.dtype}"
            )
        return False
