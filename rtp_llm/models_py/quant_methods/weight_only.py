from typing import Any, Optional

import torch
from torch import nn

from rtp_llm.models_py.quant_methods.base import LinearMethodBase, register_quant_method


@register_quant_method(
    "weight_only_per_col", "weight_only_per_group", "w4a8_int4_per_channel"
)
class WeightOnlyLinearMethod(LinearMethodBase):
    def __init__(self, quant_config: Any = None):
        self.quant_config = quant_config
        self.source_config = getattr(quant_config, "source_config", None)

    def create_weights(
        self,
        layer,
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **kwargs,
    ):
        layer.register_parameter(
            "weight",
            nn.Parameter(
                torch.empty(0, dtype=torch.int8),
                requires_grad=False,
            ),
        )
        layer.register_parameter(
            "scales",
            nn.Parameter(
                torch.empty(0, dtype=params_dtype),
                requires_grad=False,
            ),
        )
        layer.register_parameter(
            "zeros",
            nn.Parameter(
                torch.empty(0, dtype=params_dtype),
                requires_grad=False,
            ),
        )

    def process_weights_after_loading(self, layer):
        pass

    def apply(
        self, layer, x: torch.Tensor, bias: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        w = layer.weight.to(x.dtype)
        if hasattr(layer, "zeros") and layer.zeros.numel() > 0:
            zeros = layer.zeros.to(x.dtype)
            if zeros.dim() == 1:
                w = w - zeros.unsqueeze(1)
            elif zeros.dim() == 2:
                w = w - zeros.repeat_interleave(w.shape[1] // zeros.shape[1], dim=1)
        if hasattr(layer, "scales") and layer.scales.numel() > 0:
            scales = layer.scales.to(x.dtype)
            if scales.dim() == 1:
                w = w * scales.unsqueeze(1)
            elif scales.dim() == 2:
                w = w * scales.repeat_interleave(w.shape[1] // scales.shape[1], dim=1)
        return torch.nn.functional.linear(x, w, bias)
