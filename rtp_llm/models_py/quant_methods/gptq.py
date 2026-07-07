from typing import Any, Optional

import torch
from torch import nn

from rtp_llm.models_py.quant_methods.base import LinearMethodBase, register_quant_method


@register_quant_method("gptq")
class GPTQLinearMethod(LinearMethodBase):
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
            "qweight",
            nn.Parameter(
                torch.empty(0, dtype=torch.int32),
                requires_grad=False,
            ),
        )
        layer.register_parameter(
            "qzeros",
            nn.Parameter(
                torch.empty(0, dtype=torch.int32),
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
        layer.register_buffer(
            "g_idx",
            torch.empty(0, dtype=torch.int32),
            persistent=True,
        )

    def process_weights_after_loading(self, layer):
        pass

    def apply(
        self, layer, x: torch.Tensor, bias: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        qweight = layer.qweight
        scales = layer.scales.to(x.dtype)
        qzeros = layer.qzeros

        K_packed, N = qweight.shape
        K = K_packed * 8

        shifts = torch.arange(0, 32, 4, device=qweight.device, dtype=torch.int32)
        unpacked = (qweight.unsqueeze(-1) >> shifts.view(1, 1, 8)) & 0xF
        unpacked = unpacked.view(K, N).to(x.dtype)

        num_groups = qzeros.shape[0]
        if qzeros.numel() > 0:
            unpacked_zeros = (qzeros.unsqueeze(-1) >> shifts.view(1, 1, 8)) & 0xF
            unpacked_zeros = unpacked_zeros.view(num_groups, N).to(x.dtype)
            unpacked_zeros = unpacked_zeros + 1
        else:
            unpacked_zeros = torch.zeros(
                num_groups, N, device=qweight.device, dtype=x.dtype
            )

        group_size = K // num_groups if num_groups > 0 else K

        if (
            hasattr(layer, "g_idx")
            and layer.g_idx is not None
            and layer.g_idx.numel() > 0
        ):
            g_idx = layer.g_idx.to(torch.long)
        else:
            g_idx = (
                torch.arange(K, device=qweight.device, dtype=torch.long) // group_size
            )

        w = (unpacked - unpacked_zeros[g_idx]) * scales[g_idx]
        return torch.nn.functional.linear(x, w.t(), bias)
