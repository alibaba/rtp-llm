from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn


class Conv3dLayer(nn.Module):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Tuple[int, int, int],
        stride: Optional[Tuple[int, int, int]] = None,
        padding: Optional[Tuple[int, int, int]] = None,
        bias: bool = True,
        params_dtype: torch.dtype = torch.float16,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        if stride is None:
            stride = kernel_size
        if padding is None:
            padding = (0, 0, 0)

        self.conv = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=bias,
        )
        self.conv = self.conv.to(dtype=params_dtype)

    def load_weights(self, weights: Dict[str, torch.Tensor]):
        for name, tensor in weights.items():
            if "weight" in name and "bias" not in name:
                self.conv.weight.data.copy_(tensor)
            elif "bias" in name:
                if self.conv.bias is not None:
                    self.conv.bias.data.copy_(tensor)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)
