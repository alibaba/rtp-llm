from dataclasses import dataclass, field
from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class BigVGANConfig:
    mel_dim: int = 80
    upsample_rates: List[int] = field(default_factory=lambda: [5, 3, 2, 2, 2, 2])
    upsample_initial_channel: int = 1536
    upsample_kernel_sizes: List[int] = field(
        default_factory=lambda: [11, 7, 4, 4, 4, 4]
    )
    resblock_kernel_sizes: List[int] = field(default_factory=lambda: [3, 7, 11])
    resblock_dilation_sizes: List[List[int]] = field(
        default_factory=lambda: [[1, 3, 5], [1, 3, 5], [1, 3, 5]]
    )
    use_bias_at_final: bool = False

    @classmethod
    def from_dict(cls, d: Dict) -> "BigVGANConfig":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


class ResBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int, dilations: List[int]):
        super().__init__()
        self.convs1 = nn.ModuleList()
        self.convs2 = nn.ModuleList()
        for d in dilations:
            padding = (kernel_size * d - d) // 2
            self.convs1.append(
                nn.Conv1d(
                    channels,
                    channels,
                    kernel_size,
                    dilation=d,
                    padding=padding,
                )
            )
            self.convs2.append(
                nn.Conv1d(
                    channels,
                    channels,
                    kernel_size,
                    dilation=1,
                    padding=(kernel_size - 1) // 2,
                )
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for c1, c2 in zip(self.convs1, self.convs2):
            xt = F.leaky_relu(x, 0.1)
            xt = c1(xt)
            xt = F.leaky_relu(xt, 0.1)
            xt = c2(xt)
            x = xt + x
        return x


class BigVGAN(nn.Module):
    def __init__(self, config: BigVGANConfig):
        super().__init__()
        self.config = config
        ch = config.upsample_initial_channel

        self.conv_pre = nn.Conv1d(config.mel_dim, ch, 7, padding=3)

        self.ups = nn.ModuleList()
        self.resblocks = nn.ModuleList()

        for i, (u, k) in enumerate(
            zip(config.upsample_rates, config.upsample_kernel_sizes)
        ):
            out_ch = ch // 2
            self.ups.append(
                nn.ConvTranspose1d(
                    ch,
                    out_ch,
                    k,
                    stride=u,
                    padding=(k - u) // 2,
                )
            )
            for j, (rk, rd) in enumerate(
                zip(config.resblock_kernel_sizes, config.resblock_dilation_sizes)
            ):
                self.resblocks.append(ResBlock(out_ch, rk, rd))
            ch = out_ch

        self.conv_post = nn.Conv1d(
            ch, 1, 7, padding=3, bias=config.use_bias_at_final
        )
        self.num_kernels = len(config.resblock_kernel_sizes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_pre(x)
        for i, up in enumerate(self.ups):
            x = F.leaky_relu(x, 0.1)
            x = up(x)
            xs = None
            for j in range(self.num_kernels):
                idx = i * self.num_kernels + j
                if xs is None:
                    xs = self.resblocks[idx](x)
                else:
                    xs += self.resblocks[idx](x)
            x = xs / self.num_kernels
        x = F.leaky_relu(x)
        x = self.conv_post(x)
        x = torch.tanh(x)
        return x
