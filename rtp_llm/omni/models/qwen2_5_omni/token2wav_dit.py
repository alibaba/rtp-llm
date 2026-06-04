import math
from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn as nn


@dataclass
class OmniDiTConfig:
    depth: int = 22
    dim: int = 1024
    heads: int = 16
    head_dim: int = 64
    ff_mult: int = 2
    mel_dim: int = 80
    num_embeds: int = 8193
    emb_dim: int = 512
    dropout: float = 0.1

    @classmethod
    def from_dict(cls, d: Dict) -> "OmniDiTConfig":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        emb = math.log(10000) / (half - 1)
        emb = torch.exp(
            torch.arange(half, device=t.device, dtype=torch.float32) * -emb
        )
        emb = t.unsqueeze(-1) * emb.unsqueeze(0)
        return torch.cat([emb.sin(), emb.cos()], dim=-1)


class DiTBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        heads: int,
        head_dim: int,
        ff_mult: int,
        dropout: float,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            dim, heads, dropout=dropout, batch_first=True
        )
        self.norm2 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, dim * ff_mult),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * ff_mult, dim),
            nn.Dropout(dropout),
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, 6 * dim),
        )

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        mod = self.adaLN_modulation(cond).unsqueeze(1)
        gamma1, beta1, alpha1, gamma2, beta2, alpha2 = mod.chunk(6, dim=-1)

        h = self.norm1(x) * (1 + gamma1) + beta1
        h, _ = self.attn(h, h, h)
        x = x + alpha1 * h

        h = self.norm2(x) * (1 + gamma2) + beta2
        h = self.ff(h)
        x = x + alpha2 * h
        return x


class OmniDiT(nn.Module):
    def __init__(self, config: OmniDiTConfig):
        super().__init__()
        self.config = config
        self.input_proj = nn.Linear(config.mel_dim, config.dim)
        self.codec_embed = nn.Embedding(config.num_embeds, config.emb_dim)
        self.codec_proj = nn.Linear(config.emb_dim, config.dim)
        self.time_embed = nn.Sequential(
            SinusoidalPosEmb(config.dim),
            nn.Linear(config.dim, config.dim),
            nn.SiLU(),
            nn.Linear(config.dim, config.dim),
        )
        self.blocks = nn.ModuleList(
            [
                DiTBlock(
                    config.dim,
                    config.heads,
                    config.head_dim,
                    config.ff_mult,
                    config.dropout,
                )
                for _ in range(config.depth)
            ]
        )
        self.final_norm = nn.LayerNorm(config.dim)
        self.output_proj = nn.Linear(config.dim, config.mel_dim)

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        codec_ids: torch.Tensor,
        spk_emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        h = self.input_proj(x)
        codec_h = self.codec_proj(self.codec_embed(codec_ids))
        h = h + codec_h
        t_emb = self.time_embed(t)

        for block in self.blocks:
            h = block(h, t_emb)

        h = self.final_norm(h)
        return self.output_proj(h)
