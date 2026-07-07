from typing import Dict, Optional

import torch
import torch.nn as nn

from rtp_llm.models_py.layers.linear import ColumnParallelLinear, RowParallelLinear
from rtp_llm.models_py.quant_methods.base import QuantizationConfig


class MMEncoderAttention(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        tp_size: int = 1,
        tp_rank: int = 0,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        bias: bool = True,
        params_dtype: torch.dtype = torch.float16,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        self.qkv = ColumnParallelLinear(
            input_size=hidden_size,
            output_size=3 * hidden_size,
            tp_size=tp_size,
            tp_rank=tp_rank,
            quant_config=quant_config,
            prefix=f"{prefix}.qkv" if prefix else "qkv",
            bias=bias,
            params_dtype=params_dtype,
        )

        self.proj = RowParallelLinear(
            input_size=hidden_size,
            output_size=hidden_size,
            tp_size=tp_size,
            tp_rank=tp_rank,
            quant_config=quant_config,
            prefix=f"{prefix}.proj" if prefix else "proj",
            bias=bias,
            params_dtype=params_dtype,
        )

    def load_weights(self, weights: Dict[str, torch.Tensor]):
        qkv_weights: Dict[str, torch.Tensor] = {}
        proj_weights: Dict[str, torch.Tensor] = {}

        for name, tensor in weights.items():
            if name.startswith("qkv."):
                qkv_weights[name[4:]] = tensor
            elif name.startswith("proj."):
                proj_weights[name[5:]] = tensor
            elif "qkv" in name:
                qkv_weights[name] = tensor
            elif "proj" in name or "out" in name:
                proj_weights[name] = tensor

        if qkv_weights:
            self.qkv.load_weights(qkv_weights)
        if proj_weights:
            self.proj.load_weights(proj_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x)

        head_dim = self.head_dim
        num_heads_per_partition = self.qkv.output_size // (3 * head_dim)
        qkv = qkv.reshape(B, N, 3, num_heads_per_partition, head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        scale = head_dim**-0.5
        attn = (q @ k.transpose(-2, -1)) * scale
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B, N, -1)

        output = self.proj(x)
        return output
