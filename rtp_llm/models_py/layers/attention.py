from typing import Dict, Optional

import torch
import torch.nn as nn

from rtp_llm.models_py.layers.linear import ColumnParallelLinear, RowParallelLinear
from rtp_llm.models_py.quant_methods.base import QuantizationConfig


class _FusedQKVColumnParallelLinear(ColumnParallelLinear):
    def __init__(self, hidden_size: int, *args, **kwargs):
        self.hidden_size = hidden_size
        super().__init__(*args, **kwargs)

    def _split_fused_qkv_rows(
        self, tensor: torch.Tensor, local_rows: int
    ) -> Optional[torch.Tensor]:
        full_rows = local_rows * self.tp_size
        if tensor.shape[0] != full_rows:
            return None
        if full_rows % 3 != 0 or local_rows % 3 != 0:
            raise ValueError(
                f"fused qkv rows must split evenly across Q/K/V and TP, got "
                f"full_rows={full_rows}, local_rows={local_rows}, tp_size={self.tp_size}"
            )
        full_rows_per_proj = full_rows // 3
        local_rows_per_proj = local_rows // 3
        chunks = []
        for offset in (0, full_rows_per_proj, 2 * full_rows_per_proj):
            start = offset + self.tp_rank * local_rows_per_proj
            chunks.append(tensor.narrow(0, start, local_rows_per_proj))
        return torch.cat(chunks, dim=0).contiguous()

    def _split_weight(self, tensor: torch.Tensor, dim: int = 0) -> torch.Tensor:
        if self.tp_size <= 1 or dim != 0:
            return super()._split_weight(tensor, dim)

        if (
            tensor.dim() == 2
            and tensor.shape[0] == self.hidden_size
            and tensor.shape[1] == 3 * self.hidden_size
        ):
            tensor = tensor.t().contiguous()

        split = self._split_fused_qkv_rows(tensor, self.output_size_per_partition)
        if split is not None:
            return split

        weight_scale_inv = getattr(self, "weight_scale_inv", None)
        if weight_scale_inv is not None:
            split = self._split_fused_qkv_rows(tensor, weight_scale_inv.shape[0])
            if split is not None:
                return split

        return super()._split_weight(tensor, dim)


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
        if num_heads <= 0:
            raise ValueError(f"num_heads must be positive, got {num_heads}")
        if tp_size <= 0:
            raise ValueError(f"tp_size must be positive, got {tp_size}")
        if tp_rank < 0 or tp_rank >= tp_size:
            raise ValueError(
                f"tp_rank must satisfy 0 <= tp_rank < tp_size, got "
                f"tp_rank={tp_rank}, tp_size={tp_size}"
            )
        if hidden_size % num_heads != 0:
            raise ValueError(
                f"hidden_size ({hidden_size}) must be divisible by num_heads ({num_heads})"
            )
        if num_heads % tp_size != 0:
            raise ValueError(
                f"num_heads ({num_heads}) must be divisible by tp_size ({tp_size})"
            )

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        self.qkv = _FusedQKVColumnParallelLinear(
            hidden_size=hidden_size,
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
