from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from rtp_llm.models_py.layers.conv import Conv3dLayer
from rtp_llm.models_py.layers.linear import ColumnParallelLinear, RowParallelLinear
from rtp_llm.models_py.layers.norm import LayerNorm
from rtp_llm.models_py.module_base import rtp_module
from rtp_llm.models_py.quant_methods.base import QuantizationConfig


@rtp_module
class Qwen2VisionPatchEmbed(nn.Module):

    def __init__(
        self,
        in_channels: int = 3,
        hidden_size: int = 1280,
        patch_size: int = 14,
        temporal_patch_size: int = 2,
        params_dtype: torch.dtype = torch.float16,
    ):
        super().__init__()
        self.proj = Conv3dLayer(
            in_channels=in_channels,
            out_channels=hidden_size,
            kernel_size=(temporal_patch_size, patch_size, patch_size),
            stride=(temporal_patch_size, patch_size, patch_size),
            bias=False,
            params_dtype=params_dtype,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        B, C = x.shape[:2]
        x = x.flatten(2).transpose(1, 2)
        return x


@rtp_module
class Qwen2VisionMLP(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        tp_size: int = 1,
        tp_rank: int = 0,
        quant_config: Optional[QuantizationConfig] = None,
        params_dtype: torch.dtype = torch.float16,
    ):
        super().__init__()
        self.fc1 = ColumnParallelLinear(
            input_size=hidden_size,
            output_size=intermediate_size,
            tp_size=tp_size,
            tp_rank=tp_rank,
            quant_config=quant_config,
            prefix="fc1",
            bias=True,
            params_dtype=params_dtype,
        )
        self.fc2 = RowParallelLinear(
            input_size=intermediate_size,
            output_size=hidden_size,
            tp_size=tp_size,
            tp_rank=tp_rank,
            quant_config=quant_config,
            prefix="fc2",
            bias=True,
            params_dtype=params_dtype,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = torch.nn.functional.gelu(x, approximate="tanh")
        x = self.fc2(x)
        return x


@rtp_module
class Qwen2VisionAttn(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        tp_size: int = 1,
        tp_rank: int = 0,
        quant_config: Optional[QuantizationConfig] = None,
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
            prefix="qkv",
            bias=True,
            params_dtype=params_dtype,
        )
        self.proj = RowParallelLinear(
            input_size=hidden_size,
            output_size=hidden_size,
            tp_size=tp_size,
            tp_rank=tp_rank,
            quant_config=quant_config,
            prefix="proj",
            bias=True,
            params_dtype=params_dtype,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x)
        num_heads_per_tp = self.num_heads // max(1, self.qkv.tp_size)
        qkv = qkv.reshape(B, N, 3, num_heads_per_tp, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        scale = self.head_dim**-0.5
        attn = (q @ k.transpose(-2, -1)) * scale
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
        x = self.proj(x)
        return x


@rtp_module
class Qwen2VisionBlock(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        intermediate_size: int,
        layer_idx: int = 0,
        tp_size: int = 1,
        tp_rank: int = 0,
        quant_config: Optional[QuantizationConfig] = None,
        params_dtype: torch.dtype = torch.float16,
    ):
        super().__init__()
        self.norm1 = LayerNorm(hidden_size, params_dtype=params_dtype)
        self.attn = Qwen2VisionAttn(
            hidden_size=hidden_size,
            num_heads=num_heads,
            tp_size=tp_size,
            tp_rank=tp_rank,
            quant_config=quant_config,
            params_dtype=params_dtype,
        )
        self.norm2 = LayerNorm(hidden_size, params_dtype=params_dtype)
        self.mlp = Qwen2VisionMLP(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            tp_size=tp_size,
            tp_rank=tp_rank,
            quant_config=quant_config,
            params_dtype=params_dtype,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


@rtp_module
class Qwen2VisionPatchMerger(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        output_size: int,
        spatial_merge_size: int = 2,
        tp_size: int = 1,
        tp_rank: int = 0,
        quant_config: Optional[QuantizationConfig] = None,
        params_dtype: torch.dtype = torch.float16,
    ):
        super().__init__()
        merge_input_size = hidden_size * (spatial_merge_size**2)
        self.ln_q = LayerNorm(hidden_size, params_dtype=params_dtype)
        self.fc1 = ColumnParallelLinear(
            input_size=merge_input_size,
            output_size=output_size,
            tp_size=tp_size,
            tp_rank=tp_rank,
            quant_config=quant_config,
            prefix="fc1",
            bias=True,
            params_dtype=params_dtype,
        )
        self.fc2 = ColumnParallelLinear(
            input_size=output_size,
            output_size=output_size,
            tp_size=tp_size,
            tp_rank=tp_rank,
            quant_config=quant_config,
            prefix="fc2",
            bias=True,
            params_dtype=params_dtype,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.ln_q(x)
        x = self.fc1(x)
        x = torch.nn.functional.gelu(x, approximate="tanh")
        x = self.fc2(x)
        return x


@rtp_module
class Qwen2VisionTransformer(nn.Module):

    def __init__(self, vit_config: dict, load_config: Any):
        super().__init__()
        hidden_size = vit_config.get("hidden_size", 1280)
        num_heads = vit_config.get("num_heads", 16)
        num_layers = vit_config.get("num_layers", 32)
        intermediate_size = vit_config.get("intermediate_size", 5120)
        patch_size = vit_config.get("patch_size", 14)
        temporal_patch_size = vit_config.get("temporal_patch_size", 2)
        in_channels = vit_config.get("in_channels", 3)
        spatial_merge_size = vit_config.get("spatial_merge_size", 2)

        tp_size = getattr(load_config, "tp_size", 1)
        tp_rank = getattr(load_config, "tp_rank", 0)
        quant_config = getattr(load_config, "quant_config", None)
        params_dtype = getattr(load_config, "compute_dtype", torch.float16)

        self.patch_embed = Qwen2VisionPatchEmbed(
            in_channels=in_channels,
            hidden_size=hidden_size,
            patch_size=patch_size,
            temporal_patch_size=temporal_patch_size,
            params_dtype=params_dtype,
        )

        self.blocks = nn.ModuleList(
            [
                Qwen2VisionBlock(
                    hidden_size=hidden_size,
                    num_heads=num_heads,
                    intermediate_size=intermediate_size,
                    layer_idx=i,
                    tp_size=tp_size,
                    tp_rank=tp_rank,
                    quant_config=quant_config,
                    params_dtype=params_dtype,
                )
                for i in range(num_layers)
            ]
        )

        llm_hidden_size = vit_config.get("llm_hidden_size", hidden_size)
        self.merger = Qwen2VisionPatchMerger(
            hidden_size=hidden_size,
            output_size=llm_hidden_size,
            spatial_merge_size=spatial_merge_size,
            tp_size=tp_size,
            tp_rank=tp_rank,
            quant_config=quant_config,
            params_dtype=params_dtype,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x)
        for block in self.blocks:
            x = block(x)
        x = self.merger(x)
        return x
