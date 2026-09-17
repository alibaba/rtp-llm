"""Dense and shared-expert MLP used by DeepSeek newloader models."""

import math
from typing import Optional

import torch

from rtp_llm.models_py.layers.linear import (
    MergedColumnParallelLinear,
    RowParallelLinear,
)
from rtp_llm.models_py.module_base import RtpModule
from rtp_llm.models_py.modules.base import FusedSiluAndMul
from rtp_llm.models_py.quant_methods.base import QuantizationConfig

_FP8_BLOCK_QUANT_TYPES = {
    "fp8_block",
    "fp8_block_online",
    "fp8_per_block",
}


def _uses_fp8_block_quant(quant_config: Optional[QuantizationConfig]) -> bool:
    quant_type = "none" if quant_config is None else quant_config.quant_type.lower()
    return quant_type in _FP8_BLOCK_QUANT_TYPES


def _pad_dimension(
    tensor: torch.Tensor,
    *,
    dim: int,
    expected_size: int,
    padded_size: int,
    label: str,
) -> torch.Tensor:
    current_size = tensor.shape[dim]
    if current_size == padded_size:
        return tensor
    if current_size != expected_size:
        raise ValueError(
            f"{label} dimension {dim} must be {expected_size} or "
            f"{padded_size}, got {current_size}"
        )

    pad_shape = list(tensor.shape)
    pad_shape[dim] = padded_size - expected_size
    padding = torch.zeros(pad_shape, dtype=torch.float32, device=tensor.device).to(
        tensor.dtype
    )
    if str(tensor.dtype).startswith("torch.float8_"):
        padded = torch.cat(
            (tensor.view(torch.uint8), padding.view(torch.uint8)), dim=dim
        ).view(tensor.dtype)
    else:
        padded = torch.cat((tensor, padding), dim=dim)
    return padded.contiguous()


class DeepSeekV32MLP(RtpModule):
    """SiLU-gated MLP with TP-safe FP8 padding and reduction semantics."""

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        tp_size: int = 1,
        tp_rank: int = 0,
        quant_config: Optional[QuantizationConfig] = None,
        params_dtype: torch.dtype = torch.bfloat16,
        reduce_output: bool = True,
        prefix: str = "mlp",
    ):
        super().__init__()
        self.intermediate_size = intermediate_size
        self.padded_intermediate_size = intermediate_size
        self.fp8_block_size = None
        if quant_config is not None and _uses_fp8_block_quant(quant_config):
            block_n, block_k = quant_config.fp8_block_size
            alignment = math.lcm(tp_size * block_n, tp_size * block_k)
            self.padded_intermediate_size = (
                (intermediate_size + alignment - 1) // alignment * alignment
            )
            self.fp8_block_size = (block_n, block_k)

        self.act_fn = FusedSiluAndMul()
        self.gate_up_proj = MergedColumnParallelLinear(
            input_size=hidden_size,
            output_size=2 * self.padded_intermediate_size,
            tp_size=tp_size,
            tp_rank=tp_rank,
            quant_config=quant_config,
            prefix=f"{prefix}.gate_up_proj",
            bias=False,
            shard_names=["gate_proj", "up_proj"],
            params_dtype=params_dtype,
        )
        self.down_proj = RowParallelLinear(
            input_size=self.padded_intermediate_size,
            output_size=hidden_size,
            tp_size=tp_size,
            tp_rank=tp_rank,
            quant_config=quant_config,
            prefix=f"{prefix}.down_proj",
            bias=False,
            reduce_output=reduce_output,
            params_dtype=params_dtype,
        )

    def _pad_checkpoint_tensor(self, name: str, tensor: torch.Tensor) -> torch.Tensor:
        if self.padded_intermediate_size == self.intermediate_size:
            return tensor

        if name in ("gate_proj.weight", "up_proj.weight"):
            return _pad_dimension(
                tensor,
                dim=0,
                expected_size=self.intermediate_size,
                padded_size=self.padded_intermediate_size,
                label=name,
            )
        if name == "down_proj.weight":
            return _pad_dimension(
                tensor,
                dim=1,
                expected_size=self.intermediate_size,
                padded_size=self.padded_intermediate_size,
                label=name,
            )
        if self.fp8_block_size is None:
            raise RuntimeError(
                "padded DeepSeek MLP weights require an FP8 block layout"
            )
        block_n, block_k = self.fp8_block_size
        if name in (
            "gate_proj.weight_scale_inv",
            "up_proj.weight_scale_inv",
        ):
            return _pad_dimension(
                tensor,
                dim=0,
                expected_size=math.ceil(self.intermediate_size / block_n),
                padded_size=self.padded_intermediate_size // block_n,
                label=name,
            )
        if name == "down_proj.weight_scale_inv":
            return _pad_dimension(
                tensor,
                dim=1,
                expected_size=math.ceil(self.intermediate_size / block_k),
                padded_size=self.padded_intermediate_size // block_k,
                label=name,
            )
        return tensor

    def load_weights(self, weights) -> None:
        items = weights.items() if isinstance(weights, dict) else weights
        for name, tensor in items:
            super().load_weights({name: self._pad_checkpoint_tensor(name, tensor)})

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act_fn(self.gate_up_proj(x)))
