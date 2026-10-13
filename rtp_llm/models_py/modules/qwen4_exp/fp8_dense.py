"""Graph-compatible consumers for hard-quantized dense Qwen parameters."""

import copy

import torch
from torch import nn
from torch.nn import functional as F

from rtp_llm.models.qwen4_exp.full_fp8_weight import FP8_STORAGE_SCALE_SUFFIX


def dequantize_parameter(weights, name):
    value = weights[name]
    scale = weights.get(name + FP8_STORAGE_SCALE_SUFFIX)
    if scale is None:
        return value
    return (value.float() * scale).to(torch.bfloat16)


def compute_weight_view(weights):
    """Norms, biases and convolution kernels compute in BF16; matrices stay FP8."""
    view = copy.copy(weights)
    view._fp8_storage_owner = weights

    def convert(items):
        return {
            name: (
                dequantize_parameter(items, name)
                if value.ndim != 2 and name + FP8_STORAGE_SCALE_SUFFIX in items
                else value
            )
            for name, value in items.items()
        }

    view.weights = [convert(items) for items in weights.weights]
    view.global_weights = convert(weights.global_weights)
    return view


class Qwen4Fp8Linear(nn.Module):
    def __init__(self, weight, scale, bias=None):
        super().__init__()
        self.weight = weight
        self.scale = scale.to(device=weight.device, dtype=torch.float32).reshape(1)
        self.bias = bias
        self.use_scaled_mm = (
            weight.is_cuda and weight.shape[0] % 16 == 0 and weight.shape[1] % 16 == 0
        )
        # Tiny unaligned gates use a BF16 compute copy, made before capture.
        self.compute_weight = (
            None
            if self.use_scaled_mm
            else (weight.float() * self.scale).to(torch.bfloat16)
        )

    def forward(self, x):
        if self.compute_weight is not None:
            return F.linear(x, self.compute_weight, self.bias)
        rows = x.reshape(-1, x.shape[-1])
        if rows.shape[0] == 0:
            return x.new_empty((*x.shape[:-1], self.weight.shape[0]))
        from rtp_llm.models_py.kernels.cuda.fp8_kernel import (
            scaled_fp8_per_tensor_quant,
        )

        quantized, scale = scaled_fp8_per_tensor_quant(rows.contiguous())
        padding = (-rows.shape[0]) % 16
        if padding:
            quantized = F.pad(quantized, (0, 0, 0, padding))
        out = torch._scaled_mm(
            quantized,
            self.weight.t(),
            scale_a=scale,
            scale_b=self.scale,
            out_dtype=torch.bfloat16,
        )
        if isinstance(out, tuple):
            out = out[0]
        out = out[: rows.shape[0]].reshape(*x.shape[:-1], self.weight.shape[0])
        if self.bias is not None:
            out = out + self.bias
        return out.to(x.dtype)


def dense_projection(x, weight, projection):
    return projection(x) if projection is not None else F.linear(x, weight)


def make_projection(weight, scale):
    return None if scale is None else Qwen4Fp8Linear(weight, scale)
