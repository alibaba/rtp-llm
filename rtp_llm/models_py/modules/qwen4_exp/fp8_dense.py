"""Graph-compatible consumers for hard-quantized dense Qwen parameters."""

import copy
import os

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
        if (
            os.environ.get("RTP_LLM_QWEN4_FP8_HC_GEMV_EXPERIMENTAL", "0") == "1"
            and rows.shape[0] <= 4
            and tuple(self.weight.shape) == (10240, 320)
        ):
            from rtp_llm.models_py.modules.qwen4_exp.fp8_dense_gemv_triton import (
                fp8_hc_gemv,
                is_supported,
            )

            if is_supported(rows, self.weight, self.scale):
                out = fp8_hc_gemv(rows, self.weight, self.scale).reshape(
                    *x.shape[:-1], self.weight.shape[0]
                )
                return out if self.bias is None else (out + self.bias).to(x.dtype)
        from rtp_llm.models_py.kernels.cuda.fp8_kernel import (
            scaled_fp8_per_tensor_quant,
        )

        padding = (-rows.shape[0]) % 16
        padded_output = None
        if (
            padding
            and os.environ.get("RTP_LLM_QWEN4_FP8_PADDED_QUANT_EXPERIMENTAL", "0")
            == "1"
        ):
            padded_output = torch.empty(
                (rows.shape[0] + padding, rows.shape[1]),
                device=rows.device,
                dtype=torch.float8_e4m3fn,
            )
        quantized, scale = scaled_fp8_per_tensor_quant(
            rows.contiguous(), output=padded_output
        )
        if padding and padded_output is None:
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
