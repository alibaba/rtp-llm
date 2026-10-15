"""Graph-compatible consumers for hard-quantized dense Qwen parameters."""

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


class Qwen4Fp8Linear(nn.Module):
    def __init__(self, weight, scale, bias=None, *, role="PDFUSION"):
        super().__init__()
        self.weight = weight
        self.scale = scale.to(device=weight.device, dtype=torch.float32).reshape(1)
        self.bias = bias
        self.role = str(role).rsplit(".", 1)[-1]
        self.use_scaled_mm = (
            weight.is_cuda and weight.shape[0] % 16 == 0 and weight.shape[1] % 16 == 0
        )
        # Tiny unaligned gates use a BF16 compute copy, made before capture.
        self.compute_weight = (
            None
            if self.use_scaled_mm
            else (weight.float() * self.scale).to(torch.bfloat16)
        )

    def maybe_cache_quant_scale(self, max_len: int) -> None:
        # Input quantization is dynamic; no persistent per-token scale is needed.
        pass

    def forward_with_bias_gelu(self, x):
        return F.gelu(self.forward(x))

    def forward(self, x, *, input_scale=None):
        if self.compute_weight is not None:
            return F.linear(x, self.compute_weight, self.bias)
        rows = x.reshape(-1, x.shape[-1])
        if rows.shape[0] == 0:
            return x.new_empty((*x.shape[:-1], self.weight.shape[0]))
        # The measured SM120 HC up-projection domain uses A16W8 GEMV.
        # Larger batches, other HC shapes and other devices use scaled MM.
        if (
            self.role in ("DECODE", "PDFUSION")
            and rows.shape[0] <= 4
            and tuple(self.weight.shape) == (10240, 320)
        ):
            from .fp8_dense_gemv_triton import fp8_hc_gemv, is_supported

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
        if padding:
            padded_output = torch.empty(
                (rows.shape[0] + padding, rows.shape[1]),
                device=rows.device,
                dtype=torch.float8_e4m3fn,
            )
        quantized, scale = scaled_fp8_per_tensor_quant(
            rows.contiguous(), scale=input_scale, output=padded_output
        )
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


def make_projection(weight, scale, *, role="PDFUSION"):
    return None if scale is None else Qwen4Fp8Linear(weight, scale, role=role)
