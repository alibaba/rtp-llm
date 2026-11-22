"""Graph-compatible consumers for per-tensor FP8 parameters."""

from contextvars import ContextVar

import torch
from torch import nn
from torch.nn import functional as F

from rtp_llm.models_py.kernels.cuda.fp8_input_quant import (
    _quantize_projection_input as _quantize_cuda_projection_input,
)
from rtp_llm.utils.fp8_constants import FP8_STORAGE_SCALE_SUFFIX

_prequantized_fp8_input: ContextVar = ContextVar("_prequantized_fp8_input", default=None)


def dequantize_parameter(weights, name):
    value = weights[name]
    scale = weights.get(name + FP8_STORAGE_SCALE_SUFFIX)
    if scale is None:
        return value
    return (value.float() * scale).to(torch.bfloat16)


def _quantize_projection_input(rows, *, role, scale=None, output=None):
    if torch.version.hip is not None:
        from .rocm_fp8 import quantize_input

        return quantize_input(rows, scale=scale, output=output)
    return _quantize_cuda_projection_input(rows, role=role, scale=scale, output=output)


class Fp8Linear(nn.Module):
    def __init__(self, weight, scale, bias=None, *, role="PDFUSION"):
        super().__init__()
        self.weight = weight
        self.scale = scale.to(device=weight.device, dtype=torch.float32).reshape(1)
        self.bias = bias
        self.role = str(role).rsplit(".", 1)[-1]
        self.use_scaled_mm = (
            weight.is_cuda and weight.shape[0] % 16 == 0 and weight.shape[1] % 16 == 0
        )
        if self.use_scaled_mm and torch.version.hip is not None:
            from .rocm_fp8 import convert_fn_weight, native_fp8_dtype

            self.weight, self.scale = convert_fn_weight(
                self.weight, self.scale, native_fp8_dtype(weight.device)
            )
        self.unit_scale = torch.ones_like(self.scale) if self.use_scaled_mm else None
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

    def _run_quantized(self, quantized, scale, shape, rows):
        out = torch._scaled_mm(
            quantized,
            self.weight.t(),
            scale_a=scale,
            scale_b=self.scale,
            out_dtype=torch.bfloat16,
        )
        if isinstance(out, tuple):
            out = out[0]
        out = out[:rows].reshape(*shape, self.weight.shape[0])
        if self.bias is not None:
            out = out + self.bias
        return out

    def forward_silu_and_mul(self, gate_up):
        """Consume a gated projection with fused activation and FP8 quantization."""
        if (
            self.compute_weight is None
            and gate_up.ndim == 2
            and (
                (
                    self.role in ("PREFILL", "PDFUSION")
                    and 4093 <= gate_up.shape[0] <= 16384
                )
                or (
                    self.role in ("DECODE", "PDFUSION")
                    and gate_up.shape[0] in (20, 24, 28, 32)
                    and self.weight.shape[1] == 640
                )
            )
            and gate_up.shape[1] == self.weight.shape[1] * 2
            and gate_up.is_cuda
            and torch.version.hip is None
        ):
            from rtp_llm.models_py.triton_kernels.common.silu_fp8_quant import (
                try_silu_mul_quant,
            )

            result = try_silu_mul_quant(gate_up)
            if result is not None:
                return self._run_quantized(
                    *result, gate_up.shape[:-1], gate_up.shape[0]
                ).to(gate_up.dtype)
        if not (
            torch.version.hip is None
            and self.compute_weight is None
            and self.weight.dtype == torch.float8_e4m3fn
            and self.role in ("DECODE", "PDFUSION")
            and gate_up.ndim == 2
            and gate_up.is_cuda
            and (
                (
                    gate_up.shape[0] in (4, 8, 12, 16)
                    and gate_up.shape[1] == 1280
                    and self.weight.shape[1] == 640
                )
                or (
                    1 <= gate_up.shape[0] <= 256
                    and gate_up.shape[1] == 640
                    and tuple(self.weight.shape) == (2560, 320)
                    and torch.cuda.get_device_capability(gate_up.device) == (9, 0)
                )
            )
            and gate_up.dtype == torch.bfloat16
            and gate_up.is_contiguous()
        ):
            return None
        from rtp_llm.models_py.triton_kernels.common.fp8_small_quant import (
            try_silu_mul_fp8_quant,
        )

        rows, width = gate_up.shape[0], self.weight.shape[1]
        padded = None
        if rows % 16:
            padded = torch.empty(
                ((rows + 15) // 16 * 16, width),
                device=gate_up.device,
                dtype=self.weight.dtype,
            )
        result = try_silu_mul_fp8_quant(gate_up, output=padded)
        if result is None:
            return None
        return self._run_quantized(*result, gate_up.shape[:-1], rows).to(gate_up.dtype)

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
            from rtp_llm.models_py.triton_kernels.gated_residual.fp8_gemv import (
                try_fp8_hc_gemv,
            )

            out = try_fp8_hc_gemv(rows, self.weight, self.scale)
            if out is not None:
                out = out.reshape(*x.shape[:-1], self.weight.shape[0])
                return out if self.bias is None else (out + self.bias).to(x.dtype)
        if input_scale is None:
            cached = _prequantized_fp8_input.get()
            if (
                cached is not None
                and x.data_ptr() == cached[0]
                and tuple(x.shape) == cached[1]
                and x.stride() == cached[2]
                and cached[3].device == x.device
                and cached[3].dtype == self.weight.dtype
            ):
                return self._run_quantized(
                    cached[3], cached[4], x.shape[:-1], rows.shape[0]
                ).to(x.dtype)
        padding = (-rows.shape[0]) % 16
        padded_output = None
        if padding:
            padded_output = torch.empty(
                (rows.shape[0] + padding, rows.shape[1]),
                device=rows.device,
                dtype=self.weight.dtype,
            )
        quantized, scale = _quantize_projection_input(
            rows.contiguous(), role=self.role, scale=input_scale, output=padded_output
        )
        return self._run_quantized(quantized, scale, x.shape[:-1], rows.shape[0]).to(
            x.dtype
        )


def prequantize_fp8_linear_input(linear, x):
    """Quantize once on the caller stream for multiple identical-input consumers."""
    if not (
        type(linear) is Fp8Linear
        and linear.use_scaled_mm
        and linear.compute_weight is None
        and x.is_cuda
        and x.dtype == torch.bfloat16
        and x.ndim == 2
        and x.shape[0] > 0
        and x.shape[1] == linear.weight.shape[1]
        and x.is_contiguous()
    ):
        return None
    rows = x.shape[0]
    padding = (-rows) % 16
    padded_output = (
        torch.empty(
            (rows + padding, x.shape[1]),
            device=x.device,
            dtype=linear.weight.dtype,
        )
        if padding
        else None
    )
    quantized, scale = _quantize_projection_input(
        x, role=linear.role, scale=None, output=padded_output
    )
    return (x.data_ptr(), tuple(x.shape), x.stride(), quantized, scale)


def dense_projection(x, weight, projection):
    return projection(x) if projection is not None else F.linear(x, weight)


def make_projection(weight, scale, *, role="PDFUSION"):
    return None if scale is None else Fp8Linear(weight, scale, role=role)


def project_fp8_pair(x, first, second):
    """Share identical input quantization, preserving both weight scale domains.

    The caller chooses the page-sized input; this never changes its amax domain
    or either GEMM's row count, weights, accumulation, or BF16 output boundary.
    """
    if not (
        type(first) is Fp8Linear
        and type(second) is Fp8Linear
        and first.use_scaled_mm
        and second.use_scaled_mm
        and first.bias is None
        and second.bias is None
        and first.weight.shape[1] == second.weight.shape[1] == x.shape[-1]
        and first.weight.dtype == second.weight.dtype
        and first.weight.device == second.weight.device == x.device
        and x.is_cuda
        and x.dtype == torch.bfloat16
        and not any(
            "forward" in projection.__dict__
            or projection._forward_pre_hooks
            or projection._forward_hooks
            for projection in (first, second)
        )
        and not torch.nn.modules.module._global_forward_pre_hooks
        and not torch.nn.modules.module._global_forward_hooks
    ):
        return None
    rows = x.reshape(-1, x.shape[-1])
    if rows.shape[0] == 0:
        return tuple(
            x.new_empty((*x.shape[:-1], p.weight.shape[0])) for p in (first, second)
        )
    padding = (-rows.shape[0]) % 16
    padded = (
        torch.empty(
            (rows.shape[0] + padding, rows.shape[1]),
            device=x.device,
            dtype=first.weight.dtype,
        )
        if padding
        else None
    )
    quantized, scale = _quantize_projection_input(
        rows.contiguous(),
        role=first.role if first.role == second.role else "PREFILL",
        output=padded,
    )
    return tuple(
        projection._run_quantized(quantized, scale, x.shape[:-1], rows.shape[0])
        for projection in (first, second)
    )
