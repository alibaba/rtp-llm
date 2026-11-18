"""Dense per-tensor FP8 methods, without importing grouped GEMM backends.

This batch exposes layer-level methods; serving admission remains unquantized
until the model/configuration integration batch. Per-channel, block FP8 and MoE
are deliberately not registered here.
"""

import importlib
from functools import lru_cache
from typing import Optional

import torch
from torch import nn

from rtp_llm.models_py.quant_methods.base import (
    QuantizeMethodBase,
    register_quant_method,
)


@lru_cache(maxsize=None)
def _quant_ops(hip: bool):
    # Cache successful resolution only. Import/symbol failures remain visible
    # at load-time preflight, not at the first inference request.
    if hip:
        module = importlib.import_module("aiter.ops.quant")
        names = ("dynamic_per_tensor_quant", "static_per_tensor_quant")
    else:
        ops = importlib.import_module("rtp_llm.ops")
        ops.ensure_compute_ops_loaded()
        module = importlib.import_module("librtp_compute_ops.rtp_llm_ops")
        names = ("per_tensor_quant_fp8",)
    result = tuple(getattr(module, name, None) for name in names)
    missing = [name for name, op in zip(names, result) if not callable(op)]
    if missing:
        raise ImportError(f"Missing FP8 per-tensor operators: {', '.join(missing)}")
    return result


def _runtime_dtype(device: torch.device) -> torch.dtype:
    device = torch.device(device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError(f"FP8 per-tensor requires an accelerator, got {device}")
    index = torch.cuda.current_device() if device.index is None else device.index
    hip = torch.version.hip is not None
    if hip:
        arch = torch.cuda.get_device_properties(index).gcnArchName
        if not any(name in arch for name in ("gfx942", "gfx950")):
            raise RuntimeError(f"Unsupported ROCm FP8 architecture: {arch}")
        dtype = torch.float8_e4m3fn if "gfx950" in arch else torch.float8_e4m3fnuz
    else:
        capability = torch.cuda.get_device_capability(index)
        if capability < (8, 9):
            raise RuntimeError(f"FP8 per-tensor requires SM89 or newer: {capability}")
        dtype = torch.float8_e4m3fn
    if not callable(getattr(torch, "_scaled_mm", None)):
        raise RuntimeError("FP8 per-tensor requires torch._scaled_mm")
    try:
        _quant_ops(hip)
    except (ImportError, OSError) as error:
        raise RuntimeError(
            "FP8 per-tensor quantization backend is unavailable"
        ) from error
    return dtype


def _positive_scale(scale: torch.Tensor, label: str) -> torch.Tensor:
    value = scale.detach().float().reshape(-1)
    if (
        value.numel() != 1
        or not bool(torch.isfinite(value).all())
        or bool((value <= 0).any())
    ):
        raise ValueError(f"{label} must contain one finite positive value")
    return value.contiguous()


def _convert_checkpoint(weight, scale, dtype):
    scale = _positive_scale(scale, "weight_scale")
    if weight.dtype == dtype:
        return weight.contiguous(), scale
    if weight.dtype == torch.float8_e4m3fn and dtype == torch.float8_e4m3fnuz:
        # Identical bits represent half the value in FNUZ. Normalize FN's
        # negative zero (FNUZ NaN), then double the dequantization scale.
        bits = weight.contiguous().view(torch.int8).clone()
        bits[bits == -128] = 0
        return bits.view(dtype), scale * 2
    raise TypeError(f"Unsupported FP8 conversion: {weight.dtype} -> {dtype}")


def _quantize(input: torch.Tensor, dtype: torch.dtype, scale=None):
    if dtype not in (torch.float8_e4m3fn, torch.float8_e4m3fnuz):
        raise TypeError("FP8 quantization output must use an E4M3 dtype")
    if torch.version.hip is None and dtype != torch.float8_e4m3fn:
        raise TypeError("CUDA per-tensor quantization requires E4M3FN output")
    if input.ndim != 2 or input.dtype not in (torch.float16, torch.bfloat16):
        raise TypeError("FP8 per-tensor input must be a 2D FP16/BF16 tensor")
    if input.device.type != "cuda":
        raise ValueError("FP8 per-tensor input must be on the accelerator")
    input = input.contiguous()
    output = torch.empty(input.shape, dtype=dtype, device=input.device)
    is_static = scale is not None
    if is_static:
        if (
            scale.numel() != 1
            or scale.dtype != torch.float32
            or scale.device != input.device
        ):
            raise ValueError(
                "input_scale must be a scalar FP32 tensor on the input device"
            )
        scale = scale.reshape(1).contiguous()
    else:
        # CUDA absmax uses atomicMax, so the dynamic output must start at zero.
        scale = torch.zeros(1, dtype=torch.float32, device=input.device)
    if input.numel() == 0:
        return output, scale
    hip = torch.version.hip is not None
    # The existing CUDA kernel requires vector alignment. Linear dimensions
    # are validated at construction; keep this boundary safe for direct calls.
    if not hip and input.numel() % (16 // input.element_size()):
        raise ValueError("CUDA per-tensor input must be 16-byte aligned")
    with torch.cuda.device(input.device):
        ops = _quant_ops(hip)
        if hip:
            ops[1 if is_static else 0](output, input, scale)
        else:
            ops[0](input, output, scale, is_static)
    return output, scale


def _replace(layer, name, tensor):
    layer.register_parameter(name, nn.Parameter(tensor, requires_grad=False))


class _PerTensorMethod(QuantizeMethodBase):
    @staticmethod
    def _check_layout(input_size, output_size, params_dtype):
        if params_dtype not in (torch.float16, torch.bfloat16):
            raise TypeError("FP8 Linear output dtype must be FP16 or BF16")
        # These are local TP dimensions, including merged/QKV partitions.
        if input_size % 16 or output_size % 16:
            raise ValueError(
                "FP8 Linear local input/output dimensions must be divisible by 16"
            )

    def validate_runtime_device(self, device):
        _runtime_dtype(device)

    def _apply_fp8(self, layer, x, bias, input_scale):
        if not hasattr(self, "runtime_dtype"):
            raise RuntimeError("FP8 weights have not been finalized")
        if (
            layer.weight.dtype != self.runtime_dtype
            or layer.weight_scale.dtype != torch.float32
        ):
            raise TypeError(
                "FP8 runtime weights/scales must retain their finalized dtypes"
            )
        if x.dtype not in (torch.float16, torch.bfloat16):
            raise TypeError("FP8 Linear input must be FP16 or BF16")
        flat = x.reshape(-1, x.shape[-1])
        shape = (*x.shape[:-1], layer.weight.shape[0])
        if flat.shape[0] == 0:
            return x.new_empty(shape)
        with torch.cuda.device(x.device):
            quantized, scale = _quantize(flat, self.runtime_dtype, input_scale)
            output = torch._scaled_mm(
                quantized,
                layer.weight.t(),
                scale_a=scale,
                scale_b=layer.weight_scale,
                bias=bias,
                out_dtype=x.dtype,
            )
        # Torch versions before the Tensor-only API returned (output, amax).
        if isinstance(output, tuple):
            output = output[0]
        return output.reshape(shape)


@register_quant_method("fp8", "FP8_PER_TENSOR_COMPRESSED")
class Fp8LinearMethod(_PerTensorMethod):
    """Prequantized E4M3 checkpoint with scalar weight/activation scales."""

    @property
    def required_checkpoint_parameters(self):
        return ("weight", "weight_scale") + (
            () if self.quant_config.activation_dynamic else ("input_scale",)
        )

    def create_weights(self, layer, input_size, output_size, params_dtype):
        self._check_layout(input_size, output_size, params_dtype)
        _replace(
            layer,
            "weight",
            torch.empty(output_size, input_size, dtype=torch.float8_e4m3fn),
        )
        _replace(layer, "weight_scale", torch.ones(1, dtype=torch.float32))
        _replace(layer, "input_scale", torch.ones(1, dtype=torch.float32))

    def process_weights_after_loading(self, layer):
        dtype = _runtime_dtype(layer.weight.device)
        weight, scale = _convert_checkpoint(
            layer.weight.detach(), layer.weight_scale, dtype
        )
        input_scale = None
        if not self.quant_config.activation_dynamic:
            input_scale = _positive_scale(layer.input_scale, "input_scale")
            if (
                layer.weight.dtype == torch.float8_e4m3fn
                and dtype == torch.float8_e4m3fnuz
            ):
                input_scale = input_scale * 2
        _replace(layer, "weight", weight)
        _replace(layer, "weight_scale", scale)
        if input_scale is not None:
            _replace(layer, "input_scale", input_scale)
        self.runtime_dtype = dtype

    def apply(self, layer, x, bias: Optional[torch.Tensor] = None):
        scale = None if self.quant_config.activation_dynamic else layer.input_scale
        return self._apply_fp8(layer, x, bias, scale)


@register_quant_method("fp8_online", "FP8_DYNAMIC_PER_TENSOR")
class Fp8OnlineLinearMethod(_PerTensorMethod):
    """Stream BF16/FP16 weights, then quantize each staged leaf once.

    LinearBase's post-load guard makes the final whole-model hook visit a
    no-op for an already converted staged leaf. Do not quantize on each shard.
    """

    requires_staged_device_postprocess = True
    required_checkpoint_parameters = ("weight",)

    def create_weights(self, layer, input_size, output_size, params_dtype):
        self._check_layout(input_size, output_size, params_dtype)
        if not self.quant_config.activation_dynamic:
            raise ValueError(
                "Online per-tensor FP8 requires dynamic activation quantization"
            )
        _replace(
            layer, "weight", torch.empty(output_size, input_size, dtype=params_dtype)
        )
        _replace(layer, "weight_scale", torch.ones(1, dtype=torch.float32))

    def process_weights_after_loading(self, layer):
        dtype = _runtime_dtype(layer.weight.device)
        weight, scale = _quantize(layer.weight.detach(), dtype)
        _replace(layer, "weight", weight)
        _replace(layer, "weight_scale", scale.reshape(1).contiguous())
        self.runtime_dtype = dtype

    def apply(self, layer, x, bias: Optional[torch.Tensor] = None):
        return self._apply_fp8(layer, x, bias, None)
