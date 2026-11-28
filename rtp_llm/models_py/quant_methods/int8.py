"""INT8 checkpoint conversion primitives, not an inference method registration.

Shared by the CPU/FT exporter and online weight-only methods. Device-specific
packing, executor selection and device admission belong to their callers, not
to the signed, unpacked per-channel quantizer below.
"""

from typing import Tuple

import torch


def quantize_weight_only_int8(
    weight: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Quantize legacy [input, output] weights independently per output column.

    Preserve legacy arithmetic (including input dtype, ties-to-even rounding,
    divisor 128, and asymmetric saturation [-128, 127]). In particular this is
    not the frequently used max/127 symmetric quantization formula. MoE callers
    quantize each expert independently after orienting it [input, output].
    """
    if weight.ndim != 2 or not weight.is_floating_point():
        raise ValueError("weight-only INT8 conversion expects a floating 2D matrix")
    amax = torch.clamp(weight.abs().max(dim=0)[0], min=1e-8)
    scale = amax / 128.0
    quantized = torch.clamp((weight / scale).round(), -128, 127).to(torch.int8)
    return quantized, scale


class Int8ExportState(torch.nn.Module):
    """Serialized module parameters, explicitly not an inference executor."""

    LAYOUTS = frozenset(
        {
            "weight_only_cuda_mixed_gemm_v1",
            "weight_only_rocm_mixed_gemm_v1",
            "int8_per_tensor_flat_storage_v1",
            "w8a8_int8_per_channel_flat_storage_v1",
            "int8_per_tensor_norm_flat_storage_v1",
        }
    )

    def export_tensors(self):
        if not self._exported:
            raise RuntimeError("INT8 parameters must be converted before serialization")
        return {name: value.detach() for name, value in self.named_parameters()}

    def export_metadata(self):
        return {
            "schema_version": 1,
            "offline_only": True,
            "layout": self.storage_layout,
            "parameters": {
                name: {"shape": list(value.shape), "dtype": str(value.dtype)}
                for name, value in self.export_tensors().items()
            },
        }

    @classmethod
    def from_export(cls, metadata, tensors):
        if (
            metadata.get("schema_version") != 1
            or metadata.get("layout") not in cls.LAYOUTS
        ):
            raise ValueError("Unsupported INT8 export schema or layout")
        if metadata.get("offline_only") is not True:
            raise ValueError("INT8 export module metadata must be offline-only")
        parameters = metadata.get("parameters")
        if not isinstance(parameters, dict) or set(parameters) != set(tensors):
            raise ValueError("INT8 export tensors do not match parameter metadata")
        if "weight" not in tensors or not set(tensors).issubset({"weight", "scale"}):
            raise ValueError("Invalid INT8 export parameter names")
        norm = metadata["layout"] == "int8_per_tensor_norm_flat_storage_v1"
        if norm and not tensors["weight"].is_floating_point():
            raise ValueError("INT8 per-tensor norm export requires floating storage")
        if not norm and tensors["weight"].dtype != torch.int8:
            raise ValueError("INT8 export requires signed INT8 storage")
        if "scale" in tensors and not tensors["scale"].is_floating_point():
            raise ValueError("INT8 export scales must be floating point")
        result = Int8ExportState()
        for name, value in tensors.items():
            actual = {"shape": list(value.shape), "dtype": str(value.dtype)}
            if parameters[name] != actual:
                raise ValueError(f"INT8 export metadata mismatch for {name}")
            result.register_parameter(
                name, torch.nn.Parameter(value, requires_grad=False)
            )
        result._exported = True
        result.storage_layout = metadata["layout"]
        return result

    def forward(self, *args, **kwargs):
        raise RuntimeError(
            "INT8 export modules are offline-only, not inference executors"
        )


class WeightOnlyInt8ExportModule(Int8ExportState):
    """Own floating projection storage through one offline INT8 conversion.

    Dense input uses legacy [K,N] orientation; stacked MoE input uses [E,N,K].
    The reader has already performed fusion, TP/EP selection and padding. This
    module does not repeat them and does not allocate an inference executor.
    """

    def __init__(self, weight: torch.Tensor, *, stacked: bool = False):
        super().__init__()
        if not weight.is_floating_point() or weight.ndim < 2:
            raise ValueError("INT8 export expects floating projection weights")
        if stacked and (weight.ndim != 3 or not weight.shape[0]):
            raise ValueError("Stacked INT8 export expects nonempty [E,N,K] weights")
        self.stacked = stacked
        self.register_parameter(
            "weight", torch.nn.Parameter(weight, requires_grad=False)
        )
        self.register_parameter("scale", None)
        self._exported = False

    def export(self, *, platform: str, arch: int = 0, device="cpu"):
        from rtp_llm.models_py.quant_methods.groupwise_export import (
            cuda_groupwise_layout,
            rocm_groupwise_layout,
        )

        if self._exported:
            raise RuntimeError("INT8 export layout conversion must run only once")
        if platform in ("Cuda", "Ppu"):
            layout = lambda tensor: cuda_groupwise_layout(tensor, 8, int(arch))
            storage_layout = "weight_only_cuda_mixed_gemm_v1"
        elif platform == "ROCm":
            layout = lambda tensor: rocm_groupwise_layout(tensor, 8)
            storage_layout = "weight_only_rocm_mixed_gemm_v1"
        else:
            raise ValueError(f"No weight-only INT8 offline layout for {platform}")

        if self.stacked:
            inputs = [expert.T.contiguous() for expert in self.weight.unbind()]
        else:
            inputs = [self.weight.reshape(self.weight.shape[0], -1)]
        converted = []
        scales = []
        for matrix in inputs:
            quantized, scale = quantize_weight_only_int8(matrix)
            converted.append(layout(quantized))
            scales.append(scale)
        weight = (
            torch.stack(converted)
            if self.stacked
            else converted[0].reshape(self.weight.shape)
        )
        scale = torch.stack(scales) if self.stacked else scales[0]
        self.weight = torch.nn.Parameter(weight.to(device), requires_grad=False)
        self.scale = torch.nn.Parameter(scale.to(device), requires_grad=False)
        self._exported = True
        self.storage_layout = storage_layout
        return self.weight, self.scale


class PrequantizedInt8ExportModule(Int8ExportState):
    """Own the INT8 checkpoint-to-FT shape ABI, without re-quantization.

    Input tensors have already been sharded and platform-processed by the
    compatibility reader. Reshape changes the view, never the stored bytes.
    Per-tensor activation scales stay owned by their source modules unchanged.
    """

    def __init__(self, weight, scale=None, *, per_channel: bool, norm: bool = False):
        super().__init__()
        if norm and (per_channel or not weight.is_floating_point() or weight.ndim < 1):
            raise ValueError("Per-tensor norm export expects floating norm weights")
        if not norm and (weight.dtype != torch.int8 or weight.ndim < 2):
            raise ValueError("Prequantized INT8 export expects signed INT8 weights")
        if scale is not None and not scale.is_floating_point():
            raise ValueError("INT8 checkpoint scales must be floating point")
        self.per_channel = per_channel
        self.norm = norm
        self._exported = False
        self.register_parameter(
            "weight", torch.nn.Parameter(weight, requires_grad=False)
        )
        self.register_parameter(
            "scale",
            None if scale is None else torch.nn.Parameter(scale, requires_grad=False),
        )

    def export(self):
        if self._exported:
            raise RuntimeError("INT8 export shape conversion must run only once")
        weight = self.weight
        if not self.per_channel or weight.ndim == 2:
            weight = weight.reshape(weight.shape[-1], -1)
        scale = self.scale
        if self.per_channel and scale is not None and scale.ndim == 2:
            scale = scale.reshape(scale.shape[-1], -1)
        self.weight = torch.nn.Parameter(weight, requires_grad=False)
        self.scale = (
            None if scale is None else torch.nn.Parameter(scale, requires_grad=False)
        )
        self._exported = True
        self.storage_layout = (
            "int8_per_tensor_norm_flat_storage_v1"
            if self.norm
            else (
                "w8a8_int8_per_channel_flat_storage_v1"
                if self.per_channel
                else "int8_per_tensor_flat_storage_v1"
            )
        )
        return self.weight, self.scale
