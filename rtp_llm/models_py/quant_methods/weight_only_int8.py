"""Online dense W8A16. FP checkpoint -> rank-local CUTLASS weights and scale."""

import torch

from rtp_llm.models_py.quant_methods.base import (
    QuantizeMethodBase,
    register_quant_method,
)
from rtp_llm.models_py.quant_methods.groupwise_export import cuda_groupwise_layout
from rtp_llm.models_py.quant_methods.int8 import quantize_weight_only_int8
from rtp_llm.models_py.read_spec import unwrap_checkpoint_tensor


@register_quant_method("INT8")
class WeightOnlyInt8LinearMethod(QuantizeMethodBase):
    # FST and scratch finalize one local projection at a time, after Q/K/V
    # or gate/up fusion and TP selection, just like the old per-column scheme.
    requires_staged_device_postprocess = True

    def create_weights(self, layer, input_size, output_size, params_dtype):
        if params_dtype not in (torch.float16, torch.bfloat16):
            raise ValueError("Dense W8A16 requires FP16 or BF16 parameters")
        if input_size % 64 or output_size % 64:
            raise ValueError(
                f"{layer.prefix}: CUTLASS W8A16 local K={input_size}, N={output_size} "
                "must both be divisible by 64"
            )
        layer.register_parameter(
            "weight",
            torch.nn.Parameter(
                torch.empty(output_size, input_size, dtype=params_dtype),
                requires_grad=False,
            ),
        )
        layer.register_parameter("weight_scale", None)

    def normalize_checkpoint_weights(self, weights):
        weights = super().normalize_checkpoint_weights(weights)
        for name, value in weights.items():
            if name.rsplit(".", 1)[-1] == "weight":
                tensor, _ = unwrap_checkpoint_tensor(value)
                if not tensor.is_floating_point():
                    raise ValueError(
                        f"Online dense W8A16 expects a floating checkpoint: {name}"
                    )
        return weights

    def validate_runtime_device(self, device):
        device = torch.device(device)
        if device.type == "cpu":  # host staging of the same packed checkpoint bytes
            return
        if (
            device.type != "cuda"
            or torch.version.hip is not None
            or torch.cuda.get_device_capability(device)[0] != 8
        ):
            raise RuntimeError("The dense W8A16 CUTLASS executor requires CUDA SM8.x")
        op = getattr(torch.ops.aten, "_mixed_dtypes_linear", None)
        if op is None or not torch._C._dispatch_has_kernel_for_dispatch_key(
            "aten::_mixed_dtypes_linear", "CUDA"
        ):
            raise RuntimeError(
                "The PyTorch build lacks the CUDA W8A16 mixed-dtype GEMM"
            )

    def prepare_weights_on_cpu(self, layer):
        self.process_weights_after_loading(layer)

    def process_weights_after_loading(self, layer):
        if layer.weight_scale is not None:
            return
        # Both the torch SM8 mixed-dtype operator and OldLoader consume these
        # exact unsigned-offset/interleaved bytes in [K,N] storage. Preserve
        # a signed parameter dtype for state inspection; .view(uint8) is free.
        q, scale = quantize_weight_only_int8(layer.weight.T)
        packed = cuda_groupwise_layout(q, bits=8, arch=80)
        layer.weight = torch.nn.Parameter(packed, requires_grad=False)
        layer.weight_scale = torch.nn.Parameter(scale, requires_grad=False)

    def apply(self, layer, x, bias=None):
        if x.device.type != "cuda" or x.device != layer.weight.device:
            raise RuntimeError(
                "Dense W8A16 forward requires colocated CUDA SM8.x tensors"
            )
        if layer.weight_scale is None:
            raise RuntimeError("Dense W8A16 forward requires finalized weights")
        shape = (*x.shape[:-1], layer.weight_scale.numel())
        if x.numel() == 0:
            return x.new_empty(shape)
        output = torch.ops.aten._mixed_dtypes_linear(
            x.reshape(-1, x.shape[-1]).contiguous(),
            layer.weight.view(torch.uint8),
            layer.weight_scale,
            bias=bias,
        )
        return output.reshape(shape)
