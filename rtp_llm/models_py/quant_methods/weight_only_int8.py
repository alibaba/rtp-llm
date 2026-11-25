"""Online dense W8A16. FP checkpoint -> rank-local CUTLASS weights and scale."""

from functools import lru_cache

import torch

from rtp_llm.models_py.quant_methods.base import (
    QuantizeMethodBase,
    register_quant_method,
)
from rtp_llm.models_py.quant_methods.groupwise_export import cuda_groupwise_layout
from rtp_llm.models_py.quant_methods.int8 import quantize_weight_only_int8
from rtp_llm.models_py.read_spec import unwrap_checkpoint_tensor
from rtp_llm.models_py.runtime_weight_update import checked_lora_sum


@lru_cache(maxsize=None)
def _cuda_executor(device, dtype):
    capability = torch.cuda.get_device_capability(device)
    if capability[0] == 8:
        op = getattr(torch.ops.aten, "_mixed_dtypes_linear", None)
        if op is None or not torch._C._dispatch_has_kernel_for_dispatch_key(
            "aten::_mixed_dtypes_linear", "CUDA"
        ):
            raise RuntimeError("The PyTorch build lacks the CUDA W8A16 mixed-dtype GEMM")
        return op
    if capability not in ((9, 0), (10, 0)):
        raise RuntimeError(
            f"No dense W8A16 executor for CUDA SM{capability[0]}{capability[1]}"
        )
    from rtp_llm.models_py.quant_methods.weight_only_int8_cuda import mixed_w8a16

    # Check the installed Triton/compiler against this card and dtype *before*
    # publishing a model. Cache only the callable, never the probe allocations.
    x = torch.zeros(1, 64, dtype=dtype, device=device)
    w = torch.full((64, 64), 128, dtype=torch.uint8, device=device)
    s = torch.ones(64, dtype=dtype, device=device)
    mixed_w8a16(x, w, s)
    torch.cuda.current_stream(device).synchronize()
    return mixed_w8a16


def pack_weight_only_int8(weight):
    """Quantize a floating [N,K] projection into canonical mixed-GEMM bytes."""
    q, scale = quantize_weight_only_int8(weight.T)
    # Keep the public shard geometry unchanged. Only the physical kernel
    # needs 64-wide tiles; zero lanes cannot affect the logical product.
    pad_k, pad_n = -q.shape[0] % 64, -q.shape[1] % 64
    if pad_k or pad_n:
        q = torch.nn.functional.pad(q, (0, pad_n, 0, pad_k))
        if pad_n:
            scale = torch.nn.functional.pad(scale, (0, pad_n), value=1)
    return cuda_groupwise_layout(q, bits=8, arch=80), scale


def apply_packed_weight_only_int8(x, weight, scale, output_size, bias=None):
    """Compute from canonical padded INT8 storage without materializing a dense weight."""
    if x.device.type != "cuda" or torch.version.hip is not None:
        raise RuntimeError("Dense W8A16 forward requires NVIDIA CUDA tensors")
    if scale is None:
        raise RuntimeError("Dense W8A16 forward requires finalized weights")
    if (
        x.ndim < 1
        or x.dtype not in (torch.float16, torch.bfloat16)
        or x.device != weight.device
        or x.device != scale.device
        or weight.dtype != torch.int8
        or weight.ndim != 2
        or not weight.is_contiguous()
        or scale.dtype != x.dtype
        or scale.shape != (weight.shape[1],)
        or not scale.is_contiguous()
    ):
        raise ValueError("Invalid dense W8A16 activation, packed weight or scale")
    k, n = weight.shape
    pad_k, pad_n = k - x.shape[-1], n - output_size
    if k % 64 or n % 64 or not 0 <= pad_k < 64 or not 0 <= pad_n < 64:
        raise ValueError("Invalid dense W8A16 logical/padded geometry")
    if bias is not None:
        if (
            bias.numel() != output_size
            or bias.device != x.device
            or bias.dtype != x.dtype
        ):
            raise ValueError("Invalid dense W8A16 bias")
        bias = bias.reshape(-1).contiguous()
    shape = (*x.shape[:-1], output_size)
    if x.numel() == 0:
        return x.new_empty(shape)
    if pad_k:
        x = torch.nn.functional.pad(x, (0, pad_k))
    if pad_n and bias is not None:
        bias = torch.nn.functional.pad(bias, (0, pad_n))
    output = _cuda_executor(x.device, x.dtype)(
        x.reshape(-1, k).contiguous(), weight.view(torch.uint8), scale, bias=bias,
    )
    if pad_n:
        output = output[..., :output_size].contiguous()
    return output.reshape(shape)


@register_quant_method("INT8")
class WeightOnlyInt8LinearMethod(QuantizeMethodBase):
    # FST and scratch finalize one local projection at a time, after Q/K/V
    # or gate/up fusion and TP selection, just like the old per-column scheme.
    requires_staged_device_postprocess = True

    def create_weights(self, layer, input_size, output_size, params_dtype):
        if params_dtype not in (torch.float16, torch.bfloat16):
            raise ValueError("Dense W8A16 requires FP16 or BF16 parameters")
        self._compute_dtype = params_dtype
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
        if device.type != "cuda" or torch.version.hip is not None:
            raise RuntimeError("Dense W8A16 serving requires NVIDIA CUDA")
        if device.index is None:
            device = torch.device("cuda", torch.cuda.current_device())
        _cuda_executor(device, self._compute_dtype)

    def prepare_weights_on_cpu(self, layer):
        self.process_weights_after_loading(layer)

    def can_stage_startup_lora(self, layer):
        return layer.weight_scale is None and layer.weight.is_floating_point()

    def stage_startup_lora(self, layer, update):
        if not self.can_stage_startup_lora(layer):
            raise ValueError("Startup LoRA requires unpacked floating point weights")
        layer._startup_lora_update = update

    def process_weights_after_loading(self, layer):
        if layer.weight_scale is not None:
            return
        # Both the torch SM8 mixed-dtype operator and OldLoader consume these
        # exact unsigned-offset/interleaved bytes in [K,N] storage. Preserve
        # a signed parameter dtype for state inspection; .view(uint8) is free.
        update = getattr(layer, "_startup_lora_update", None)
        weight = layer.weight
        if update is not None:
            # TP/fused factors are in checkpoint orientation. Match the base
            # merge: FP32 product, cast the delta, then add before quantization.
            delta = update.delta().to(weight)
            if delta.shape != weight.shape:
                raise ValueError(f"Invalid startup LoRA delta for {update.target!r}")
            weight = checked_lora_sum(update.target, weight, delta)
        packed, scale = pack_weight_only_int8(weight)
        layer.weight = torch.nn.Parameter(packed, requires_grad=False)
        layer.weight_scale = torch.nn.Parameter(scale, requires_grad=False)
        if update is not None:
            del layer._startup_lora_update

    def apply(self, layer, x, bias=None):
        return apply_packed_weight_only_int8(
            x, layer.weight, layer.weight_scale, layer.output_size, bias
        )
