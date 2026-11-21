"""Quark MXFP4 checkpoint storage for the existing gfx950 AITER MoE executor.

Weights remain two E2M1 values per byte. Scales remain one UE8M0 byte per
32 logical K elements. No unpack/dequantize path is used during loading.
The shared MoE runtime-layout boundary performs AITER weight/scale shuffles
after the method has assembled the local [up | gate] expert parameters.
"""

from typing import Any

import torch
import torch.nn as nn

from rtp_llm.models_py.quant_methods.base import (
    FusedMoEMethodBase,
    register_moe_quant_method,
)
from rtp_llm.utils.model_weight import W


@register_moe_quant_method("mxfp4_quark", "QuarkMXFP4")
class MXFp4QuarkMoEMethod(FusedMoEMethodBase):
    uses_packed_weights = True
    GROUP_SIZE = 32

    def __init__(self, quant_config: Any = None):
        super().__init__(quant_config)
        source = getattr(quant_config, "source_config", None)
        if source is not None:
            is_quanted = getattr(source, "is_quanted", lambda: True)
            if not is_quanted():
                raise ValueError("Quark MXFP4 requires an already-quantized checkpoint")
            group_size = getattr(source, "group_size", lambda: 0)()
            if group_size not in (0, self.GROUP_SIZE):
                raise ValueError("Quark MXFP4 requires groups of 32 logical K elements")

    def validate_runtime_device(self, device: torch.device) -> None:
        if getattr(self, "_runtime_topology", (1, 1)) != (1, 1):
            raise RuntimeError(
                "The existing ROCm MXFP4 executor/router supports pure TP only "
                "(EP=1, DP=1); checkpoint placement support does not imply "
                "an executable EP/DP backend"
            )
        device = torch.device(device)
        if device.type != "cuda" or not getattr(torch.version, "hip", None):
            raise RuntimeError("Quark MXFP4 execution requires ROCm gfx950 (MI355)")
        if not torch.cuda.is_available():
            raise RuntimeError("Quark MXFP4 requires an available ROCm accelerator")
        properties = torch.cuda.get_device_properties(device)
        arch = getattr(properties, "gcnArchName", "").split(":", 1)[0]
        if arch != "gfx950":
            raise RuntimeError(
                f"Quark MXFP4 requires gfx950 (MI355), got {arch or 'unknown'}; "
                "gfx942/MI308 cannot execute this backend"
            )
        if not hasattr(torch, "float4_e2m1fn_x2"):
            raise RuntimeError("Quark MXFP4 requires torch.float4_e2m1fn_x2")
        try:
            import aiter
            from aiter.fused_moe import fused_moe
            from aiter.ops.shuffle import shuffle_weight
            from aiter.utility.fp4_utils import e8m0_shuffle

            symbols = (fused_moe, shuffle_weight, e8m0_shuffle)
            if not all(callable(symbol) for symbol in symbols):
                raise AttributeError("AITER MXFP4 conversion/execution symbols missing")
            aiter.QuantType.per_1x32
            aiter.ActivationType.Silu
        except (ImportError, OSError, RuntimeError, AttributeError) as exc:
            raise RuntimeError("Quark MXFP4 requires the AITER MXFP4 runtime") from exc

    def create_weights(
        self, layer, num_experts, hidden_size, intermediate_size, params_dtype, **kwargs
    ):
        self._runtime_topology = (
            layer.ep_size,
            getattr(layer._parallelism_config, "dp_size", 1),
        )
        if hidden_size % self.GROUP_SIZE or intermediate_size % self.GROUP_SIZE:
            raise ValueError(
                "Quark MXFP4 hidden size and per-rank intermediate size must be "
                f"multiples of 32, got H={hidden_size}, M={intermediate_size}"
            )
        E, H, M = num_experts, hidden_size, intermediate_size
        layer.w13 = nn.Parameter(
            torch.empty(E, 2 * M, H // 2, dtype=torch.uint8), requires_grad=False
        )
        layer.w2 = nn.Parameter(
            torch.empty(E, H, M // 2, dtype=torch.uint8), requires_grad=False
        )
        layer.register_buffer(
            "w13_scale", torch.empty(E, 2 * M, H // 32, dtype=torch.uint8)
        )
        layer.register_buffer("w2_scale", torch.empty(E, H, M // 32, dtype=torch.uint8))

    def required_aux_parameters(self):
        return ("weight_scale",)

    @staticmethod
    def _storage_bytes(tensor, *, scale):
        shell_dtype = getattr(
            torch, "float8_e8m0fnu" if scale else "float4_e2m1fn_x2", None
        )
        if shell_dtype is not None and tensor.dtype == shell_dtype:
            tensor = tensor.view(torch.uint8)
        if tensor.dtype != torch.uint8:
            kind = "UE8M0 scales" if scale else "packed E2M1 weights"
            raise TypeError(
                f"Quark MXFP4 {kind} require byte storage, got {tensor.dtype}"
            )
        if tensor.dim() != 2:
            raise ValueError("Quark MXFP4 per-expert tensors must be matrices")
        # UE8M0 byte 255 denotes NaN (0 is a valid small exponent, not zero).
        if scale and bool((tensor == 255).any()):
            raise ValueError("Quark MXFP4 UE8M0 scale contains NaN encoding 255")
        return tensor

    def _copy_projection(self, layer, local_id, proj, tensor, *, scale):
        tensor = self._storage_bytes(tensor, scale=scale)
        divisor = self.GROUP_SIZE if scale else 2
        H, M = layer.hidden_size, layer.moe_inter
        local_m = layer.moe_inter_tp
        rank = layer.moe_expert_tp_rank
        if proj == "gate_up_proj":
            if tuple(tensor.shape) != (2 * M, H // divisor):
                raise ValueError(
                    f"Invalid Quark MXFP4 gate_up shape {tuple(tensor.shape)}"
                )
            self._copy_projection(layer, local_id, "gate_proj", tensor[:M], scale=scale)
            self._copy_projection(layer, local_id, "up_proj", tensor[M:], scale=scale)
            return
        if proj in ("gate_proj", "up_proj"):
            expected = (M, H // divisor)
            target = layer.w13_scale if scale else layer.w13
            row_offset = local_m if proj == "gate_proj" else 0
            destination = target.data[local_id, row_offset : row_offset + local_m]
            selected = tensor[rank * local_m : (rank + 1) * local_m]
        elif proj == "down_proj":
            expected = (H, M // divisor)
            target = layer.w2_scale if scale else layer.w2
            destination = target.data[local_id]
            local_k = local_m // divisor
            selected = tensor[:, rank * local_k : (rank + 1) * local_k]
        else:
            raise ValueError(f"Unsupported Quark MXFP4 projection {proj!r}")
        if tuple(tensor.shape) != expected:
            raise ValueError(
                f"Quark MXFP4 {proj} {'scale' if scale else 'weight'} shape "
                f"{tuple(tensor.shape)} != expected {expected}"
            )
        destination.copy_(selected)

    def dispatch_weight(self, layer, local_id, projection, parameter_name, tensor):
        if parameter_name != "weight":
            return False
        self._copy_projection(layer, local_id, projection, tensor, scale=False)
        return True

    def dispatch_scale(self, layer, local_id, projection, parameter_name, tensor):
        if parameter_name != "weight_scale":
            return False
        self._copy_projection(layer, local_id, projection, tensor, scale=True)
        return True

    def process_weights_after_loading(self, layer):
        # Kernel-specific gate/up and UE8M0 swizzles run at _build_weights_dict,
        # once, on the final target device, just like other ROCm MoE methods.
        pass

    def add_weight_tensors(self, layer, weights):
        weights[W.moe_s1] = layer.w13_scale
        weights[W.moe_s2] = layer.w2_scale
