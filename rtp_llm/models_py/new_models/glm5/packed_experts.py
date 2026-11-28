"""GLM-5.3 packed FP4 routed weights (dense/shared projections remain FP8)."""

import torch
from torch import nn

from rtp_llm.models_py.quant_methods.base import FusedMoEMethodBase
from rtp_llm.utils.model_weight import W


class Glm5PackedExpertMethod(FusedMoEMethodBase):
    uses_packed_weights = True

    def create_weights(
        self, layer, num_experts, hidden_size, intermediate_size, params_dtype
    ):
        if layer.moe_expert_tp_size != 1 or hidden_size % 32 or intermediate_size % 32:
            raise ValueError(
                "GLM-5 packed FP4 requires unsharded, group-32 expert matrices"
            )
        for name, rows, cols in (
            ("w13", 2 * intermediate_size, hidden_size),
            ("w2", hidden_size, intermediate_size),
        ):
            layer.register_parameter(
                name,
                nn.Parameter(
                    torch.empty((num_experts, rows, cols // 2), dtype=torch.int8),
                    requires_grad=False,
                ),
            )
            # Keep raw exponent bytes. Modern files supply E8M0 and the older
            # writer used FP32 powers of two, both exact in this representation.
            layer.register_buffer(
                name + "_scale",
                torch.empty(
                    (num_experts, rows, cols // 32),
                    dtype=torch.float8_e8m0fnu,
                ),
            )

    def required_aux_parameters(self):
        return ("scale",)

    def _target(self, layer, slot, proj, suffix):
        if proj == "down_proj":
            return getattr(layer, "w2" + suffix)[slot]
        if proj not in ("up_proj", "gate_proj"):
            raise ValueError(f"unsupported GLM-5 packed expert projection {proj!r}")
        rows = layer.moe_inter_tp
        return getattr(layer, "w13" + suffix)[slot].narrow(
            0, rows if proj == "gate_proj" else 0, rows
        )

    def dispatch_weight(self, layer, slot, proj, name, tensor):
        if name != "weight":
            return False
        target = self._target(layer, slot, proj, "")
        if tensor.dtype != torch.int8 or tensor.shape != target.shape:
            raise ValueError(
                f"invalid packed GLM-5 FP4 {proj} weight: {tensor.dtype}, {tensor.shape}"
            )
        target.copy_(tensor)
        return True

    def dispatch_scale(self, layer, slot, proj, name, tensor):
        if name != "scale":
            return False
        target = self._target(layer, slot, proj, "_scale")
        if tensor.shape != target.shape:
            raise ValueError(
                f"invalid GLM-5 FP4 {proj} group-32 scale shape: {tensor.shape}"
            )
        if tensor.dtype == torch.float32:
            mantissa, exponent = torch.frexp(tensor)
            if not ((mantissa == 0.5) & (exponent >= -126) & (exponent <= 128)).all():
                raise ValueError(
                    "GLM-5 FP4 scales must be positive UE8M0 powers of two"
                )
            tensor = tensor.to(torch.float8_e8m0fnu)
        if tensor.dtype != torch.float8_e8m0fnu:
            raise TypeError(f"invalid GLM-5 FP4 scale dtype {tensor.dtype}")
        target.view(torch.uint8).copy_(tensor.view(torch.uint8))
        return True

    def process_weights_after_loading(self, layer):
        # The MegaMoE executor packs the complete rank-local stack at startup
        # and again on private staging storage when EPLB changes the placement.
        pass

    def add_weight_tensors(self, layer, weights):
        weights[W.moe_s1] = layer.w13_scale
        weights[W.moe_s2] = layer.w2_scale
