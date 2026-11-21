"""Floating checkpoint -> per-expert, per-output-channel W8A16 weights."""

import torch

from rtp_llm.models_py.quant_methods import groupwise_runtime
from rtp_llm.models_py.quant_methods.base import register_moe_quant_method
from rtp_llm.models_py.quant_methods.int8 import quantize_weight_only_int8
from rtp_llm.models_py.quant_methods.unquantized import UnquantizedFusedMoEMethod


@register_moe_quant_method("INT8")
class WeightOnlyInt8MoEMethod(UnquantizedFusedMoEMethod):
    requires_staged_device_postprocess = True

    def validate_runtime_device(self, device):
        if torch.device(device).type != "cpu":
            groupwise_runtime.validate_integer_device(device)

    def process_weights_after_loading(self, layer):
        if hasattr(layer, "_groupwise_runtime"):
            return
        states = {}
        for name in ("w13", "w2"):
            param = getattr(layer, name)
            e, n, k = param.shape
            packed = torch.empty_like(param, dtype=torch.uint8)
            scales = param.new_empty((e, n, 1))
            for i in range(e):
                q, s = quantize_weight_only_int8(param[i].T)
                # The grouped executor consumes unsigned codes minus zero point.
                packed[i].copy_(q.T.view(torch.uint8).bitwise_xor(128))
                scales[i, :, 0].copy_(s)
            param.data = packed
            zero = torch.full((e, n, 1), 128, dtype=torch.int16, device=packed.device)
            layer.register_buffer(name + "_int8_scale", scales)
            layer.register_buffer(name + "_int8_zero", zero)
            states[name] = groupwise_runtime.GroupwiseRuntimeState(
                param,
                scales,
                zero,
                groupwise_runtime.GroupwiseLayout("INT8", 8, k, k, n),
                8,
                None,
            )
        layer._groupwise_runtime = states

    def build_moe_executor(self, layer):
        return groupwise_runtime.build_moe_executor(layer)
