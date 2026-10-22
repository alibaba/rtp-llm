"""Bind PPU routing arithmetic to the common MoE gate without copying weights."""

from functools import partial

import torch.nn.functional as F

from rtp_llm.models_py.modules.factory.fused_moe.utils.fp8_fp4.gate import Gate
from rtp_llm.utils.model_weight import W


class PpuGate(Gate):
    def __init__(self, *args, layer_weights, platform_provider, **kwargs):
        if platform_provider is None:
            raise ValueError("PPU Gate requires an instance-owned operator adapter")
        keys = {
            W.v4_router_w: W.moe_gate,
            W.v4_router_bias: W.moe_gate_bias,
            W.v4_router_tid2eid: W.moe_gate_tid2eid,
        }
        weights = {
            dst: layer_weights[src] for src, dst in keys.items() if src in layer_weights
        }
        options = platform_provider.execution_options
        super().__init__(
            *args,
            layer_weights=weights,
            linear=partial(platform_provider.run_bf16_fp32_linear, F.linear),
            fp32_gemm=options.get("DSV4_GATE_FP32", "0") == "1",
            fused_gate=options.get("DSV4_GATE_FUSED", "1") != "0",
            **kwargs,
        )
