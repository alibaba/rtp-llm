"""Adapt DSv4 checkpoint weights only after selecting the baseline MoE."""

from functools import partial
import os

from rtp_llm.models_py.modules.dsv4.chunk_env import chunked_moe_enabled
from rtp_llm.models_py.modules.factory.fused_moe.utils.fp8_fp4.chunked_layer import (
    ChunkedFp8Fp4MoeLayer,
)
from rtp_llm.models_py.modules.factory.fused_moe.utils.fp8_fp4.gate import Gate
from rtp_llm.models_py.modules.factory.fused_moe.utils.fp8_fp4.weight_adapter import (
    adapt_split_moe_weights,
)
from rtp_llm.utils.model_weight import W

_MOE_WEIGHT_NAMES = {
    "routed_gate": W.v4_routed_w1_w,
    "routed_gate_scale": W.v4_routed_w1_s,
    "routed_up": W.v4_routed_w3_w,
    "routed_up_scale": W.v4_routed_w3_s,
    "routed_down": W.v4_routed_w2_w,
    "routed_down_scale": W.v4_routed_w2_s,
    "router": W.v4_router_w,
    "router_bias": W.v4_router_bias,
    "router_tid2eid": W.v4_router_tid2eid,
    "shared_gate_up": W.v4_shared_w13_w,
    "shared_gate_up_scale": W.v4_shared_w13_s,
    "shared_down": W.v4_shared_w2_w,
    "shared_down_scale": W.v4_shared_w2_s,
}


def build_baseline_moe(*, tp_size=1, tp_rank=0, platform_provider=None, **kwargs):
    # PPU builders receive the untouched checkpoint dictionary and therefore
    # retain their loader-owned TP layout. Only this selected baseline packs
    # weights into the generic fused-MoE convention.
    options = getattr(platform_provider, "execution_options", os.environ)
    adapt_split_moe_weights(
        kwargs["layer_weights"],
        kwargs["moe_inter_dim"],
        kwargs["n_shared_experts"],
        _MOE_WEIGHT_NAMES,
    )
    return ChunkedFp8Fp4MoeLayer(
        chunking_enabled=chunked_moe_enabled(options),
        model_type="deepseek_v4",
        moe_w1_layout="gate_up",
        gate_factory=partial(
            Gate,
            fp32_gemm=options.get("DSV4_GATE_FP32", options.get("MOE_GATE_FP32", "0"))
            == "1",
            fused_gate=options.get(
                "DSV4_GATE_FUSED", options.get("MOE_GATE_FUSED", "1")
            )
            != "0",
        ),
        **kwargs,
    )
