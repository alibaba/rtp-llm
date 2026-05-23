"""GLM-5 MegaMoE: DeepGEMM fp8_fp4_mega_moe adapted for GLM-5 shapes.

Ported from feat/dsv4_on_dev branch's rtp_llm/models_py/modules/dsv4/moe/ module.
Adapted for GLM-5: hidden=6144, moe_inter=2048, experts=256, top_k=8, FP8 weights.

Usage:
    from rtp_llm.models_py.modules.glm5_mega_moe import GLM5MegaMoE

    moe = GLM5MegaMoE(
        layer_id=0,
        dim=6144,
        moe_inter_dim=2048,
        n_routed_experts=256,
        n_activated_experts=8,
        ep_size=4,
        ep_rank=0,
        max_tokens_per_rank=8192,
    )
    moe.setup_weights(layer_weights)
    y = moe.forward(x, weights, indices)
"""

from .input_packer import get_mega_moe_input_packer
from .jit_warmup import generate_mega_moe_jit_token_counts, mega_moe_jit_warmup_enabled
from .mega_buf import (
    get_or_create_mega_buf,
    get_or_create_mega_output,
    mega_moe_available,
    mega_moe_enabled,
)
from .mega_moe import GLM5MegaMoE

__all__ = [
    "GLM5MegaMoE",
    "get_or_create_mega_buf",
    "get_or_create_mega_output",
    "mega_moe_available",
    "mega_moe_enabled",
    "get_mega_moe_input_packer",
    "mega_moe_jit_warmup_enabled",
    "generate_mega_moe_jit_token_counts",
]
