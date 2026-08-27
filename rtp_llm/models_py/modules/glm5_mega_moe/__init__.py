"""GLM-5 MegaMoE: DeepGEMM fp8_fp4_mega_moe for GLM-5.3-Flash.

The released model uses hidden=4096, moe_inter=2048, experts=288, top_k=8,
and FP8 checkpoint weights converted to FP4 at load time.

Usage:
    from rtp_llm.models_py.modules.glm5_mega_moe import GLM5MegaMoE

    moe = GLM5MegaMoE(
        layer_id=0,
        dim=4096,
        moe_inter_dim=2048,
        n_routed_experts=288,
        n_activated_experts=8,
        ep_size=8,
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
