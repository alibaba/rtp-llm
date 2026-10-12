"""Construct GLM MegaMoE wrappers at the model's strategy boundary."""

MEGA_MOE_STRATEGIES = frozenset(
    {"mega_moe", "mega_moe_se", "mega_moe_fp8", "mega_moe_fp8_se", "mega_moe_fused"}
)
FUSED_SHARED_STRATEGIES = frozenset(
    {"mega_moe_se", "mega_moe_fp8_se", "mega_moe_fused"}
)


def create_mega_moe_wrapper(
    config,
    parallelism_config,
    weights,
    moe_config,
    *,
    layer_idx: int,
    max_generate_batch_size: int,
):
    strategy = moe_config.moe_strategy
    if strategy in FUSED_SHARED_STRATEGIES:
        if config.moe_style != 2:
            raise ValueError(f"moe_strategy={strategy} requires shared experts")
        from rtp_llm.utils.model_weight import W

        if weights.get(W.shared_expert_gate) is not None:
            raise ValueError(
                f"moe_strategy={strategy} does not support shared_expert_gate"
            )
        if parallelism_config.ep_size > 1 and parallelism_config.get_ffn_tp_size() > 1:
            raise ValueError(
                f"moe_strategy={strategy} requires replicated shared-expert weights"
            )
    # Lazy imports keep backend selection out of model-module initialization.
    from .mega_moe_fp8_se_wrapper import MegaMoeFp8SEWrapper
    from .mega_moe_fp8_wrapper import MegaMoeFp8Wrapper
    from .mega_moe_fused_wrapper import MegaMoeFusedWrapper
    from .mega_moe_se_wrapper import MegaMoeSEWrapper
    from .mega_moe_wrapper import MegaMoeWrapper

    wrappers = {
        "mega_moe": MegaMoeWrapper,
        "mega_moe_se": MegaMoeSEWrapper,
        "mega_moe_fp8": MegaMoeFp8Wrapper,
        "mega_moe_fp8_se": MegaMoeFp8SEWrapper,
        "mega_moe_fused": MegaMoeFusedWrapper,
    }
    return wrappers[moe_config.moe_strategy](
        config,
        parallelism_config,
        weights,
        moe_config,
        layer_idx=layer_idx,
        max_generate_batch_size=max_generate_batch_size,
    )
