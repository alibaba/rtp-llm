"""Construct the runtime for a recipe selected by the public MoE registry."""

from rtp_llm.config.mega_moe import FUSED_SHARED_STRATEGIES


def create_mega_moe_wrapper(
    config,
    parallelism_config,
    weights,
    moe_config,
    *,
    layer_idx: int,
    max_generate_batch_size: int,
    execution_policy,
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
    from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.executors.mega_moe_fp8_runtime import (
        MegaMoeFp8Wrapper,
    )
    from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.executors.mega_moe_fp8_se_runtime import (
        MegaMoeFp8SEWrapper,
    )
    from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.executors.mega_moe_fused_runtime import (
        MegaMoeFusedWrapper,
    )
    from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.executors.mega_moe_runtime import (
        MegaMoeWrapper,
    )
    from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.executors.mega_moe_shared_runtime import (
        MegaMoeSEWrapper,
    )

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
        execution_policy=execution_policy,
    )
