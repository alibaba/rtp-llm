"""Baseline builders, imported only after protocol verification."""


def validate_arguments(build_ctx, request, kwargs):
    metadata = build_ctx.selection.model_metadata
    kind = request.module_id.rsplit(".", 1)[1]
    if kind == "model":
        config, parallel = kwargs["model_config"], kwargs["parallelism_config"]
        actual = (
            int(config.num_layers),
            int(config.hidden_size),
            int(parallel.tp_size),
            int(parallel.ep_size),
        )
        expected = tuple(
            metadata[key] for key in ("num_layers", "hidden_size", "tp_size", "ep_size")
        )
    else:
        actual = (int(kwargs["layer_id"]), int(kwargs["dim"]), int(kwargs["tp_size"]))
        expected = (
            request.metadata["layer_id"],
            metadata["hidden_size"],
            metadata["tp_size"],
        )
    if actual != expected:
        raise ValueError(
            f"Actual construction arguments differ from preflight at {request.path}: {actual} != {expected}"
        )


def validate_instance(module, build_ctx, request):
    from torch import nn

    if request.module_id == "rtp.dsv4.model":
        from rtp_llm.models_py.model_desc.module_base import GptModelBase

        if not isinstance(module, GptModelBase):
            raise TypeError("Whole model implementations must inherit GptModelBase")
        if module.module_build_context is not build_ctx:
            raise TypeError("Model did not retain its instance build context")
    elif not isinstance(module, nn.Module):
        raise TypeError("DSV4 implementation must be a torch.nn.Module")


def build_model(*, build_ctx, request, **kwargs):
    from rtp_llm.models_py.model_desc.deepseek_v4_model import DeepSeekV4Model

    validate_arguments(build_ctx, request, kwargs)
    return DeepSeekV4Model(module_build_context=build_ctx, **kwargs)


def build_block(*, build_ctx, request, **kwargs):
    from rtp_llm.models_py.modules.dsv4.block import Block

    validate_arguments(build_ctx, request, kwargs)
    return Block(module_build_context=build_ctx, **kwargs)


def build_attention(*, build_ctx, request, platform_provider=None, **kwargs):
    from rtp_llm.models_py.modules.dsv4.fp8.attention import AttentionFP8

    validate_arguments(build_ctx, request, kwargs)
    if platform_provider is not None:
        return platform_provider.build_attention(AttentionFP8, **kwargs)
    return AttentionFP8(**kwargs)


def build_moe(*, build_ctx, request, platform_provider=None, **kwargs):
    from rtp_llm.models_py.modules.dsv4.moe.moe_layer import MoE

    validate_arguments(build_ctx, request, kwargs)
    if platform_provider is not None:
        return platform_provider.build_moe(MoE, **kwargs)
    return MoE(**kwargs)
